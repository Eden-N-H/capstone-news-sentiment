"""
FinBERT sentiment scoring for the cleaned news corpus.

Model choice: ProsusAI/finbert, a BERT model fine-tuned on financial text.
Benchmarks put it around 85-93% F1 on financial sentiment, versus roughly 50%
for lexicon methods like Loughran-McDonald, which cannot tell that "beats
estimates" is positive. Instruction-tuned LLMs score a few points higher but
cost 300-500x more per document — not a trade worth making for ~39,000
articles on a laptop. FinBERT is also a BERT variant, so it stays consistent
with what the proposal specifies.

FULL ARTICLE BODIES
-------------------
FinBERT accepts 512 tokens; articles run well past that. Each article is split
into overlapping windows, every window scored, and the results recombined as a
token-weighted mean — so a long article is not reduced to its first paragraph,
and a passing negative aside in a positive article does not flip the verdict.

The output score is P(positive) - P(negative), in [-1, 1]. Neutral probability
is kept as a separate column: a strongly neutral article and a genuinely mixed
one both score near zero, and they are not the same thing.

RESUMABLE
---------
Scores are appended to a JSONL cache after every batch. Re-running skips
articles already scored, so an interrupted run costs one batch.

Requirements:
    pip install torch transformers pandas

Run:
    python sentiment_score.py                      # news corpus
    python sentiment_score.py --source reddit      # Reddit corpus
    python sentiment_score.py --source reddit --only NVDA
    python sentiment_score.py --text title         # headlines instead of bodies
"""

import argparse
import json
import os

import pandas as pd

from config import END_DATE_STR, REPORT_DIR, START_DATE_STR, select_stocks, ticker_dir

MODEL_NAME = "ProsusAI/finbert"

# The two corpora use different column names and live in different files, but
# are scored by the SAME model on purpose. The research question compares
# platforms, so holding the scorer constant means a difference in results is a
# difference between platforms rather than between models. Swapping in a
# social-media-tuned model for Reddit would confound exactly the comparison
# being made. (Worth running as a robustness check separately, not as the
# primary scorer.)
SOURCES = {
    "news": {
        "input": "{sym}_news_clean_{start}_{end}.csv",
        "output": "{sym}_news_scored_{start}_{end}.csv",
        "cache": "{sym}_sentiment_cache.jsonl",
        "title": "Article_title",
        "body": ["Article", "Lsa_summary"],
        "id": "Url",
        "needs": "news_quality_filter.py",
    },
    "reddit": {
        "input": "{sym}_reddit_clean_{start}_{end}.csv",
        "output": "{sym}_reddit_scored_{start}_{end}.csv",
        "cache": "{sym}_reddit_sentiment_cache.jsonl",
        "title": "title",
        "body": ["selftext"],
        # Reddit post ids are globally unique and stable, so they make a
        # better cache key than a URL.
        "id": "post_id",
        "needs": "reddit_quality_filter.py",
    },
}

MAX_TOKENS = 512
# 510 leaves room for [CLS] and [SEP].
WINDOW = 510
# Overlap so a sentence split across a boundary is still seen intact once.
STRIDE = 64

BATCH_SIZE = 32
SAVE_EVERY_BATCHES = 20


def pick_device(requested: str):
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():      # Apple silicon
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def choose_text_column(df: pd.DataFrame, preference: str, spec: dict) -> str:
    """
    Resolve which column to score, falling back if the preferred one is absent.

    FNSPID builds vary in which columns they ship, so this checks rather than
    assumes — and reports what it picked, because scoring headlines when you
    meant to score bodies is a silent and serious difference.
    """
    order = {
        "body": list(spec["body"]) + [spec["title"]],
        "summary": list(spec["body"]) + [spec["title"]],
        "title": [spec["title"]],
    }[preference]

    for col in order:
        if col in df.columns and df[col].notna().any():
            return col
    raise SystemExit(f"None of {order} present with data. Columns: {list(df.columns)}")


def build_text(df: pd.DataFrame, col: str, title_col: str = "Article_title") -> pd.Series:
    """
    Title plus body when scoring bodies.

    Headlines carry disproportionate sentiment signal and are the part a reader
    reacts to, so prepending the title rather than discarding it is worth the
    handful of extra tokens.
    """
    body = df[col].fillna("").astype(str).str.strip()
    if col == title_col or title_col not in df.columns:
        return body

    title = df[title_col].fillna("").astype(str).str.strip()
    # Join only where both sides have content — otherwise two empty fields
    # produce a bare ". ", which is not empty and gets confidently scored.
    joined = title.where(body == "", title + ". " + body)
    joined = joined.where(title != "", body)
    return joined.str.strip()


def chunk_ids(ids, window=WINDOW, stride=STRIDE):
    """Split a token-id list into overlapping windows."""
    if len(ids) <= window:
        return [ids]
    out = []
    start = 0
    step = window - stride
    while start < len(ids):
        out.append(ids[start:start + window])
        if start + window >= len(ids):
            break
        start += step
    return out


def score_articles(df, text_col, tokenizer, model, device, cache_path,
                   done_ids, batch_size=BATCH_SIZE, title_col="Article_title"):
    """
    Score every unscored article. Returns nothing; results stream to the cache.

    Chunks from many articles are batched together rather than one article at a
    time, because a batch of 32 short chunks costs about the same as a batch of
    one on GPU/MPS.
    """
    import torch

    # Special-token ids read off the tokenizer rather than via
    # build_inputs_with_special_tokens(), which transformers 5.x removed.
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def wrap(chunk):
        ids = list(chunk)
        if cls_id is not None:
            ids = [cls_id] + ids
        if sep_id is not None:
            ids = ids + [sep_id]
        return ids

    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    pos_i = labels.index("positive")
    neg_i = labels.index("negative")
    neu_i = labels.index("neutral")

    texts = build_text(df, text_col, title_col)
    todo = [(i, t) for i, t in zip(df["_uid"], texts) if i not in done_ids]
    print(f"  {len(todo):,} to score ({len(done_ids):,} already cached)")
    if not todo:
        return

    # Flatten to (uid, chunk_ids) so batches are uniform in size.
    flat = []
    empty = []
    for uid, text in todo:
        # Punctuation-only strings survive .strip() but carry no signal.
        has_content = bool(text) and any(ch.isalnum() for ch in text)
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=False) if has_content else []
        if not ids:
            # An empty body fed to the model as a bare [CLS][SEP] still gets a
            # confident label — meaningless. Record it as null so it reads as
            # "no text" downstream rather than as sentiment.
            empty.append(uid)
            continue
        for c in chunk_ids(ids):
            flat.append((uid, c))

    if empty:
        print(f"  {len(empty):,} articles have no text — recorded as null")

    print(f"  {len(flat):,} chunks (mean {len(flat)/max(len(todo),1):.1f} per article)")

    acc = {}          # uid -> [weighted pos, neg, neu, total weight]
    fh = open(cache_path, "a", encoding="utf-8")
    for uid in empty:
        fh.write(json.dumps({"_uid": uid, "prob_positive": None,
                             "prob_negative": None, "prob_neutral": None,
                             "sentiment": None}) + "\n")
    batches_done = 0

    try:
        for start in range(0, len(flat), batch_size):
            batch = flat[start:start + batch_size]

            # Padding built by hand so this does not depend on tokenizer.pad()
            # or other helpers whose signatures shift between transformers
            # major versions.
            seqs = [wrap(c) for _, c in batch]
            width = max(len(s_) for s_ in seqs)
            input_ids = torch.tensor(
                [s_ + [pad_id] * (width - len(s_)) for s_ in seqs], dtype=torch.long)
            attention_mask = torch.tensor(
                [[1] * len(s_) + [0] * (width - len(s_)) for s_ in seqs], dtype=torch.long)

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

            for (uid, chunk), p in zip(batch, probs):
                w = max(len(chunk), 1)
                a = acc.setdefault(uid, [0.0, 0.0, 0.0, 0.0])
                # float() rather than the numpy scalar: numpy float32 is not
                # JSON-serialisable, and the cache write happens at the very
                # end of a long run — a bad place to discover that.
                a[0] += float(p[pos_i]) * w
                a[1] += float(p[neg_i]) * w
                a[2] += float(p[neu_i]) * w
                a[3] += w

            batches_done += 1
            if batches_done % SAVE_EVERY_BATCHES == 0:
                fh.flush()
                os.fsync(fh.fileno())
                pct = min(start + batch_size, len(flat)) / len(flat) * 100
                print(f"\r  {pct:5.1f}%  ({min(start+batch_size, len(flat)):,}/{len(flat):,} chunks)",
                      end="", flush=True)

        print()
        # An article is only written once all of its chunks are in.
        for uid, (p, n, u, w) in acc.items():
            rec = {
                "_uid": uid,
                "prob_positive": round(p / w, 6),
                "prob_negative": round(n / w, 6),
                "prob_neutral": round(u / w, 6),
                "sentiment": round((p - n) / w, 6),
            }
            fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    finally:
        fh.close()


def load_cache(path):
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[rec["_uid"]] = rec
    return out


def main():
    ap = argparse.ArgumentParser(description="Score the cleaned news corpus with FinBERT.")
    ap.add_argument("--only", nargs="*", help="Restrict to these tickers")
    ap.add_argument("--source", choices=list(SOURCES), default="news",
                    help="Which corpus to score (default news)")
    ap.add_argument("--text", choices=["body", "summary", "title"], default="body",
                    help="Which text to score (default body)")
    ap.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--restart", action="store_true", help="Discard cached scores")
    args = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import logging as hf_logging

    # Encoding without truncation is intentional — chunking is done here — so
    # the "sequence longer than 512" warning is noise, once per long article.
    hf_logging.set_verbosity_error()

    device = pick_device(args.device)
    print(f"Device: {device}")
    print(f"Loading {MODEL_NAME} (first run downloads ~440 MB)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval().to(device)
    print(f"Labels: {model.config.id2label}")

    spec = SOURCES[args.source]
    stocks = select_stocks(args.only)
    summary = []

    for stock in stocks:
        fmt = dict(sym=stock.symbol, start=START_DATE_STR, end=END_DATE_STR)
        src = os.path.join(ticker_dir(stock.symbol), spec["input"].format(**fmt))
        if not os.path.exists(src):
            print(f"\n{stock.symbol}: {src} missing — run {spec['needs']} first")
            continue

        print(f"\n=== {stock.symbol} ({args.source}) ===")
        df = pd.read_csv(src)
        if df.empty:
            print("  no rows")
            continue

        text_col = choose_text_column(df, args.text, spec)
        print(f"  scoring column: {text_col}")

        # Stable per-row id so the cache survives re-runs, reordering, and
        # adding more subreddits later.
        id_col = spec["id"] if spec["id"] in df.columns else None
        df["_uid"] = (df[id_col].fillna("").astype(str) if id_col
                      else df.index.astype(str))
        dupes = df["_uid"].duplicated() | (df["_uid"] == "")
        if dupes.any():
            df.loc[dupes, "_uid"] = (df.loc[dupes, "_uid"] + "#"
                                     + df.index[dupes].astype(str))

        cache_path = os.path.join(ticker_dir(stock.symbol), spec["cache"].format(**fmt))
        if args.restart and os.path.exists(cache_path):
            os.remove(cache_path)

        cached = load_cache(cache_path)
        score_articles(df, text_col, tokenizer, model, device, cache_path,
                       set(cached), args.batch_size, title_col=spec["title"])

        cached = load_cache(cache_path)
        scores = pd.DataFrame(cached.values())
        out = df.merge(scores, on="_uid", how="left") if not scores.empty else df

        dst = os.path.join(ticker_dir(stock.symbol), spec["output"].format(**fmt))
        out.to_csv(dst, index=False)

        got = out["sentiment"].notna().sum() if "sentiment" in out.columns else 0
        mean = float(out["sentiment"].mean()) if got else float("nan")
        print(f"  scored {got:,}/{len(out):,}   mean {mean:+.4f}")
        print(f"  -> {dst}")

        summary.append({
            "ticker": stock.symbol,
            "source": args.source,
            "items": len(out),
            "scored": int(got),
            "mean_sentiment_per_document": round(mean, 4) if got else None,
            "pct_positive": round(float((out["sentiment"] > 0.1).mean()) * 100, 1) if got else None,
            "pct_negative": round(float((out["sentiment"] < -0.1).mean()) * 100, 1) if got else None,
            "text_column": text_col,
        })

    if summary:
        rep = pd.DataFrame(summary)
        path = os.path.join(REPORT_DIR, f"sentiment_summary_{args.source}.csv")
        rep.to_csv(path, index=False)
        print(f"\n=== Sentiment summary ===\n{rep.to_string(index=False)}")
        print(f"Saved -> {path}")


if __name__ == "__main__":
    main()