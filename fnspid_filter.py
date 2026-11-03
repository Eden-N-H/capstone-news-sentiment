"""
FNSPID download + filter -> all selected NASDAQ tickers in one pass.

Differences from the single-ticker version:
  * Streams the 15.7M-row news file ONCE and splits it across every ticker,
    instead of re-reading a ~5 GB CSV per stock.
  * Optionally does a title/summary keyword scan in the same pass, which picks
    up articles that discuss a company without carrying its symbol tag.
  * Pulls each ticker's price CSV out of full_history.zip using an exact
    filename-stem match (substring matching would have made "AFI" collide with
    dozens of unrelated files).
  * Writes a per-ticker coverage report so gaps are visible before modelling.

Requirements:
    pip install requests pandas

Run:
    python fnspid_filter.py                 # everything
    python fnspid_filter.py --only NVDA AAPL
    python fnspid_filter.py --skip-keyword-scan
"""

import argparse
import os
import zipfile

import pandas as pd

from checkpoint import Checkpoint, download_file

from config import (
    DATA_DIR,
    END_DATE_STR,
    FNSPID_SYMBOL_MAP,
    RAW_DIR,
    REPORT_DIR,
    START_DATE_STR,
    select_stocks,
    ticker_dir,
)

NEWS_URL = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv"
PRICE_ZIP_URL = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_price/full_history.zip"

NEWS_RAW_PATH = os.path.join(RAW_DIR, "nasdaq_exteral_data.csv")
PRICE_ZIP_PATH = os.path.join(RAW_DIR, "full_history.zip")

CHUNK_SIZE = 200_000

# Columns worth keeping if the file has them. The header is inspected first, so
# a schema change upstream degrades gracefully instead of raising on usecols.
WANTED_COLUMNS = [
    "Date",
    "Article_title",
    "Stock_symbol",
    "Url",
    "Publisher",
    "Author",
    "Article",
    "Lsa_summary",
    "Luhn_summary",
    "Textrank_summary",
    "Lexrank_summary",
]

# The full article body roughly triples the on-disk size of the filtered output.
# Keep it: BERT sentiment over a headline alone is a much weaker signal than
# over the article text, and after filtering it is only a few hundred MB.
INCLUDE_ARTICLE_TEXT = True

CKPT_PATH = os.path.join(REPORT_DIR, "_filter_checkpoint.json")
LOCK_PATH = os.path.join(REPORT_DIR, "_filter.lock")
SAVE_EVERY_CHUNKS = 10


def resolve_columns(path: str) -> list:
    """Intersect WANTED_COLUMNS with what the file actually has."""
    header = pd.read_csv(path, nrows=0)
    available = [c for c in WANTED_COLUMNS if c in header.columns]
    if not INCLUDE_ARTICLE_TEXT and "Article" in available:
        available.remove("Article")
    missing = [c for c in WANTED_COLUMNS if c not in header.columns]
    if missing:
        print(f"  note: columns absent from this FNSPID build: {', '.join(missing)}")
    return available


def build_keyword_masks(stocks):
    """
    Regex per stock for the keyword scan.

    Word boundaries matter: without them "GS" matches inside "GSK" and "Apple"
    matches "Applebee's".
    """
    import re

    masks = {}
    for s in stocks:
        if not s.title_keywords:
            continue
        pattern = "|".join(rf"\b{re.escape(k)}\b" for k in s.title_keywords)
        masks[s.symbol] = re.compile(pattern, re.IGNORECASE)
    return masks


def non_ascii_ratio(series):
    """
    Rough flag for non-English articles.

    The corpus contains Russian and German items, which an English sentiment
    model will score as noise. This is a cheap proxy, not language detection —
    it catches non-Latin scripts reliably and misses German entirely. Treat it
    as a first pass; use langdetect during preprocessing if it matters.
    """
    return series.fillna("").map(
        lambda t: sum(1 for c in str(t) if ord(c) > 127) / max(len(str(t)), 1)
    )


class RunLock:
    """
    Refuse to start if another copy of this script is already running.

    Two concurrent runs append to the same partial CSVs, interleaving at chunk
    boundaries — that corrupts rows rather than merely duplicating them, and
    deduplication afterwards will not repair it. Easy to trigger by accident
    when a shell aborts a command line and you retype it.
    """

    def __init__(self, path):
        self.path = path
        self.acquired = False

    def __enter__(self):
        if os.path.exists(self.path):
            try:
                pid = int(open(self.path).read().strip())
            except (ValueError, OSError):
                pid = None
            if pid and self._alive(pid):
                raise SystemExit(
                    f"Another fnspid_filter.py is already running (pid {pid}).\n"
                    f"Wait for it, or stop it with:  kill {pid}\n"
                    f"If you are sure it is dead:    rm {self.path}"
                )
            print(f"  clearing stale lock from pid {pid}")
            os.remove(self.path)

        with open(self.path, "w") as f:
            f.write(str(os.getpid()))
        self.acquired = True
        return self

    def __exit__(self, *exc):
        if self.acquired and os.path.exists(self.path):
            os.remove(self.path)

    @staticmethod
    def _alive(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


def part_path(symbol: str) -> str:
    """Where in-progress matches for one ticker are appended."""
    return os.path.join(ticker_dir(symbol), f"{symbol}_news_partial.csv")


def filter_news(stocks, keyword_scan: bool = True, resume: bool = True) -> dict:
    """
    One pass over the news CSV, fanning rows out to each ticker.

    Matches are appended to a per-ticker partial CSV as they are found rather
    than accumulated in memory — with full article bodies for six tickers over
    five years, holding everything in RAM until the end is how this dies on a
    laptop. It also means an interrupted run keeps what it already had.

    Rows carry a `match_type` of "symbol" (Stock_symbol tag) or "keyword" (text
    match only) so the two can be kept apart during preprocessing; keyword hits
    are noisier.
    """
    download_file(NEWS_URL, NEWS_RAW_PATH)
    usecols = resolve_columns(NEWS_RAW_PATH)

    symbol_map = {a: c for a, c in FNSPID_SYMBOL_MAP.items()
                  if c in {s.symbol for s in stocks}}
    keyword_masks = build_keyword_masks(stocks) if keyword_scan else {}
    text_cols = [c for c in ("Article_title", "Lsa_summary") if c in usecols]

    ckpt = Checkpoint(CKPT_PATH)
    state = ckpt.load() if resume else {}
    rows_done = int(state.get("rows_done", 0))

    if not resume:
        for s in stocks:
            if os.path.exists(part_path(s.symbol)):
                os.remove(part_path(s.symbol))

    if rows_done:
        print(f"Resuming filter from row {rows_done:,}")

    print(f"Filtering news {START_DATE_STR} -> {END_DATE_STR}")
    print(f"  symbol tags: {sorted(symbol_map)}")
    if keyword_masks:
        print(f"  keyword scan: {sorted(keyword_masks)}")

    start_ts = pd.Timestamp(START_DATE_STR, tz="UTC")
    end_ts = pd.Timestamp(END_DATE_STR, tz="UTC") + pd.Timedelta(days=1)

    def append(symbol, frame):
        path = part_path(symbol)
        frame.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

    read_kwargs = dict(chunksize=CHUNK_SIZE, usecols=usecols, low_memory=False)
    if rows_done:
        read_kwargs["skiprows"] = range(1, rows_done + 1)

    scanned = rows_done
    found = int(state.get("found", 0))
    chunk_no = 0

    try:
        for chunk in pd.read_csv(NEWS_RAW_PATH, **read_kwargs):
            scanned += len(chunk)
            chunk_no += 1

            chunk["Date"] = pd.to_datetime(chunk["Date"], errors="coerce",
                                           utc=True, format="mixed")
            chunk = chunk[(chunk["Date"] >= start_ts) & (chunk["Date"] < end_ts)]

            if not chunk.empty:
                sym = chunk["Stock_symbol"].astype("string").str.upper().str.strip()

                # Title and summary are matched separately. A keyword in the
                # headline usually means the article is about that company; a
                # keyword in the summary often means a passing mention, and
                # measured on a sample roughly 70-80% of summary-only matches
                # for GOOG were about some other company entirely.
                title_txt = (chunk["Article_title"].astype("string").fillna("")
                             if "Article_title" in chunk.columns else None)
                summary_cols = [c for c in ("Lsa_summary", "Luhn_summary")
                                if c in chunk.columns]
                summary_txt = None
                if summary_cols:
                    summary_txt = chunk[summary_cols[0]].astype("string").fillna("")

                for alias, canonical in symbol_map.items():
                    hit = chunk[sym == alias]
                    if not hit.empty:
                        hit = hit.copy()
                        hit["match_type"] = "symbol"
                        hit["matched_alias"] = alias
                        append(canonical, hit)
                        found += len(hit)

                for canonical, rx in keyword_masks.items():
                    own = [a for a, c in symbol_map.items() if c == canonical]
                    not_tagged = ~sym.isin(own)

                    in_title = (title_txt.str.contains(rx, na=False)
                                if title_txt is not None else None)
                    in_summary = (summary_txt.str.contains(rx, na=False)
                                  if summary_txt is not None else None)

                    if in_title is not None:
                        hit = chunk[in_title & not_tagged]
                        if not hit.empty:
                            hit = hit.copy()
                            hit["match_type"] = "keyword_title"
                            hit["matched_alias"] = ""
                            append(canonical, hit)
                            found += len(hit)

                    if in_summary is not None:
                        only_summary = in_summary & not_tagged
                        if in_title is not None:
                            only_summary = only_summary & ~in_title
                        hit = chunk[only_summary]
                        if not hit.empty:
                            hit = hit.copy()
                            hit["match_type"] = "keyword_summary"
                            hit["matched_alias"] = ""
                            append(canonical, hit)
                            found += len(hit)

            if chunk_no % SAVE_EVERY_CHUNKS == 0:
                ckpt.save({"rows_done": scanned, "found": found})

            print(f"\r  scanned {scanned:,} rows, {found:,} matches so far",
                  end="", flush=True)
    except KeyboardInterrupt:
        ckpt.save({"rows_done": scanned, "found": found})
        print(f"\n\nInterrupted at row {scanned:,}. Re-run to resume from here.")
        raise

    print()
    ckpt.clear()

    # Consolidate the partials: dedupe (an article can be tagged GOOG and
    # GOOGL, or hit both the symbol and keyword passes) and sort by date.
    out = {}
    for s in stocks:
        path = part_path(s.symbol)
        if not os.path.exists(path):
            out[s.symbol] = pd.DataFrame()
            print(f"  {s.symbol}: 0 articles")
            continue
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True, format="mixed")
        # Keep the most trustworthy tier when the same article matched twice.
        rank = {"symbol": 0, "keyword_title": 1, "keyword_summary": 2}
        df["_rank"] = df["match_type"].map(rank).fillna(3)
        subset = ["Url"] if "Url" in df.columns else ["Date", "Article_title"]
        df = df.sort_values("_rank").drop_duplicates(subset=subset, keep="first")
        df = df.drop(columns="_rank")

        if "Article_title" in df.columns:
            df["non_ascii_ratio"] = non_ascii_ratio(df["Article_title"]).round(3)

        df = df.sort_values("Date").reset_index(drop=True)
        out[s.symbol] = df
        tiers = df["match_type"].value_counts()
        print(f"  {s.symbol}: {len(df):,} articles  "
              f"(symbol {tiers.get('symbol', 0):,}, "
              f"title {tiers.get('keyword_title', 0):,}, "
              f"summary {tiers.get('keyword_summary', 0):,})")
    return out


def list_price_members(zf: zipfile.ZipFile) -> dict:
    """Map upper-cased filename stem -> member path."""
    members = {}
    for name in zf.namelist():
        if not name.lower().endswith(".csv"):
            continue
        stem = os.path.splitext(os.path.basename(name))[0].upper()
        members[stem] = name
    return members


def extract_prices(stocks) -> dict:
    """Pull each NASDAQ ticker's CSV out of full_history.zip."""
    wanted = [s for s in stocks if s.fnspid_price_files]
    if not wanted:
        return {}

    download_file(PRICE_ZIP_URL, PRICE_ZIP_PATH)
    out = {}

    with zipfile.ZipFile(PRICE_ZIP_PATH) as zf:
        members = list_price_members(zf)
        for s in wanted:
            frames = []
            for stem in s.fnspid_price_files:
                member = members.get(stem.upper())
                if member is None:
                    print(f"  {s.symbol}: no '{stem}.csv' in full_history.zip — skipping")
                    continue
                with zf.open(member) as f:
                    df = pd.read_csv(f)
                df.columns = [c.strip().lower() for c in df.columns]
                if "date" not in df.columns:
                    df = df.rename(columns={df.columns[0]: "date"})
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
                df["source_file"] = stem.upper()
                frames.append(df)

            if not frames:
                out[s.symbol] = pd.DataFrame()
                continue

            # Where a ticker has two share classes, keep the primary listing
            # (first entry in fnspid_price_files) as the price series.
            df = frames[0]
            df = df[(df["date"] >= pd.Timestamp(START_DATE_STR))
                    & (df["date"] <= pd.Timestamp(END_DATE_STR))]
            df = df.sort_values("date").reset_index(drop=True)
            out[s.symbol] = df
            print(f"  {s.symbol}: {len(df):,} trading days")
    return out


def coverage_report(news: dict, prices: dict) -> pd.DataFrame:
    """Monthly article counts per ticker — the fastest way to spot dead spans."""
    rows = []
    for symbol, df in news.items():
        if df.empty:
            rows.append({"ticker": symbol, "articles": 0, "months_covered": 0,
                         "first_article": None, "last_article": None,
                         "price_days": len(prices.get(symbol, []))})
            continue
        months = df["Date"].dt.tz_convert(None).dt.to_period("M").nunique()
        rows.append({
            "ticker": symbol,
            "articles": len(df),
            "months_covered": months,
            "first_article": df["Date"].min().date(),
            "last_article": df["Date"].max().date(),
            "price_days": len(prices.get(symbol, [])),
        })
    return pd.DataFrame(rows)


def monthly_report(news: dict) -> pd.DataFrame:
    """
    Articles per ticker per month, split by how they were matched.

    The census only reports how many months a symbol appears in, which cannot
    distinguish a single contiguous 37-month hole (fatal for a daily model)
    from scattered quiet weeks (survivable). It also counts only symbol-tagged
    rows, so it understates any ticker whose coverage the keyword scan fills in.
    This table settles both questions.
    """
    frames = []
    for symbol, df in news.items():
        if df.empty:
            continue
        tmp = df.copy()
        tmp["month"] = tmp["Date"].dt.strftime("%Y-%m")
        counts = (tmp.groupby(["month", "match_type"]).size()
                  .unstack(fill_value=0).reset_index())
        for col in ("symbol", "keyword_title", "keyword_summary"):
            if col not in counts.columns:
                counts[col] = 0
        counts["keyword"] = counts["keyword_title"] + counts["keyword_summary"]
        counts["total"] = counts["symbol"] + counts["keyword"]
        counts.insert(0, "ticker", symbol)
        frames.append(counts[["ticker", "month", "symbol", "keyword", "total"]])

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    # Reindex onto every month in the window so gaps show up as explicit zero
    # rows rather than as absent rows you have to notice are missing.
    all_months = pd.period_range(START_DATE_STR, END_DATE_STR, freq="M").astype(str)
    filled = []
    num_cols = ["symbol", "keyword", "total"]
    for ticker in out["ticker"].unique():
        sub = (out[out["ticker"] == ticker]
               .set_index("month")[num_cols]
               .reindex(all_months, fill_value=0))
        sub.index.name = "month"
        sub = sub.reset_index()
        sub.insert(0, "ticker", ticker)
        filled.append(sub)
    return pd.concat(filled, ignore_index=True)


def publisher_report(news: dict) -> pd.DataFrame:
    """
    Which outlets the corpus actually contains, per ticker.

    Section 3.2.3 of the proposal treats each platform as its own model input.
    FNSPID is a mixed-publisher corpus, so this table is what determines whether
    a per-platform split is viable or whether it collapses to "financial news"
    as a single source.
    """
    frames = []
    for symbol, df in news.items():
        if df.empty or "Publisher" not in df.columns:
            continue
        counts = df["Publisher"].fillna("(unknown)").value_counts().reset_index()
        counts.columns = ["publisher", "articles"]
        counts.insert(0, "ticker", symbol)
        frames.append(counts)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description="Filter FNSPID down to the capstone tickers.")
    ap.add_argument("--only", nargs="*", help="Restrict to these tickers, e.g. --only NVDA AAPL")
    ap.add_argument("--skip-keyword-scan", action="store_true",
                    help="Symbol-tag matching only — faster, but misses untagged coverage")
    ap.add_argument("--restart", action="store_true",
                    help="Ignore the checkpoint and partial files; rescan from row 0")
    args = ap.parse_args()

    stocks = select_stocks(args.only)
    print(f"Tickers: {', '.join(s.symbol for s in stocks)}\n")

    with RunLock(LOCK_PATH):
        run(stocks, args)


def run(stocks, args):
    news = filter_news(stocks, keyword_scan=not args.skip_keyword_scan,
                       resume=not args.restart)
    for symbol, df in news.items():
        path = os.path.join(ticker_dir(symbol), f"{symbol}_news_{START_DATE_STR}_{END_DATE_STR}.csv")
        df.to_csv(path, index=False)
        print(f"Saved -> {path}")
        if os.path.exists(part_path(symbol)):
            os.remove(part_path(symbol))

    print("\nExtracting price history...")
    prices = extract_prices(stocks)
    for symbol, df in prices.items():
        path = os.path.join(ticker_dir(symbol), f"{symbol}_prices_{START_DATE_STR}_{END_DATE_STR}.csv")
        df.to_csv(path, index=False)
        print(f"Saved -> {path}")

    cov = coverage_report(news, prices)
    cov_path = os.path.join(REPORT_DIR, "fnspid_coverage.csv")
    cov.to_csv(cov_path, index=False)
    print("\n=== Coverage ===")
    print(cov.to_string(index=False))
    print(f"Saved -> {cov_path}")

    monthly = monthly_report(news)
    if not monthly.empty:
        mpath = os.path.join(REPORT_DIR, "fnspid_monthly_counts.csv")
        monthly.to_csv(mpath, index=False)
        print(f"Saved -> {mpath}")

        print("\n=== Month coverage (of 60) ===")
        for symbol, group in monthly.groupby("ticker"):
            nonzero = (group["total"] > 0).sum()
            kw = int(group["keyword"].sum())
            # Longest run of consecutive empty months — the number that decides
            # whether a ticker is usable, not the total count of empty months.
            longest, run = 0, 0
            for v in group["total"]:
                run = run + 1 if v == 0 else 0
                longest = max(longest, run)
            print(f"  {symbol:5} {nonzero:2}/60 months  "
                  f"longest gap {longest:2} months  "
                  f"({kw:,} of {int(group['total'].sum()):,} via keyword scan)")

    pub = publisher_report(news)
    if not pub.empty:
        pub_path = os.path.join(REPORT_DIR, "fnspid_publishers.csv")
        pub.to_csv(pub_path, index=False)
        print(f"Saved -> {pub_path}")

    missing_prices = [s.symbol for s in stocks if s.symbol not in prices or prices[s.symbol].empty]
    if missing_prices:
        print(f"\nNo FNSPID prices for: {', '.join(missing_prices)} — run market_prices.py "
              f"to fetch these (and to cross-check the NASDAQ series).")


if __name__ == "__main__":
    main()