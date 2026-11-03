"""
Filter Arctic Shift subreddit dumps -> the same per-ticker CSVs as reddit_pull.py.

Why this exists: the Arctic Shift API times out unpredictably. Not by window
width, page size, field set or subreddit size — the same request succeeds and
fails minutes apart. It is a free, single-maintainer service and its own docs
make no uptime promise. That is fine for exploratory queries and unworkable for
a fixed-deadline data collection.

The dumps are the deterministic alternative. They are files: no rate limits, no
timeouts, no retries, and they can be re-processed as often as you like once
downloaded. Slower to start, but the runtime is predictable, which is the
property that actually matters with a week-3 deadline.

GETTING THE DUMPS
-----------------
Per-subreddit downloads (what you want — pick the subreddits from config.py):

    https://arctic-shift.photon-reddit.com/download-tool

Request *submissions* (posts) for each subreddit. Save everything into one
folder. Files may be .jsonl, .jsonl.zst, .ndjson or .zst — all are handled.

For very large subreddits the download tool may refuse and point you at the
monthly dumps instead:

    https://github.com/ArthurHeitmann/arctic_shift  (see the download links)

Those are all-of-Reddit per month and very large; only go there if the
per-subreddit tool will not serve r/wallstreetbets.

Requirements:
    pip install zstandard pandas

Run:
    python reddit_dump_filter.py --dumps ./reddit_dumps
    python reddit_dump_filter.py --dumps ./reddit_dumps --only AAPL NVDA
"""

import argparse
import datetime as dt
import glob
import io
import json
import os
import re

import pandas as pd

from config import (
    END_DATE,
    END_DATE_STR,
    REPORT_DIR,
    START_DATE,
    START_DATE_STR,
    select_stocks,
    ticker_dir,
)

try:
    import zstandard as zstd
except ImportError:
    zstd = None


# ---------------------------------------------------------------------------
# Quality filtering
# ---------------------------------------------------------------------------
# Removed and deleted posts keep their title but lose their body, so they would
# be scored on a headline alone while counting as a full post.
DEAD_BODIES = {"[deleted]", "[removed]", "[deleted by user]", "", "nan"}

# Bots post on a schedule regardless of what the market is doing, so their
# sentiment is constant noise weighted by posting frequency. The author field
# is read for this check and then discarded — never written to the output —
# which keeps the ethics commitment in section 4.1.1 while still allowing the
# filtering.
BOT_AUTHORS = {
    "automoderator", "wsbvotebot", "vote_bot", "stockbot",
    "thestockbot", "remindmebot", "sneakpeekbot", "totesmessenger",
    "b0trank", "gifreversingbot", "nice-scores", "wsbgod",
}

# NOT a bot. "[deleted]" as the author means the ACCOUNT was deleted, which
# says nothing about the post: the title and body are usually intact and the
# content is as valid as any other. Including it here removed 690,715 posts —
# 28% of the corpus — on a criterion unrelated to quality. Counted separately
# now so the distinction stays visible rather than hidden inside "bot".
DELETED_AUTHOR = "[deleted]"

# Recurring threads: same title every day, no event-specific content, and
# hundreds of comments that make them look important by engagement metrics.
ROUTINE_TITLES = [
    r"^daily discussion", r"^weekend discussion", r"^weekly discussion",
    r"^what are your moves tomorrow", r"^daily thread", r"^rate my portfolio",
    r"^daily general discussion", r"^moves tomorrow", r"^weekly thread",
    r"^monthly thread", r"^(mon|tues|wednes|thurs|fri|satur|sun)day\b.*thread$",
    r"^daily advice thread", r"^wallstreetbets\b.*thread", r"^megathread",
    r"^\(?daily\)? discussion thread", r"^options questions safe haven",
]
ROUTINE_RE = re.compile("|".join(f"(?:{p})" for p in ROUTINE_TITLES), re.IGNORECASE)

# Low on purpose. "NVDA 🚀🚀🚀" is 11 characters and is a real expression of
# retail sentiment, even if a crude one. The threshold exists to drop posts
# with no text at all, not to judge how much was said.
MIN_TEXT_CHARS = 5


def open_dump(path):
    """Yield decoded lines from .jsonl, .ndjson, or zstd-compressed variants."""
    if path.endswith(".zst"):
        if zstd is None:
            raise SystemExit("zstandard is not installed. Run: pip install zstandard")
        with open(path, "rb") as fh:
            # max_window_size raised because Reddit dumps use a large window;
            # the default rejects them with "Frame requires too much memory".
            dctx = zstd.ZstdDecompressor(max_window_size=2**31)
            with dctx.stream_reader(fh) as reader:
                for line in io.TextIOWrapper(reader, encoding="utf-8", errors="replace"):
                    yield line
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            yield from fh


def build_matchers(stocks):
    """
    Compile one regex per stock from its search terms.

    Terms are matched with word boundaries so "GTA" does not fire on "gotta"
    and "BLK" does not fire inside a longer token. Quoted phrases from the
    config ("Larry Fink") have their quotes stripped and match as phrases.
    """
    matchers = {}
    for s in stocks:
        terms = set(s.home_terms) | set(s.strict_terms)
        cleaned = [t.strip('"') for t in terms if t.strip('"')]
        pattern = "|".join(rf"\b{re.escape(t)}\b" for t in sorted(cleaned, key=len, reverse=True))
        matchers[s.symbol] = re.compile(pattern, re.IGNORECASE)
    return matchers


def subreddits_for(stocks, use_all=False):
    """
    Map lower-cased subreddit -> tickers that care about it.

    Defaults to the DUMP_SUBREDDITS shortlist. Matching against all 170
    API-era subreddits would report 160 of them as empty, which reads like
    "these subreddits are dead" when it actually means "no dump was
    downloaded" — a distinction worth keeping visible.
    """
    wanted = {}
    for s in stocks:
        subs = s.all_subreddits if use_all else s.dump_subreddits
        for sub in subs:
            wanted.setdefault(sub.lower(), []).append(s.symbol)
    return wanted


def main():
    ap = argparse.ArgumentParser(description="Filter Arctic Shift dumps into per-ticker CSVs.")
    ap.add_argument("--dumps", required=True, help="Folder containing the downloaded dump files")
    ap.add_argument("--only", nargs="*", help="Restrict to these tickers")
    ap.add_argument("--no-quality-filter", action="store_true",
                    help="Keep bots, recurring daily threads and empty posts")
    ap.add_argument("--all-subreddits", action="store_true",
                    help="Match against the full 170-subreddit list rather than "
                         "the DUMP_SUBREDDITS shortlist")
    ap.add_argument("--keep-all-subreddits", action="store_true",
                    help="Do not filter on the subreddit list; match on terms alone")
    args = ap.parse_args()

    stocks = select_stocks(args.only)
    matchers = build_matchers(stocks)
    wanted_subs = subreddits_for(stocks, use_all=args.all_subreddits)

    files = []
    for ext in ("*.jsonl", "*.ndjson", "*.jsonl.zst", "*.zst", "*.json"):
        files.extend(glob.glob(os.path.join(args.dumps, "**", ext), recursive=True))
    files = sorted(set(files))

    if not files:
        raise SystemExit(f"No dump files found in {args.dumps}. See the module docstring "
                         "for where to download them.")

    present = {os.path.basename(f).lower() for f in files}
    missing = [sub for sub in wanted_subs
               if not any(sub in name for name in present)]
    if missing:
        print(f"No dump file found for: {', '.join(sorted(missing))}")
        print("  (filenames are matched loosely; rename if the tool used a "
              "different convention)\n")

    print(f"Found {len(files)} dump file(s)")
    for f in files:
        print(f"  {os.path.basename(f)}  ({os.path.getsize(f)/1e6:.0f} MB)")

    start_ts = int(dt.datetime.combine(START_DATE, dt.time.min, tzinfo=dt.timezone.utc).timestamp())
    end_ts = int(dt.datetime.combine(END_DATE + dt.timedelta(days=1), dt.time.min,
                                     tzinfo=dt.timezone.utc).timestamp())

    # Append as we go rather than accumulating — these files are large and this
    # also means an interrupted run keeps whatever it had already matched.
    handles = {}

    def writer(symbol):
        if symbol not in handles:
            path = os.path.join(ticker_dir(symbol), f"{symbol}_reddit_from_dump.jsonl")
            handles[symbol] = open(path, "w", encoding="utf-8")
        return handles[symbol]

    counts = {s.symbol: 0 for s in stocks}
    stats = {"bot": 0, "routine": 0, "too_short": 0, "deleted_author_kept": 0}
    seen_ids = set()
    scanned = kept = 0

    try:
        for path in files:
            print(f"\nScanning {os.path.basename(path)}")
            for line in open_dump(path):
                line = line.strip()
                if not line:
                    continue
                scanned += 1
                if scanned % 200_000 == 0:
                    print(f"\r  {scanned:,} posts scanned, {kept:,} kept", end="", flush=True)

                try:
                    post = json.loads(line)
                except json.JSONDecodeError:
                    continue

                created = post.get("created_utc")
                if created is None:
                    continue
                created = int(created)
                if not (start_ts <= created < end_ts):
                    continue

                sub = (post.get("subreddit") or "").lower()
                if not args.keep_all_subreddits and sub not in wanted_subs:
                    continue

                if not args.no_quality_filter:
                    author = str(post.get("author") or "").lower()
                    if author in BOT_AUTHORS:
                        stats["bot"] += 1
                        continue
                    if author == DELETED_AUTHOR:
                        stats["deleted_author_kept"] += 1

                    title = str(post.get("title") or "").strip()
                    if ROUTINE_RE.search(title):
                        stats["routine"] += 1
                        continue

                    body = str(post.get("selftext") or "").strip()
                    if body.lower() in DEAD_BODIES:
                        body = ""
                    # A link post with no body is still a valid utterance — its
                    # title carries the claim — so only drop it when the title
                    # is too short to say anything either.
                    if len(title) + len(body) < MIN_TEXT_CHARS:
                        stats["too_short"] += 1
                        continue
                    post["selftext"] = body

                blob = f"{post.get('title', '')} {post.get('selftext', '')}"
                candidates = (wanted_subs.get(sub, [])
                              if not args.keep_all_subreddits else list(counts))

                # Crossposts and re-scans repeat post ids; a duplicate would
                # be counted twice in the daily aggregate.
                pid = post.get("id")
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)

                for symbol in candidates:
                    if matchers[symbol].search(blob):
                        # Same schema reddit_pull.py produces, so everything
                        # downstream works unchanged. No author fields.
                        row = {
                            "ticker": symbol,
                            "platform": "reddit",
                            "subreddit": post.get("subreddit"),
                            "post_id": post.get("id"),
                            "created_utc": created,
                            "date": dt.datetime.fromtimestamp(
                                created, tz=dt.timezone.utc).isoformat(),
                            "title": post.get("title", ""),
                            "selftext": post.get("selftext", ""),
                            "score": post.get("score"),
                            "num_comments": post.get("num_comments"),
                            "flair": post.get("link_flair_text"),
                            "over_18": post.get("over_18"),
                            "url": post.get("url"),
                        }
                        writer(symbol).write(json.dumps(row, ensure_ascii=False) + "\n")
                        counts[symbol] += 1
                        kept += 1
            print()
    finally:
        for h in handles.values():
            h.close()

    print(f"\nScanned {scanned:,} posts, kept {kept:,}")
    if any(stats.values()):
        print(f"  filtered out: {stats['bot']:,} bot posts, "
              f"{stats['routine']:,} recurring threads, "
              f"{stats['too_short']:,} empty/too short")
        if stats["deleted_author_kept"]:
            print(f"  kept {stats['deleted_author_kept']:,} posts whose author "
                  f"account was deleted (content intact)")

    summary = []
    for s in stocks:
        path = os.path.join(ticker_dir(s.symbol), f"{s.symbol}_reddit_from_dump.jsonl")
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"  {s.symbol}: no matches")
            summary.append({"ticker": s.symbol, "posts": 0, "subreddits": 0})
            continue

        df = pd.read_json(path, lines=True).drop_duplicates(subset=["post_id"])
        df = df.sort_values("created_utc")
        out = os.path.join(ticker_dir(s.symbol),
                           f"{s.symbol}_reddit_{START_DATE_STR}_{END_DATE_STR}.csv")
        df.to_csv(out, index=False)
        print(f"  {s.symbol}: {len(df):,} posts -> {out}")
        summary.append({"ticker": s.symbol, "posts": len(df),
                        "subreddits": df["subreddit"].nunique()})

    sdf = pd.DataFrame(summary)
    spath = os.path.join(REPORT_DIR, "reddit_dump_summary.csv")
    sdf.to_csv(spath, index=False)
    print(f"\nSaved -> {spath}")


if __name__ == "__main__":
    main()