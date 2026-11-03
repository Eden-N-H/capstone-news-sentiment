"""
Arctic Shift -> full Reddit post pull for every selected stock.

This retrieves the text needed for sentiment analysis. It walks
/api/posts/search with an ascending cursor, one quarter at a time, paginating
until the window is exhausted.

Three things worth knowing about the request shape:

  * `fields` deliberately omits `author` and `author_fullname`. Section 4.1.1 of
    the proposal commits to storing only publication content and a timestamp,
    so the identifying fields are never fetched rather than fetched and dropped.
  * `query` searches title and selftext, and Arctic Shift only supports keyword
    search alongside a subreddit filter — which is why every request is scoped
    to one subreddit.
  * A window is checkpointed as done only if every request inside it succeeded.
    An earlier version recorded the window regardless, so a resumed pull
    silently skipped the quarters that had failed and baked holes into the
    dataset that nothing downstream would flag. Incomplete windows are now left
    unrecorded and listed in reports/reddit_incomplete_windows.csv.

Output is JSONL per ticker, appended and fsynced as it goes, plus a combined
per-ticker CSV at the end. Re-running skips (ticker, subreddit, quarter) triples
already recorded in the progress file, so an interrupted pull resumes — whether
it was Ctrl-C, a dropped connection, or the laptop dying. A retried window
re-appends posts it had already written; the CSV step deduplicates on post_id,
so that is harmless. Network errors get escalating waits up to five minutes
before the run gives up on a window, which covers most wifi outages.

Requirements:
    pip install requests pandas

Run:
    python reddit_pull.py
    python reddit_pull.py --only NVDA --subreddits wallstreetbets nvidia
    python reddit_pull.py --restart          # ignore saved progress
    python reddit_pull.py --only NVDA --subreddits nvidia --max-windows 1   # smoke test
"""

import argparse
import datetime as dt
import json
import os
import time

import pandas as pd
import requests

from config import (
    DATA_DIR,
    END_DATE,
    END_DATE_STR,
    REPORT_DIR,
    START_DATE,
    START_DATE_STR,
    select_stocks,
    ticker_dir,
)
from jsonl_csv import collapse_jsonl_to_csv, count_by_subreddit, date_span

BASE_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"
USER_AGENT = "uts-capstone-sentiment-research/1.0 (academic; contact via UTS)"

# Two-stage retrieval.
#
# Searching is the expensive operation, and asking it for post bodies makes it
# far more expensive — the docs say fewer fields means faster responses, and
# every request that succeeded in testing used a minimal field set while the
# body-carrying ones timed out. So search returns IDs only, and the content is
# fetched afterwards from /api/posts/ids, which is a direct key lookup with no
# text matching and takes 500 IDs per request.
FIELDS_SEARCH = "id,created_utc"

# No author fields — see module docstring.
FIELDS_FULL = ",".join([
    "id",
    "created_utc",
    "title",
    "selftext",
    "subreddit",
    "score",
    "num_comments",
    "url",
    "link_flair_text",
    "over_18",
])

IDS_URL = "https://arctic-shift.photon-reddit.com/api/posts/ids"
IDS_BATCH = 500

REQUEST_DELAY_SEC = 2.0
MAX_RETRIES = 8

# A dropped connection should not end the run. These are the waits between
# attempts for network-level failures, which is what a wifi outage looks like;
# the last few give a laptop time to reconnect on its own.
NETWORK_RETRY_WAITS = (5, 10, 20, 40, 60, 90, 120, 300)
PAGE_LIMIT = 100             # "auto" (up to 1000) makes each request heavier
MAX_PAGES_PER_WINDOW = 500   # circuit breaker

# Timeouts are not predicted by span, page size or subreddit size — measured
# directly, a 90-day query succeeded where the identical 7-day query failed.
# They are server load, and the docs note a second attempt often works. So the
# aim is the fewest possible requests, each retried patiently; splitting would
# add load to the thing that is already overloaded.
SEARCH_SLICE_DAYS = 90

PROGRESS_PATH = os.path.join(DATA_DIR, "_reddit_progress.json")
INCOMPLETE_PATH = os.path.join(REPORT_DIR, "reddit_incomplete_windows.csv")


def quarter_windows(start: dt.date, end: dt.date):
    cur = start
    while cur <= end:
        q_end_month = ((cur.month - 1) // 3 + 1) * 3
        last_day = dt.date(cur.year + (q_end_month == 12), (q_end_month % 12) + 1, 1) - dt.timedelta(days=1)
        window_end = min(last_day, end)
        yield cur, window_end
        cur = window_end + dt.timedelta(days=1)


def months_in_window(start: dt.date, end: dt.date) -> int:
    """Whole months spanned, for the posts-per-month column."""
    return max(1, (end.year - start.year) * 12 + (end.month - start.month) + 1)


def load_progress(restart: bool) -> set:
    if restart or not os.path.exists(PROGRESS_PATH):
        return set()
    with open(PROGRESS_PATH) as f:
        return set(tuple(k) for k in json.load(f))


def save_progress(done: set) -> None:
    tmp = PROGRESS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sorted(list(k) for k in done), f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, PROGRESS_PATH)   # atomic; a kill mid-write cannot truncate it


def request_page(session, params, url=BASE_URL):
    """
    One GET with retry/backoff.

    Returns a list of posts on success, or None if every attempt failed.

    An empty list means the server genuinely had nothing for this query. It
    does NOT mean "the server said no": a 200 carrying
    {"data": null, "error": "Timeout. Maybe slow down a bit"} is the service's
    normal failure mode, and returning [] for it would end pagination early and
    look exactly like a finished window. That case is retried like an HTTP
    error, and exhausts to None.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        pause = NETWORK_RETRY_WAITS[min(attempt - 1, len(NETWORK_RETRY_WAITS) - 1)]

        try:
            resp = session.get(url, params=params, timeout=120)
        except requests.RequestException as e:
            print(f"      {type(e).__name__} (attempt {attempt}/{MAX_RETRIES}), "
                  f"waiting {pause}s — connection may be down")
            time.sleep(pause)
            continue

        if resp.ok:
            # Back off pre-emptively when the budget is nearly spent.
            remaining = resp.headers.get("X-RateLimit-Remaining")
            if remaining is not None and remaining.isdigit() and int(remaining) < 10:
                time.sleep(5)

            try:
                body = resp.json()
            except ValueError:
                print(f"      HTTP 200 with unparseable body (attempt {attempt}/{MAX_RETRIES}), "
                      f"waiting {pause}s")
                time.sleep(pause)
                continue

            data = body.get("data")
            if data is None:
                err = body.get("error")
                if err:
                    print(f"      HTTP 200 but error={err!r} (attempt {attempt}/{MAX_RETRIES}), "
                          f"waiting {pause}s")
                    time.sleep(pause)
                    continue
                return []          # null data, no error: a real empty result
            return data

        if resp.status_code == 429:
            reset = resp.headers.get("X-RateLimit-Reset", "30")
            time.sleep(int(reset) if reset.isdigit() else 30)
            continue

        # A "Timeout" here is usually transient server load — confirmed live,
        # where an identical single-term query succeeded on a subreddit ten
        # times the size of one that had just failed. Worth waiting out.
        print(f"      HTTP {resp.status_code} (attempt {attempt}/{MAX_RETRIES}), "
              f"waiting {pause}s  {resp.text[:200]}")
        time.sleep(pause)

    return None


def _pull_term(session, subreddit, term, after: dt.date, before_ts: int, seen: set):
    """
    Page through one term.

    Returns (posts, complete). `complete` is False if any request in this term
    exhausted its retries or the page cap was hit — meaning the result is a
    floor, not the full set, and the caller must not checkpoint the window.

    The cursor is the created_utc of the last post received. Posts sharing that
    exact second would be re-served forever, so the cursor is nudged forward by
    a second whenever a page adds nothing new.
    """
    cursor = int(dt.datetime.combine(after, dt.time.min, tzinfo=dt.timezone.utc).timestamp())
    collected = []

    for _ in range(MAX_PAGES_PER_WINDOW):
        params = {
            "subreddit": subreddit,
            "query": term,
            "after": cursor,
            "before": before_ts,
            "limit": PAGE_LIMIT,
            "sort": "asc",
            "fields": FIELDS_SEARCH,
        }
        page = request_page(session, params)
        if page is None:
            print(f"      giving up on r/{subreddit} term '{term}' after repeated errors "
                  "— window will NOT be marked done")
            return collected, False
        if not page:
            return collected, True

        fresh = [p for p in page if p.get("id") not in seen]
        for p in fresh:
            seen.add(p["id"])
        collected.extend(fresh)

        last_ts = max(int(p.get("created_utc", cursor)) for p in page)
        cursor = last_ts if fresh else last_ts + 1
        if cursor >= before_ts:
            return collected, True

        time.sleep(REQUEST_DELAY_SEC)

    print(f"      hit the {MAX_PAGES_PER_WINDOW}-page cap on r/{subreddit} term '{term}' "
          "— treating the window as incomplete")
    return collected, False


def pull_window(session, subreddit, terms, after: dt.date, before: dt.date):
    """
    Page through one quarter per search term, ascending by created_utc.

    Returns (posts, complete). One request per term rather than a single
    "A OR B" query: Arctic Shift's docs flag their boolean-syntax section as
    out of date, so a combined query may not be parsed as a union. Single-term
    queries are also cheaper for the server, which is what times out. Results
    are unioned on post ID, so a post matching several terms is stored once.
    """
    seen = set()
    found = []
    complete = True

    for term in terms:
        slice_start = after
        while slice_start <= before:
            slice_end = min(slice_start + dt.timedelta(days=SEARCH_SLICE_DAYS - 1), before)
            slice_before_ts = int(dt.datetime.combine(
                slice_end + dt.timedelta(days=1), dt.time.min,
                tzinfo=dt.timezone.utc).timestamp())
            got, ok = _pull_term(session, subreddit, term, slice_start,
                                 slice_before_ts, seen)
            found.extend(got)
            complete = complete and ok
            slice_start = slice_end + dt.timedelta(days=1)

    posts, ids_ok = fetch_posts_by_id(session, [p["id"] for p in found])
    return posts, (complete and ids_ok)


def fetch_posts_by_id(session, ids):
    """
    Fetch full post content for a list of IDs, 500 at a time.

    Returns (posts, complete). This endpoint does no keyword matching, so it is
    dramatically cheaper than search and should not hit the timeouts that
    search does — but if a batch does fail we have IDs whose text we never
    retrieved, which is a hole like any other, so the window stays open.
    """
    posts = []
    complete = True
    for i in range(0, len(ids), IDS_BATCH):
        batch = ids[i:i + IDS_BATCH]
        params = {"ids": ",".join(batch), "fields": FIELDS_FULL}
        page = request_page(session, params, url=IDS_URL)
        if page is None:
            print(f"      could not fetch content for {len(batch)} posts "
                  "— window will NOT be marked done")
            complete = False
            continue
        posts.extend(page)
        time.sleep(REQUEST_DELAY_SEC)
    return posts, complete


def normalise(posts, stock, subreddit):
    rows = []
    for p in posts:
        created = p.get("created_utc")
        actual_sub = p.get("subreddit", subreddit)
        rows.append({
            "ticker": stock.symbol,
            "platform": "reddit",
            "subreddit": actual_sub,
            # Which term tier matched this post, so precision can be measured
            # per tier later. Matches the column reddit_dump_filter.py writes.
            "scope": stock.scope_for(subreddit),
            "post_id": p.get("id"),
            "created_utc": created,
            "date": dt.datetime.fromtimestamp(int(created), tz=dt.timezone.utc).isoformat()
            if created else None,
            "title": p.get("title", ""),
            "selftext": p.get("selftext", ""),
            "score": p.get("score"),
            "num_comments": p.get("num_comments"),
            "flair": p.get("link_flair_text"),
            "over_18": p.get("over_18"),
            "url": p.get("url"),
        })
    return rows


def per_subreddit_breakdown(stocks, done) -> pd.DataFrame:
    """
    Post counts per (ticker, subreddit), read back from the pulled data.

    Counts are streamed rather than loaded into a DataFrame, and every row is
    labelled `complete` — meaning every quarter in the window is recorded in the
    progress file. Anything else is a floor, not a measurement.

    Completeness is derived from the progress file rather than from this run's
    failures, so a pair that was never attempted at all (because of --only,
    --subreddits or --max-windows) is correctly reported as incomplete instead
    of showing up as a measured zero and inviting a bad prune.
    """
    months = months_in_window(START_DATE, END_DATE)
    quarters = [w_start.isoformat() for w_start, _ in quarter_windows(START_DATE, END_DATE)]
    rows = []
    for stock in stocks:
        jsonl = os.path.join(ticker_dir(stock.symbol), f"{stock.symbol}_reddit_raw.jsonl")
        counts = count_by_subreddit(jsonl) if os.path.exists(jsonl) else {}
        for sub in stock.all_subreddits:
            n = int(counts.get(sub.lower(), 0))
            covered = sum(1 for q in quarters if (stock.symbol, sub, q) in done)
            rows.append({
                "ticker": stock.symbol,
                "subreddit": sub,
                "scope": stock.scope_for(sub),
                "posts": n,
                "posts_per_month": round(n / months, 1),
                "quarters_done": covered,
                "quarters_total": len(quarters),
                "complete": covered == len(quarters),
            })
    df = pd.DataFrame(rows)
    return df.sort_values(["ticker", "posts"], ascending=[True, False]) if not df.empty else df


def main():
    ap = argparse.ArgumentParser(description="Pull Reddit posts for the capstone tickers.")
    ap.add_argument("--only", nargs="*", help="Restrict to these tickers")
    ap.add_argument("--subreddits", nargs="*", help="Restrict to these subreddits")
    ap.add_argument("--restart", action="store_true", help="Ignore saved progress and refetch")
    ap.add_argument("--max-windows", type=int, default=0,
                    help="Stop each subreddit after this many quarters. For smoke tests.")
    args = ap.parse_args()

    stocks = select_stocks(args.only)
    done = load_progress(args.restart)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    summary = []
    incomplete = []      # (ticker, subreddit, quarter_start)

    for stock in stocks:
        subs = [s for s in stock.all_subreddits
                if not args.subreddits or s.lower() in {a.lower() for a in args.subreddits}]
        out_dir = ticker_dir(stock.symbol)
        jsonl_path = os.path.join(out_dir, f"{stock.symbol}_reddit_raw.jsonl")

        print(f"\n=== {stock.symbol} ===")
        ticker_total = 0

        for subreddit in subs:
            terms = stock.terms_for(subreddit)
            sub_total = 0
            sub_holes = 0
            for n, (w_start, w_end) in enumerate(quarter_windows(START_DATE, END_DATE)):
                if args.max_windows and n >= args.max_windows:
                    break
                key = (stock.symbol, subreddit, w_start.isoformat())
                if key in done:
                    continue

                posts, complete = pull_window(session, subreddit, terms, w_start, w_end)
                rows = normalise(posts, stock, subreddit)
                if rows:
                    with open(jsonl_path, "a", encoding="utf-8") as f:
                        for r in rows:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                        f.flush()
                        os.fsync(f.fileno())   # survive a hard kill, not just Ctrl-C
                sub_total += len(rows)

                # Only checkpoint a window that fully succeeded. Whatever was
                # retrieved from a failed window stays on disk and is deduped
                # later, so retrying it costs requests but never correctness.
                if complete:
                    done.add(key)
                    save_progress(done)
                else:
                    incomplete.append(key)
                    sub_holes += 1
                time.sleep(REQUEST_DELAY_SEC)

            ticker_total += sub_total
            flag = f"  [{sub_holes} incomplete window(s)]" if sub_holes else ""
            print(f"  r/{subreddit}: +{sub_total} posts{flag}")

        # Collapse the JSONL into one CSV per ticker, deduped across subreddits
        # (crossposts, multi-sub reposts and retried windows share a post_id).
        if os.path.exists(jsonl_path):
            csv_path = os.path.join(out_dir,
                                    f"{stock.symbol}_reddit_{START_DATE_STR}_{END_DATE_STR}.csv")
            n_rows, n_subs = collapse_jsonl_to_csv(jsonl_path, csv_path)
            first, last = date_span(csv_path)
            print(f"  Saved -> {csv_path} ({n_rows:,} unique posts)")
            summary.append({
                "ticker": stock.symbol,
                "unique_posts": n_rows,
                "subreddits": n_subs,
                "first": first,
                "last": last,
            })
        else:
            print(f"  no posts collected for {stock.symbol}")
            summary.append({"ticker": stock.symbol, "unique_posts": 0, "subreddits": 0,
                            "first": None, "last": None})

    out = pd.DataFrame(summary)
    path = os.path.join(REPORT_DIR, "reddit_pull_summary.csv")
    out.to_csv(path, index=False)
    print("\n=== Reddit pull summary ===")
    print(out.to_string(index=False))

    # Per-subreddit breakdown, which is what reddit_volume_check.py used to
    # provide. Now that both scripts hit the same endpoint, running the check
    # first would just pay for every request twice — so the volume numbers come
    # out of the real pull instead, and the posts are kept rather than counted
    # and discarded.
    breakdown = per_subreddit_breakdown(stocks, done)
    if not breakdown.empty:
        bpath = os.path.join(REPORT_DIR, "reddit_subreddit_volume.csv")
        breakdown.to_csv(bpath, index=False)
        print("\n=== Posts per subreddit ===")
        print(breakdown[breakdown["quarters_done"] > 0].to_string(index=False)
              if (breakdown["quarters_done"] > 0).any() else "  (nothing pulled yet)")

        # Only pairs with every quarter recorded can be called a measured zero.
        empty = breakdown[(breakdown["posts"] == 0) & breakdown["complete"]]
        if not empty.empty:
            print(f"\n{len(empty)} subreddit-ticker pairs returned nothing across all "
                  f"{len(list(quarter_windows(START_DATE, END_DATE)))} quarters — "
                  "candidates to drop from config.py:")
            for ticker, group in empty.groupby("ticker"):
                print(f"  {ticker}: {', '.join(group['subreddit'])}")

        unreliable = breakdown[~breakdown["complete"]]
        if not unreliable.empty:
            print(f"\n{len(unreliable)} pair(s) are not fully covered — failed windows or "
                  "never attempted. Their counts are floors, not measurements; do NOT "
                  f"prune on them. Full detail in {bpath}")

    if incomplete:
        idf = pd.DataFrame(incomplete, columns=["ticker", "subreddit", "quarter_start"])
        idf.to_csv(INCOMPLETE_PATH, index=False)
        print(f"\n{len(incomplete)} window(s) did not complete and were left unrecorded. "
              f"Re-run to retry them — see {INCOMPLETE_PATH}")
    elif os.path.exists(INCOMPLETE_PATH):
        os.remove(INCOMPLETE_PATH)   # a clean run should not leave a stale warning

    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
