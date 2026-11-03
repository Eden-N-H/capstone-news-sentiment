"""
Arctic Shift -> post volume check for every selected stock.

Uses /api/posts/search/aggregate to get monthly post COUNTS per (ticker,
subreddit) pair without downloading post content. Run this before
reddit_pull.py: it tells you which subreddit lists are worth pulling and which
return so little that a per-platform model would be trained on noise.

Three mechanisms handle the fact that Arctic Shift answers overloaded keyword
queries with HTTP 422 and "Timeout. Maybe slow down a bit":

  1. Binary window splitting. A window that times out is halved and each half
     retried, recursively, down to a single day. A calendar-month floor was not
     enough for r/wallstreetbets, which is the largest subreddit in the list by
     an order of magnitude.
  2. Adaptive throttling. Every timeout raises the inter-request delay; a run
     of successes decays it back down. The server's own advice is to slow down,
     so the script listens to it rather than hammering at a fixed rate.
  3. Checkpointing. Results are appended per (ticker, subreddit) pair, so a
     crash or Ctrl-C costs you one pair rather than the whole sweep. Re-running
     resumes; --restart forces a clean sweep.

Some subreddits may still refuse to aggregate at any window width. Those are
reported as `failed_days` rather than silently counted as zero — the
distinction matters, because a zero means "nobody posted" and a failure means
"we don't know". For a subreddit that fails persistently, the fallback is
Arctic Shift's per-subreddit dump rather than the API.

Run:
    python reddit_volume_check.py
    python reddit_volume_check.py --only BLK JPM
    python reddit_volume_check.py --delay 5 --window month
"""

import argparse
import datetime as dt
import os
import time

import pandas as pd
import requests

from config import END_DATE, REPORT_DIR, START_DATE, select_stocks

BASE_URL = "https://arctic-shift.photon-reddit.com/api/posts/search/aggregate"
USER_AGENT = "uts-capstone-sentiment-research/1.0 (academic; contact via UTS)"

MAX_RETRIES = 3
RETRY_WAIT_SEC = 8
MAX_DELAY_SEC = 20.0

OUT_PATH = os.path.join(REPORT_DIR, "reddit_volume_check.csv")

# Adaptive pacing: `base` is the floor set by --delay, `current` rises on
# timeouts and decays on success.
_pace = {"base": 2.5, "current": 2.5}


def throttle_up():
    _pace["current"] = min(_pace["current"] * 1.6, MAX_DELAY_SEC)


def throttle_down():
    _pace["current"] = max(_pace["current"] * 0.92, _pace["base"])


def wait():
    time.sleep(_pace["current"])


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
# Confirmed live shape: {"created_utc": "2018-12-31T23:00:00.000Z", "count": "250"}
# Note the count arrives as a string. Keys are still read defensively, since the
# docs do not pin the shape down.
COUNT_KEYS = ("count", "doc_count", "total", "value", "n")
KEY_KEYS = ("key", "key_as_string", "_id", "date", "created_utc")
_shape_logged = [False]


def _as_int(value) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return 0
    return 0


def row_count(row) -> int:
    """Pull the count out of one aggregate row, whatever shape it arrives in."""
    if isinstance(row, (int, float, str)):
        return _as_int(row)
    if isinstance(row, dict):
        for k in COUNT_KEYS:
            if k in row:
                return _as_int(row[k])
        for k, v in row.items():
            if k not in KEY_KEYS and _as_int(v):
                return _as_int(v)
        return 0
    if isinstance(row, (list, tuple)) and len(row) == 2:
        return _as_int(row[1])
    return 0


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------
def quarter_windows(start: dt.date, end: dt.date):
    cur = start
    while cur <= end:
        q_end_month = ((cur.month - 1) // 3 + 1) * 3
        last_day = dt.date(cur.year + (q_end_month == 12), (q_end_month % 12) + 1, 1) - dt.timedelta(days=1)
        window_end = min(last_day, end)
        yield cur, window_end
        cur = window_end + dt.timedelta(days=1)


def month_windows(start: dt.date, end: dt.date):
    cur = start
    while cur <= end:
        last_day = dt.date(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1) - dt.timedelta(days=1)
        window_end = min(last_day, end)
        yield cur, window_end
        cur = window_end + dt.timedelta(days=1)


def year_windows(start: dt.date, end: dt.date):
    cur = start
    while cur <= end:
        window_end = min(dt.date(cur.year, 12, 31), end)
        yield cur, window_end
        cur = window_end + dt.timedelta(days=1)


WINDOW_FUNCS = {"year": year_windows, "quarter": quarter_windows, "month": month_windows}


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def request_once(session, subreddit, query, after, before):
    """One aggregate request with retries. Returns (rows, "ok"|"timeout"|"error")."""
    params = {
        "subreddit": subreddit,
        "query": query,
        "after": after.isoformat(),
        "before": before.isoformat(),
        "aggregate": "created_utc",
        "frequency": "month",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(BASE_URL, params=params, timeout=90)
        except requests.RequestException as e:
            print(f"      {type(e).__name__} on {after}->{before} "
                  f"(attempt {attempt}/{MAX_RETRIES})")
            throttle_up()
            time.sleep(RETRY_WAIT_SEC * attempt)
            continue

        if resp.ok:
            data = resp.json().get("data") or []
            if data and not _shape_logged[0]:
                _shape_logged[0] = True
                print(f"    [shape] first aggregate row: {data[0]!r}")
            throttle_down()
            return data, "ok"

        body = resp.text[:160]
        is_timeout = "timeout" in body.lower() or "timed out" in body.lower()

        if resp.status_code == 429:
            throttle_up()
            reset = resp.headers.get("X-RateLimit-Reset", "30")
            time.sleep(int(reset) if reset.isdigit() else 30)
            continue

        if is_timeout:
            # Retrying an identical query that just timed out rarely helps.
            # Back off once and hand control back to the splitter.
            throttle_up()
            return [], "timeout"

        print(f"      HTTP {resp.status_code} on {after}->{before} "
              f"(attempt {attempt}/{MAX_RETRIES}) {body}")
        time.sleep(RETRY_WAIT_SEC * attempt)

    return [], "error"


# Below this width, splitting stops helping. The docs warn that the AGGREGATE
# endpoint times out for very active subreddits — that is a property of the
# endpoint, not of the window, so halving a 7-day window into two 3-day windows
# just doubles the request count and makes the rate limiting worse.
MIN_SPLIT_DAYS = 7

# Patient retries at the floor, since failures there are intermittent.
FLOOR_RETRY_WAITS = (20, 45, 90)

SEARCH_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"

# Waits between attempts on a single search request. Timeouts here are usually
# transient load rather than an unanswerable query.
SEARCH_RETRY_WAITS = (5, 15, 30, 60)


def count_via_search(session, subreddit, terms, after, before):
    """
    Fallback: count posts through /api/posts/search instead of the aggregate.

    One request per term, with post IDs unioned locally. This sidesteps the
    question of whether the server parses "A OR B" as a union at all — the docs
    flag their boolean-syntax section as out of date — and a single-term query
    is cheaper for the server, which is the thing timing out. Deduplicating on
    ID means a post matching three terms is still counted once.

    Returns (month -> count dict, ok) — ok is False if the search also failed.
    """
    before_ts = int(dt.datetime.combine(before + dt.timedelta(days=1), dt.time.min,
                                        tzinfo=dt.timezone.utc).timestamp())
    seen = set()
    ok_any = False

    for term in terms:
        cursor = int(dt.datetime.combine(after, dt.time.min,
                                         tzinfo=dt.timezone.utc).timestamp())
        term_ok, ids = _search_term(session, subreddit, term, cursor, before_ts)
        ok_any = ok_any or term_ok
        seen.update(ids)

    return _months_from_ids(seen), ok_any


def _months_from_ids(id_ts_pairs):
    months = {}
    for _pid, ts in id_ts_pairs:
        key = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).strftime("%Y-%m")
        months[key] = months.get(key, 0) + 1
    return months


def _search_term(session, subreddit, term, cursor, before_ts):
    """Paginate one single-term search. Returns (ok, {(id, created_utc), ...})."""
    found = set()
    seen_ids = set()

    for _ in range(200):   # circuit breaker
        params = {
            "subreddit": subreddit,
            "query": term,
            "after": cursor,
            "before": before_ts,
            "limit": 100,
            "sort": "asc",
            "fields": "id,created_utc",
        }
        # The docs note the database sometimes needs to warm up and that a
        # second attempt often succeeds, so a timeout here is retried rather
        # than treated as a verdict. Confirmed live: a single-term query on
        # r/wallstreetbets works while the same query on the much smaller
        # r/stocks can time out — this is load, not subreddit size.
        page = None
        for attempt, pause in enumerate(SEARCH_RETRY_WAITS, start=1):
            try:
                resp = session.get(SEARCH_URL, params=params, timeout=90)
            except requests.RequestException as e:
                print(f"      {type(e).__name__} on r/{subreddit} '{term}' "
                      f"(attempt {attempt}/{len(SEARCH_RETRY_WAITS)}), waiting {pause}s")
                throttle_up()
                time.sleep(pause)
                continue

            if resp.ok:
                page = resp.json().get("data") or []
                throttle_down()
                break

            body = resp.text[:120]
            throttle_up()
            if resp.status_code == 429:
                reset = resp.headers.get("X-RateLimit-Reset", "30")
                time.sleep(int(reset) if reset.isdigit() else 30)
                continue
            print(f"      r/{subreddit} '{term}' {resp.status_code} "
                  f"(attempt {attempt}/{len(SEARCH_RETRY_WAITS)}) {body}")
            time.sleep(pause)

        if page is None:
            return False, found

        if not page:
            break

        fresh = [p for p in page if p.get("id") not in seen_ids]
        for p in fresh:
            seen_ids.add(p["id"])
            found.add((p["id"], int(p.get("created_utc", 0))))

        last_ts = max(int(p.get("created_utc", cursor)) for p in page)
        # Posts sharing a second would be re-served forever, so nudge the
        # cursor whenever a page adds nothing new.
        cursor = last_ts if fresh else last_ts + 1
        if cursor >= before_ts:
            break
        throttle_down()
        wait()

    throttle_down()
    return True, found


def months_from_rows(rows):
    """Aggregate-endpoint rows -> {month: count}."""
    out = {}
    for r in rows:
        c = row_count(r)
        key = None
        if isinstance(r, dict):
            for k in KEY_KEYS:
                if k in r:
                    key = str(r[k])[:7]
                    break
        key = key or "unknown"
        out[key] = out.get(key, 0) + c
    return out


def merge(into: dict, other: dict):
    for k, v in other.items():
        into[k] = into.get(k, 0) + v


# Once a subreddit has proven it will not aggregate, stop asking. Grinding
# through 300 seven-day windows at three requests each costs hours per
# subreddit; switching wholesale to search over the original wide windows costs
# minutes.
GIVE_UP_ON_AGGREGATE_AFTER = 2


def fetch_range(session, subreddit, query, terms, after, before, state, verbose=True):
    """
    Get {month: count} for a window.

    Strategy: aggregate first; halve on timeout down to MIN_SPLIT_DAYS; below
    that, retry patiently; if it still fails, count via the search endpoint —
    and if that happens repeatedly, abandon the aggregate endpoint for this
    subreddit entirely.

    Returns (months dict, failed_days, searched_days).
    """
    if state["mode"] == "search":
        months, ok = count_via_search(session, subreddit, terms, after, before)
        wait()
        span = (before - after).days + 1
        return (months, 0, span) if ok else (months, span, 0)

    rows, status = request_once(session, subreddit, query, after, before)
    if status == "ok":
        wait()
        return months_from_rows(rows), 0, 0

    span = (before - after).days

    if span >= MIN_SPLIT_DAYS:
        mid = after + dt.timedelta(days=span // 2)
        if verbose and span > 20:
            print(f"      splitting {after}->{before} at {mid} "
                  f"(delay now {_pace['current']:.1f}s)")
        wait()
        left = fetch_range(session, subreddit, query, terms, after, mid, state, verbose)
        right = fetch_range(session, subreddit, query, terms,
                            mid + dt.timedelta(days=1), before, state, verbose)
        merged = dict(left[0])
        merge(merged, right[0])
        return merged, left[1] + right[1], left[2] + right[2]

    # At the floor. Failures here are intermittent rather than structural, so
    # wait properly instead of subdividing further.
    for pause in FLOOR_RETRY_WAITS:
        if verbose:
            print(f"      {after}->{before} at floor, waiting {pause}s")
        time.sleep(pause)
        rows, status = request_once(session, subreddit, query, after, before)
        if status == "ok":
            wait()
            return months_from_rows(rows), 0, 0

    state["agg_failures"] += 1
    if verbose:
        print(f"      aggregate will not serve {after}->{before}; counting via search")
    if state["agg_failures"] >= GIVE_UP_ON_AGGREGATE_AFTER and state["mode"] != "search":
        state["mode"] = "search"
        print(f"      -> r/{subreddit} will not aggregate; using the search "
              f"endpoint for the rest of this subreddit")

    months, ok = count_via_search(session, subreddit, terms, after, before)
    wait()
    if ok:
        return months, 0, span + 1
    return months, span + 1, 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_done(restart: bool):
    if restart or not os.path.exists(OUT_PATH):
        return set()
    df = pd.read_csv(OUT_PATH)
    return set(zip(df["ticker"], df["subreddit"]))


def append_row(row: dict):
    """Write incrementally so a crash costs one pair, not the whole sweep."""
    header = not os.path.exists(OUT_PATH)
    pd.DataFrame([row]).to_csv(OUT_PATH, mode="a", header=header, index=False)


def main():
    ap = argparse.ArgumentParser(description="Check Reddit post volume before the full pull.")
    ap.add_argument("--only", nargs="*", help="Restrict to these tickers")
    ap.add_argument("--window", choices=["year", "quarter", "month"], default="quarter",
                    help="Starting chunk size; timeouts halve it further (default quarter)")
    ap.add_argument("--delay", type=float, default=2.5,
                    help="Minimum seconds between requests (default 2.5)")
    ap.add_argument("--restart", action="store_true",
                    help="Ignore the existing CSV and re-check every pair")
    ap.add_argument("--search-only", action="store_true",
                    help="Skip the aggregate endpoint entirely and count via "
                         "per-term search. Slower per subreddit, but avoids the "
                         "retry preamble when the aggregate is known to be down "
                         "or unsupported.")
    args = ap.parse_args()

    _pace["base"] = _pace["current"] = args.delay
    windows = WINDOW_FUNCS[args.window]

    if args.restart and os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)
    done = load_done(args.restart)
    if done:
        print(f"Resuming — {len(done)} pairs already checked. --restart to redo.")

    stocks = select_stocks(args.only)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    for stock in stocks:
        print(f"\n=== {stock.symbol} ===")
        for subreddit in stock.all_subreddits:
            if (stock.symbol, subreddit) in done:
                continue

            terms = stock.terms_for(subreddit)
            query = " OR ".join(terms)
            failed_days = 0

            all_months = {}
            searched_days = 0
            state = {"mode": "search" if args.search_only else "aggregate",
                     "agg_failures": 0}
            for w_start, w_end in windows(START_DATE, END_DATE):
                months, failed, searched = fetch_range(session, subreddit, query, terms,
                                                       w_start, w_end, state)
                merge(all_months, months)
                failed_days += failed
                searched_days += searched
            total = sum(all_months.values())
            months_with_data = sum(1 for v in all_months.values() if v > 0)

            # A count is only trustworthy if the endpoints actually answered.
            # r/wallstreetbets returning single digits for AAPL is not a finding
            # about Reddit, it is the API declining keyword search on a very
            # active subreddit — the docs warn about exactly this. Labelling it
            # matters, because "3 posts" would get the subreddit deleted from
            # config.py when it is in fact one of the busiest sources here.
            if failed_days > 0:
                status = "incomplete"
            elif state["mode"] == "search" and total < 100:
                status = "suspect-use-dump"
            elif searched_days > 0:
                status = "ok-via-search"
            else:
                status = "ok"

            append_row({
                "ticker": stock.symbol,
                "subreddit": subreddit,
                "status": status,
                "scope": "home" if subreddit in stock.subreddits else "discovery",
                "query": query,
                "total_posts": total,
                "months_with_activity": months_with_data,
                "avg_per_month": round(total / max(months_with_data, 1), 1),
                "failed_days": failed_days,
                "search_fallback_days": searched_days,
            })

            flag = ""
            if searched_days:
                flag += f"  [{searched_days}d via search]"
            if failed_days:
                flag += f"  [{failed_days}d unretrievable]"
            if status in ("incomplete", "suspect-use-dump"):
                flag += f"  <- {status.upper()}, do not trust this count"
            print(f"  r/{subreddit}: {total} posts, {months_with_data} active months{flag}")

    if not os.path.exists(OUT_PATH):
        print("\nNothing to report.")
        return

    df = pd.read_csv(OUT_PATH).sort_values(["ticker", "total_posts"], ascending=[True, False])
    df.to_csv(OUT_PATH, index=False)

    print("\n=== Summary ===")
    print(df.to_string(index=False))
    print("\nPer-ticker totals:")
    print(df.groupby("ticker")["total_posts"].sum().to_string())

    unusable = df[df["status"].isin(["incomplete", "suspect-use-dump"])]
    if not unusable.empty:
        print(f"\n{len(unusable)} pairs the API would not serve properly. Their counts are"
              " NOT evidence of low volume — do not prune these from config.py:")
        for _, r in unusable.iterrows():
            print(f"  {r['ticker']} / r/{r['subreddit']}: reported {r['total_posts']},"
                  f" status {r['status']}")
        print("  These subreddits need Arctic Shift's per-subreddit dumps:")
        print("  https://arctic-shift.photon-reddit.com/download-tool")

    dead = df[(df["total_posts"] == 0) & (df["failed_days"] == 0)
              & (df["status"] == "ok")]
    if not dead.empty:
        print(f"\n{len(dead)} pairs genuinely returned nothing — drop from config.py:")
        for ticker, group in dead.groupby("ticker"):
            print(f"  {ticker}: {', '.join(group['subreddit'])}")

    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
