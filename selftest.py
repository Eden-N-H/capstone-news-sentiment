"""
Offline self-test for the Reddit collection scripts. No network, no dumps.

Run this after any edit to config.py, reddit_pull.py or reddit_dump_filter.py,
and before starting a real collection run:

    python selftest.py

It builds a small synthetic dump in a temp directory and asserts that the
matching rules behave, then drives reddit_pull.py against a scripted fake HTTP
session that reproduces Arctic Shift's failure modes. Nothing here touches the
real data/ or reports/ directories.

Exit code 0 means every check passed.
"""

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def ts(y, m, d):
    return int(dt.datetime(y, m, d, 12, tzinfo=dt.timezone.utc).timestamp())


# Each case is (post, must_match_ticker_or_None, why).
FIXTURE = [
    # Tier discipline: a loose term must not fire in a discovery subreddit.
    (dict(id="a1", subreddit="wallstreetbets", title="YOLO into TTWO calls",
          selftext="", created_utc=ts(2021, 3, 1)), "TTWO", "ticker in a home sub"),
    (dict(id="a2", subreddit="Games_Deals", title="GTA V is 70% off",
          selftext="", created_utc=ts(2021, 3, 1)), None, "loose 'GTA' in a discovery sub"),
    (dict(id="a3", subreddit="Games_Deals", title="Grand Theft Auto VI leak",
          selftext="", created_utc=ts(2021, 3, 2)), "TTWO", "strict phrase in a discovery sub"),
    (dict(id="a4", subreddit="GTA", title="GTA online is dead",
          selftext="", created_utc=ts(2021, 3, 3)), "TTWO", "loose 'GTA' in a home sub"),
    (dict(id="a5", subreddit="conspiracy", title="BLK owns everything",
          selftext="", created_utc=ts(2021, 4, 1)), None, "bare 'BLK' in a discovery sub"),
    (dict(id="a6", subreddit="conspiracy", title="BlackRock owns everything",
          selftext="", created_utc=ts(2021, 4, 2)), "BLK", "strict 'BlackRock' in discovery"),
    (dict(id="a7", subreddit="stocks", title="BLK earnings",
          selftext="", created_utc=ts(2021, 4, 3)), "BLK", "bare 'BLK' in a home sub"),

    # The "Chase" problem.
    (dict(id="b1", subreddit="churning", title="Chase 5/24 rule question",
          selftext="", created_utc=ts(2020, 6, 1)), None, "consumer-banking 'Chase'"),
    (dict(id="b2", subreddit="wallstreetbets", title="gonna chase this pump",
          selftext="", created_utc=ts(2020, 6, 2)), None, "'chase' as a verb"),
    (dict(id="b3", subreddit="wallstreetbets", title="Chase Bank fees are insane",
          selftext="", created_utc=ts(2020, 6, 3)), "JPM", "'Chase Bank' phrase"),
    (dict(id="b4", subreddit="investing", title="Jamie Dimon speaks",
          selftext="", created_utc=ts(2020, 6, 4)), "JPM", "executive name"),

    # Tokenisation.
    (dict(id="c1", subreddit="stocks", title="I gotta buy something",
          selftext="", created_utc=ts(2021, 1, 1)), None, "'gotta' must not fire 'GTA'"),
    (dict(id="c2", subreddit="investing", title="thoughts on Larry\nFink memo",
          selftext="", created_utc=ts(2021, 1, 2)), "BLK", "phrase split by a newline"),

    # Window edges.
    (dict(id="d1", subreddit="stocks", title="AAPL to the moon",
          selftext="", created_utc=ts(2018, 12, 31)), None, "day before the window"),
    (dict(id="d2", subreddit="stocks", title="AAPL to the moon",
          selftext="", created_utc=ts(2024, 1, 1)), None, "day after the window"),
    (dict(id="d3", subreddit="stocks", title="AAPL earnings",
          selftext="", created_utc=ts(2019, 1, 1)), "AAPL", "first day of the window"),
    (dict(id="d4", subreddit="stocks", title="NVDA earnings",
          selftext="", created_utc=ts(2023, 12, 31)), "NVDA", "last day of the window"),

    # Robustness.
    (dict(id="e1", subreddit="knitting", title="AAPL iPhone case pattern",
          selftext="", created_utc=ts(2021, 5, 1)), None, "subreddit not in config"),
    (dict(id="e2", subreddit="stocks", title="AAPL and NVDA both up",
          selftext="", created_utc=ts(2021, 5, 2)), "AAPL", "one post, two tickers"),
    (dict(id="e3", subreddit="stocks", title="AAPL dupe",
          selftext="", created_utc=ts(2021, 5, 3)), "AAPL", "duplicate id, kept once"),
    (dict(id="e3", subreddit="stocks", title="AAPL dupe",
          selftext="", created_utc=ts(2021, 5, 3)), "AAPL", "duplicate id, kept once"),
    (dict(id="e4", subreddit="stocks", title=None, selftext=None,
          created_utc=ts(2021, 5, 4)), None, "null title and body"),
    (dict(id="e5", subreddit="stocks", title="no ticker here",
          created_utc=None), None, "null timestamp"),
]


def run_dump_filter_checks(workdir):
    import csv
    import glob

    dumps = os.path.join(workdir, "dumps")
    os.makedirs(dumps, exist_ok=True)
    with open(os.path.join(dumps, "selftest_submissions.jsonl"), "w") as fh:
        for post, _, _ in FIXTURE:
            fh.write(json.dumps(post) + "\n")
        fh.write("\n")             # blank line
        fh.write("{not json\n")    # malformed line

    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "reddit_dump_filter.py"), "--dumps", dumps],
        cwd=workdir, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit("reddit_dump_filter.py exited non-zero")

    kept = {}
    for path in glob.glob(os.path.join(workdir, "data", "*", "*_reddit_*.csv")):
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                kept.setdefault(row["post_id"], set()).add(row["ticker"])

    failures = []
    for post, expect, why in FIXTURE:
        got = kept.get(post["id"], set())
        if expect is None and got:
            failures.append(f"  FAIL  {post['id']}: {why} — matched {sorted(got)}, wanted nothing")
        elif expect is not None and expect not in got:
            failures.append(f"  FAIL  {post['id']}: {why} — wanted {expect}, matched {sorted(got)}")
        else:
            verdict = "excluded" if expect is None else f"kept as {expect}"
            print(f"  ok    {post['id']}: {why} — {verdict}")

    # Schema parity with the API path.
    from jsonl_csv import CSV_COLUMNS
    sample = sorted(glob.glob(os.path.join(workdir, "data", "*", "*_reddit_*.csv")))[0]
    with open(sample, newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    if header != CSV_COLUMNS:
        failures.append(f"  FAIL  CSV header drifted from the shared schema: {header}")
    else:
        print("  ok    output schema matches jsonl_csv.CSV_COLUMNS")

    return failures


def run_pull_checks():
    import requests
    import reddit_pull as rp

    rp.NETWORK_RETRY_WAITS = (0,) * 8      # no real sleeping in a test
    rp.REQUEST_DELAY_SEC = 0
    rp.MAX_RETRIES = 3

    class Resp:
        def __init__(self, code, payload):
            self.status_code, self._p = code, payload
            self.text, self.headers = json.dumps(payload), {}

        @property
        def ok(self):
            return self.status_code < 400

        def json(self):
            return self._p

    class Scripted:
        headers = {}

        def __init__(self, handler):
            self.handler, self.calls = handler, []

        def get(self, url, params=None, timeout=None):
            self.calls.append((url, dict(params or {})))
            return self.handler(url, self.calls)

    failures = []

    def check(label, cond, detail=""):
        if cond:
            print(f"  ok    {label}")
        else:
            failures.append(f"  FAIL  {label} {detail}")

    soft = Resp(200, {"data": None, "error": "Timeout. Maybe slow down a bit"})
    s = Scripted(lambda u, c: soft)
    _, ok = rp._pull_term(s, "stocks", "AAPL", dt.date(2021, 1, 1), ts(2021, 4, 1), set())
    check("HTTP 200 with a soft error is retried, not read as an empty window",
          ok is False and len(s.calls) == rp.MAX_RETRIES, f"(ok={ok}, calls={len(s.calls)})")

    s = Scripted(lambda u, c: Resp(200, {"data": None}))
    _, ok = rp._pull_term(s, "stocks", "AAPL", dt.date(2021, 1, 1), ts(2021, 4, 1), set())
    check("null data with no error is a genuine empty result, no retry storm",
          ok is True and len(s.calls) == 1, f"(ok={ok}, calls={len(s.calls)})")

    s = Scripted(lambda u, c: Resp(422, {"data": None, "error": "Timeout"}))
    _, ok = rp.pull_window(s, "stocks", ["AAPL"], dt.date(2021, 1, 1), dt.date(2021, 3, 31))
    check("a window whose search exhausted its retries is incomplete", ok is False)

    def ids_fail(url, calls):
        if url == rp.IDS_URL:
            return Resp(422, {"data": None, "error": "Timeout"})
        searches = len([c for c in calls if c[0] != rp.IDS_URL])
        return (Resp(200, {"data": [{"id": "x1", "created_utc": ts(2021, 2, 1)}]})
                if searches == 1 else Resp(200, {"data": []}))

    s = Scripted(ids_fail)
    _, ok = rp.pull_window(s, "stocks", ["AAPL"], dt.date(2021, 1, 1), dt.date(2021, 3, 31))
    check("a good search with a failed /ids fetch is still incomplete", ok is False)

    full = {"id": "x1", "created_utc": ts(2021, 2, 1), "title": "AAPL up", "selftext": "",
            "subreddit": "stocks", "score": 5, "num_comments": 2, "url": "u", "over_18": False}

    def all_good(url, calls):
        if url == rp.IDS_URL:
            return Resp(200, {"data": [full]})
        searches = len([c for c in calls if c[0] != rp.IDS_URL])
        return (Resp(200, {"data": [{"id": "x1", "created_utc": ts(2021, 2, 1)}]})
                if searches == 1 else Resp(200, {"data": []}))

    s = Scripted(all_good)
    posts, ok = rp.pull_window(s, "stocks", ["AAPL"], dt.date(2021, 1, 1), dt.date(2021, 3, 31))
    check("a fully successful window is complete", ok is True and len(posts) == 1,
          f"(ok={ok}, posts={len(posts)})")

    leaked = [c for c in s.calls if "author" in str(c[1].get("fields", ""))]
    check("no request ever asks for author fields (proposal 4.1.1)", not leaked,
          f"({len(leaked)} request(s) did)")

    return failures


def main():
    sys.path.insert(0, HERE)
    workdir = tempfile.mkdtemp(prefix="capstone_selftest_")
    try:
        print("Dump-filter matching rules")
        failures = run_dump_filter_checks(workdir)
        print("\nAPI failure handling")
        failures += run_pull_checks()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print()
    if failures:
        print("\n".join(failures))
        print(f"\n{len(failures)} check(s) FAILED.")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
