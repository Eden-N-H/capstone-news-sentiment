"""
Clean the Reddit corpus, using the filters already validated on the news arm.

A sample of matched posts showed the same three contamination classes that the
news corpus had, in similar proportions:

  MENTION, NOT SUBJECT — "Coinbase Releasing Earnings On November 02" matched
    BLK because BlackRock appears somewhere in the body. Five of six sampled
    BLK posts were about other companies.

  THIRD-PARTY ANALYST MENTIONS — "What y'all think of FUBO? JPMorgan analyst
    Anna..." is a post about FUBO. Identical to the "JP Morgan Maintains
    Prologis Overweight" pattern that accounted for 2,435 news articles.

  ROUNDUPS AND RECAPS — "Here is a Market Recap for today Monday, Nov 22",
    "Today's Pre-Market News", "Battle of the Big Tech Stocks". Market-wide
    sentiment attached to whichever tickers happen to be listed.

The same remedy applies: require the company in the TITLE, drop roundups, drop
third-party analyst mentions. Patterns are imported from news_quality_filter so
both arms are filtered identically — which matters, because the whole point of
the project is comparing the two platforms, and a difference in filtering
would confound a difference in signal.

Reddit needs a few patterns of its own. Its recaps and daily threads are worded
differently from a newswire's, and cashtag lists ("$AAPL $MSFT $TSLA") are a
Reddit-specific way of writing a multi-ticker post.

Run:
    python reddit_quality_filter.py
    python reddit_quality_filter.py --allow-untitled     # keep body-only matches
"""

import argparse
import glob
import os
import re

import pandas as pd

from config import END_DATE_STR, REPORT_DIR, START_DATE_STR, select_stocks, ticker_dir
from news_quality_filter import (
    NON_ASCII_THRESHOLD, ROUNDUP_RE, TICKER_LIST_RE, _no_groups,
    company_title_regex, compile_excludes,
)

# Reddit's own roundup and recap vocabulary.
REDDIT_ROUNDUP_PATTERNS = [
    r"\bmarket recap\b",
    r"\bpre-?market news\b",
    r"\bpre-?market movers\b",
    r"\bmost anticipated earnings\b",
    r"\bearnings (?:this|next) week\b",
    r"\bstocks? to watch\b",
    r"\bwatchlist\b",
    r"\bdaily (?:discussion|thread|chat)\b",
    r"\bwhat are your moves\b",
    r"\bportfolio (?:advice|review|critique|rate)\b",
    r"\brate my portfolio\b",
    r"\bbattle of the\b",
    r"\bbiggest (?:movers|gainers|losers)\b",
    r"\btop \d+ stocks\b",
    r"\b(?:list of|these) \d+ stocks\b",
    r"\bmy portfolio\b",
    r"\bfed\b.*\b(?:rate|meeting|decision)\b",
]
REDDIT_ROUNDUP_RE = re.compile(
    "|".join(f"(?:{_no_groups(p)})" for p in REDDIT_ROUNDUP_PATTERNS), re.IGNORECASE)

# Three or more cashtags is a listing, not a post about one company.
CASHTAG_LIST_RE = re.compile(r"\$[A-Za-z]{1,5}\b(?:\W+\$[A-Za-z]{1,5}\b){2,}")

ANY_CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,5})\b")

# The company named as an ACTOR in a post about a different stock. Reddit's
# version of the analyst-note problem: "What y'all think of FUBO? JPMorgan
# analyst Anna raised target" is a post about FUBO.
ACTOR_TEMPLATES = [
    r"\b{kw}\b\s+(?:analysts?|strategists?|research|note)\b",
    r"\b{kw}\b\s+(?:reports?|raises?|lowers?|cuts?|initiates?|upgrades?|downgrades?)\b"
    r".{{0,50}}\b(?:ownership|stake|position|target|rating|coverage)\b",
    r"\b{kw}\b\s+just\s+(?:bought|sold|dumped|loaded)\b",
]


def actor_regexes(stock):
    out = []
    for kw in set(stock.title_keywords) | {stock.symbol}:
        k = re.escape(kw).replace(r"\ ", r"\s?")
        out += [re.compile(t.format(kw=k), re.IGNORECASE) for t in ACTOR_TEMPLATES]
    return out


def other_ticker_cashtag(titles, stock):
    """
    True when the title cashtags some OTHER company and not this one.

    "$MILE is ready for the moon! And Blackrock just bought in" is a post about
    MILE. A cashtag is Reddit's way of declaring what a post is about, so one
    pointing elsewhere is strong evidence the company is incidental.
    """
    own = {stock.symbol.lower()} | {k.lower() for k in stock.title_keywords}

    def check(t):
        tags = {m.lower() for m in ANY_CASHTAG_RE.findall(t)}
        return bool(tags) and not (tags & own)

    return titles.map(check)


def flag(df: pd.DataFrame, stock, check_title=True) -> pd.DataFrame:
    titles = df["title"].fillna("").astype(str)

    excludes = compile_excludes(stock) + actor_regexes(stock)
    df["is_analyst_note"] = (titles.apply(lambda t: any(rx.search(t) for rx in excludes))
                             | other_ticker_cashtag(titles, stock))

    df["is_roundup"] = (titles.str.contains(ROUNDUP_RE, na=False)
                        | titles.str.contains(TICKER_LIST_RE, na=False)
                        | titles.str.contains(REDDIT_ROUNDUP_RE, na=False)
                        | titles.str.contains(CASHTAG_LIST_RE, na=False))

    df["is_non_english"] = titles.map(
        lambda t: sum(1 for c in t if ord(c) > 127) / max(len(t), 1)) > NON_ASCII_THRESHOLD

    if check_title:
        rx = company_title_regex(stock)
        df["no_company_in_title"] = ~titles.str.contains(rx, na=False)
    else:
        df["no_company_in_title"] = False

    df["flagged"] = (df["is_analyst_note"] | df["is_roundup"]
                     | df["is_non_english"] | df["no_company_in_title"])
    return df


def main():
    ap = argparse.ArgumentParser(description="Clean the Reddit corpus.")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--allow-untitled", action="store_true",
                    help="Keep posts whose title never names the company")
    ap.add_argument("--keep-flagged", action="store_true",
                    help="Write the flag columns without dropping rows")
    args = ap.parse_args()

    rows, samples = [], []

    for stock in select_stocks(args.only):
        src = os.path.join(ticker_dir(stock.symbol),
                           f"{stock.symbol}_reddit_{START_DATE_STR}_{END_DATE_STR}.csv")
        if not os.path.exists(src):
            print(f"{stock.symbol}: no Reddit file — run reddit_dump_filter.py first")
            continue

        df = pd.read_csv(src)
        before = len(df)
        df = flag(df, stock, check_title=not args.allow_untitled)

        counts = {r: int(df[f"is_{r}"].sum()) if r != "no_company_in_title"
                  else int(df[r].sum())
                  for r in ("analyst_note", "roundup", "non_english", "no_company_in_title")}

        for reason in ("is_analyst_note", "is_roundup", "is_non_english",
                       "no_company_in_title"):
            hit = df[df[reason]]
            if not hit.empty:
                take = hit.sample(min(10, len(hit)), random_state=0)
                samples.append(pd.DataFrame({
                    "ticker": stock.symbol,
                    "reason": reason.replace("is_", ""),
                    "subreddit": take["subreddit"].values,
                    "title": take["title"].values,
                }))

        if not args.keep_flagged:
            df = df[~df["flagged"]].copy()

        dst = os.path.join(ticker_dir(stock.symbol),
                           f"{stock.symbol}_reddit_clean_{START_DATE_STR}_{END_DATE_STR}.csv")
        df.to_csv(dst, index=False)

        days = pd.to_datetime(df["date"], utc=True, format="mixed").dt.date.nunique() if len(df) else 0
        print(f"\n{stock.symbol}")
        print(f"  raw        {before:6,}")
        print(f"  flagged    analyst {counts['analyst_note']:,}, "
              f"roundup {counts['roundup']:,}, "
              f"non-English {counts['non_english']:,}, "
              f"unnamed-in-title {counts['no_company_in_title']:,}")
        print(f"  clean      {len(df):6,}  across {days:,} days")
        print(f"  -> {dst}")

        rows.append({"ticker": stock.symbol, "raw": before,
                     **{k: v for k, v in counts.items()},
                     "clean": len(df), "days_with_posts": days})

    if samples:
        spath = os.path.join(REPORT_DIR, "reddit_flagged_samples.csv")
        pd.concat(samples, ignore_index=True).to_csv(spath, index=False)
        print(f"\nFlagged examples for auditing -> {spath}")

    if rows:
        rep = pd.DataFrame(rows)
        path = os.path.join(REPORT_DIR, "reddit_quality.csv")
        rep.to_csv(path, index=False)
        print(f"\n=== Summary ===\n{rep.to_string(index=False)}")
        print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
