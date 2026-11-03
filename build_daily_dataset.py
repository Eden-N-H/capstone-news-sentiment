"""
Article-level sentiment -> daily series aligned to trading days, merged with prices.

Two things here are easy to get wrong and both would corrupt the modelling.

MARKET TIMING
-------------
FNSPID timestamps are UTC. An article stamped 21:00 UTC is 16:00 New York —
at or after the closing bell, so it cannot have moved that day's close. Simply
truncating the timestamp to a date would assign it to a session that had
already finished, leaking future information into the features. Every article
is therefore converted to US/Eastern and assigned to the next session that was
still open when it published: same day if before the cutoff, otherwise the
following trading day. Weekend and holiday articles roll forward to the next
session, which is why Monday often carries three days of news.

MISSING DAYS
------------
Trading days with no articles are left as NaN rather than filled. Neutral-zero,
forward-fill and rolling-mean all encode different assumptions about how
sentiment persists, and that choice belongs to the modelling stage where it can
be tested. NaN keeps all three options open; `article_count` is 0 on those days
so they are distinguishable from genuinely neutral days.

Outputs per ticker, plus a combined panel across all six.

Run:
    python build_daily_dataset.py                    # news
    python build_daily_dataset.py --source reddit    # Reddit
    python build_daily_dataset.py --only NVDA
"""

import argparse
import os
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config import END_DATE_STR, REPORT_DIR, START_DATE_STR, select_stocks, ticker_dir

MARKET_TZ = ZoneInfo("America/New_York")

# Both corpora are aggregated the same way so the platform comparison is not
# confounded by a difference in processing. Reddit carries one extra field the
# news corpus has no equivalent for: upvote score, which is a rough proxy for
# how many people actually saw a post.
SOURCES = {
    "news": {
        "scored": "{sym}_news_scored_{start}_{end}.csv",
        "output": "{sym}_daily_{start}_{end}.csv",
        "panel": "daily_panel_{start}_{end}.csv",
        "coverage": "daily_coverage.csv",
        "date_col": "Date",
        "weight_col": None,
        "needs": "sentiment_score.py",
    },
    "reddit": {
        "scored": "{sym}_reddit_scored_{start}_{end}.csv",
        "output": "{sym}_daily_reddit_{start}_{end}.csv",
        "panel": "daily_panel_reddit_{start}_{end}.csv",
        "coverage": "daily_coverage_reddit.csv",
        "date_col": "date",
        "weight_col": "score",
        "needs": "sentiment_score.py --source reddit",
    },
}
CUTOFF_HOUR = 16          # 4pm ET closing bell


def load_prices(stock) -> pd.DataFrame:
    """Prefer the yfinance series: one schema and one calendar for all six."""
    d = ticker_dir(stock.symbol)
    candidates = [
        os.path.join(d, f"{stock.symbol}_prices_yf_{START_DATE_STR}_{END_DATE_STR}.csv"),
        os.path.join(d, f"{stock.symbol}_prices_{START_DATE_STR}_{END_DATE_STR}.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
            return df.sort_values("date").reset_index(drop=True), path
    raise SystemExit(f"No price file for {stock.symbol}. Run market_prices.py first.")


def assign_sessions(published_utc: pd.Series, sessions: np.ndarray,
                    cutoff_hour: int) -> pd.Series:
    """
    Map each article to the trading session it could first have influenced.

    An article published before the cutoff on a trading day belongs to that
    day. Anything at or after the cutoff — and anything on a weekend or holiday
    — belongs to the next session.
    """
    local = published_utc.dt.tz_convert(MARKET_TZ)

    # Push post-cutoff articles into the next calendar day, then let the
    # searchsorted below find the next actual session.
    effective = local.dt.normalize()
    after_close = local.dt.hour >= cutoff_hour
    effective = effective.where(~after_close, effective + pd.Timedelta(days=1))

    naive = effective.dt.tz_localize(None).values.astype("datetime64[ns]")
    idx = np.searchsorted(sessions, naive, side="left")

    out = pd.Series(pd.NaT, index=published_utc.index, dtype="datetime64[ns]")
    valid = idx < len(sessions)
    out.loc[valid] = sessions[idx[valid]]
    return out


def aggregate(df: pd.DataFrame, weight_col=None) -> pd.DataFrame:
    """Per-session sentiment statistics."""
    g = df.groupby("session")
    out = pd.DataFrame({
        "sentiment_mean": g["sentiment"].mean(),
        "sentiment_median": g["sentiment"].median(),
        "sentiment_std": g["sentiment"].std(),
        "sentiment_min": g["sentiment"].min(),
        "sentiment_max": g["sentiment"].max(),
        "article_count": g["sentiment"].size(),
        "prob_positive_mean": g["prob_positive"].mean(),
        "prob_negative_mean": g["prob_negative"].mean(),
        "prob_neutral_mean": g["prob_neutral"].mean(),
    })
    # Share of the day's articles on each side, which separates "three mildly
    # positive articles" from "one glowing and two scathing".
    out["pct_positive"] = g["sentiment"].apply(lambda s: (s > 0.1).mean())
    out["pct_negative"] = g["sentiment"].apply(lambda s: (s < -0.1).mean())

    if weight_col and weight_col in df.columns:
        # Upvote-weighted mean. A post with 900 upvotes was seen by far more
        # people than one with 2, so weighting by reach is at least as
        # defensible as treating every post equally — but which is right is an
        # empirical question, so both are kept and can be compared.
        #
        # log1p, not raw score: upvotes are heavy-tailed, and a single viral
        # post would otherwise decide the entire day on its own. Clipped at
        # zero first because Reddit scores can be negative, and a negative
        # weight would flip the sign of that post's sentiment.
        w = np.log1p(df[weight_col].fillna(0).clip(lower=0))
        tmp = df.assign(_w=w, _sw=df["sentiment"] * w)
        gw = tmp.groupby("session")
        wsum = gw["_w"].sum()
        out["sentiment_weighted"] = (gw["_sw"].sum() / wsum.replace(0, np.nan))
        # Falls back to the unweighted mean when every post that day scored 0.
        out["sentiment_weighted"] = out["sentiment_weighted"].fillna(out["sentiment_mean"])
        out["score_sum"] = gw[weight_col].sum()
        out["score_median"] = gw[weight_col].median()
    return out


def main():
    ap = argparse.ArgumentParser(description="Build the daily sentiment + price dataset.")
    ap.add_argument("--only", nargs="*", help="Restrict to these tickers")
    ap.add_argument("--source", choices=list(SOURCES), default="news",
                    help="Which scored corpus to aggregate (default news)")
    ap.add_argument("--cutoff-hour", type=int, default=CUTOFF_HOUR,
                    help="Market close hour in US/Eastern (default 16)")
    args = ap.parse_args()

    spec = SOURCES[args.source]
    stocks = select_stocks(args.only)
    panels, summary = [], []

    for stock in stocks:
        fmt = dict(sym=stock.symbol, start=START_DATE_STR, end=END_DATE_STR)
        scored = os.path.join(ticker_dir(stock.symbol), spec["scored"].format(**fmt))
        if not os.path.exists(scored):
            print(f"{stock.symbol}: no scored file — run {spec['needs']} first")
            continue

        prices, price_path = load_prices(stock)
        sessions = prices["date"].values.astype("datetime64[ns]")

        keep = {spec["date_col"], "sentiment", "prob_positive", "prob_negative",
                "prob_neutral"}
        if spec["weight_col"]:
            keep.add(spec["weight_col"])
        news = pd.read_csv(scored, usecols=lambda c: c in keep)
        news = news.rename(columns={spec["date_col"]: "Date"})
        news["Date"] = pd.to_datetime(news["Date"], utc=True, format="mixed", errors="coerce")
        news = news.dropna(subset=["Date", "sentiment"])

        news["session"] = assign_sessions(news["Date"], sessions, args.cutoff_hour)
        dropped = news["session"].isna().sum()   # published after the last session
        news = news.dropna(subset=["session"])

        daily = aggregate(news, weight_col=spec["weight_col"])

        # Reindex onto every trading day: days with no news become NaN rows,
        # which is the point — they must be visible, not absent.
        out = prices.merge(daily, left_on="date", right_index=True, how="left")
        out["article_count"] = out["article_count"].fillna(0).astype(int)
        out["source"] = args.source

        # market_prices.py already writes ticker/exchange columns; the FNSPID
        # price files do not. Set the values either way, then move them to the
        # front rather than inserting, so neither case collides.
        out["ticker"] = stock.symbol
        out["category"] = stock.category
        lead = ["date", "ticker", "category"]
        out = out[lead + [c for c in out.columns if c not in lead]]

        # Forward-looking return, for convenience at modelling time. Sentiment
        # on session d is known before d closes, so d's own return is the first
        # it could plausibly predict; next_day_return is the stricter target.
        price_col = "adj_close" if "adj_close" in out.columns else "close"
        out["next_day_return"] = out[price_col].pct_change().shift(-1)

        dst = os.path.join(ticker_dir(stock.symbol), spec["output"].format(**fmt))
        out.to_csv(dst, index=False)
        panels.append(out)

        with_news = int((out["article_count"] > 0).sum())
        # Sanity check, not a result: if sentiment carries any signal at all,
        # this should not be exactly zero. Do not read much into the magnitude.
        sub = out.dropna(subset=["sentiment_mean", "next_day_return"])
        corr = (sub["sentiment_mean"].corr(sub["next_day_return"])
                if len(sub) > 30 else float("nan"))

        print(f"\n{stock.symbol}  ({os.path.basename(price_path)})")
        print(f"  sessions            {len(out):,}")
        print(f"  with news           {with_news:,} ({with_news/len(out)*100:.1f}%)")
        print(f"  articles assigned   {len(news):,}" +
              (f"  ({dropped} after last session)" if dropped else ""))
        print(f"  median articles/day {out.loc[out.article_count>0,'article_count'].median():.0f}")
        print(f"  corr(sentiment, next-day return) {corr:+.4f}")
        print(f"  -> {dst}")

        summary.append({
            "ticker": stock.symbol,
            "category": stock.category,
            "sessions": len(out),
            "sessions_with_news": with_news,
            "pct_covered": round(with_news / len(out) * 100, 1),
            "articles": len(news),
            "source": args.source,
            "mean_sentiment_per_session": round(float(out["sentiment_mean"].mean()), 4),
            "corr_next_day_return": round(float(corr), 4) if corr == corr else None,
        })

    if panels:
        panel = pd.concat(panels, ignore_index=True)
        ppath = os.path.join(REPORT_DIR, spec["panel"].format(
            sym="", start=START_DATE_STR, end=END_DATE_STR))
        panel.to_csv(ppath, index=False)

        rep = pd.DataFrame(summary)
        spath = os.path.join(REPORT_DIR, spec["coverage"])
        rep.to_csv(spath, index=False)

        print(f"\n=== Daily dataset ===\n{rep.to_string(index=False)}")
        print(f"\nPanel -> {ppath}")
        print(f"Summary -> {spath}")
        print("\nDays without news are NaN, not 0 — article_count distinguishes them.")


if __name__ == "__main__":
    main()