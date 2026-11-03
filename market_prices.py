"""
Daily OHLCV prices for every selected stock, via yfinance.

Why this exists alongside fnspid_filter.py: FNSPID's full_history.zip is built
around NASDAQ listings, and BLK and JPM are NYSE. If either is missing from the
archive this is the fallback, and running it for all six also guarantees one
schema and one adjustment convention across the whole set.

All six stocks are USD on the US trading calendar, so the series align without
reindexing. Still scale each stock separately — the proposal compares model
accuracy across stocks, and a shared scaler would let NVDA's range dominate.

Requirements:
    pip install yfinance pandas

Run:
    python market_prices.py
    python market_prices.py --missing-only
"""

import argparse
import os

import pandas as pd

from config import END_DATE, END_DATE_STR, REPORT_DIR, START_DATE_STR, select_stocks, ticker_dir

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    raise SystemExit("yfinance is not installed. Run: pip install yfinance")

# yfinance treats `end` as exclusive, so push it one day past the window.
YF_END = (END_DATE + pd.Timedelta(days=1)).isoformat()


def fetch(stock) -> pd.DataFrame:
    print(f"Fetching {stock.yahoo_symbol} ({stock.exchange})...")
    df = yf.download(
        stock.yahoo_symbol,
        start=START_DATE_STR,
        end=YF_END,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        print(f"  no data returned for {stock.yahoo_symbol}")
        return pd.DataFrame()

    # Recent yfinance versions return a MultiIndex column even for one ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    df["ticker"] = stock.symbol
    df["yahoo_symbol"] = stock.yahoo_symbol
    df["exchange"] = stock.exchange

    # Daily average price, which is what section 3.3.2 specifies as the datapoint.
    if {"high", "low"}.issubset(df.columns):
        df["avg_price"] = (df["high"] + df["low"]) / 2
    if "adj_close" in df.columns:
        df["daily_return"] = df["adj_close"].pct_change()
    elif "close" in df.columns:
        df["daily_return"] = df["close"].pct_change()

    print(f"  {len(df):,} trading days, {df['date'].min().date()} -> {df['date'].max().date()}")
    return df


def main():
    ap = argparse.ArgumentParser(description="Fetch daily prices for the capstone tickers.")
    ap.add_argument("--only", nargs="*", help="Restrict to these tickers")
    ap.add_argument("--missing-only", action="store_true",
                    help="Only fetch stocks with no FNSPID price file configured")
    args = ap.parse_args()

    stocks = select_stocks(args.only)
    if args.missing_only:
        stocks = [s for s in stocks if not s.fnspid_price_files]

    summary = []
    for s in stocks:
        df = fetch(s)
        if df.empty:
            summary.append({"ticker": s.symbol, "exchange": s.exchange, "rows": 0,
                            "first": None, "last": None})
            continue
        path = os.path.join(ticker_dir(s.symbol),
                            f"{s.symbol}_prices_yf_{START_DATE_STR}_{END_DATE_STR}.csv")
        df.to_csv(path, index=False)
        print(f"  Saved -> {path}")
        summary.append({"ticker": s.symbol, "exchange": s.exchange, "rows": len(df),
                        "first": df["date"].min().date(), "last": df["date"].max().date()})

    out = pd.DataFrame(summary)
    path = os.path.join(REPORT_DIR, "price_coverage.csv")
    out.to_csv(path, index=False)
    print("\n=== Price coverage ===")
    print(out.to_string(index=False))
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
