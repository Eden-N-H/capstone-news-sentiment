"""
Which tickers does FNSPID actually cover, and how well?

One streaming pass over the news CSV counting articles per Stock_symbol inside
the project date window. Two uses:

  1. Confirm the four NASDAQ names have enough coverage to model.
  2. Find replacement candidates for AFI and CBA. The proposal needs at least
     two "immaterial" companies — banks, insurers, asset managers, closed-end
     funds — with real article volume. Rather than guessing which are in the
     corpus, this lists them.

Only Date and Stock_symbol are read, so this is much faster than the full
filter, but it still reads the whole ~23 GB file. Both the download and the
scan are resumable: an interrupted run picks up where it stopped, so a dropped
connection costs minutes rather than hours.

Run:
    python fnspid_symbol_census.py
    python fnspid_symbol_census.py --candidates      # only the financials watchlist
    python fnspid_symbol_census.py --restart         # ignore the checkpoint
"""

import argparse
import os

import pandas as pd

from checkpoint import Checkpoint, download_file
from config import END_DATE_STR, RAW_DIR, REPORT_DIR, START_DATE_STR
from fnspid_filter import NEWS_RAW_PATH, NEWS_URL

CHUNK_SIZE = 500_000
CKPT_PATH = os.path.join(REPORT_DIR, "_census_checkpoint.json")
SAVE_EVERY_CHUNKS = 10

# Immaterial-organisation candidates: no consumer product driving the price, so
# they play the same role in the research design that AFI and CBA were meant to.
# Presence in FNSPID is exactly what this script checks — nothing here is
# assumed to be covered.
FINANCIALS_WATCHLIST = [
    # Big banks
    "JPM", "BAC", "WFC", "C", "USB", "PNC", "TFC", "SCHW",
    # Investment banks / asset managers
    "GS", "MS", "BLK", "BX", "KKR", "APO", "TROW", "BEN", "IVZ", "AMP",
    # Insurers
    "BRK-B", "BRK.B", "AIG", "MET", "PRU", "ALL", "TRV", "CB",
    # Exchanges, index and ratings businesses
    "SPGI", "MCO", "MSCI", "ICE", "CME", "NDAQ", "CBOE",
    # Closed-end funds and diversified holdings (closest AFI analogues)
    "ADX", "CET", "GAB", "RIT", "USA", "SOR",
]


def census(resume: bool = True) -> pd.DataFrame:
    download_file(NEWS_URL, NEWS_RAW_PATH)

    ckpt = Checkpoint(CKPT_PATH)
    state = ckpt.load() if resume else {}
    totals = {k: int(v) for k, v in state.get("totals", {}).items()}
    months = {k: set(v) for k, v in state.get("months", {}).items()}
    rows_done = int(state.get("rows_done", 0))

    if rows_done:
        print(f"Resuming census from row {rows_done:,} "
              f"({len(totals):,} symbols counted so far)")

    start_ts = pd.Timestamp(START_DATE_STR, tz="UTC")
    end_ts = pd.Timestamp(END_DATE_STR, tz="UTC") + pd.Timedelta(days=1)

    def persist(scanned):
        ckpt.save({
            "rows_done": scanned,
            "totals": totals,
            "months": {k: sorted(v) for k, v in months.items()},
        })

    # skiprows re-reads the already-processed lines but skips parsing them into
    # a frame, which is much cheaper than redoing the date parsing and grouping.
    read_kwargs = dict(chunksize=CHUNK_SIZE, usecols=["Date", "Stock_symbol"],
                       low_memory=False)
    if rows_done:
        read_kwargs["skiprows"] = range(1, rows_done + 1)

    scanned = rows_done
    chunk_no = 0

    try:
        for chunk in pd.read_csv(NEWS_RAW_PATH, **read_kwargs):
            scanned += len(chunk)
            chunk_no += 1

            chunk["Date"] = pd.to_datetime(chunk["Date"], errors="coerce",
                                           utc=True, format="mixed")
            chunk = chunk[(chunk["Date"] >= start_ts) & (chunk["Date"] < end_ts)]

            if not chunk.empty:
                chunk["sym"] = chunk["Stock_symbol"].astype("string").str.upper().str.strip()
                chunk = chunk.dropna(subset=["sym"])
                chunk["ym"] = chunk["Date"].dt.strftime("%Y-%m")

                for sym, count in chunk["sym"].value_counts().items():
                    totals[sym] = totals.get(sym, 0) + int(count)
                for sym, group in chunk.groupby("sym")["ym"]:
                    months.setdefault(sym, set()).update(group.unique())

            if chunk_no % SAVE_EVERY_CHUNKS == 0:
                persist(scanned)

            print(f"\r  scanned {scanned:,} rows, {len(totals):,} distinct symbols",
                  end="", flush=True)
    except KeyboardInterrupt:
        persist(scanned)
        print(f"\n\nInterrupted at row {scanned:,}. Re-run to resume from here.")
        raise

    print()
    ckpt.clear()

    # 60 months in the window; a symbol covering few of them cannot support a
    # daily-resolution model no matter how large its total count looks.
    rows = [{
        "ticker": sym,
        "articles": n,
        "months_covered": len(months.get(sym, ())),
        "pct_of_months": round(len(months.get(sym, ())) / 60 * 100, 1),
        "avg_per_month": round(n / max(len(months.get(sym, ())), 1), 1),
    } for sym, n in totals.items()]

    return pd.DataFrame(rows).sort_values("articles", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Count FNSPID articles per ticker.")
    ap.add_argument("--candidates", action="store_true",
                    help="Print only the financials watchlist")
    ap.add_argument("--top", type=int, default=40, help="Rows to print (default 40)")
    ap.add_argument("--restart", action="store_true",
                    help="Ignore the saved checkpoint and rescan from row 0")
    args = ap.parse_args()

    df = census(resume=not args.restart)
    path = os.path.join(REPORT_DIR, "fnspid_symbol_census.csv")
    df.to_csv(path, index=False)

    if args.candidates:
        watch = {w.upper() for w in FINANCIALS_WATCHLIST}
        view = df[df["ticker"].isin(watch)]
        print("\n=== Immaterial-organisation candidates present in FNSPID ===")
        print(view.to_string(index=False) if not view.empty else "  none of the watchlist found")
        missing = sorted(watch - set(df["ticker"]))
        if missing:
            print(f"\nNot in FNSPID: {', '.join(missing)}")
    else:
        print(f"\n=== Top {args.top} tickers by article count ===")
        print(df.head(args.top).to_string(index=False))

    print(f"\n{len(df):,} distinct symbols in window. Saved -> {path}")


if __name__ == "__main__":
    main()
