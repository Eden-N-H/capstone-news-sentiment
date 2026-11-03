"""
Clean the filtered news CSVs — no 23 GB rescan required.

fnspid_filter.py casts a wide net on purpose, because FNSPID's Stock_symbol
tagging is too incomplete to rely on: AAPL is tagged in 23 of 60 months, BLK
and JPM in 18. The keyword scan takes every ticker to 60/60, but brings three
kinds of contamination with it. This script removes them, working from the
saved CSVs so iterating is seconds rather than another full pass.

  1. ANALYST NOTES. "JP Morgan Maintains Prologis (PLD) Overweight" mentions
     JPMorgan but is about Prologis. Scoring it measures JPMorgan's opinion of
     another company. Structural for investment banks, which appear in
     headlines about everyone else. Driven by title_exclude in config.py.

  2. MULTI-TICKER ROUNDUPS. "Zacks Analyst Blog Highlights: Alphabet, Apple,
     Amazon" and "Technology Sector Update for 06/26/2023: ENFN, ..." carry
     market-wide sentiment, not company-specific sentiment.

  3. NON-ENGLISH. The corpus contains Russian and German articles. An English
     sentiment model scores them as noise.

Rows are flagged and, by default, dropped. Use --keep-flagged to write the
flags without dropping, if you would rather decide during preprocessing.

Run:
    python news_quality_filter.py
    python news_quality_filter.py --tiers symbol keyword_title
    python news_quality_filter.py --keep-flagged
"""

import argparse
import glob
import os
import re

import pandas as pd

from config import END_DATE_STR, REPORT_DIR, START_DATE_STR, select_stocks, ticker_dir

# Headlines that are inherently about a basket rather than one company.
ROUNDUP_PATTERNS = [
    r"\bZacks\b.{0,30}\b(Analyst Blog|Highlights)\b",
    r"\bSector Update\b",
    r"\bCavalcade of Earnings\b",
    r"^US STOCKS-",
    r"\bMarket (Update|Wrap|Recap|Close)\b",
    r"\b(Top|Best|Worst)\s+\d+\s+Stocks\b",
    r"\b\d+\s+Stocks?\s+(To|That|Are)\b",
    r"\bMidday (Movers|Update)\b",
    r"\bStocks? (Moving|Making) (Big )?Moves\b",
    r"\bPre-?Market (Movers|Update)\b",
]
def _no_groups(pattern: str) -> str:
    """Turn capturing groups into non-capturing ones.

    pandas' str.contains warns when a pattern has match groups, since it looks
    like the caller meant str.extract.
    """
    return re.sub(r"\((?!\?)", "(?:", pattern)


ROUNDUP_RE = re.compile(
    "|".join(f"(?:{_no_groups(p)})" for p in ROUNDUP_PATTERNS), re.IGNORECASE
)

# Three or more comma-separated ticker-like tokens is a listing, not a story.
# {2,5} rather than {1,5}: single capital letters match ordinary punctuation and
# initialisms, which cost AAPL 1,295 articles and a month of coverage.
TICKER_LIST_RE = re.compile(r"\b[A-Z]{2,5}\b(?:\s*,\s*\b[A-Z]{2,5}\b){2,}")

# Headline shapes where the company is mentioned but is not the subject. These
# are templated per stock from its title_keywords, because they generalise:
# any company large enough to move another company's stock shows up this way.
#
#   "Energizer shares are trading lower after JP Morgan downgrade"
#       -> sentiment belongs to Energizer, not JPMorgan
#   "Weatherly Asset Management L. P. Buys JPMorgan"
#       -> a 13F filing report about Weatherly's portfolio
GENERIC_EXCLUDE_TEMPLATES = [
    # Another company's shares reacting to an action by this one.
    r"\bshares?\b.{{0,40}}\b(?:trading|moving|jump|jumped|fall|fell|rise|rose|"
    r"climb|climbed|drop|dropped|slide|slid|surge|surged|sink|sank)\b.{{0,60}}\b{kw}\b",
    r"\b{kw}\b.{{0,60}}\b(?:downgrade|upgrade|price target|PT)\b.{{0,40}}\bshares?\b",
    # 13F filing reports: "Wedgewood Partners Buys NVIDIA, Alphabet, Sells ..."
    # Plural verbs only. Bare "Buy" and "Sell" appear in ordinary headlines
    # ("Time to Buy Goldman Sachs or BlackRock?") and must not match.
    r"\b(?:Buys|Sells|Acquires|Trims|Divests|Adds to)\b.{{0,70}}\b{kw}\b",
    # The company acting on someone else's stock: "BlackRock Increases Position
    # in Inseego". Requires an explicit holdings noun, so "Alphabet Boosts
    # Google Cloud Offerings" is untouched.
    r"\b{kw}\b.{{0,40}}\b(?:Increases|Cuts|Trims|Updates|Boosts|Reduces|Raises|Lowers)\b"
    r"\s+(?:its\s+)?(?:Position|Stake|Holdings?)\b",
    r"\b{kw}\b.{{0,50}}\b13F\b",
    # Fund and ETF products named after the company rather than the company.
    r"\b{kw}\b\s+(?:ETF|Trust|Fund|Index)\b",
]

NON_ASCII_THRESHOLD = 0.10


def company_title_regex(stock):
    """
    Any name the company goes by, for testing whether the headline names it.

    Draws on the ticker, title_keywords and the Reddit search terms, so
    "Rockstar" and "GTA" count as naming Take-Two.
    """
    names = {stock.symbol}
    names.update(stock.title_keywords)
    names.update(t.strip('"') for t in stock.home_terms)
    names.update(t.strip('"') for t in stock.strict_terms)
    names = {n for n in names if len(n) >= 2}
    pattern = "|".join(rf"\b{re.escape(n)}\b" for n in sorted(names, key=len, reverse=True))
    return re.compile(pattern, re.IGNORECASE)


def compile_excludes(stock):
    """Stock-specific patterns from config, plus the templated generic ones."""
    pats = list(stock.title_exclude)
    for kw in stock.title_keywords:
        kw_rx = re.escape(kw).replace(r"\ ", r"\s?")   # "JP Morgan" ~ "JPMorgan"
        for tmpl in GENERIC_EXCLUDE_TEMPLATES:
            pats.append(tmpl.format(kw=kw_rx))
    return [re.compile(p, re.IGNORECASE) for p in pats]


def flag(df: pd.DataFrame, stock, keyword_tiers_only: bool = False,
         check_title_names_company: bool = True) -> pd.DataFrame:
    """Add boolean columns for each contamination type."""
    titles = df["Article_title"].fillna("").astype(str)

    excludes = compile_excludes(stock)
    df["is_analyst_note"] = (
        titles.apply(lambda t: any(rx.search(t) for rx in excludes))
        if excludes else False
    )

    df["is_roundup"] = titles.str.contains(ROUNDUP_RE, na=False) | \
        titles.str.contains(TICKER_LIST_RE, na=False)

    if "non_ascii_ratio" in df.columns:
        df["is_non_english"] = df["non_ascii_ratio"].fillna(0) > NON_ASCII_THRESHOLD
    else:
        df["is_non_english"] = (
            titles.map(lambda t: sum(1 for c in t if ord(c) > 127) / max(len(t), 1))
            > NON_ASCII_THRESHOLD
        )

    # Applied to every tier, including symbol-tagged rows.
    #
    # These were originally restricted to keyword matches, on the assumption
    # that FNSPID's own tagging was trustworthy. It is not, for this purpose:
    # "Technology Sector Update for 03/10/2020: TTWO, MDLA, BLIN, CY" is tagged
    # TTWO and is still a roundup, and FinBERT scored two such headlines at
    # +0.93 and -0.97 — among the strongest sentiment in the whole TTWO
    # corpus, attached to text that says nothing about Take-Two. A correct tag
    # means the article mentions the company, not that the article is about it.
    if keyword_tiers_only:
        is_kw = df["match_type"].isin(["keyword_title", "keyword_summary"])
        df["is_analyst_note"] &= is_kw
        df["is_roundup"] &= is_kw

    # If the headline never names the company, the article is probably not
    # about it. FNSPID tags on a body mention, which is a relevance marker, not
    # a topicality one — "Video game publisher Electronic Arts raises annual
    # sales forecast" is tagged TTWO and scored +0.93, and "Wall St set to fall
    # as trade tensions add to virus woes" scored -0.96. Neither says anything
    # about Take-Two. This one rule catches both without pattern whack-a-mole.
    if check_title_names_company:
        name_rx = company_title_regex(stock)
        df["no_company_in_title"] = ~titles.str.contains(name_rx, na=False)
    else:
        df["no_company_in_title"] = False

    df["flagged"] = (df["is_analyst_note"] | df["is_roundup"]
                     | df["is_non_english"] | df["no_company_in_title"])
    return df


def summarise(df, label):
    d = pd.to_datetime(df["Date"], utc=True, format="mixed")
    return {
        "stage": label,
        "articles": len(df),
        "months": d.dt.tz_convert(None).dt.to_period("M").nunique(),
        "days": d.dt.date.nunique(),
    }


def main():
    ap = argparse.ArgumentParser(description="Clean the filtered news CSVs.")
    ap.add_argument("--only", nargs="*", help="Restrict to these tickers")
    ap.add_argument("--tiers", nargs="*",
                    default=["symbol", "keyword_title"],
                    help="match_type tiers to keep "
                         "(default: symbol keyword_title)")
    ap.add_argument("--allow-untitled", action="store_true",
                    help="Keep articles whose headline never names the company. "
                         "Raises recall, lowers precision.")
    ap.add_argument("--keyword-tiers-only", action="store_true",
                    help="Apply the analyst/roundup heuristics only to keyword "
                         "matches, trusting FNSPID's symbol tags. Not the "
                         "default: symbol-tagged roundups score strongly and "
                         "are not about the company.")
    ap.add_argument("--keep-flagged", action="store_true",
                    help="Write the flag columns but do not drop the rows")
    args = ap.parse_args()

    stocks = select_stocks(args.only)
    rows = []
    samples = []

    for stock in stocks:
        src = os.path.join(ticker_dir(stock.symbol),
                           f"{stock.symbol}_news_{START_DATE_STR}_{END_DATE_STR}.csv")
        if not os.path.exists(src):
            print(f"{stock.symbol}: {src} not found — run fnspid_filter.py first")
            continue

        df = pd.read_csv(src)
        before = summarise(df, "raw")

        df = df[df["match_type"].isin(args.tiers)].copy()
        after_tier = summarise(df, f"tiers={'+'.join(args.tiers)}")

        df = flag(df, stock, keyword_tiers_only=args.keyword_tiers_only,
                  check_title_names_company=not args.allow_untitled)
        n_analyst = int(df["is_analyst_note"].sum())
        n_roundup = int(df["is_roundup"].sum())
        n_foreign = int(df["is_non_english"].sum())
        n_untitled = int(df["no_company_in_title"].sum())

        # Keep a sample of what was removed, so the decision can be audited
        # rather than taken on trust.
        for reason in ("is_analyst_note", "is_roundup", "is_non_english",
                       "no_company_in_title"):
            hit = df[df[reason]]
            if not hit.empty:
                take = hit.sample(min(15, len(hit)), random_state=0)
                samples.append(pd.DataFrame({
                    "ticker": stock.symbol,
                    "reason": reason.replace("is_", ""),
                    "match_type": take["match_type"].values,
                    "Article_title": take["Article_title"].values,
                }))

        if not args.keep_flagged:
            df = df[~df["flagged"]].copy()
        final = summarise(df, "clean")

        out = os.path.join(ticker_dir(stock.symbol),
                           f"{stock.symbol}_news_clean_{START_DATE_STR}_{END_DATE_STR}.csv")
        df.to_csv(out, index=False)

        print(f"\n{stock.symbol}")
        print(f"  raw            {before['articles']:6,} arts  "
              f"{before['months']:2}/60 mo  {before['days']:4}/1826 days")
        print(f"  after tiers    {after_tier['articles']:6,} arts  "
              f"{after_tier['months']:2}/60 mo  {after_tier['days']:4}/1826 days")
        print(f"  flagged        analyst {n_analyst:,}, roundup {n_roundup:,}, "
              f"non-English {n_foreign:,}, unnamed-in-title {n_untitled:,}")
        print(f"  clean          {final['articles']:6,} arts  "
              f"{final['months']:2}/60 mo  {final['days']:4}/1826 days")
        print(f"  -> {out}")

        rows.append({
            "ticker": stock.symbol,
            "raw": before["articles"],
            "after_tiers": after_tier["articles"],
            "analyst_notes": n_analyst,
            "roundups": n_roundup,
            "non_english": n_foreign,
            "no_company_in_title": n_untitled,
            "clean": final["articles"],
            "months": final["months"],
            "days": final["days"],
        })

    if samples:
        spath = os.path.join(REPORT_DIR, "news_flagged_samples.csv")
        pd.concat(samples, ignore_index=True).to_csv(spath, index=False)
        print(f"\nFlagged examples for auditing -> {spath}")

    if rows:
        rep = pd.DataFrame(rows)
        path = os.path.join(REPORT_DIR, "news_quality.csv")
        rep.to_csv(path, index=False)
        print(f"\n=== Summary ===\n{rep.to_string(index=False)}")
        print(f"Saved -> {path}")

        thin = rep[rep["months"] < 60]
        if not thin.empty:
            print("\nLost month coverage — reconsider the tier policy for: "
                  + ", ".join(thin["ticker"]))


if __name__ == "__main__":
    main()