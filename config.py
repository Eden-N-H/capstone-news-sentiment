"""
Shared configuration for the capstone data collection pipeline.

Every collection script imports from here so that the ticker list, the date
window and the output layout stay consistent across news, price and Reddit
pulls. Edit this file, not the individual scripts.

Stocks are adapted from section 3.2.2 of the project proposal:

  * Pearl Abyss (263750:KOSDAQ) dropped — its narrative hinges on a March 2026
    launch, outside the FNSPID-aligned window below.
  * AFI and CBA (both ASX) replaced by BLK and JPM. FNSPID is a US corpus and
    carries no news for ASX listings, which would have left the two
    "immaterial organisation" stocks with a price series and Reddit posts but
    no news arm — exactly the comparison section 3.4 depends on. BLK is an
    asset manager (the AFI analogue), JPM a retail-and-investment bank (the
    CBA analogue).

FNSPID's Stock_symbol tagging is badly incomplete, and this drives the
title_keywords below. Measured over this window (reports/fnspid_symbol_census.csv
vs reports/fnspid_coverage.csv):

    ticker   symbol-tagged        with keyword scan
    AAPL     9,338 / 23 months    16,371 / 60 months
    NVDA     9,533 / 47 months    13,821 / 60 months
    JPM      2,384 / 18 months    12,448 / 60 months
    BLK        876 / 18 months     4,119 / 60 months

The articles were always in the corpus; they simply were not tagged. Several
unrelated symbols cluster at exactly 18 months too (BAC, IVZ, MET, AMP), so
this is systematic. Never judge a candidate ticker on its symbol-tagged count
alone — and keep title_keywords populated for every stock.
"""

from dataclasses import dataclass, field
import datetime as dt
import os

# ---------------------------------------------------------------------------
# Date window
# ---------------------------------------------------------------------------
# FNSPID's news coverage effectively stops at the end of 2023, so the original
# proposal window (May 2021 - May 2026) is not achievable with this corpus.
# This is the shifted 5-year window. Section 3.2.4 of the proposal needs to be
# updated to match.
START_DATE = dt.date(2019, 1, 1)
END_DATE = dt.date(2023, 12, 31)

START_DATE_STR = START_DATE.isoformat()
END_DATE_STR = END_DATE.isoformat()

# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------
RAW_DIR = "fnspid_raw"          # big downloads, gitignore this
DATA_DIR = "data"               # per-ticker filtered outputs
REPORT_DIR = "reports"          # coverage summaries, volume checks

for _d in (RAW_DIR, DATA_DIR, REPORT_DIR):
    os.makedirs(_d, exist_ok=True)


def ticker_dir(symbol: str) -> str:
    """data/<TICKER>/ — created on demand."""
    path = os.path.join(DATA_DIR, symbol)
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Subreddit pools
# ---------------------------------------------------------------------------
# Arctic Shift has no global full-text search: keyword queries only work when
# scoped to a subreddit. So coverage is entirely a function of how good this
# list is, and there is no way to "search all of Reddit" as a safety net.
#
# Two tiers per stock:
#   home      -> the stock's natural venues, searched with the loose query
#   discovery -> plausible but unproven venues, searched with the strict query
#
# Cast the net wide here and let reddit_volume_check.py prune it. Counting is
# cheap; pulling full post text from 40 dead subreddits is not.
#
# Watch for subreddits that did not exist or were inactive across 2019-2023 —
# the volume check will show them as zero, and they should be dropped rather
# than carried into the pull.

# ---------------------------------------------------------------------------
# Reddit: dump-based collection
# ---------------------------------------------------------------------------
# The Arctic Shift API times out unpredictably — it failed on a one-week,
# single-term, minimal-field query against a medium subreddit — so Reddit data
# comes from downloaded per-subreddit dumps instead.
#
# That changes the economics of the subreddit list completely. Over the API an
# extra subreddit costs a handful of requests; as a dump it costs a download
# measured in gigabytes. The 170-subreddit list below was built for the API and
# is unusable for dumps, so DUMP_SUBREDDITS is the shortlist actually collected:
# the five core finance subs (which serve all six tickers) plus each stock's
# main venues.
#
# Investment-focused subreddits only. Product subreddits (r/nvidia, r/apple,
# r/GTA) were tested and rejected: a sample of 15 matched r/nvidia posts was
# 15/15 hardware support — "Is my GPU dying?", "Annoying coil whine", "driver
# causing Bluescreen". FinBERT scores those strongly negative, and r/nvidia
# alone contributed 70,572 posts, enough to swamp every finance subreddit
# combined. Worse, driver complaints spike at product launches, so the
# contamination correlates with exactly the events of interest rather than
# averaging out.
#
# r/personalfinance and r/banking are excluded for the same reason in a
# different form: they are consumer-banking subs where "Chase closed my
# account" dominates, which is not sentiment about JPMorgan as an equity.
#
# Download submissions (NOT comments) for these from
#     https://arctic-shift.photon-reddit.com/download-tool
# into one folder, then run reddit_dump_filter.py.
DUMP_SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "StockMarket",
    "options",
    "ValueInvesting",
    "SecurityAnalysis",
    "Bogleheads",
    "ETFs",
    "dividends",
]


# Searched for every stock, loose query.
CORE_FINANCE_SUBS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "StockMarket",
    "options",
    "thetagang",
    "SecurityAnalysis",
    "ValueInvesting",
    "Daytrading",
    "swingtrading",
    "stockstobuytoday",
]

# Searched for every stock, strict query — broader and noisier.
BROAD_FINANCE_SUBS = [
    "finance",
    "economics",
    "Economics",
    "economy",
    "business",
    "Bogleheads",
    "dividends",
    "ETFs",
    "financialindependence",
    "personalfinance",
    "InvestmentClub",
    "FinancialCareers",
    "wallstreetbetsOGs",
    "Wallstreetbetsnew",
    "smallstreetbets",
    "TheRaceTo10Million",
]

# General technology venues shared by AAPL, GOOG and NVDA.
TECH_SUBS = [
    "technology",
    "technews",
    "gadgets",
    "hardware",
    "computers",
    "Futurology",
    "singularity",
    "artificial",
]


# ---------------------------------------------------------------------------
# Stock definitions
# ---------------------------------------------------------------------------
@dataclass
class Stock:
    symbol: str                  # canonical label used for directories/files
    name: str
    category: str                # material | immaterial (the 3.4 contrast)
    exchange: str
    yahoo_symbol: str            # for yfinance
    fnspid_symbols: list = field(default_factory=list)      # Stock_symbol values to match
    fnspid_price_files: list = field(default_factory=list)  # stems inside full_history.zip
    title_keywords: list = field(default_factory=list)
    # Regexes for headlines that mention the company but are not ABOUT it.
    # Investment banks are the acute case: "JP Morgan Maintains Prologis
    # Overweight" is an analyst note about Prologis, and scoring its sentiment
    # measures JPMorgan's opinion of another company. Applied to the keyword
    # tiers only; symbol-tagged rows are left alone.
    title_exclude: list = field(default_factory=list)      # fallback title scan
    # Terms are stored as lists, not as a pre-joined "A OR B" string. Arctic
    # Shift's docs warn that the boolean syntax section is out of date and that
    # post search is currently "simply keyword search" — so an OR query may not
    # be parsed as a union at all. Keeping terms separate lets the scripts
    # issue one request per term and union the post IDs locally, which is
    # correct regardless of how the server interprets operators, and cheaper
    # per request as a side effect.
    home_terms: list = field(default_factory=list)
    strict_terms: list = field(default_factory=list)
    subreddits: list = field(default_factory=list)
    discovery_subreddits: list = field(default_factory=list)

    @property
    def in_fnspid(self) -> bool:
        return bool(self.fnspid_symbols)

    @property
    def all_subreddits(self) -> list:
        seen, out = set(), []
        for s in list(self.subreddits) + list(self.discovery_subreddits):
            if s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
        return out

    @property
    def dump_subreddits(self) -> list:
        """
        The stock's subreddits, narrowed to those actually downloaded.

        Keeps reddit_dump_filter.py from reporting 170 subreddits as empty when
        only 14 were collected — a missing dump and a dead subreddit are not
        the same thing and should not look alike in the output.
        """
        want = {s.lower() for s in DUMP_SUBREDDITS}
        return [s for s in self.all_subreddits if s.lower() in want]

    def terms_for(self, subreddit: str) -> list:
        """
        Loose terms in home subreddits, strict terms everywhere else.

        Bare tickers are badly ambiguous outside their own context: "GTA" is the
        Greater Toronto Area, "BLK" is used as slang, "Chase" is a common verb
        and surname. Using the loose terms in r/economics would bury the real
        posts under noise that a BERT sentiment pass would happily score anyway.
        """
        if not self.strict_terms:
            return list(self.home_terms)
        home = {s.lower() for s in self.subreddits}
        return list(self.home_terms if subreddit.lower() in home else self.strict_terms)

    def query_for(self, subreddit: str) -> str:
        """The same terms joined with OR, for the combined-query fast path."""
        return " OR ".join(self.terms_for(subreddit))


STOCKS = [
    Stock(
        symbol="AAPL",
        name="Apple Inc",
        category="material",
        exchange="NASDAQ",
        yahoo_symbol="AAPL",
        fnspid_symbols=["AAPL"],
        fnspid_price_files=["AAPL"],
        title_keywords=["Apple Inc", "iPhone", "AAPL"],
        home_terms=['AAPL', 'Apple', 'iPhone', 'Macbook'],
        strict_terms=['AAPL', '"Apple stock"', '"Apple Inc"', 'iPhone'],
        subreddits=CORE_FINANCE_SUBS + [
            "apple", "iphone", "ios", "ipad", "mac", "macbook", "MacOS",
            "AppleWatch", "airpods", "applehelp", "AppleTV", "iOSProgramming",
            "AppleMusic", "iphonehelp",
        ],
        discovery_subreddits=BROAD_FINANCE_SUBS + TECH_SUBS + [
            "Android", "smartphones", "GooglePixel", "apolloapp",
            "consumerism", "BuyItForLife", "apple_jailbreak",
        ],
    ),
    Stock(
        symbol="TTWO",
        name="Take-Two Interactive Software, Inc",
        category="material",
        exchange="NASDAQ",
        yahoo_symbol="TTWO",
        fnspid_symbols=["TTWO"],
        fnspid_price_files=["TTWO"],
        title_keywords=["Take-Two", "Rockstar Games", "TTWO"],
        home_terms=['TTWO', '"Take-Two"', 'Rockstar', 'GTA'],
        strict_terms=['TTWO', '"Take-Two"', '"Rockstar Games"', '"Grand Theft Auto"'],
        subreddits=CORE_FINANCE_SUBS + [
            # Take-Two labels: Rockstar, 2K, Private Division, Zynga (2022+)
            "GTA", "GTAV", "GTA6", "gtaonline", "rockstar", "GtaMemes",
            "reddeadredemption", "RedDeadOnline", "NBA2k", "borderlands",
            "civ", "Xcom", "Mafia_Game", "KerbalSpaceProgram", "bioshock",
            "gaming", "Games", "videogames", "pcgaming", "truegaming",
            "GamingLeaksAndRumours", "gamingnews", "patientgamers",
        ],
        discovery_subreddits=BROAD_FINANCE_SUBS + [
            "PS5", "PS4", "xbox", "XboxSeriesX", "PlayStation", "consoles",
            "Steam", "SteamDeck", "gamingcirclejerk", "Games_Deals",
            "MMORPG", "zynga",
        ],
    ),
    Stock(
        symbol="GOOG",
        name="Alphabet Inc Class C",
        category="material",
        exchange="NASDAQ",
        yahoo_symbol="GOOG",
        # FNSPID tags Alphabet articles under both share classes.
        fnspid_symbols=["GOOG", "GOOGL"],
        fnspid_price_files=["GOOG", "GOOGL"],
        title_keywords=["Alphabet", "Google"],
        home_terms=['GOOG', 'GOOGL', 'Alphabet', 'Google'],
        strict_terms=['GOOG', 'GOOGL', 'Alphabet', '"Google stock"'],
        subreddits=CORE_FINANCE_SUBS + [
            "google", "Android", "GooglePixel", "chrome", "chromeos",
            "GoogleAssistant", "googlehome", "GoogleMaps", "youtube",
            "androidapps", "androiddev", "GoogleFi", "nestthermostat",
            "GoogleCloud", "GoogleWorkspace", "waymo",
        ],
        discovery_subreddits=BROAD_FINANCE_SUBS + TECH_SUBS + [
            "MachineLearning", "deeplearning", "selfdrivingcars", "Bard",
            "privacy", "degoogle", "SEO", "PPC", "advertising", "antitrust",
            "youtubers", "NewTubers",
        ],
    ),
    Stock(
        symbol="NVDA",
        name="NVIDIA Corp",
        category="material",
        exchange="NASDAQ",
        yahoo_symbol="NVDA",
        fnspid_symbols=["NVDA"],
        fnspid_price_files=["NVDA"],
        title_keywords=["Nvidia", "NVDA"],
        home_terms=['NVDA', 'Nvidia', 'GeForce', 'RTX'],
        strict_terms=['NVDA', 'Nvidia', 'GeForce'],
        subreddits=CORE_FINANCE_SUBS + [
            "nvidia", "pcmasterrace", "buildapc", "buildapcsales", "hardware",
            "pcgaming", "overclocking", "sffpc", "PcBuild", "Amd", "intel",
            "GraphicsProgramming", "linux_gaming", "battlestations",
            "hardwareswap", "PCBuilds",
        ],
        discovery_subreddits=BROAD_FINANCE_SUBS + TECH_SUBS + [
            "MachineLearning", "deeplearning", "LocalLLaMA",
            "StableDiffusion", "datascience", "CUDA", "gpumining",
            "EtherMining", "CryptoCurrency", "VFIO", "editors",
            "3Dmodeling", "blender",
        ],
    ),
    Stock(
        symbol="BLK",
        name="BlackRock Inc",
        category="immaterial",
        exchange="NYSE",
        yahoo_symbol="BLK",
        fnspid_symbols=["BLK"],
        fnspid_price_files=["BLK"],
        # These keywords are what make BLK viable. Symbol-tagged rows alone
        # cover just 18 of 60 months; adding the title/summary scan takes it to
        # 60/60 and from 876 articles to 4,119.
        title_keywords=["BlackRock"],
        title_exclude=[
            # Requires a rating word as well as the verb — "Can BlackRock
            # Maintain Its Profitability?" is genuine BLK analysis and must
            # not be caught by "Maintain" alone.
            r"\bBlackRock\b.{0,60}\b(?:Maintains|Upgrades|Downgrades|Initiates)\b"
            r".{0,40}\b(?:Overweight|Underweight|Neutral|Buy|Sell|Hold|Outperform|"
            r"Underperform|Rating|Coverage|Price Target|PT)\b",
        ],
        home_terms=['BLK', 'BlackRock', '"Larry Fink"'],
        strict_terms=['BlackRock', '"Larry Fink"'],
        subreddits=CORE_FINANCE_SUBS + BROAD_FINANCE_SUBS + [
            "ETFs", "IndexFunds", "portfolios", "AssetManagement", "quant",
            "Fire", "Vanguard",
        ],
        discovery_subreddits=[
            "banking", "Banking", "Superstonk", "REBubble", "RealEstate",
            "conspiracy", "Economics", "politics", "news", "worldnews",
            "ESG", "sustainability", "antiwork", "LateStageCapitalism",
            "Bitcoin", "CryptoCurrency",
        ],
    ),
    Stock(
        symbol="JPM",
        name="JPMorgan Chase & Co",
        category="immaterial",
        exchange="NYSE",
        yahoo_symbol="JPM",
        fnspid_symbols=["JPM"],
        fnspid_price_files=["JPM"],
        # As with BLK: 18/60 months symbol-tagged, 60/60 with the keyword scan,
        # and 2,384 articles becomes 12,448.
        title_keywords=["JPMorgan", "JP Morgan"],
        title_exclude=[
            # Analyst actions issued BY JPMorgan about other companies.
            r"\bJ\.?\s?P\.?\s?Morgan\b.{0,60}\b(?:Maintains|Upgrades|Downgrades|"
            r"Initiates|Reiterates|Resumes|Assumes|Reinstates)\b",
            # Rating-word form catches "JP Morgan Raises PT on X" without
            # catching "JPMorgan Raises Dividend".
            r"\bJ\.?\s?P\.?\s?Morgan\b.{0,60}\b(?:Raises|Lowers|Cuts|Boosts|Starts|Begins)\b"
            r".{0,40}\b(?:Overweight|Underweight|Neutral|Buy|Sell|Hold|Outperform|"
            r"Underperform|Rating|Coverage|Price Target|PT)\b",
            r"\b(Maintains?|Upgrades?|Downgrades?|Initiates?)\b.{0,40}\bby\s+J\.?P\.?\s?Morgan\b",
        ],
        # "Chase" is a verb and a surname, so it only appears in the loose terms.
        home_terms=['JPM', 'JPMorgan', '"JP Morgan"', '"Jamie Dimon"'],
        strict_terms=['JPM', 'JPMorgan', '"JP Morgan"', '"Jamie Dimon"'],
        subreddits=CORE_FINANCE_SUBS + BROAD_FINANCE_SUBS + [
            "banking", "Banking", "CreditCards", "churning", "creditcardchurning",
            "personalfinance", "FinancialCareers", "quant", "Accounting",
        ],
        discovery_subreddits=[
            "Economics", "economy", "news", "worldnews", "politics",
            "Superstonk", "wallstreetbetsELITE", "SecurityAnalysis",
            "financialmodelling", "CFA", "MBA", "consulting",
            "Bitcoin", "CryptoCurrency", "RealEstate", "Mortgages",
        ],
    ),
]

STOCKS_BY_SYMBOL = {s.symbol: s for s in STOCKS}

# Every Stock_symbol value we care about, mapped back to its canonical ticker.
FNSPID_SYMBOL_MAP = {
    alias: s.symbol
    for s in STOCKS
    for alias in s.fnspid_symbols
}


def select_stocks(only=None):
    """Return the stock list, optionally narrowed to a list of symbols."""
    if not only:
        return list(STOCKS)
    missing = [s for s in only if s not in STOCKS_BY_SYMBOL]
    if missing:
        raise SystemExit(f"Unknown ticker(s): {', '.join(missing)}")
    return [STOCKS_BY_SYMBOL[s] for s in only]