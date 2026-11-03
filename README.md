# Predicting Stock Fluctuations from News and Social Media Sentiment

Engineering capstone, University of Technology Sydney (41030, Spring 2026).
Eden Hallett · supervisor Prof. Valerie Gay.

Tests whether sentiment extracted from financial news and from Reddit improves
next-session stock direction prediction beyond price history alone, across six
US-listed stocks over 2019–2023.

**Headline finding:** neither platform's sentiment beat an equal number of
random features in any tested configuration. The pipeline carries the controls
needed to make that a defensible null rather than an absence of effort —
leakage validation, a dimensionality-matched control arm, paired significance
testing, and a success criterion fixed before any model was run.

**Secondary findings:** news sentiment tracks *same-session* returns more
consistently than Reddit does (positive for 6 of 6 tickers against 5 of 6);
Reddit sentiment peaks at a **negative** lag, correlating better with the
previous session's return than the current one; and Reddit post volume is a
more robust correlate of next-session volatility than news volume is.

## Design

| | |
|---|---|
| **Window** | 2019-01-01 to 2023-12-31 — 1,258 trading sessions |
| **Stocks** | AAPL, GOOG, NVDA, TTWO (material) · BLK, JPM (immaterial) |
| **News** | FNSPID — 65,418 raw matches, 34,725 after filtering |
| **Social** | Reddit, 5 investment subreddits — 66,877 matched posts, 32,406 after filtering |
| **Prices** | Yahoo Finance via `yfinance`, daily OHLCV |
| **Sentiment** | FinBERT (`ProsusAI/finbert`), full text, chunked |
| **Models** | Gradient boosting and LSTM, walk-forward, six arms each |

"Material" stocks derive value from consumer products; "immaterial" ones do not.
The contrast was intended to test whether sentiment matters more where a product
narrative drives the price.

## Repository layout

**The Python modules are deliberately flat.** Every script does
`from config import ...`, and Python puts the running script's own directory on
`sys.path` rather than the repository root — so moving any module into a
subpackage breaks that import for every script that runs it directly. An earlier
attempt at a `src/collection/`, `src/preprocessing/` structure was reverted for
exactly this reason. The modules are grouped below by pipeline stage instead.

```
.                        pipeline modules (flat, by necessity — see above)
├── docs/                methodology, results, operations
│   └── archive/         superseded project documents
├── reports/             derived results — the report's tables are quoted from here
│   └── superseded/      output of stages later work replaced
├── figures/             report figures, 300dpi PNG + vector PDF
│   └── data/            the numbers behind each figure, one CSV per figure
├── data/                per-ticker corpora (git-ignored, ~3 GB)
├── fnspid_raw/          FNSPID download (git-ignored, ~24 GB)
└── reddit_dumps/        Arctic Shift dumps (git-ignored)
```

### Modules by stage

| Stage | Module | Role |
|---|---|---|
| **Configuration** | `config.py` | Stocks, dates, search terms, subreddit lists. **The only file to edit.** |
| | `checkpoint.py` | Self-resuming downloads and atomic checkpoints |
| **Collection** | `fnspid_symbol_census.py` | Counts FNSPID articles per ticker |
| | `fnspid_filter.py` | Filters FNSPID to the selected tickers, three match tiers |
| | `market_prices.py` | Daily prices via `yfinance` |
| | `reddit_dump_filter.py` | Filters Arctic Shift dumps to the selected tickers |
| **Filtering** | `news_quality_filter.py` | Removes analyst notes, roundups, off-topic articles |
| | `reddit_quality_filter.py` | Same filters plus Reddit-specific patterns |
| **Scoring** | `sentiment_score.py` | FinBERT scoring for either corpus, cached by document ID |
| | `build_daily_dataset.py` | Aggregates to trading sessions, merges with prices |
| **Analysis** | `correlation_analysis.py` | Same-day, next-day and volume–volatility correlations |
| | `model_tuned.py` | **The definitive six-arm comparison** with the pre-registered criterion |
| | `make_figures.py` | All report figures, from `reports/` |
| **Superseded** | `model_lstm.py` | Return regression. Retained — METHODOLOGY §6 explains why it failed |
| | `model_direction.py` | Earlier classifier. **Still imported** by `model_tuned.py` for its shared helpers |
| | `reddit_pull.py`, `reddit_volume_check.py` | Arctic Shift API collection. Retained as evidence for METHODOLOGY §3 |
| **Support** | `jsonl_csv.py` | Format conversion utility |
| | `selftest.py` | Pipeline self-checks — see `TESTING.md` |

The superseded modules are kept on purpose. The API scripts document why the
offline-dump route was necessary, having failed even on a one-week,
single-term, minimal-field query; `model_lstm.py` documents a specification
error that the report discusses; and `model_direction.py` is a live dependency
despite being superseded as an entry point.

## Pipeline

```bash
pip install -r requirements.txt
./run_pipeline.sh          # or run the stages below individually
```

**1. Collection**
```bash
python fnspid_symbol_census.py --candidates    # which tickers FNSPID covers
python fnspid_filter.py                        # news + prices (~23 GB download)
python market_prices.py                        # daily OHLCV, all six
python reddit_dump_filter.py --dumps ./reddit_dumps
```

**2. Relevance filtering**
```bash
python news_quality_filter.py
python reddit_quality_filter.py
```

**3. Sentiment**
```bash
python sentiment_score.py                      # news
python sentiment_score.py --source reddit
```

**4. Daily datasets**
```bash
python build_daily_dataset.py
python build_daily_dataset.py --source reddit
```

**5. Analysis, modelling and figures**
```bash
python correlation_analysis.py --log-count
python correlation_analysis.py --source reddit --log-count
python model_tuned.py --model gbm  --target open_to_close
python model_tuned.py --model lstm --target open_to_close
python model_tuned.py --model gbm  --target open_to_close --reddit-weighted
python make_figures.py
```

## Documentation

- [docs/METHODOLOGY.md](docs/METHODOLOGY.md) — every decision, what forced it, and what each control was for
- [docs/RESULTS.md](docs/RESULTS.md) — all findings with the numbers behind them
- [docs/OPERATIONS.md](docs/OPERATIONS.md) — runtimes, disk, resuming, known issues

## Data

Raw data is not in the repository. `fnspid_raw/` is ~24 GB and `reddit_dumps/`
several more; both are reproducible from the sources below.

- FNSPID: https://huggingface.co/datasets/Zihan1004/FNSPID
- Reddit dumps: https://arctic-shift.photon-reddit.com/download-tool
- Prices: `yfinance` (Yahoo Finance)

`reports/` and `figures/` **are** tracked, since those are the results and the
report's tables and figures are quoted from them. Each figure's underlying
numbers sit beside it in `figures/data/`, so any mark in any figure can be
traced to a CSV.

## Status

News arm complete. Reddit arm collected from 5 of 10 planned subreddits; the
sentiment cache is keyed by document ID, so the remainder could be added at the
cost of only the new posts. Note however that tripling the Reddit corpus from 2
subreddits to 5 made its model arm *worse* relative to its noise control
(−0.48pp → −2.46pp), which argues against more data helping — see RESULTS §5.
