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

**Secondary finding:** news sentiment does track *same-session* returns, more
consistently and more strongly than Reddit does (positive for 6 of 6 tickers
against 5 of 6; strongest +0.267 against +0.109). The platforms are equally
poor at forecasting tomorrow; news is measurably better at reflecting today.

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

## Pipeline

Run in order. Every long step is resumable — see [docs/OPERATIONS.md](docs/OPERATIONS.md).

```bash
pip install -r requirements.txt
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

**5. Analysis and modelling**
```bash
python correlation_analysis.py --log-count
python correlation_analysis.py --source reddit --log-count
python model_tuned.py --model gbm  --target open_to_close
python model_tuned.py --model lstm --target open_to_close
python model_tuned.py --model gbm  --target open_to_close --reddit-weighted
```

## Files

| File | Role |
|---|---|
| `config.py` | Stocks, dates, search terms, subreddit lists — **the only file to edit** |
| `checkpoint.py` | Self-resuming downloads and atomic checkpoints |
| `fnspid_symbol_census.py` | Counts FNSPID articles per ticker |
| `fnspid_filter.py` | Filters FNSPID to the selected tickers, three match tiers |
| `market_prices.py` | Daily prices via `yfinance` |
| `reddit_dump_filter.py` | Filters Arctic Shift dumps to the selected tickers |
| `news_quality_filter.py` | Removes analyst notes, roundups, off-topic articles |
| `reddit_quality_filter.py` | Same filters plus Reddit-specific patterns |
| `sentiment_score.py` | FinBERT scoring for either corpus, cached by document ID |
| `build_daily_dataset.py` | Aggregates to trading sessions, merges with prices |
| `correlation_analysis.py` | Same-day / next-day / volume–volatility correlations |
| `model_tuned.py` | **The definitive six-arm comparison** with the pre-registered criterion |
| `model_direction.py` | Earlier direction classifier plus the ablation grid |
| `model_lstm.py` | Regression baseline (superseded — see METHODOLOGY §6) |
| `reddit_pull.py`, `reddit_volume_check.py` | Arctic Shift API collection (superseded) |

The Arctic Shift API scripts are kept deliberately: they document why the
offline-dump route was necessary, having failed even on a one-week, single-term,
minimal-field query.

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

`reports/` **is** tracked, since those files are the results and the report's
tables are quoted from them.

## Status

News arm complete. Reddit arm collected from 5 of 10 planned subreddits;
the sentiment cache is keyed by document ID, so the remainder could be added
at the cost of only the new posts. Note however that tripling the Reddit corpus
from 2 subreddits to 5 made its model arm *worse* relative to its noise control
(−0.48pp → −2.46pp), which argues against more data helping — see RESULTS §5.
