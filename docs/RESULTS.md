# Results

Every figure here is read from a file in `reports/`, named where it matters.
All model figures are out-of-sample, from walk-forward validation. Every
comparison is paired — both arms predict the same sessions — and tested with
McNemar's exact test plus a percentile bootstrap confidence interval.

---

## 1. Corpora

`reports/news_quality.csv`, `reports/reddit_quality.csv`, `reports/daily_coverage*.csv`

| Ticker | Category | News articles | News sessions | Reddit posts | Reddit sessions |
|---|---|---|---|---|---|
| AAPL | material | 8,181 | 55.2% | 14,899 | **97.5%** |
| GOOG | material | 10,811 | 98.9% | 7,379 | 89.8% |
| NVDA | material | 9,217 | 84.6% | 6,399 | 81.1% |
| TTWO | material | 1,250 | 52.1% | 503 | 18.9% |
| BLK | immaterial | 2,152 | 44.6% | 1,120 | 33.4% |
| JPM | immaterial | 3,114 | 53.7% | 2,106 | 56.2% |

Percentages are of 1,258 trading sessions. Totals: 34,725 articles and 32,406
posts.

**AAPL is covered on 97.5% of sessions by Reddit against 55.2% by a
professional financial news feed.** Retail investors discuss Apple daily; the
financial press does not publish company-specific Apple coverage every day.

Three counts differ slightly in the modelled panel — NVDA news 9,215, NVDA
Reddit 6,393, TTWO Reddit 501 — being items that were cleaned and scored but
could not be assigned to a session inside the window, chiefly those published
after 16:00 ET on the final sessions. The cleaned corpus and the modelled corpus
are different quantities and both figures are correct for what they measure.

### Filtering

| | News | Reddit |
|---|---|---|
| Raw matches | 65,418 | 66,877 |
| After tier policy | 44,559 | — |
| Removed (unique) | 9,834 | 34,471 |
| **Cleaned** | **34,725** | **32,406** |

The per-reason columns in `news_quality.csv` and `reddit_quality.csv` are
**independent flag counts and overlap** — one document may be flagged under
several reasons but is removed once. News flags sum to 9,835 against 9,834
removed, a single double-flagged article. Reddit flags sum to 38,633 against
34,471 removed, so 4,162 posts carry more than one flag. Sequential subtraction
is not a valid way to read either table.

### By subreddit

`reports/reddit_dump_summary.csv` and the per-ticker cleaned files

| Ticker | wallstreetbets | stocks | investing | StockMarket | options | Total |
|---|---|---|---|---|---|---|
| AAPL | 8,898 | 2,788 | 1,253 | 1,164 | 796 | 14,899 |
| GOOG | 4,475 | 1,398 | 728 | 592 | 186 | 7,379 |
| NVDA | 4,153 | 1,201 | 311 | 443 | 291 | 6,399 |
| TTWO | 345 | 93 | 21 | 29 | 15 | 503 |
| BLK | 764 | 127 | 146 | 72 | 11 | 1,120 |
| JPM | 1,276 | 309 | 257 | 207 | 57 | 2,106 |
| **Total** | **19,911** | **5,916** | **2,716** | **2,507** | **1,356** | **32,406** |

The `wallstreetbets + stocks` subtotal is 25,827, which reproduces the earlier
two-subreddit corpus exactly — a useful consistency check on both tables.

---

## 2. Sentiment levels

`reports/sentiment_summary_news.csv`, `reports/sentiment_summary_reddit.csv`.
Document-level means — every article or post weighted equally. "Neutral" is the
share scoring within ±0.1 of zero, **not** FinBERT's neutral class.

| Ticker | News mean | pos % | neu % | neg % | Reddit mean | pos % | neu % | neg % |
|---|---|---|---|---|---|---|---|---|
| AAPL | −0.0643 | 32.8 | 29.5 | 37.7 | −0.0382 | 19.7 | 58.0 | 22.3 |
| GOOG | +0.0503 | 41.5 | 31.9 | 26.6 | −0.0823 | 15.3 | 56.6 | 28.1 |
| NVDA | +0.0919 | 48.7 | 25.4 | 25.9 | −0.0119 | 22.6 | 56.2 | 21.2 |
| TTWO | 0.0000 | 41.7 | 26.9 | 31.4 | −0.0198 | 20.3 | 62.2 | 17.5 |
| BLK | −0.1054 | 24.7 | 39.6 | 35.7 | −0.0332 | 17.5 | 59.9 | 22.6 |
| JPM | +0.0069 | 30.9 | 43.7 | 25.4 | −0.1169 | 17.3 | 47.5 | 35.2 |

**Reddit is negative for all six tickers; news is mixed.** Two explanations,
probably both operating: Reddit genuinely skews negative, and FinBERT misreads
Reddit's register (§4 of METHODOLOGY). Weighted by document count the corpora
are **31% neutral for news against 57% for Reddit**, which supports the second.

> **Two means exist in `reports/` and they are not interchangeable.**
> `sentiment_summary_*.csv` holds `mean_sentiment_per_document`;
> `daily_coverage*.csv` holds `mean_sentiment_per_session`, which averages
> per-session means and so weights a 40-article session the same as a
> 1-article one. They can differ in sign: BLK's news sentiment is −0.1054
> document-level but +0.0001 session-level. The table above is document-level,
> because it describes the scorer rather than the market.

---

## 3. Correlation analysis

`reports/correlation_analysis_news.csv`, `reports/correlation_analysis_reddit.csv`,
produced by `correlation_analysis.py`. Sentiment columns use sessions carrying
items; volume columns use all 1,257 sessions with a next-day return.

### News

| Ticker | Same-day | Next-day | Volume → volatility | Controlled |
|---|---|---|---|---|
| AAPL | +0.022 | −0.038 | −0.037 | −0.018 |
| GOOG | +0.102\*\*\* | +0.039 | +0.073\*\* | +0.023 |
| NVDA | +0.267\*\*\* | +0.020 | +0.029 | +0.000 |
| TTWO | +0.058 | +0.012 | +0.163\*\*\* | **+0.101\*\*\*** |
| BLK | +0.074 | −0.047 | −0.009 | −0.002 |
| JPM | +0.182\*\*\* | −0.049 | +0.068\* | +0.023 |

### Reddit

| Ticker | Same-day | Next-day | Volume → volatility | Controlled |
|---|---|---|---|---|
| AAPL | +0.088\*\* | −0.010 | +0.108\*\*\* | +0.050 |
| GOOG | +0.083\*\* | +0.016 | +0.075\*\* | +0.043 |
| NVDA | +0.109\*\*\* | +0.043 | +0.019 | −0.024 |
| TTWO | +0.085 | −0.071 | +0.080\*\* | +0.061\* |
| BLK | **−0.101\*** | +0.012 | +0.031 | +0.013 |
| JPM | +0.029 | +0.009 | +0.084\*\* | +0.042 |

`*** p<0.001 · ** p<0.01 · * p<0.05`

| | News | Reddit |
|---|---|---|
| Same-day sign test | **6/6 positive, p = 0.031** | 5/6 positive, p = 0.219 |
| Bonferroni α (6 controlled tests) | 0.0083 | 0.0083 |
| Clears it | **TTWO** (+0.101, p = 0.0003) | none |
| Pooled controlled, all six | +0.005, p = 0.673 | +0.021, p = 0.070 |
| Pooled controlled, excluding TTWO | +0.001, p = 0.932 | +0.022, p = 0.083 |

**Same-day: news is positive for 6 of 6, Reddit for 5 of 6.** Both measurements
capture something real, but news does so more consistently and more strongly
(+0.267 against +0.109 at best). Causal direction is ambiguous because FNSPID
timestamps are date-only.

**Next-day: zero on both platforms.** No forecasting relationship anywhere.

**Volume → volatility** looks promising raw and mostly reflects volatility
clustering: turbulent sessions generate more news *and* are followed by more
turbulence. After partialling out today's absolute return and trailing 5-day
volatility, **only TTWO survives, and only on news**. Pooled across all six
tickers there is no effect at all (p = 0.673). TTWO is the least-covered stock
and AAPL the most, with AAPL showing nothing: a plausible mechanism is that for
a thinly-covered stock a burst of articles signals a genuine event, while for a
constantly-covered one articles are background. One ticker, one platform,
in-sample — a hypothesis, not a finding.

The pooled figures residualise **within** each ticker before concatenating, so
that differences in article volume *between* stocks cannot register as a
relationship *within* them.

---

## 4. Primary result

`reports/tuned_results_all_runs.csv`. Canonical runs: gradient boosting
`2026-10-14T11:25:56` (n = 3,336) and LSTM `2026-09-28T14:29:25` (n = 3,294).

Majority-class baseline **52.49%**.

| Arm | Features | Gradient boosting | LSTM |
|---|---|---|---|
| `price_only` | 5 | **52.79%** | 49.97% |
| `price_plus_news` | 10 | 50.51% | 50.09% |
| `price_plus_reddit` | 10 | 49.31% | 49.97% |
| `price_plus_noise` *(control)* | 10 | 51.77% | 50.82% |
| `price_plus_both` | 15 | 49.73% | 50.03% |
| `price_plus_noise10` *(control)* | 15 | 49.67% | 49.39% |

### The pre-registered test

| Comparison | GBM | p | LSTM | p |
|---|---|---|---|---|
| news vs noise | −1.26pp [−3.30, +0.81] | 0.241 | −0.73pp [−2.64, +1.24] | 0.478 |
| reddit vs noise | −2.46pp [−4.53, −0.42] | **0.019** | −0.85pp [−2.61, +0.91] | 0.356 |
| both vs noise10 | +0.06pp [−2.01, +2.16] | 0.978 | +0.64pp [−1.15, +2.37] | 0.504 |
| reddit vs news | −1.20pp [−3.18, +0.72] | 0.238 | −0.12pp [−2.03, +1.88] | 0.928 |

**The criterion is not met in any configuration.** No arm beats its noise
control.

### Neither model has skill at all

Gradient boosting's best arm beats the majority-class baseline by **0.30pp**
(52.79% against 52.49%), and the LSTM's price-only arm is **2.52pp below**
simply always predicting "up". This is not "sentiment lost to a strong price
model" — *nothing* predicted next-session direction, which is what makes the
null a finding rather than a failure.

### Ablation

Before tuning, 12 valid configurations were tested across representation
(surprise / level / counts), horizon (1 / 5 days) and movement threshold (all
days / >1% moves). **0 of 12 beat noise.** Differences ranged −1.64 to +1.69pp,
centred near zero — the signature of no effect rather than a weak one. Sentiment
vs noise pooled: +0.93pp, CI [−0.75, +2.57], p = 0.30.

---

## 5. The one result that is not simply null

Reddit sentiment is **worse** than an equal number of random features in
gradient boosting: −2.46pp (p = 0.019) unweighted, −2.64pp (p = 0.012)
upvote-weighted. Three things constrain it: it appears in one model family only
(the LSTM gives −0.85pp, p = 0.356), neither p survives Bonferroni correction
for the ten comparisons in that run (α = 0.005), and no prior hypothesis
predicted it. It does **not** satisfy the criterion, which requires *beating*
noise.

What makes it worth reporting is a dose-response relationship already in the
log. On the **2-subreddit** corpus, Reddit-vs-noise was **−0.48pp (p = 0.670)**.
On the **5-subreddit** corpus it is **−2.46pp (p = 0.019)**. *Tripling the
Reddit corpus made the arm worse.* More data degrading performance is the
signature of **bias, not variance**: a random feature carries no structure to
overfit, whereas a systematically mis-scored one does, and adding more
consistently mis-scored data reinforces the wrong pattern. That connects
directly to the measured FinBERT domain gap — the negative skew across all six
tickers, the inversion of positioning language, and the 57% neutral rate.

Offered as a mechanism consistent with measured scoring error, not as a finding.

---

## 6. Robustness

| Variation | Arm | vs its noise control | p |
|---|---|---|---|
| Unweighted daily mean | Reddit | −2.46pp | 0.019 |
| Upvote-weighted mean | Reddit | −2.64pp | 0.012 |
| Two-subreddit corpus | Reddit | −0.48pp | 0.670 |
| Five-subreddit corpus | Reddit | −2.46pp | 0.019 |
| Target: open-to-close | News | −1.26pp | 0.241 |
| Target: close-to-close | News | +0.21pp | 0.865 |
| Family: gradient boosting | News | −1.26pp | 0.241 |
| Family: LSTM | News | −0.73pp | 0.478 |

### Validity controls

| Control | Result |
|---|---|
| Leakage test — pure random walk | 51.3% (chance, as required) |
| Leakage sensitivity — answer planted in a feature | 83.1% (test detects leakage) |
| Positive control — tuned pipeline, planted signal | 100% vs 48.9% |
| Walk-forward | Expanding window, scaler fit per fold on training rows only, never shuffled |
| Paired testing | McNemar's exact test plus bootstrap CIs on the same sessions |
| Power | n = 3,336 / 3,294, SE ≈ 0.9pp — adequate for a 2pp effect |

### Reproducibility

**Both model families reproduce exactly.** Across seven gradient-boosting runs
spanning three weeks, `price_only` returned 52.79%, `price_plus_noise` 51.77%
and `price_plus_news` 50.51% every time, with news-vs-noise at −1.26pp,
p = 0.241 on all seven. The tuned three-seed LSTM returned identical
`price_only`, `price_plus_news` and `price_plus_noise` accuracies across all
four of its runs. The only arm that moved was Reddit, and it moved for two
identifiable reasons — the corpus growing from two subreddits to five, and the
upvote-weighting flag — each of which reproduces exactly when held fixed.

An earlier draft of this document claimed the LSTM varied ±1–2pp between
identical runs. That was true of the **untuned, unseeded** model used in early
exploration and is not true of the tuned pipeline.

---

## 7. Interpretation

**News and Reddit sentiment capture real price-relevant content but do not
forecast next-session direction beyond price history**, for six large-cap US
stocks at daily horizons, in either model family, across 24+ configurations,
individually or combined, and under both plain and upvote-weighted Reddit
aggregation. Consistent with weak-form market efficiency: heavily-covered large
caps price public news within the session.

**Why this need not contradict the literature.** Much published work reporting
sentiment "improving performance" measures *contemporaneous* association — the
same-day relationship in §3, which is strong here too. That is explanation, not
prediction. Studies that do forecast often lack a price-only baseline, split
time series randomly (leaking future into past), predict price level rather than
direction, or report a single favourable configuration. The leakage test here
returns 51.3% on a random walk, which many published pipelines would not pass.

**On the platform comparison.** The forecasting question returns a null for both
platforms, so the research question's primary form cannot be answered with a
ranking. The contemporaneous question can: news sentiment tracks same-session
returns more consistently and more strongly than Reddit sentiment does. Reddit
nonetheless provides far better *coverage* — 97.5% of sessions for AAPL against
55.2% — so the platforms differ in opposite directions on breadth and fidelity.

**On stock type.** With nothing clearing the noise bar on either platform, there
is no effect whose size could differ between material and immaterial stocks. The
sub-question is not answerable from these data, and is reported as such.

**Contribution.** A rigorously-controlled null, plus four measured and
transferable findings about FNSPID and Reddit sentiment pipelines: incomplete
symbol tagging, institutional name contamination, the distinction between an
article *mentioning* a company and being *about* one, and a dose-response
degradation that suggests systematic scorer error rather than insufficient data.

---

## 8. Limitations

- **FNSPID timestamps are date-only** — all articles stored at midnight UTC. No
  intraday analysis is possible, and the direction of the same-day correlation
  is ambiguous. This also creates a mild leak that *favoured* finding an effect,
  making the null conservative.
- **Heuristic filtering** with a measured residual error rate in both
  directions: roughly 1 in 12 sampled exclusions was a false positive.
- **Non-English detection is a character-script heuristic**, not language
  detection — a non-ASCII ratio above 0.10 on the headline. It catches
  non-Latin scripts and misses German and French entirely, so the 145 articles
  removed are a lower bound.
- **Coverage varies across stocks** (news 44.6–98.9%, Reddit 18.9–97.5% of
  sessions), confounding cross-stock comparison.
- **Overlapping windows** at the 5-day horizon break the independence McNemar
  assumes, so those p-values are optimistic.
- **`model_direction.py --pooled` is buggy** — sequences interleave tickers.
  Twelve of the 24 ablation configurations used that mode and are invalid; the
  twelve per-ticker configurations are valid and returned 0 of 12 beating noise.
- **Scope**: six stocks, one market, one five-year window, five of ten planned
  subreddits, investment-focused communities only.
