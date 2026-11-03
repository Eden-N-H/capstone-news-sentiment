# Methodology

Every decision below was forced by something measured, not assumed. Where a
choice was later shown to be wrong, both the choice and the correction are
recorded — the sequence is part of the method.

---

## 1. Study design

**Window: 2019-01-01 to 2023-12-31.** FNSPID's coverage ends in December 2023,
so the originally proposed May 2021 – May 2026 window was unachievable. 1,258
trading sessions.

**Stocks.** Six, split on whether a consumer product drives the price:

| Material | Immaterial |
|---|---|
| AAPL, TTWO, GOOG, NVDA | BLK, JPM |

Three changes from the proposal. Pearl Abyss was dropped — its rationale was a
March 2026 product launch, outside the window. AFI and CBA (both ASX) were
replaced because FNSPID carries no news for ASX listings, which would have left
both immaterial stocks with prices and social data but no news arm, making the
comparison the design rests on untestable.

BLK and JPM were themselves nearly rejected: the census showed both tagged in
only 18 of 60 months. They were retained once the keyword scan (§2) restored
full coverage.

---

## 2. News corpus construction

### FNSPID's ticker tagging is incomplete

Measured across 7,176 symbols within the window:

| Ticker | Symbol-tagged | Months covered | With keyword scan |
|---|---|---|---|
| AAPL | 9,338 | 23 / 60 | 16,220 / 60 |
| NVDA | 9,533 | 47 / 60 | 13,820 / 60 |
| JPM | 2,384 | 18 / 60 | 12,438 / 60 |
| BLK | 876 | 18 / 60 | 4,115 / 60 |

The articles were in the corpus all along, untagged. Several unrelated symbols
cluster at exactly 18 months (BAC, IVZ, MET, AMP), so the gaps are systematic
rather than random.

**Consequence for anyone using FNSPID:** filtering by `Stock_symbol` alone
silently discards most of a multi-year window. A supplementary keyword scan
over titles and summaries is required.

### Three-tier match quality

The keyword scan restores coverage at the cost of precision, so matches are
recorded in tiers:

| Tier | Meaning | Kept |
|---|---|---|
| `symbol` | FNSPID's own tag | yes |
| `keyword_title` | company named in the headline | yes |
| `keyword_summary` | named only in the body | **no** |

A sample of summary-only GOOG matches was roughly 70–80% about other
companies — "Atlassian: AI May Not Save the Stock", "US STOCKS-Amazon results
boost futures". A headline mention is far stronger evidence of topicality than
a body mention. Dropping this tier removed 20,859 articles.

### Contamination classes removed

**Analyst notes.** 2,435 JPM matches were ratings *issued by* JPMorgan about
other companies — "JP Morgan Maintains Prologis Overweight". Sentiment there
belongs to Prologis. Structural for investment banks, which appear in headlines
about everyone else. Also covers 13F filings: "BlackRock Increases Position in
Inseego".

**Multi-ticker roundups.** "Technology Sector Update for 12/10/2019: MDB, NLOK,
INTC, TSM, MSFT, AAPL, IBM, CSCO, GOOG" carries market-wide sentiment.

*Evidence this mattered:* FinBERT scored two sector-update headlines at **+0.93
and −0.97** — among the strongest sentiment in TTWO's entire corpus, on text
saying nothing about Take-Two. Removing roundups shifted TTWO's mean sentiment
from −0.032 to 0.000.

**Company not named in the headline.** A general rule that subsumes several
specific ones: if the headline never names the company, the article is probably
not about it. This is what a dataset tag actually means — that an article
*mentions* a company, not that it is *about* one.

**Non-English articles.** Detected by a **character-script heuristic, not
language detection**: an article is flagged when more than 10% of the
characters in its headline fall outside 7-bit ASCII. This catches non-Latin
scripts such as Cyrillic and CJK reliably and misses Latin-script languages
such as German and French entirely, and it inspects only the headline. The 145
articles removed are therefore a lower bound on non-English content, not a
measurement of it.

**Outcome:** 65,418 raw matches → 44,559 after the tier policy → 34,725
cleaned.

### Filters were audited, not assumed

Patterns were run, then checked against a sample of what they removed
(`reports/news_flagged_samples.csv`, 15 examples per ticker per reason). Two
false-positive classes were found and corrected:

- "Can BlackRock **Maintain** Its Profitability?" was caught by a rating
  pattern matching the plain English verb. Narrowed to require a rating term
  (Overweight, Neutral, Price Target…) alongside the verb.
- "Time to **Buy** Goldman Sachs or BlackRock?" was caught by a 13F pattern.
  Narrowed to plural verbs only — funds "Buys" in headlines; advice columns say
  "Buy".

Residual error was measured, not eliminated: roughly 1 in 12 sampled exclusions
was a false positive. This is heuristic filtering with a known error rate, not
a validated classifier.

---

## 3. Reddit corpus construction

**The Arctic Shift API was abandoned.** It returned HTTP 422 timeouts
unpredictably. Six hypotheses were tested and falsified by direct measurement:

| Hypothesis | Falsified by |
|---|---|
| Window too wide | A 7-day query failed while the same query over 90 days succeeded |
| Request rate too high | Delays up to 20s did not help |
| Subreddit too large | r/wallstreetbets succeeded while the smaller r/stocks failed |
| Page size too large | `limit=5` failed on a week where `limit=100` succeeded |
| Field set too heavy | Minimal `fields=id,created_utc` also failed |
| Boolean OR unsupported | Single-term queries also failed |

The decisive observation: **the same two URLs gave opposite results on
consecutive runs.** The failures are server load on a free, single-maintainer
service. This has a practical consequence — splitting a failed request into
smaller ones makes things worse, since it adds load to an overloaded service.
Offline per-subreddit dumps were used instead.

**Product subreddits were tested and rejected.** A sample of 15 matched
r/nvidia posts was 15/15 hardware support — "Is my GPU dying?", "Annoying coil
whine", "driver causing Bluescreen". FinBERT scores those strongly negative,
r/nvidia alone contributed 70,572 posts (enough to swamp every finance
subreddit combined), and driver complaints spike at product launches — so the
contamination correlates with the events of interest rather than averaging out.

Collection is therefore restricted to investment-focused subreddits, which also
removes the coverage imbalance that affected the news arm, since all ten serve
all six tickers equally.

**Reddit-specific filtering**, in addition to everything applied to news:
bots (AutoModerator and similar), recurring threads ("Daily Discussion", "What
Are Your Moves Tomorrow"), posts whose entire text is `[deleted]`/`[removed]`,
duplicate post IDs, and two Reddit-native patterns — the company named as an
*actor* in a post about another stock ("What y'all think of FUBO? JPMorgan
analyst…") and titles cashtagging a *different* ticker ("$MILE is ready for the
moon! And Blackrock just bought in").

Posts whose **author account** was deleted are kept. `[deleted]` as an author
means the account is gone, not that the post is junk; the content is intact.
Treating it as a bot signal removed 690,715 posts — 28% of the corpus — on a
criterion unrelated to quality.

**Outcome:** 66,877 matched posts → 32,406 cleaned, from 5 of 10 planned
subreddits: r/wallstreetbets, r/stocks, r/investing, r/StockMarket and
r/options. `reddit_dump_summary.csv` records matched posts per ticker but not a
scanned total, so no scanned figure is quoted here.

**The per-reason counts overlap.** `reddit_quality_filter.py` computes each
column as an independent boolean sum and removes the union, so one post
flagged under several reasons is counted several times but removed once. The
Reddit flags sum to 38,633 against 34,471 posts removed — 4,162 posts carry
more than one flag. The same mechanism accounts for exactly one article on the
news side. Sequential subtraction is not a valid way to read either table.

---

## 4. Sentiment scoring

**FinBERT (ProsusAI/finbert)**, chosen over general-purpose BERT (finance-tuned,
~85–93% F1 vs ~50% for lexicon methods) and over instruction-tuned LLMs
(300–500× cost per document for a few points).

**The same model scores both corpora.** FinBERT is trained on news, not social
text, so a social-tuned model would likely read Reddit better in isolation. But
the research question compares *platforms* — swapping models between them would
confound a platform difference with a model difference. Holding the scorer
constant is the correct choice for this design, and the domain gap is reported
as a limitation rather than silently corrected.

**Full text, chunked.** FinBERT accepts 512 tokens; articles run longer. Each
document is split into overlapping 510-token windows and recombined as a
token-weighted mean, so a long article is not reduced to its first paragraph.
News averaged 3.1 chunks per article; Reddit posts averaged 1.0.

**Score** = P(positive) − P(negative), in [−1, 1]. `prob_neutral` is retained
separately, because a confidently neutral document and a genuinely mixed one
both score near zero and are not the same thing.

### Measured domain transfer

FinBERT reads Reddit's *reporting* language accurately:

| Post | Score |
|---|---|
| "$TTWO Earnings - Stock Currently -17% down to 93.75" | −0.95 |
| "Take-Two stock dropping after GTA6 trailer" | −0.95 |

But misreads its *positioning* language and gaming vocabulary:

| Post | Score | Problem |
|---|---|---|
| "Time to short take 2 since the cracks are showing" | **+0.89** | Bearish thesis read as description |
| "First time options trading with $TTWO with the expected GTA6 trailer **dropping** in December" | **−0.96** | "dropping" = releasing, not falling |

Consistent with this, 57% of Reddit posts score within ±0.1 of neutral against
31% of news articles — less confident outside its training domain. Note that
this band is not FinBERT's neutral *class*: the thresholds are `sentiment > 0.1`
and `< -0.1`, whereas the class would be an argmax over three probabilities.

---

## 5. Daily alignment

Articles and posts are converted UTC → US/Eastern and assigned to the next
session that was **still open** at publication: same day before 16:00 ET, next
session after, weekends rolling forward to Monday. Truncating to a date would
assign an evening article to a session that had already closed, leaking future
information into the features.

Sessions with no items are left **NaN**, with `article_count = 0` so they remain
distinguishable from genuinely neutral days. Neutral-zero, forward-fill and
rolling-mean encode different assumptions about how sentiment persists; that
choice belongs to modelling, where it can be tested.

Reddit additionally carries an **upvote-weighted mean** alongside the plain one.
A post with 900 upvotes reached far more people than one with 2. Weights use
`log1p` of the score clipped at zero — upvotes are heavy-tailed, so a single
viral post would otherwise decide the day, and negative scores would flip the
sign of that post's sentiment. Both columns are kept so the choice is testable.

---

## 6. Modelling

### Why the target changed

The first model predicted next-day **return** with MSE loss. Every configuration
lost to a constant-zero baseline (TTWO: RMSE 0.02347 vs 0.02294).

This was not a training failure. Daily returns are roughly N(0, 0.02), and the
MSE-optimal prediction for noise is its mean — so the model correctly learned to
output approximately zero. Directional accuracy read from the sign of a
near-zero number is then a coin flip by construction.

Reframed as **binary direction classification** with cross-entropy and class
weighting, which optimises the metric actually reported and cannot collapse to a
constant.

### The dimensionality control

Adding sentiment initially appeared to *hurt* significantly (pooled −2.99pp,
p = 0.003). A control arm carrying the same number of **random** features hurt
more (−3.91pp), showing the penalty came from added dimensions, not from
sentiment.

This is why every comparison is made against noise, not against price-only.
Five extra features change accuracy on their own; only a comparison at equal
dimensionality isolates whether those features carry information.

Demonstrated concretely: on synthetic random data, a news arm scored **+3.42pp
against price-only at p = 0.034** — apparently significant — while measuring
+2.88pp at p = 0.13 against noise. Judged against price-only alone, that is a
false positive.

### Validity controls

| Control | Result |
|---|---|
| **Leakage test** — pure random walk | 51.3% (chance, as required) |
| **Leakage sensitivity** — answer planted in a feature | 83.1% (test detects leakage) |
| **Positive control** — tuned pipeline, planted signal | 100% vs 48.9% |
| **Walk-forward** | Expanding window, scaler fit per fold on training rows only, never shuffled |
| **Paired testing** | McNemar plus bootstrap CIs on the same predicted sessions |
| **Power** | Pooled n ≈ 3,300, SE ≈ 0.9pp — adequate for a 2pp effect |
| **Reproducibility** | Both families reproduce exactly — see below |

### Pre-registered criterion

> A platform helps **only** if its arm beats the **noise** arm at p < 0.05 on
> the pooled test.

Stated before running and printed at the start of every run. Every
configuration is appended to `reports/tuned_results_all_runs.csv`, so the full
history of what was tried is auditable rather than selectively reported.

Significance is McNemar's exact test: a paired comparison over the sessions on
which the two arms disagreed, where under the null of equal accuracy each such
session is equally likely to favour either arm, and p is the two-sided exact
binomial probability of a split at least as extreme as observed. Because that
test is two-sided while the criterion is directional, the criterion requires
**both** a positive point estimate and p < 0.05 — an arm that differs
significantly from its control in the *negative* direction does not satisfy it.
Confidence intervals come separately from a 10,000-sample percentile bootstrap
of the paired per-session difference.

### Reproducibility, measured

Across seven gradient-boosting runs spanning three weeks, `price_only` returned
52.79%, `price_plus_noise` 51.77% and `price_plus_news` 50.51% on every run,
with news-vs-noise at −1.26pp, p = 0.241 seven times identically. The tuned
three-seed LSTM returned identical `price_only`, `price_plus_news` and
`price_plus_noise` accuracies across all four of its runs. The only arm that
moved was Reddit, for two identifiable reasons — the corpus growing from two
subreddits to five, and the upvote-weighting flag — each of which reproduces
exactly when held fixed.

An earlier version of this document claimed the LSTM varied ±1–2pp between
identical runs. That was true of the **untuned, unseeded** model used in early
exploration and is not true of the tuned pipeline.

### Correlation analysis

`correlation_analysis.py` produces all four columns of the correlation tables
from one source. It exists because those columns were originally computed by an
uncommitted diagnostic, leaving the table irreproducible from the repository and
one of its columns taken from a different file with a different sample
definition.

Sample rules differ between its rows deliberately. Sentiment rows use only
sessions carrying items, because sentiment is undefined on a quiet session and
imputing a value would manufacture observations. Volume rows use every session,
because an article count of zero is a real measurement and dropping quiet
sessions would remove the contrast the test depends on.

The partial correlation residualises on today's absolute return and trailing
5-day volatility, with `.shift(1)` before `.rolling()` so session *d*'s control
never contains session *d*'s own return. Degrees of freedom are n − 4, not
n − 2, reflecting the two controls. Pooled correlations residualise **within**
each ticker before concatenating: pooling raw series first would let the fact
that NVDA simply carries more articles than TTWO register as a relationship,
which is a between-ticker effect rather than the within-ticker one the research
question asks about. An earlier pooled figure of p = 0.020 was computed without
that separation and is superseded by p = 0.673.

The script is validated on synthetic data with a planted relationship: it
recovers the planted effect through the control (+0.274, p < 0.001) and reports
nothing where nothing was planted (−0.015, p = 0.59). Its next-day column
reproduces `daily_coverage.csv` exactly, and its same-day column reproduces the
figures the uncommitted diagnostic had produced.

### Known bug, disclosed

`model_direction.py --pooled` sorted all tickers by date and slid a window over
the result, so each sequence interleaved six stocks instead of following one.
**Twelve of the 24 ablation configurations used this mode and are invalid.** The
twelve per-ticker configurations are valid and returned 0 of 12 beating noise.
`model_tuned.py` builds sequences per ticker before pooling, and steps the
walk-forward by calendar date.

---

## 7. Limitations

**FNSPID timestamps are date-only.** All articles are stored at midnight UTC, so
no intraday analysis is possible and the direction of the same-day
sentiment/return correlation is ambiguous — an article dated Tuesday may have
been written after Tuesday's close, reporting the move. This also creates a mild
leak that **favoured** finding an effect, making the null conservative.

**Heuristic filtering** with a measured residual error rate in both directions.

**Coverage varies across stocks** (news 44.6–98.9%, Reddit 18.9–97.5% of
sessions), confounding cross-stock comparison.

**Overlapping windows** at the 5-day horizon break the independence McNemar
assumes, so those p-values are optimistic.

**Scope**: six stocks, one market, one five-year window, five of ten planned
subreddits, investment-focused communities only.
