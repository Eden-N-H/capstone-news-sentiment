# Handoff — Reddit retrieval problem

Context for a fresh session. Project: UTS capstone, "Leveraging News and Media
Outlets to Predict Stock Market Fluctuations" (Eden Hallett, supervisor Prof.
Valerie Gay). Comparing financial news vs. broader social media as inputs to
LSTM stock-price prediction models.

## Project parameters

- **Window:** 2019-01-01 to 2023-12-31 (shifted from the proposal's May
  2021–May 2026 because FNSPID news coverage stops at the end of 2023).
- **Stocks:** AAPL, TTWO, GOOG, NVDA (material — product-driven) and BLK, JPM
  (immaterial — no consumer product). Pearl Abyss dropped; AFI and CBA replaced
  by BLK/JPM because FNSPID carries no news for ASX listings, which would have
  left both immaterial stocks with no news arm to compare against.
- **Sources:** FNSPID (news, treated as one combined corpus), Reddit, price
  data. Twitter/X and Facebook have no accessible historical research tier and
  are out of scope.

## The problem

Reddit data comes from the Arctic Shift API
(`https://arctic-shift.photon-reddit.com`), a free archive of Reddit history.
It returns **HTTP 422 `{"data":null,"error":"Timeout. Maybe slow down a bit"}`**
on most requests, unpredictably.

**The failures are random, not deterministic.** This was established by direct
curl testing and is the single most important fact for anyone picking this up.
The following hypotheses were each proposed and then falsified by measurement:

| Hypothesis | Falsified by |
|---|---|
| Window too wide | A 7-day query failed while the identical query over 90 days succeeded |
| Request rate too high | Delays up to 20s between requests did not help |
| Subreddit too large | r/wallstreetbets succeeded while the far smaller r/stocks timed out |
| Page size too large | `limit=5` succeeded on a week where `limit=100` failed |
| Field set too heavy | Minimal `fields=id,created_utc` requests also fail |
| Boolean `OR` unsupported | Single-term queries fail too |

The clincher: **the same two URLs gave opposite results on consecutive runs.**
Arctic Shift's own docs note the database sometimes needs to warm up and that a
second attempt often works, and the service makes no uptime guarantee (status:
`https://status.arctic-shift.photon-reddit.com`).

**Consequence for strategy:** splitting a failed request into smaller ones makes
things worse, because it adds load to an already-overloaded service. Correct
approach is the fewest possible requests, each retried patiently. Several hours
were lost to fixes that did the opposite.

## Current state

The API path is implemented and hardened as far as it usefully can be, but has
never completed a real subreddit. **The recommended path is now the offline
dumps**, which are files and cannot time out.

Nothing is blocked on code. The blocker is whether the API is usable at all.

## Files

All in the same folder; scripts import from `config.py`.

| File | Role | Status |
|---|---|---|
| `config.py` | Stocks, dates, subreddit lists, search terms. **The only file to edit.** | Working |
| `checkpoint.py` | Resumable downloads (HTTP Range) + atomic JSON checkpoints | Working |
| `jsonl_csv.py` | Shared JSONL -> deduped, sorted CSV. Both Reddit scripts use it, so their output is identical | Working |
| `reddit_dump_filter.py` | **Recommended.** Filters locally downloaded dumps into per-ticker CSVs | Working, tested on synthetic data |
| `reddit_pull.py` | API pull. Two-stage: search for IDs, then `/api/posts/ids` for content | Blocked by API timeouts |
| `reddit_volume_check.py` | Optional pre-flight counts. **Normally skip** — now makes the same requests as the pull and discards the posts | Blocked, and redundant |
| `fnspid_symbol_census.py` | Counts FNSPID articles per ticker; confirms BLK/JPM coverage | Unaffected, ~23 GB download |
| `fnspid_filter.py` | FNSPID news + NASDAQ prices | Unaffected |
| `market_prices.py` | Daily OHLCV for all six via yfinance | Unaffected |
| `selftest.py` | Offline checks for both Reddit paths. Run after any config edit | Working |
| `TESTING.md` | Staged test procedure to run before the full collection | Current |
| `README.md` | Full run order, disk requirements, caveats | Current |

### Relevant config details

- Search terms are stored as **lists** (`home_terms`, `strict_terms`), not
  joined `"A OR B"` strings, so scripts can issue one request per term and
  union post IDs locally.
- Two subreddit tiers per stock: `subreddits` (home, loose terms) and
  `discovery_subreddits` (strict terms). Bare tickers are ambiguous elsewhere —
  "GTA" is the Greater Toronto Area, "Chase" is a verb, "BLK" is slang.
  `Stock.terms_for()` picks the tier, `Stock.scope_for()` labels it, and **both**
  the API and dump paths use them. Every output row carries a `scope` column.
- 331 subreddit-ticker pairs total. Deliberately wide; meant to be pruned
  using measured volume, *after* collection rather than before.
- Neither Reddit script ever stores `author`/`author_fullname`, matching the
  ethics commitment in section 4.1.1 of the proposal. The API path excludes them
  from the `fields` request; the dump path never copies them out of the dumps.

### Fixed since the last handoff

Five defects, all found by review rather than by a run:

1. **Dump filter ignored the term tiers.** It unioned `home_terms` and
   `strict_terms` into one regex per stock and applied it everywhere, so bare
   "GTA" matched in r/Games_Deals and bare "BLK" in r/conspiracy — exactly what
   the two-tier design exists to prevent. Now uses `scope_for()`.
2. **Failed windows were checkpointed as done.** `reddit_pull.py` recorded every
   window regardless of outcome, so a resumed pull skipped the quarters that had
   failed. Windows now checkpoint only on full success; failures go to
   `reports/reddit_incomplete_windows.csv` and are retried.
3. **Soft errors read as empty results.** A 200 carrying
   `{"data": null, "error": ...}` was returned as `[]`, which ends pagination
   and looks identical to a finished window. Now retried like an HTTP error.
4. **BLK and JPM ran loose terms across the broad finance subs**, which were
   listed as home subs. Moved to discovery. Bare `Chase` replaced with
   `"Chase Bank"` / `"Chase stock"` — see the README note before reverting that.
5. **Whole-ticker DataFrames.** Both scripts did
   `pd.read_json(path, lines=True)` over the full JSONL. Replaced with a
   two-pass offset index in `jsonl_csv.py`.

## Recommended next steps

1. Download per-subreddit **submissions** dumps from
   `https://arctic-shift.photon-reddit.com/download-tool` into one folder.
   Start with the core finance subs plus each stock's main product subs.
2. `pip install zstandard pandas`
3. Smoke-test on one subreddit before committing to the full set —
   `python reddit_dump_filter.py --dumps ./reddit_dumps --only NVDA --limit 200000`
   — and eyeball 20 matches for precision. See the test procedure below.
4. `python reddit_dump_filter.py --dumps ./reddit_dumps` for the real run.
5. Prune `config.py` using the resulting counts, then re-run the filter —
   re-filtering local files takes seconds, unlike re-pulling. Do not prune a sub
   that the filter reported as absent from every dump; that means the download
   is missing, not that the sub is dead.
6. Optionally retry `reddit_pull.py` later if the API recovers; output schema is
   identical either way, so downstream work is unaffected by which path is used.

## Known limitations to write up

- Arctic Shift is a free single-maintainer service with documented reliability
  caveats. Belongs in section 3.3.2 and the limitations section, with the
  offline dumps named as the deterministic alternative.
- Keyword matching cannot disambiguate context ("GTA" as Grand Theft Auto vs.
  Toronto). Sample and report an error rate.
- Reddit post volume will not be evenly distributed across the window (NVDA will
  skew to 2021 and 2023). Section 3.3.2 currently claims an even distribution;
  that claim needs softening.
- BlackRock and Blackstone are routinely conflated on Reddit. Spot-check BLK
  matches.
