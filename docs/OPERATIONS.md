# Operations

Practical notes for running the pipeline.

## Disk

The FNSPID news file is **~23 GB** — it carries full article bodies, not just
headlines. Budget **40 GB free**, because the download writes to a `.part` file
and only renames on completion, so both exist during the transfer.

| | Size |
|---|---|
| `nasdaq_exteral_data.csv` | ~23 GB |
| `full_history.zip` | ~600 MB |
| `reddit_dumps/` | ~15 GB for 5 subreddits; r/wallstreetbets alone is ~6 GB |
| `data/` | ~1–3 GB |

Once `fnspid_filter.py` has run and the coverage report looks right, the raw
CSV can be deleted — nothing downstream reads it again.

## Runtimes

Measured on an M-series MacBook.

| Step | Time |
|---|---|
| FNSPID download | Hours, network-bound |
| `fnspid_symbol_census.py` | ~30 min scan |
| `fnspid_filter.py` | ~30 min scan |
| `market_prices.py` | ~2 min |
| `reddit_dump_filter.py` | ~10 min for 7 GB |
| `sentiment_score.py` (news) | 1–2 h — 34,725 articles at 3.1 chunks each |
| `sentiment_score.py --source reddit` | ~25 min — 32,406 posts at 1.0 chunks |
| `build_daily_dataset.py` | Seconds |
| `correlation_analysis.py` | Seconds |
| `model_tuned.py --model gbm` | ~5 min |
| `model_tuned.py --model lstm` | 30–60 min |

## Resuming

Every long step survives interruption. Ctrl-C is safe throughout.

| Step | State | Resume |
|---|---|---|
| Downloads | `<file>.part` | HTTP Range; also auto-retries up to 100 times |
| Census scan | `reports/_census_checkpoint.json` | Re-run |
| News filter | `reports/_filter_checkpoint.json` + `*_news_partial.csv` | Re-run |
| Sentiment | `*_sentiment_cache.jsonl` (keyed by URL / post ID) | Re-run |
| Reddit pull (API) | `data/_reddit_progress.json` | Re-run |

Each script takes `--restart` to discard its state. **Use it after changing a
query, a date range or a filter** — no checkpoint notices config edits.

The sentiment cache is keyed by document ID, so **adding more subreddits later
costs only the new posts**. Re-filtering and re-scoring an expanded corpus is
close to free for anything already seen.

## Known issues

**Arctic Shift API is unreliable.** Returns HTTP 422 timeouts unpredictably —
the same URL succeeds and fails minutes apart. Use the dump route. See
METHODOLOGY §3 for the full diagnosis. `reddit_pull.py` and
`reddit_volume_check.py` are retained as documentation of the attempt.

**`model_direction.py --pooled` is buggy** — sequences interleave tickers. Use
`model_tuned.py`, which builds sequences per ticker before pooling.

**zsh aborts a whole command line when a glob matches nothing.** `rm -f
data/*/*_partial.csv` will not run at all if no such file exists, including the
rest of the line. Use `find data -name '*_partial.csv' -delete`.

**Both model families reproduce exactly.** Verified across seven
gradient-boosting runs and four LSTM runs spanning three weeks — see
METHODOLOGY §6. The ±1–2pp variance noted in earlier versions of this file
belonged to the untuned, unseeded LSTM used in early exploration.

**`model_tuned.py` output filenames carry the Reddit weighting flag.** Runs
with and without `--reddit-weighted` write to `*_rwt.csv` and `*.csv`
respectively. Before that fix a weighted run silently overwrote an unweighted
one; if you see results you cannot account for, check
`reports/tuned_results_all_runs.csv`, which is append-only.

**Two different sentiment means live in `reports/`.**
`sentiment_summary_*.csv` holds `mean_sentiment_per_document` and
`daily_coverage*.csv` holds `mean_sentiment_per_session`. They can differ in
sign. Quote the document-level figure when describing the scorer and the
session-level one when describing model inputs.

**`fnspid_filter.py` holds a PID lock** at `reports/_filter.lock` so two copies
cannot corrupt the partial files. Stale locks from a killed process are cleared
automatically.

## Verification

After downloading an updated script, confirm it landed:

```bash
ls -lt *.py | head -5
```

Most-recently-modified first. If the file you just saved is not at the top with
a fresh timestamp, it went somewhere else — usually `~/Downloads`.

## Backups

The FNSPID download plus two full scans represent hours of work:

```bash
BACKUP=~/Capstone/backup_$(date +%Y%m%d)
mkdir -p "$BACKUP" && cp -r reports data *.py docs "$BACKUP"/
```

Skip `fnspid_raw/` and `reddit_dumps/` — large and re-downloadable.
