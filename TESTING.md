# Testing before the full collection

Four stages, roughly 90 minutes end to end, most of it download time. Do not
skip stage 3 — it is the only one that measures precision, and precision is what
the limitations section has to report a number for.

Stop at the first stage that fails.

---

## Stage 0 — Offline self-test (30 seconds, no downloads)

```bash
python selftest.py
```

Builds a synthetic dump and a scripted fake API in a temp directory, then checks
the things that are easy to get silently wrong:

- loose terms fire in home subs and **not** in discovery subs (`GTA` in r/GTA
  yes, `GTA` in r/Games_Deals no)
- `gotta` does not match `GTA`; `Larry\nFink` does match across the newline
- posts on 2019-01-01 and 2023-12-31 are kept, 2018-12-31 and 2024-01-01 are not
- duplicate post IDs collapse to one row
- a `200` carrying `{"data": null, "error": ...}` is retried rather than read as
  an empty window
- a window with any failed request is reported incomplete
- no request ever asks for `author`

Expected last line: `All checks passed.`

Re-run this after **any** edit to `config.py`, `reddit_pull.py` or
`reddit_dump_filter.py`. It costs nothing.

---

## Stage 1 — One subreddit, end to end (15 minutes)

Download a single mid-sized dump — **r/nvidia** is a good choice: big enough to
be realistic, small enough to finish quickly, and NVDA is the ticker whose
volume is most skewed across the window, so it also previews that problem.

From `https://arctic-shift.photon-reddit.com/download-tool`, request
**submissions** for `nvidia`, and put the file in `./reddit_dumps/`.

```bash
pip install zstandard pandas
python reddit_dump_filter.py --dumps ./reddit_dumps --only NVDA
```

Check, in order:

| Check | Expected |
|---|---|
| Script finishes without a traceback | yes |
| `Scanned N posts, kept M` | M is well below N — a high keep rate means the terms are too loose |
| `data/NVDA/NVDA_reddit_2019-01-01_2023-12-31.csv` exists | yes |
| Its header | matches the 14 columns in `jsonl_csv.py` |
| `home` vs `discovery` counts in the summary line | both plausible; r/nvidia is a home sub so discovery should be 0 here |

Then confirm the ethics commitment holds on real data:

```bash
head -1 data/NVDA/NVDA_reddit_2019-01-01_2023-12-31.csv | grep -i author && echo "PROBLEM" || echo "no author columns — good"
```

---

## Stage 2 — Distribution sanity (5 minutes)

Still on the single subreddit. This is the check that catches a broken date
filter or a cursor that silently stopped early.

```bash
python - <<'EOF'
import pandas as pd
df = pd.read_csv("data/NVDA/NVDA_reddit_2019-01-01_2023-12-31.csv")
df["date"] = pd.to_datetime(df["date"], utc=True)
print(df.groupby(df["date"].dt.year).size())
print("\nrange:", df["date"].min(), "->", df["date"].max())
print("unique post_ids:", df["post_id"].nunique(), "of", len(df), "rows")
EOF
```

What you want to see:

- **All five years present.** A missing year means the window filter or the
  pagination is wrong. A thin year is fine and expected.
- **Range inside 2019-01-01 .. 2023-12-31.** Anything outside is a bug.
- **Unique IDs == row count.** Otherwise deduplication failed.
- **NVDA skewed toward 2021 and 2023.** This is real, not a bug — it is the
  crypto-GPU period and the AI run-up. It is also the thing that section 3.3.2
  currently gets wrong by claiming an even distribution, so record the actual
  per-year numbers now; they go straight into the write-up.

---

## Stage 3 — Precision sample (20 minutes, manual — do not skip)

Keyword matching cannot disambiguate context, and the report needs a measured
error rate rather than an assertion. Sample 20 rows per tier and judge each one
as "actually about the company as an investable entity" or not.

```bash
python - <<'EOF'
import pandas as pd
df = pd.read_csv("data/NVDA/NVDA_reddit_2019-01-01_2023-12-31.csv")
for scope, group in df.groupby("scope"):
    print(f"\n===== {scope} ({len(group)} rows) =====")
    for _, r in group.sample(min(20, len(group)), random_state=0).iterrows():
        print(f"[{r['date'][:10]}] r/{r['subreddit']}: {str(r['title'])[:100]}")
EOF
```

Count the false positives by hand and write the fraction down. Judge the two
tiers separately — they have different error rates by design, and reporting one
blended number hides that.

Rules of thumb:

- **Home tier above ~20% false positives** — a loose term is too broad. Move it
  to `strict_terms` only.
- **Discovery tier above ~10%** — that subreddit is not worth keeping. Drop it.
- **Near 0% in a discovery sub with decent volume** — consider promoting it to
  home, since strict terms are probably costing you recall there.

Then repeat this stage for **BLK** and **JPM** specifically once you have their
dumps, because they carry the two known ambiguity problems:

- **BlackRock vs Blackstone.** Reddit conflates them constantly. Grep your BLK
  sample for `Blackstone` and `BX` and count.
- **r/conspiracy and r/REBubble** discuss BlackRock heavily but about housing
  and institutional ownership, not the equity. High volume, low relevance.
  Decide deliberately whether to keep them and say which way you went.

---

## Stage 4 — Full run

Download the rest of the dumps, then:

```bash
python reddit_dump_filter.py --dumps ./reddit_dumps
```

Read the tail of the output before doing anything else. Two lists come out and
they mean opposite things:

- **Subreddits with no rows in any dump** — the download is missing. Fetch it or
  delete the entry from `config.py` deliberately. Do **not** treat this as a
  measured zero.
- **Per-ticker home/discovery counts** — the real numbers. Prune `config.py`
  from these, then re-run; re-filtering local files takes seconds.

Re-run `python selftest.py` after the prune to confirm nothing in `config.py`
broke the matching rules.

---

## If you go back to the API instead

Only if Arctic Shift recovers. Smoke-test it first rather than launching the
overnight job:

```bash
python reddit_pull.py --only NVDA --subreddits nvidia --max-windows 1 --restart
```

One ticker, one subreddit, one quarter. Then check:

```bash
cat data/_reddit_progress.json                  # should hold exactly one entry
ls reports/reddit_incomplete_windows.csv        # should NOT exist
```

If the incomplete-windows file exists, the service is still failing and the
window was correctly left unrecorded — re-run and it will retry. If the progress
file has an entry and the incomplete file is absent, the API is answering and a
full run is worth attempting.

Never prune a subreddit on counts from a run where `reddit_subreddit_volume.csv`
shows `complete = False` for that pair. Those numbers are floors.
