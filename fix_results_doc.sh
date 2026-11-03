#!/usr/bin/env bash
# docs/RESULTS.md used gradient boosting's 52.49% baseline when describing the
# LSTM, which pools a different number of sessions and has its own baseline of
# 52.25%. The gap is 2.28pp, not 2.52pp. Run from the repo root.
set -euo pipefail
f=docs/RESULTS.md
[ -f "$f" ] || { echo "run from the repo root"; exit 1; }

python3 - "$f" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); t = p.read_text(); n = 0
def sub(a, b):
    global t, n
    if a in t: t = t.replace(a, b); n += 1
    else: print("  (already applied or not found):", a[:60])

sub("Majority-class baseline **52.49%**.",
    "Majority-class baselines: **52.49%** (gradient boosting, n = 3,336) and\n**52.25%** (LSTM, n = 3,294). The two families pool a different number of\nsessions, so one shared baseline would be wrong for one of them.")
sub('and the LSTM\'s price-only arm is **2.52pp below** simply always predicting "up"',
    'and the LSTM\'s price-only arm sits **2.28pp below its own baseline**\n(49.97% against 52.25%) — worse than always predicting "up"')
p.write_text(t); print(f"  {n} replacement(s) applied to {p}")
PY
