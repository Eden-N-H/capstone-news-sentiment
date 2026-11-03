#!/usr/bin/env bash
# The pipeline, in order, as one command.
#
# This exists because the order matters and is not obvious from a directory of
# twenty flat modules: filtering must follow collection, scoring must follow
# filtering, and the daily panels must exist before anything is modelled or
# plotted. Every stage is independently resumable, so re-running after an
# interruption is safe and cheap — see docs/OPERATIONS.md.
#
#   ./run_pipeline.sh              everything
#   ./run_pipeline.sh analysis     just the analysis and figures
#   ./run_pipeline.sh --list       show the stages
set -euo pipefail

run() { echo; echo "=== $* ==="; "$@"; }

collection() {
  run python fnspid_symbol_census.py --candidates
  run python fnspid_filter.py
  run python market_prices.py
  run python reddit_dump_filter.py --dumps ./reddit_dumps
}

filtering() {
  run python news_quality_filter.py
  run python reddit_quality_filter.py
}

sentiment() {
  run python sentiment_score.py
  run python sentiment_score.py --source reddit
}

datasets() {
  run python build_daily_dataset.py
  run python build_daily_dataset.py --source reddit
}

analysis() {
  run python correlation_analysis.py --log-count
  run python correlation_analysis.py --source reddit --log-count
  run python model_tuned.py --model gbm  --target open_to_close
  run python model_tuned.py --model lstm --target open_to_close
  run python model_tuned.py --model gbm  --target open_to_close --reddit-weighted
  run python make_figures.py
}

STAGES=(collection filtering sentiment datasets analysis)

case "${1:-all}" in
  --list|-l) printf '%s\n' "${STAGES[@]}"; exit 0 ;;
  all)       for s in "${STAGES[@]}"; do "$s"; done ;;
  *)         if printf '%s\n' "${STAGES[@]}" | grep -qx "$1"; then "$1";
             else echo "unknown stage: $1"; printf '%s\n' "${STAGES[@]}"; exit 1; fi ;;
esac

echo
echo "Done. Results in reports/, figures in figures/."
