# Superseded results

Output from stages that later work replaced. Kept because the report cites the
sequence — a specification error found and corrected is part of the method —
and because `docs/METHODOLOGY.md` refers to these runs.

| File | What produced it | Why it is superseded |
|---|---|---|
| `model_results.csv`, `model_predictions.csv` | `model_lstm.py` | Next-day **return** regression. Lost to a constant-zero baseline, which was a specification error rather than a training failure: the MSE-optimal prediction for noise is its mean. Replaced by direction classification. See METHODOLOGY §6. |
| `direction_results.csv`, `direction_predictions.csv`, `ablation_results.csv` | `model_direction.py` | The 24-configuration ablation grid. Twelve configurations used `--pooled`, which interleaved tickers within a sequence and is invalid; the twelve per-ticker configurations are valid and returned 0 of 12 beating noise. Superseded by `model_tuned.py`, which builds sequences per ticker before pooling. |
| `final_platform_comparison.csv` | an earlier comparison script | Predates the dimensionality-matched control arms. Superseded by `reports/tuned_results_*.csv`. |
| `sentiment_summary.csv` | `sentiment_score.py`, before the news/Reddit split | Replaced by `sentiment_summary_news.csv` and `sentiment_summary_reddit.csv`. |

Nothing in the report quotes these files. Current results are in `reports/`.
