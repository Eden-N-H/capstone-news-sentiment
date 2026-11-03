"""
Correlation analysis for Section 4.3 — one script, one source of truth.

Why this file exists: the four columns of the correlation table were originally
produced by an ad-hoc diagnostic that was never committed, so the table could
not be reproduced from the repository and one of its columns had been taken from
reports/daily_coverage.csv instead. Mixed provenance inside a single table is not
defensible, and "we cannot regenerate this" is worse than any individual number
in it. Everything in that table now comes from here.

Four quantities, per ticker:

  1. same_day      corr(sentiment on session d, return of session d)
  2. next_day      corr(sentiment on session d, return of session d+1)
  3. volume_vol    corr(article count on d, |return| of session d+1)
  4. volume_vol_pc the same, partialling out today's |return| and trailing
                   5-day volatility

(3) looks like a forecasting result and mostly is not. Turbulent sessions both
generate more news and are followed by more turbulence, so volatility clustering
produces (3) on its own. (4) is the test that matters: it asks whether news
volume says anything about tomorrow that today's own turbulence does not already
say.

Sample definitions differ between the sentiment and volume rows, deliberately:

  - Sentiment rows use sessions that carry news. On a session with no articles
    sentiment is undefined, and imputing a value would manufacture observations.
  - Volume rows use every session. An article count of zero is a real
    measurement, not a missing one, and dropping quiet sessions would remove
    exactly the comparison the test depends on.

Run:
    python correlation_analysis.py                     # news
    python correlation_analysis.py --source reddit
    python correlation_analysis.py --source reddit --sentiment-col sentiment_weighted
    python correlation_analysis.py --log-count         # log1p(article_count)
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from config import END_DATE_STR, REPORT_DIR, START_DATE_STR

PANELS = {
    "news": "daily_panel_{start}_{end}.csv",
    "reddit": "daily_panel_reddit_{start}_{end}.csv",
}

# Two controls, so a partial correlation on n observations has n-4 degrees of
# freedom. Spelled out because getting this wrong inflates every p-value in the
# fourth column.
N_CONTROLS = 2


def stars(p):
    """Significance markers, matching the legend printed under the table."""
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def pearson(x, y):
    """Pearson r with its two-sided p-value, on rows where both are present."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30:
        return np.nan, np.nan, int(m.sum())
    r, p = stats.pearsonr(x[m], y[m])
    return float(r), float(p), int(m.sum())


def residualise(y, controls):
    """
    Strip the linear influence of `controls` out of `y`, returning the residual.

    Least squares with an intercept. Rows with any missing value are returned as
    NaN rather than dropped, so the residual series stays aligned with the frame
    it came from — alignment is what makes the pooled step below valid.
    """
    y = np.asarray(y, dtype=float)
    C = np.column_stack([np.asarray(c, dtype=float) for c in controls])
    m = np.isfinite(y) & np.isfinite(C).all(axis=1)
    out = np.full(len(y), np.nan)
    if m.sum() < 30:
        return out
    X = np.column_stack([np.ones(m.sum()), C[m]])
    beta, *_ = np.linalg.lstsq(X, y[m], rcond=None)
    out[m] = y[m] - X @ beta
    return out


def partial_corr(x, y, controls):
    """
    Correlation between x and y once both have been residualised on `controls`.

    Equivalent to the coefficient on x in a regression of y on x plus controls,
    but reported as a correlation so it sits on the same scale as the raw column
    beside it. The p-value uses n-2-N_CONTROLS degrees of freedom, not n-2;
    using n-2 would understate the uncertainty the controls cost.
    """
    rx = residualise(x, controls)
    ry = residualise(y, controls)
    m = np.isfinite(rx) & np.isfinite(ry)
    n = int(m.sum())
    if n < 30 + N_CONTROLS:
        return np.nan, np.nan, n
    r = float(np.corrcoef(rx[m], ry[m])[0, 1])
    df = n - 2 - N_CONTROLS
    if abs(r) >= 1.0:
        return r, 0.0, n
    t = r * np.sqrt(df / (1.0 - r * r))
    return r, float(2 * stats.t.sf(abs(t), df)), n


def features(df, sentiment_col, log_count):
    """
    Derive the series the four tests need, per ticker, in date order.

    Every lookback is causal: trailing volatility uses .shift(1) before
    .rolling() so session d's control never contains session d's own return.
    Without that shift the control would absorb part of what the test is trying
    to measure, and the partial correlations would be biased toward zero.
    """
    d = df.sort_values("date").copy()

    price = "adj_close" if "adj_close" in d.columns else "close"
    d["ret"] = d[price].pct_change()                       # return of session d

    # next_day_return is written by build_daily_dataset.py as
    # pct_change().shift(-1); recomputed here so this script does not depend on
    # that column existing, and asserted against it when it does.
    d["ret_next"] = d[price].pct_change().shift(-1)
    if "next_day_return" in d.columns:
        a, b = d["ret_next"], d["next_day_return"]
        both = a.notna() & b.notna()
        if both.any() and not np.allclose(a[both], b[both], atol=1e-9):
            print("  ! recomputed next-day return disagrees with the panel column")

    d["abs_ret"] = d["ret"].abs()
    d["abs_ret_next"] = d["ret_next"].abs()               # tomorrow's volatility
    d["vol_5d"] = d["ret"].shift(1).rolling(5).std()      # causal, see above

    count = d["article_count"].fillna(0)
    d["count_feat"] = np.log1p(count) if log_count else count

    d["sent"] = d[sentiment_col] if sentiment_col in d.columns else np.nan
    return d


def analyse(d):
    """The four tests for one ticker. Sample rules are in the module docstring."""
    has_sent = d[d["sent"].notna()]

    r1, p1, n1 = pearson(has_sent["sent"].to_numpy(), has_sent["ret"].to_numpy())
    r2, p2, n2 = pearson(has_sent["sent"].to_numpy(), has_sent["ret_next"].to_numpy())
    r3, p3, n3 = pearson(d["count_feat"].to_numpy(), d["abs_ret_next"].to_numpy())
    r4, p4, n4 = partial_corr(
        d["count_feat"].to_numpy(), d["abs_ret_next"].to_numpy(),
        [d["abs_ret"].to_numpy(), d["vol_5d"].to_numpy()],
    )
    return {
        "same_day": r1, "same_day_p": p1, "same_day_n": n1,
        "next_day": r2, "next_day_p": p2, "next_day_n": n2,
        "volume_vol": r3, "volume_vol_p": p3, "volume_vol_n": n3,
        "volume_vol_pc": r4, "volume_vol_pc_p": p4, "volume_vol_pc_n": n4,
    }


def pooled_partial(frames):
    """
    One partial correlation across every ticker's sessions.

    Residualising happens **within** each ticker before the residuals are
    concatenated. Pooling the raw series first would let differences in level
    between tickers — NVDA simply carries more articles than TTWO — masquerade
    as a relationship, which is a between-ticker effect rather than the
    within-ticker one the research question asks about.
    """
    rxs, rys = [], []
    for d in frames:
        controls = [d["abs_ret"].to_numpy(), d["vol_5d"].to_numpy()]
        rx = residualise(d["count_feat"].to_numpy(), controls)
        ry = residualise(d["abs_ret_next"].to_numpy(), controls)
        m = np.isfinite(rx) & np.isfinite(ry)
        rxs.append(rx[m])
        rys.append(ry[m])
    x, y = np.concatenate(rxs), np.concatenate(rys)
    n = len(x)
    if n < 30:
        return np.nan, np.nan, n
    r = float(np.corrcoef(x, y)[0, 1])
    df = n - 2 - N_CONTROLS
    t = r * np.sqrt(df / (1.0 - r * r))
    return r, float(2 * stats.t.sf(abs(t), df)), n


def main():
    ap = argparse.ArgumentParser(description="Section 4.3 correlation analysis.")
    ap.add_argument("--source", choices=list(PANELS), default="news")
    ap.add_argument("--only", nargs="*", help="Restrict to these tickers")
    ap.add_argument("--sentiment-col", default="sentiment_mean",
                    help="sentiment_mean (default) or sentiment_weighted")
    ap.add_argument("--log-count", action="store_true",
                    help="Use log1p(article_count) instead of the raw count")
    ap.add_argument("--exclude-from-pooled", nargs="*", default=["TTWO"],
                    help="Tickers to drop from the pooled test (default TTWO)")
    args = ap.parse_args()

    path = os.path.join(REPORT_DIR, PANELS[args.source].format(
        start=START_DATE_STR, end=END_DATE_STR))
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found — run build_daily_dataset.py first")

    panel = pd.read_csv(path, parse_dates=["date"])
    tickers = args.only or list(dict.fromkeys(panel["ticker"]))

    rows, frames = [], {}
    for t in tickers:
        sub = panel[panel["ticker"] == t]
        if sub.empty:
            print(f"{t}: not in the panel")
            continue
        d = features(sub, args.sentiment_col, args.log_count)
        frames[t] = d
        rows.append({"ticker": t,
                     "category": sub["category"].iloc[0] if "category" in sub else "",
                     **analyse(d)})

    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("nothing to analyse")

    count_label = "log1p(article_count)" if args.log_count else "article_count"
    print(f"\n=== Correlation analysis: {args.source} "
          f"({args.sentiment_col}, {count_label}) ===\n")
    hdr = f"{'Ticker':<7}{'Same-day':>14}{'Next-day':>14}{'Volume-vol':>14}{'Controlled':>14}"
    print(hdr)
    print("-" * len(hdr))
    for _, r in res.iterrows():
        def cell(v, p):
            return "" if not np.isfinite(v) else f"{v:+.3f}{stars(p)}"
        print(f"{r.ticker:<7}"
              f"{cell(r.same_day, r.same_day_p):>14}"
              f"{cell(r.next_day, r.next_day_p):>14}"
              f"{cell(r.volume_vol, r.volume_vol_p):>14}"
              f"{cell(r.volume_vol_pc, r.volume_vol_pc_p):>14}")
    print("\n*** p<0.001  **  p<0.01  *  p<0.05")
    print(f"n: sentiment rows {res.same_day_n.min()}-{res.same_day_n.max()} "
          f"(sessions with {args.source}); volume rows {res.volume_vol_n.max()} (all sessions)")

    # Same-day sign test. With six tickers the interesting question is not
    # whether any one correlation clears 0.05 but whether they all lean the same
    # way, which a sign test answers without assuming anything about magnitude.
    sd = res["same_day"].dropna()
    k = int((sd > 0).sum())
    sp = stats.binomtest(k, len(sd), 0.5).pvalue
    print(f"\nSame-day sign test: {k}/{len(sd)} positive, p = {sp:.4f}")

    # Bonferroni over the per-ticker controlled tests, which is the family the
    # TTWO result has to survive to be worth reporting as more than an outlier.
    m = len(res)
    print(f"Bonferroni over {m} controlled tests: alpha = {0.05/m:.4f}")
    survive = res[(res.volume_vol_pc_p < 0.05 / m) & res.volume_vol_pc.notna()]
    print("  clears it: " + (", ".join(
        f"{r.ticker} ({r.volume_vol_pc:+.3f}, p={r.volume_vol_pc_p:.4f})"
        for _, r in survive.iterrows()) or "none"))

    r_all, p_all, n_all = pooled_partial(list(frames.values()))
    print(f"\nPooled controlled correlation, all {len(frames)} tickers: "
          f"{r_all:+.4f}, p = {p_all:.4f}  (n = {n_all:,})")

    ex = [t for t in args.exclude_from_pooled if t in frames]
    if ex:
        keep = [d for t, d in frames.items() if t not in ex]
        r_ex, p_ex, n_ex = pooled_partial(keep)
        print(f"Pooled excluding {', '.join(ex)}: "
              f"{r_ex:+.4f}, p = {p_ex:.4f}  (n = {n_ex:,})")
        print("\nIf the pooled result holds only with a single ticker included, the\n"
              "finding is about that ticker, not about news volume in general.")

    out = os.path.join(REPORT_DIR, f"correlation_analysis_{args.source}.csv")
    res.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
    print("Quote Section 4.3 from this file, not from daily_coverage.csv — that\n"
          "column is computed over a different sample.")


if __name__ == "__main__":
    main()
