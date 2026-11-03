"""
Tuned comparison: does sentiment help once models are properly optimised?

Pre-registered success criterion
--------------------------------
Sentiment counts as helping ONLY if the sentiment arm beats the noise-control
arm at p < 0.05 on the pooled test (McNemar, paired). This is stated before
running, and the script prints it first. Anything else — a sentiment arm that
beats price-only but not noise, or a single model family that clears the bar by
a hair after many tries — does not count.

What this adds over model_direction.py
--------------------------------------
1. TWO MODEL FAMILIES. The LSTM, plus HistGradientBoostingClassifier. Tree
   ensembles routinely beat LSTMs on small tabular financial datasets and are
   much less damaged by extra uninformative features — precisely the
   dimensionality penalty measured earlier. scikit-learn's implementation is
   used rather than LightGBM, which needs libomp on macOS.

2. TUNING INSIDE EACH FOLD. Every fold picks its hyperparameters on a
   validation slice taken from the END of its own training window, then refits.
   The test block is never seen during tuning. All three arms get the identical
   search budget, so none is advantaged.

3. SEED ENSEMBLING. The final LSTM for each fold averages several random
   initialisations, so the comparison is not decided by initialisation luck.

4. CORRECT POOLING. Sequences are built per ticker and only then combined.
   model_direction.py --pooled sorted all tickers by date and slid a window over
   the result, so each "sequence" interleaved six stocks — those pooled results
   are invalid. Here the walk-forward also steps by calendar date, not by row.

5. CLEANER TARGET. FNSPID timestamps are date-only, so a day-D article may have
   been written after day D's close. Predicting close(D) -> close(D+1) therefore
   lets evening news leak into a window that had already opened. --target
   open_to_close predicts open(D+1) -> close(D+1) instead: the first move a
   trader could actually act on.

Run:
    python model_tuned.py                          # both families, default target
    python model_tuned.py --target open_to_close   # leak-free target
    python model_tuned.py --model gbm              # fast: trees only
"""

import argparse
import os
import zlib

import numpy as np
import pandas as pd

from config import END_DATE_STR, REPORT_DIR, START_DATE_STR, select_stocks, ticker_dir
from model_direction import (
    PRICE_FEATURES, SENTIMENT_SETS, NOISE_FEATURES, DirectionLSTM,
    bootstrap_ci, build_features, fill_sentiment, mcnemar,
)

# Four arms on identical folds. Each sentiment arm carries exactly five extra
# features, the same as the noise arm, so no arm is advantaged or penalised by
# dimensionality — which was the confound that made the first direction results
# uninterpretable.
NEWS_FEATURES = ["news_sent", "news_sent_std", "news_pct_pos",
                 "news_pct_neg", "news_count_log"]
REDDIT_FEATURES = ["rdt_sent", "rdt_sent_std", "rdt_pct_pos",
                   "rdt_pct_neg", "rdt_count_log"]
# Swapped in for rdt_sent with --reddit-weighted: the upvote-weighted daily
# mean rather than the plain one, so reach is accounted for.
REDDIT_WEIGHTED = "rdt_sent_weighted"

# The combined arm carries 15 features against price_only's 5, so it needs its
# OWN control with 10 random features. Comparing a 15-feature arm against the
# 10-feature noise arm would confound "does combining platforms help" with
# "do five more features hurt" — exactly the confound that made the first
# direction results uninterpretable.
EXTRA_NOISE = [f"noise_{i}" for i in range(5, 10)]
NOISE10 = NOISE_FEATURES + EXTRA_NOISE

INITIAL_TRAIN_FRAC = 0.55
STEP_DATES = 63
VAL_FRAC = 0.15
TREE_LAGS = 5                 # trees see today's features plus 4 lags

LSTM_GRID = [{"hidden": h, "lookback": lb} for h in (16, 32) for lb in (10, 20)]
GBM_GRID = [{"learning_rate": lr, "max_leaf_nodes": ml, "min_samples_leaf": ms}
            for lr in (0.03, 0.1) for ml in (7, 31) for ms in (20, 80)]

SUCCESS = ("A platform helps ONLY if its arm beats the NOISE arm at p<0.05 on "
           "the pooled test. Beating price_only is not enough — five extra "
           "features change accuracy on their own. The platform comparison "
           "(news vs Reddit) is only meaningful if at least one clears that bar.")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def reddit_features(path, weighted=False) -> pd.DataFrame:
    """
    Daily Reddit sentiment, filled and named so it cannot collide with news.

    Uses the same causal 3-day fill as the news arm — identical processing on
    both platforms, so any difference in results is a difference between
    platforms rather than between pipelines.
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = fill_sentiment(df)

    base = df["sent_filled"]
    if weighted and "sentiment_weighted" in df.columns:
        # Falls back to the plain mean wherever weighting is undefined (a
        # session where every post scored zero).
        base = df["sentiment_weighted"].fillna(df["sent_filled"])

    out = pd.DataFrame({
        "date": df["date"],
        "rdt_sent": base,
        "rdt_sent_std": df["sent_std_filled"],
        "rdt_pct_pos": df["pct_pos_filled"],
        "rdt_pct_neg": df["pct_neg_filled"],
        "rdt_count_log": np.log1p(df.get("article_count", 0)),
    })
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def load(args):
    frames = []
    for stock in select_stocks(args.only):
        path = os.path.join(ticker_dir(stock.symbol),
                            f"{stock.symbol}_daily_{START_DATE_STR}_{END_DATE_STR}.csv")
        if not os.path.exists(path):
            print(f"{stock.symbol}: no daily file — run build_daily_dataset.py")
            continue
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        f = build_features(df, horizon=1)

        # News sentiment, renamed so both platforms can coexist in one frame.
        f = f.rename(columns={
            "sent_filled": "news_sent", "sent_std_filled": "news_sent_std",
            "pct_pos_filled": "news_pct_pos", "pct_neg_filled": "news_pct_neg",
            "article_count_log": "news_count_log",
        })

        rpath = os.path.join(ticker_dir(stock.symbol),
                             f"{stock.symbol}_daily_reddit_{START_DATE_STR}_{END_DATE_STR}.csv")
        if os.path.exists(rpath):
            f = f.merge(reddit_features(rpath, args.reddit_weighted), on="date", how="left")
        else:
            print(f"  {stock.symbol}: no Reddit daily file — Reddit arm unavailable")
            for c in REDDIT_FEATURES:
                f[c] = np.nan
        f[REDDIT_FEATURES] = f[REDDIT_FEATURES].fillna(0.0)

        # Reproducible noise. model_direction seeds from hash(), which Python
        # randomises per process, so the control would differ run to run.
        rng = np.random.default_rng(zlib.crc32(stock.symbol.encode()))
        for c in NOISE10:
            f[c] = rng.standard_normal(len(f))

        if args.target == "open_to_close":
            nxt = f["close"].shift(-1) / f["open"].shift(-1) - 1
            f["target_return"] = nxt
            f["target"] = (nxt > 0).astype(float)
            f.loc[nxt.isna(), "target"] = np.nan

        f["ticker"] = stock.symbol
        f["category"] = stock.category
        frames.append(f)
    return frames


def arm_columns(sentiment_set=None, combined=True):
    arms = {
        "price_only": PRICE_FEATURES,                        # 5
        "price_plus_news": PRICE_FEATURES + NEWS_FEATURES,   # 10
        "price_plus_reddit": PRICE_FEATURES + REDDIT_FEATURES,  # 10
        "price_plus_noise": PRICE_FEATURES + NOISE_FEATURES,    # 10 — control
    }
    if combined:
        arms["price_plus_both"] = PRICE_FEATURES + NEWS_FEATURES + REDDIT_FEATURES  # 15
        arms["price_plus_noise10"] = PRICE_FEATURES + NOISE10                        # 15 — control
    return arms


# ---------------------------------------------------------------------------
# Sample construction — always per ticker, then pooled
# ---------------------------------------------------------------------------
def tabular_samples(frames, cols, lags=TREE_LAGS):
    """Today's features plus lags, one row per (ticker, date)."""
    parts = []
    for f in frames:
        d = f[["date", "ticker", "target"] + cols].copy()
        lagged = [d[cols]]
        for k in range(1, lags):
            lagged.append(d[cols].shift(k).add_suffix(f"_lag{k}"))
        X = pd.concat(lagged, axis=1)
        out = pd.concat([d[["date", "ticker", "target"]], X], axis=1).dropna()
        parts.append(out)
    s = pd.concat(parts, ignore_index=True)
    feat = [c for c in s.columns if c not in ("date", "ticker", "target")]
    return s["date"].to_numpy(), s["ticker"].to_numpy(), s[feat].to_numpy(float), \
        s["target"].to_numpy(float)


def sequence_samples(frames, cols, lookback):
    """Windows built within each ticker, so no sequence spans two stocks."""
    D, T, X, Y = [], [], [], []
    for f in frames:
        d = f.dropna(subset=cols + ["target"]).reset_index(drop=True)
        if len(d) <= lookback:
            continue
        arr = d[cols].to_numpy(float)
        for i in range(lookback - 1, len(d)):
            X.append(arr[i - lookback + 1:i + 1])
            Y.append(d["target"].iat[i])
            D.append(d["date"].iat[i])
            T.append(d["ticker"].iat[i])
    return np.array(D), np.array(T), np.array(X), np.array(Y, dtype=float)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def fit_gbm(Xtr, ytr, params, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    # early_stopping off: its validation split is random, which would leak
    # the future into a time series.
    m = HistGradientBoostingClassifier(max_iter=200, early_stopping=False,
                                       class_weight="balanced", random_state=seed,
                                       **params)
    m.fit(Xtr, ytr)
    return m


def gbm_proba(m, X):
    return m.predict_proba(X)[:, 1]


def fit_lstm(Xtr, ytr, Xva, yva, params, seed, args):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    pos = float(ytr.mean())
    m = DirectionLSTM(Xtr.shape[2], hidden=params["hidden"], lr=args.lr,
                      pos_weight=(1 - pos) / pos if 0 < pos < 1 else 1.0)
    m.fit(Xtr, ytr, Xva, yva, epochs=args.epochs, patience=args.patience)
    return m


def scale_seq(train, *others):
    from sklearn.preprocessing import StandardScaler
    n, L, F = train.shape
    sc = StandardScaler().fit(train.reshape(-1, F))
    tf = lambda a: sc.transform(a.reshape(-1, F)).reshape(a.shape) if len(a) else a
    return (tf(train),) + tuple(tf(o) for o in others)


# ---------------------------------------------------------------------------
# Walk-forward by calendar date, with tuning inside each fold
# ---------------------------------------------------------------------------
def date_folds(all_dates):
    u = np.array(sorted(pd.unique(all_dates)))
    start = int(len(u) * INITIAL_TRAIN_FRAC)
    for i in range(start, len(u), STEP_DATES):
        lo, hi = u[i], u[min(i + STEP_DATES, len(u)) - 1]
        train_dates = u[:i]
        val_cut = train_dates[int(len(train_dates) * (1 - VAL_FRAC))]
        yield lo, hi, val_cut


def run_gbm(frames, cols, args):
    dates, ticks, X, y = tabular_samples(frames, cols)
    out = []
    for lo, hi, val_cut in date_folds(dates):
        tr = dates < lo
        te = (dates >= lo) & (dates <= hi)
        inner, val = dates < val_cut, (dates >= val_cut) & (dates < lo)
        if tr.sum() < 200 or te.sum() == 0:
            continue

        best, best_acc = GBM_GRID[0], -1
        for p in GBM_GRID:
            m = fit_gbm(X[inner], y[inner], p, seed=0)
            acc = float(((gbm_proba(m, X[val]) >= .5) == y[val]).mean())
            if acc > best_acc:
                best, best_acc = p, acc

        m = fit_gbm(X[tr], y[tr], best, seed=0)
        out.append(pd.DataFrame({"date": dates[te], "ticker": ticks[te],
                                 "y_true": y[te], "proba_up": gbm_proba(m, X[te])}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def run_lstm(frames, cols, args):
    # Build every lookback once; the grid reuses them.
    cache = {lb: sequence_samples(frames, cols, lb)
             for lb in sorted({p["lookback"] for p in LSTM_GRID})}
    ref_dates = cache[max(cache)][0]
    out = []

    for lo, hi, val_cut in date_folds(ref_dates):
        best, best_acc = LSTM_GRID[0], -1
        for p in LSTM_GRID:
            d, t, X, y = cache[p["lookback"]]
            inner, val = d < val_cut, (d >= val_cut) & (d < lo)
            if inner.sum() < 200 or val.sum() < 20:
                continue
            Xi, Xv = scale_seq(X[inner], X[val])
            m = fit_lstm(Xi, y[inner], Xv, y[val], p, seed=0, args=args)
            acc = float(((m.predict_proba(Xv) >= .5) == y[val]).mean())
            if acc > best_acc:
                best, best_acc = p, acc

        d, t, X, y = cache[best["lookback"]]
        tr, te = d < lo, (d >= lo) & (d <= hi)
        if tr.sum() < 200 or te.sum() == 0:
            continue
        cut = d[tr][int(tr.sum() * (1 - VAL_FRAC))]
        fit_i, fit_v = d < cut, (d >= cut) & (d < lo)
        Xf, Xfv, Xte = scale_seq(X[fit_i], X[fit_v], X[te])

        # Seed ensemble: average probabilities across initialisations.
        probs = [fit_lstm(Xf, y[fit_i], Xfv, y[fit_v], best, seed=s, args=args)
                 .predict_proba(Xte) for s in range(args.seeds)]
        out.append(pd.DataFrame({"date": d[te], "ticker": t[te], "y_true": y[te],
                                 "proba_up": np.mean(probs, axis=0)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def compare(a, b, a_name, b_name):
    m = a.merge(b, on=["date", "ticker"], suffixes=("_a", "_b"))
    if m.empty:
        return None
    ca = ((m.proba_up_a >= .5).astype(float) == m.y_true_a).to_numpy()
    cb = ((m.proba_up_b >= .5).astype(float) == m.y_true_b).to_numpy()
    p, _, _ = mcnemar(ca, cb)
    lo, hi = bootstrap_ci(ca, cb)
    return {
        "comparison": f"{b_name} vs {a_name}",
        # Fixed column names. Naming the accuracy columns after the models
        # being compared meant every comparison produced a different schema,
        # which made the appended all-runs log unparseable once two runs with
        # different arm sets were written under one header.
        "model_a": a_name, "model_b": b_name,
        "acc_a": round(float(ca.mean()), 4), "acc_b": round(float(cb.mean()), 4),
        "n": len(m),
        "diff_pp": round((cb.mean() - ca.mean()) * 100, 2),
        "ci_low_pp": round(lo * 100, 2), "ci_high_pp": round(hi * 100, 2),
        "p": round(p, 4),
        # Kept so existing readers of these files keep working.
        a_name: round(float(ca.mean()), 4), b_name: round(float(cb.mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser(description="Tuned, ensembled sentiment comparison.")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--model", choices=["lstm", "gbm", "both"], default="both")
    ap.add_argument("--target", choices=["close_to_close", "open_to_close"],
                    default="close_to_close")
    ap.add_argument("--sentiment-set", choices=list(SENTIMENT_SETS), default="level")
    ap.add_argument("--no-combined", action="store_true",
                    help="Skip the news+Reddit arm and its 10-feature control "
                         "(faster: 4 arms instead of 6)")
    ap.add_argument("--reddit-weighted", action="store_true",
                    help="Use the upvote-weighted daily Reddit mean instead of "
                         "the plain mean")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    print("PRE-REGISTERED CRITERION:", SUCCESS, "\n")
    frames = load(args)
    if not frames:
        return

    families = ["gbm", "lstm"] if args.model == "both" else [args.model]
    arms = arm_columns(args.sentiment_set, combined=not args.no_combined)
    rows = []

    for fam in families:
        print(f"=== {fam.upper()}  target={args.target}  "
              f"sentiment={args.sentiment_set}  pooled over {len(frames)} tickers ===")
        preds = {}
        for arm, cols in arms.items():
            print(f"  training {arm} ...", flush=True)
            preds[arm] = (run_gbm if fam == "gbm" else run_lstm)(frames, cols, args)

        if preds["price_only"].empty:
            print("  not enough data")
            continue

        maj = preds["price_only"]["y_true"].mean()
        print(f"  majority baseline {max(maj, 1-maj)*100:.2f}%")

        pairs = [
            # Each platform against price_only: informative, but confounded by
            # dimensionality, so never decisive on its own.
            ("price_only", "price_plus_news"),
            ("price_only", "price_plus_reddit"),
            ("price_only", "price_plus_noise"),
            # Against noise: the pre-registered test.
            ("price_plus_noise", "price_plus_news"),
            ("price_plus_noise", "price_plus_reddit"),
            # The research question itself, at equal dimensionality.
            ("price_plus_news", "price_plus_reddit"),
            # Combined arm, judged against its OWN 15-feature control.
            ("price_only", "price_plus_both"),
            ("price_plus_noise10", "price_plus_both"),
            # Does combining beat either platform alone? Not dimensionality-
            # matched (15 vs 10), so read alongside the noise10 comparison.
            ("price_plus_news", "price_plus_both"),
            ("price_plus_reddit", "price_plus_both"),
        ]
        for a, b in pairs:
            if preds.get(a) is None or preds.get(b) is None:
                continue
            if preds[a].empty or preds[b].empty:
                continue
            r = compare(preds[a], preds[b], a, b)
            if r is None:
                continue
            r["family"] = fam
            rows.append(r)
            print(f"  {r['comparison']:52} {r['diff_pp']:+.2f}pp  "
                  f"CI [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}]  p={r['p']}")

        for arm, df in preds.items():
            wtag = "_rwt" if args.reddit_weighted else ""
            df.assign(model=arm, family=fam, target=args.target,
                      reddit_weighted=args.reddit_weighted).to_csv(
                os.path.join(REPORT_DIR,
                             f"tuned_preds_{fam}_{args.target}{wtag}_{arm}.csv"),
                index=False)

    res = pd.DataFrame(rows)
    res["target"] = args.target
    res["sentiment_set"] = args.sentiment_set
    res["reddit_weighted"] = args.reddit_weighted
    fam_tag = "_".join(families)
    # Every flag that changes the result must appear in the filename, or a
    # second configuration silently overwrites the first.
    wtag = "_rwt" if args.reddit_weighted else ""
    path = os.path.join(REPORT_DIR, f"tuned_results_{fam_tag}_{args.target}{wtag}.csv")
    res.to_csv(path, index=False)

    # Running log across every configuration, appended rather than replaced, so
    # the full history of what was tried survives any single re-run. That record
    # is also what makes the pre-registered criterion auditable.
    log_path = os.path.join(REPORT_DIR, "tuned_results_all_runs.csv")
    LOG_COLS = ["run_at", "family", "target", "sentiment_set", "reddit_weighted",
                "comparison", "model_a", "model_b", "acc_a", "acc_b", "n",
                "diff_pp", "ci_low_pp", "ci_high_pp", "p"]
    log = res.assign(run_at=pd.Timestamp.now().isoformat(timespec="seconds"))
    # Fixed schema, so appends from runs with different arm sets stay parseable.
    log = log.reindex(columns=LOG_COLS)
    log.to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)

    print("\n=== Verdict against the pre-registered criterion ===")
    for fam in families:
        for platform, ctrl in (("news", "noise"), ("reddit", "noise"),
                               ("both", "noise10")):
            r = res[(res.family == fam) &
                    (res.comparison == f"price_plus_{platform} vs price_plus_{ctrl}")]
            if r.empty:
                continue
            r = r.iloc[0]
            ok = r["p"] < 0.05 and r["diff_pp"] > 0
            print(f"  {fam.upper():5} {platform:6} vs {ctrl:7} {r['diff_pp']:+.2f}pp, "
                  f"p={r['p']}  -> " + ("MEETS criterion" if ok
                                        else "does not meet criterion"))

        head = res[(res.family == fam) &
                   (res.comparison == "price_plus_reddit vs price_plus_news")]
        if not head.empty:
            h = head.iloc[0]
            print(f"  {fam.upper():5} reddit vs news   {h['diff_pp']:+.2f}pp, p={h['p']}"
                  + ("  (neither platform cleared the noise bar, so this "
                     "comparison is between two null signals)"
                     if not any(res[(res.family == fam) &
                                    res.comparison.str.contains("vs price_plus_noise") &
                                    (res.p < 0.05) & (res.diff_pp > 0)].index)
                     else ""))

    print(f"\nSaved -> {path}")
    print(f"Appended -> {log_path}")


if __name__ == "__main__":
    main()