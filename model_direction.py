"""
Direction-classification LSTM: does news sentiment improve next-day direction?

WHY THIS REPLACES THE REGRESSION VERSION
----------------------------------------
model_lstm.py trained with MSE on daily returns, which are roughly N(0, 0.02).
The MSE-optimal prediction for noise is its mean, so the model correctly
learned to output near-zero — which is why `baseline_zero` beat every model on
RMSE. Directional accuracy was then read off the sign of a number that was
essentially zero plus jitter, landing near 50% by construction. The model was
not failing to learn; it was optimising the wrong objective for the question.

Three changes:

  1. DIRECTION AS THE TARGET. Binary classification, cross-entropy loss,
     predicting P(up). The model is optimised for the metric actually reported
     and cannot collapse to a constant zero.

  2. SENTIMENT SURPRISE, NOT LEVEL. Sentiment levels are strongly
     autocorrelated and largely already priced in: if a stock has had positive
     coverage for weeks, today's positive article is not news. The features
     here are deviation from a trailing mean, change since yesterday, and
     volume-weighted variants, all computed causally.

  3. POOLING. 558 test predictions per ticker gives a standard error of about
     2.1pp on a proportion, so every per-ticker difference in the old results
     sat inside the noise band. Pooling six tickers gives ~3,300 and roughly
     0.9pp — enough to detect a 2pp effect. --pooled trains one model across
     all tickers; the significance test pools predictions either way.

SIGNIFICANCE
------------
Reports McNemar's test on paired predictions (the two models predict the same
days, so the pairing matters) plus a bootstrap CI on the accuracy difference.
An improvement that fails these is not an improvement.

Run:
    python model_direction.py                 # per-ticker
    python model_direction.py --pooled        # one model across all six
    python model_direction.py --only GOOG NVDA
"""

import argparse
import os

import numpy as np
import pandas as pd

from config import END_DATE_STR, REPORT_DIR, START_DATE_STR, select_stocks, ticker_dir

PRICE_FEATURES = ["return_1d", "return_5d", "volatility_5d", "volume_z", "close_ma_ratio"]

# Several sentiment representations, because which one carries signal is an
# empirical question rather than something to assume. The surprise transform
# was a judgement call — that levels are autocorrelated and priced in — and it
# may equally have destroyed the signal. Most published work uses raw levels or
# article counts, so all three are testable here.
SENTIMENT_SETS = {
    # Deviation from recent norms.
    "surprise": ["sent_surprise", "sent_change", "sent_weighted",
                 "article_count_z", "sent_dispersion"],
    # Raw levels, closest to what most of the literature uses.
    "level": ["sent_filled", "sent_std_filled", "pct_pos_filled",
              "pct_neg_filled", "article_count_log"],
    # Counts and balance rather than the continuous score.
    "counts": ["pos_minus_neg", "pct_pos_filled", "pct_neg_filled",
               "article_count_log", "article_count_z"],
    "all": ["sent_surprise", "sent_change", "sent_weighted", "article_count_z",
            "sent_dispersion", "sent_filled", "sent_std_filled",
            "pct_pos_filled", "pct_neg_filled", "article_count_log",
            "pos_minus_neg"],
}
SENTIMENT_FEATURES = SENTIMENT_SETS["surprise"]

# Control features: pure noise, same count as the sentiment block.
#
# Adding sentiment takes the input from 5 features to 10, which doubles the
# parameter count on ~600 training sequences per fold. Any five uninformative
# features would hurt under those conditions. Without this control, a negative
# sentiment result cannot be distinguished from a dimensionality penalty — so
# the comparison that matters is sentiment vs. noise, not sentiment vs. nothing.
NOISE_FEATURES = [f"noise_{i}" for i in range(len(
    ["sent_surprise", "sent_change", "sent_weighted",
     "article_count_z", "sent_dispersion"]))]

LOOKBACK = 20
HIDDEN = 32
DROPOUT = 0.2
EPOCHS = 60
PATIENCE = 10
LR = 1e-3
BATCH = 64

INITIAL_TRAIN_FRAC = 0.55
STEP_DAYS = 63
VAL_FRAC = 0.15
SEED = 42


def fill_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling 3-day causal fill: .shift(1) first, so day t never sees itself."""
    out = df.copy()
    for col, target in [("sentiment_mean", "sent_filled"),
                        ("sentiment_std", "sent_std_filled"),
                        ("pct_positive", "pct_pos_filled"),
                        ("pct_negative", "pct_neg_filled")]:
        s = out[col] if col in out.columns else pd.Series(np.nan, index=out.index)
        out[target] = s.fillna(s.shift(1).rolling(3, min_periods=1).mean()).fillna(0.0)
    return out


def build_features(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    out = df.sort_values("date").reset_index(drop=True).copy()
    price = "adj_close" if "adj_close" in out.columns else "close"

    out["return_1d"] = out[price].pct_change()
    out["return_5d"] = out[price].pct_change(5)
    out["volatility_5d"] = out["return_1d"].rolling(5).std()
    out["close_ma_ratio"] = out[price] / out[price].rolling(20).mean() - 1

    vol = out["volume"].replace(0, np.nan)
    lv = np.log(vol)
    out["volume_z"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()

    out = fill_sentiment(out)
    s = out["sent_filled"]

    # Trailing statistics exclude today, so nothing here uses the present or
    # future to describe the present.
    trail_mean = s.shift(1).rolling(20, min_periods=5).mean()
    trail_std = s.shift(1).rolling(20, min_periods=5).std()

    # A floor on the denominator, and a clip on the result. Without the floor,
    # a flat stretch of sentiment gives std = 0 and the surprise on the day the
    # regime changes divides by zero — losing the single most informative
    # observation. Near-zero std would otherwise produce enormous spikes that
    # dominate the scaler.
    SURPRISE_STD_FLOOR = 0.05
    out["sent_surprise"] = ((s - trail_mean) / trail_std.fillna(0).clip(lower=SURPRISE_STD_FLOOR)
                            ).clip(-5, 5)
    out["sent_change"] = s - s.shift(1)

    counts = out.get("article_count", pd.Series(0, index=out.index)).astype(float)
    log_count = np.log1p(counts)
    out["sent_weighted"] = s * log_count
    c_mean = log_count.shift(1).rolling(20, min_periods=5).mean()
    c_std = log_count.shift(1).rolling(20, min_periods=5).std()
    out["article_count_z"] = ((log_count - c_mean) / c_std.fillna(0).clip(lower=0.1)).clip(-5, 5)

    # Within-day disagreement: three articles at +0.9/-0.9/0.0 average to zero
    # but mean nothing like three neutral ones.
    out["sent_dispersion"] = out["sent_std_filled"].fillna(0.0)
    out["article_count_log"] = np.log1p(counts)
    out["pos_minus_neg"] = out["pct_pos_filled"] - out["pct_neg_filled"]

    out[SENTIMENT_FEATURES] = out[SENTIMENT_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Seeded per ticker so the control is reproducible but not accidentally
    # correlated across tickers.
    rng = np.random.default_rng(abs(hash(str(out.get("ticker", pd.Series(["x"])).iloc[0]))) % (2**32))
    for c in NOISE_FEATURES:
        out[c] = rng.standard_normal(len(out))

    # Horizon: cumulative return over the next `horizon` sessions. Daily
    # direction may simply be too fast for news sentiment to matter, so the
    # horizon is a parameter rather than a fixed choice.
    fwd = out[price].pct_change(horizon).shift(-horizon)
    out["target_return"] = fwd
    out["target"] = (fwd > 0).astype(float)
    out.loc[fwd.isna(), "target"] = np.nan
    return out


def make_sequences(X, y, extra, lookback):
    if len(X) <= lookback:
        return np.empty((0, lookback, X.shape[1])), np.empty((0,)), np.empty((0,))
    xs = np.stack([X[i:i + lookback] for i in range(len(X) - lookback + 1)])
    return xs, y[lookback - 1:], extra[lookback - 1:]


def apply_min_move(Xs, ys, rs, min_move):
    """
    Keep only samples whose target move exceeds a threshold.

    Days where a stock moved 0.05% are coin flips: no signal exists to find and
    the label is close to arbitrary, so they drag accuracy toward 50% in both
    training and evaluation. Restricting to meaningful moves asks the more
    answerable — and more useful — question of whether direction is predictable
    when the stock actually moves. Filtering happens after sequence
    construction, so the lookback window is never broken.
    """
    if not min_move:
        return Xs, ys, rs
    keep = np.abs(rs) >= min_move
    return Xs[keep], ys[keep], rs[keep]


class DirectionLSTM:
    def __init__(self, n_features, hidden=HIDDEN, dropout=DROPOUT, lr=LR, pos_weight=1.0):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.device = self._pick_device()

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
                self.drop = nn.Dropout(dropout)
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(self.drop(out[:, -1])).squeeze(-1)

        self.net = Net().to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        # Class weighting stops the model from simply always predicting the
        # majority direction, which is the classification equivalent of the
        # collapse-to-zero failure in the regression version.
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, device=self.device))

    @staticmethod
    def _pick_device():
        import torch
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def fit(self, Xtr, ytr, Xva, yva, epochs=EPOCHS, patience=PATIENCE, batch=BATCH):
        torch = self.torch
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=self.device)
        Xtr, ytr, Xva, yva = t(Xtr), t(ytr), t(Xva), t(yva)

        best, best_state, bad = float("inf"), None, 0
        for _ in range(epochs):
            self.net.train()
            perm = torch.randperm(len(Xtr), device=self.device)
            for i in range(0, len(Xtr), batch):
                idx = perm[i:i + batch]
                self.opt.zero_grad()
                self.loss_fn(self.net(Xtr[idx]), ytr[idx]).backward()
                self.opt.step()

            self.net.eval()
            with torch.no_grad():
                vloss = float(self.loss_fn(self.net(Xva), yva)) if len(Xva) else 0.0
            if vloss < best - 1e-9:
                best, bad = vloss, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        return best

    def predict_proba(self, X):
        torch = self.torch
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.tensor(X, dtype=torch.float32, device=self.device))
            return torch.sigmoid(logits).cpu().numpy()


def walk_forward(data, feature_cols, lookback, args):
    """Expanding window; scaler fitted per fold on training rows only."""
    import torch
    from sklearn.preprocessing import StandardScaler

    data = data.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
    if len(data) < lookback + 120:
        return pd.DataFrame(), 0

    X_all = data[feature_cols].to_numpy(float)
    y_all = data["target"].to_numpy(float)
    r_all = data["target_return"].to_numpy(float)
    dates = data["date"].to_numpy()
    tick = data["ticker"].to_numpy() if "ticker" in data.columns else np.array([""] * len(data))

    start = int(len(data) * INITIAL_TRAIN_FRAC)
    rows, folds = [], 0

    for ts in range(start, len(data), args.step):
        te = min(ts + args.step, len(data))
        if te - ts < 5:
            break
        torch.manual_seed(SEED + folds)
        np.random.seed(SEED + folds)

        scaler = StandardScaler().fit(X_all[:ts])
        Xs = scaler.transform(X_all)

        Xtr_s, ytr_s, rtr_s = make_sequences(Xs[:ts], y_all[:ts], r_all[:ts], lookback)
        Xtr_s, ytr_s, rtr_s = apply_min_move(Xtr_s, ytr_s, rtr_s, args.min_move)
        if len(Xtr_s) < 80:
            continue
        cut = int(len(Xtr_s) * (1 - VAL_FRAC))
        Xtr, ytr, Xva, yva = Xtr_s[:cut], ytr_s[:cut], Xtr_s[cut:], ytr_s[cut:]

        pos = float(ytr.mean())
        pos_weight = (1 - pos) / pos if 0 < pos < 1 else 1.0

        lo = max(0, ts - lookback + 1)
        Xte_s, yte_s, rte_s = make_sequences(Xs[lo:te], y_all[lo:te], r_all[lo:te], lookback)
        keep = max(len(Xte_s) - (te - ts), 0)
        Xte, yte, rte = Xte_s[keep:], yte_s[keep:], rte_s[keep:]
        te_dates = dates[te - len(Xte):te]
        te_tick = tick[te - len(Xte):te]
        if args.min_move:
            m = np.abs(rte) >= args.min_move
            Xte, yte, rte, te_dates, te_tick = Xte[m], yte[m], rte[m], te_dates[m], te_tick[m]
        if not len(Xte):
            continue

        model = DirectionLSTM(len(feature_cols), hidden=args.hidden, lr=args.lr,
                              pos_weight=pos_weight)
        model.fit(Xtr, ytr, Xva, yva, epochs=args.epochs, patience=args.patience)
        proba = model.predict_proba(Xte)

        rows.append(pd.DataFrame({
            "date": te_dates,
            "ticker": te_tick,
            "y_true": yte,
            "y_return": rte,
            "proba_up": proba,
        }))
        folds += 1

    if not rows:
        return pd.DataFrame(), 0
    return pd.concat(rows, ignore_index=True), folds


def accuracy(oos):
    pred = (oos["proba_up"] >= 0.5).astype(float)
    return float((pred == oos["y_true"]).mean())


def mcnemar(a_correct, b_correct):
    """
    Paired test. Both models predict the same days, so only the days where they
    disagree carry information — an unpaired t-test would throw that away.
    Exact binomial, which is valid at any discordant count.
    """
    from scipy import stats

    b_only = int(((~a_correct) & b_correct).sum())   # sentiment right, price wrong
    a_only = int((a_correct & (~b_correct)).sum())   # price right, sentiment wrong
    n = a_only + b_only
    if n == 0:
        return 1.0, a_only, b_only
    p = float(stats.binomtest(b_only, n, 0.5).pvalue)
    return p, a_only, b_only


def bootstrap_ci(a_correct, b_correct, n_boot=10000, seed=SEED):
    rng = np.random.default_rng(seed)
    diff = b_correct.astype(float) - a_correct.astype(float)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(price_oos, sent_oos, label):
    merged = price_oos.merge(sent_oos, on=["date", "ticker"], suffixes=("_p", "_s"))
    if merged.empty:
        return None

    a_correct = ((merged["proba_up_p"] >= 0.5).astype(float) == merged["y_true_p"]).to_numpy()
    b_correct = ((merged["proba_up_s"] >= 0.5).astype(float) == merged["y_true_s"]).to_numpy()

    acc_a, acc_b = float(a_correct.mean()), float(b_correct.mean())
    majority = max(float(merged["y_true_p"].mean()), 1 - float(merged["y_true_p"].mean()))
    p, a_only, b_only = mcnemar(a_correct, b_correct)
    lo, hi = bootstrap_ci(a_correct, b_correct)

    return {
        "scope": label,
        "n": len(merged),
        "majority_baseline": round(majority, 4),
        "price_only": round(acc_a, 4),
        "price_plus_sentiment": round(acc_b, 4),
        "difference_pp": round((acc_b - acc_a) * 100, 2),
        "ci_low_pp": round(lo * 100, 2),
        "ci_high_pp": round(hi * 100, 2),
        "mcnemar_p": round(p, 4),
        "price_only_wins": a_only,
        "sentiment_wins": b_only,
        "significant": bool(p < 0.05),
    }


def run_one(args, sent_set, horizon, min_move, pooled):
    """One configuration -> pooled sentiment-vs-price and sentiment-vs-noise."""
    global SENTIMENT_FEATURES
    SENTIMENT_FEATURES = SENTIMENT_SETS[sent_set]

    cfg = argparse.Namespace(**vars(args))
    cfg.horizon, cfg.min_move = horizon, min_move

    frames = []
    for stock in select_stocks(args.only):
        path = os.path.join(ticker_dir(stock.symbol),
                            f"{stock.symbol}_daily_{START_DATE_STR}_{END_DATE_STR}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        f = build_features(df, horizon=horizon)
        f["ticker"] = stock.symbol
        f["category"] = stock.category
        frames.append(f)
    if not frames:
        return None

    if pooled:
        data = pd.concat(frames, ignore_index=True).sort_values(
            ["date", "ticker"]).reset_index(drop=True)
        p_oos, _ = walk_forward(data, PRICE_FEATURES, cfg.lookback, cfg)
        s_oos, _ = walk_forward(data, PRICE_FEATURES + SENTIMENT_FEATURES, cfg.lookback, cfg)
        n_oos, _ = walk_forward(data, PRICE_FEATURES + NOISE_FEATURES, cfg.lookback, cfg)
    else:
        ps, ss, ns = [], [], []
        for f in frames:
            a, _ = walk_forward(f, PRICE_FEATURES, cfg.lookback, cfg)
            b, _ = walk_forward(f, PRICE_FEATURES + SENTIMENT_FEATURES, cfg.lookback, cfg)
            c, _ = walk_forward(f, PRICE_FEATURES + NOISE_FEATURES, cfg.lookback, cfg)
            if not a.empty:
                ps.append(a); ss.append(b); ns.append(c)
        if not ps:
            return None
        p_oos, s_oos, n_oos = (pd.concat(ps, ignore_index=True),
                               pd.concat(ss, ignore_index=True),
                               pd.concat(ns, ignore_index=True))

    if p_oos.empty or s_oos.empty:
        return None

    vs_price = evaluate(p_oos, s_oos, "x")
    vs_noise = evaluate(n_oos, s_oos, "x") if not n_oos.empty else None

    return {
        "sentiment_set": sent_set, "horizon": horizon,
        "min_move": min_move, "pooled": pooled,
        "n": vs_price["n"],
        "majority": vs_price["majority_baseline"],
        "price_only": vs_price["price_only"],
        "with_sentiment": vs_price["price_plus_sentiment"],
        "vs_price_pp": vs_price["difference_pp"],
        "vs_price_p": vs_price["mcnemar_p"],
        # The comparison that isolates signal from dimensionality: both arms
        # carry the same number of features.
        "vs_noise_pp": vs_noise["difference_pp"] if vs_noise else None,
        "vs_noise_p": vs_noise["mcnemar_p"] if vs_noise else None,
    }


def run_grid(args):
    """Ablation across representation, horizon, movement threshold and pooling."""
    sets = ["surprise", "level", "counts"]
    horizons = [1, 5]
    moves = [0.0, 0.01]
    pooled_opts = [True, False]

    total = len(sets) * len(horizons) * len(moves) * len(pooled_opts)
    print(f"Running {total} configurations. Each trains 3 models per arm.\n")

    rows, i = [], 0
    for pooled in pooled_opts:
        for sset in sets:
            for h in horizons:
                for mm in moves:
                    i += 1
                    tag = (f"[{i}/{total}] {'pooled' if pooled else 'per-ticker'} "
                           f"{sset} h={h} min_move={mm}")
                    print(f"{tag} ...", flush=True)
                    r = run_one(args, sset, h, mm, pooled)
                    if r:
                        rows.append(r)
                        print(f"    n={r['n']:5}  price {r['price_only']*100:.2f}%  "
                              f"sent {r['with_sentiment']*100:.2f}%  "
                              f"vs_price {r['vs_price_pp']:+.2f}pp (p={r['vs_price_p']})  "
                              f"vs_noise {r['vs_noise_pp']:+.2f}pp (p={r['vs_noise_p']})")

    if not rows:
        print("No configurations produced results.")
        return

    res = pd.DataFrame(rows).sort_values("vs_noise_pp", ascending=False)
    path = os.path.join(REPORT_DIR, "ablation_results.csv")
    res.to_csv(path, index=False)

    print("\n=== Ablation, best first by sentiment-vs-noise ===")
    print(res.to_string(index=False))

    hits = res[(res["vs_noise_p"] < 0.05) & (res["vs_noise_pp"] > 0)]
    print("\n" + ("Configurations where sentiment beats an equal number of random "
                  f"features (p<0.05): {len(hits)}"))
    if len(hits):
        print(hits.to_string(index=False))
        print("\nNote: with {} configurations tested, expect ~{:.1f} false positives "
              "at p<0.05 by chance. Treat single hits as hypotheses to re-test, "
              "not findings.".format(len(res), len(res) * 0.05))
    else:
        print("None. Across every representation, horizon and threshold tried, "
              "sentiment is indistinguishable from noise of the same dimension.")
    print(f"\nSaved -> {path}")


def main():
    ap = argparse.ArgumentParser(description="Direction-classification LSTM with significance tests.")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--pooled", action="store_true",
                    help="Train one model across all tickers instead of one each")
    ap.add_argument("--lookback", type=int, default=LOOKBACK)
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--step", type=int, default=STEP_DAYS)
    ap.add_argument("--sentiment-set", choices=list(SENTIMENT_SETS), default="surprise",
                    help="Which sentiment representation to test (default surprise)")
    ap.add_argument("--horizon", type=int, default=1,
                    help="Predict direction over the next N sessions (default 1)")
    ap.add_argument("--min-move", type=float, default=0.0,
                    help="Only use days whose absolute target return exceeds this "
                         "(e.g. 0.01 for 1%%). Tiny moves are coin flips.")
    ap.add_argument("--grid", action="store_true",
                    help="Run the full ablation across sets, horizons and thresholds")
    ap.add_argument("--no-control", action="store_true",
                    help="Skip the random-noise control arm (faster, but then a "
                         "negative sentiment result is uninterpretable)")
    args = ap.parse_args()

    if args.grid:
        run_grid(args)
        return

    global SENTIMENT_FEATURES
    SENTIMENT_FEATURES = SENTIMENT_SETS[args.sentiment_set]

    stocks = select_stocks(args.only)
    frames = []
    for stock in stocks:
        path = os.path.join(ticker_dir(stock.symbol),
                            f"{stock.symbol}_daily_{START_DATE_STR}_{END_DATE_STR}.csv")
        if not os.path.exists(path):
            print(f"{stock.symbol}: no daily file — run build_daily_dataset.py first")
            continue
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        f = build_features(df, horizon=args.horizon)
        f["ticker"] = stock.symbol
        f["category"] = stock.category
        frames.append(f)

    if not frames:
        return

    results, preds = [], []

    if args.pooled:
        # Sorted by date so the walk-forward boundary is a point in time, not a
        # point in the file — otherwise one ticker's future trains on another's.
        pooled = pd.concat(frames, ignore_index=True).sort_values(
            ["date", "ticker"]).reset_index(drop=True)
        print(f"=== POOLED ({len(frames)} tickers, {len(pooled):,} rows) ===")
        p_oos, folds = walk_forward(pooled, PRICE_FEATURES, args.lookback, args)
        s_oos, _ = walk_forward(pooled, PRICE_FEATURES + SENTIMENT_FEATURES,
                                args.lookback, args)
        n_oos, _ = (walk_forward(pooled, PRICE_FEATURES + NOISE_FEATURES,
                                 args.lookback, args)
                    if not args.no_control else (pd.DataFrame(), 0))
        if not p_oos.empty:
            r = evaluate(p_oos, s_oos, "pooled")
            results.append(r)
            preds += [p_oos.assign(model="price_only"), s_oos.assign(model="price_plus_sentiment")]
            if not n_oos.empty:
                rc = evaluate(p_oos, n_oos, "pooled_noise_control")
                rc["model_b"] = "price_plus_noise"
                results.append(rc)
                preds.append(n_oos.assign(model="price_plus_noise"))
                print(f"  price_plus_noise     {rc['price_plus_sentiment']*100:.2f}%  "
                      f"(control: {rc['difference_pp']:+.2f}pp vs price_only)")
            print(f"  folds {folds}  majority {r['majority_baseline']*100:.2f}%")
            print(f"  price_only           {r['price_only']*100:.2f}%")
            print(f"  price_plus_sentiment {r['price_plus_sentiment']*100:.2f}%")
            print(f"  difference {r['difference_pp']:+.2f}pp  "
                  f"95% CI [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}]  "
                  f"McNemar p={r['mcnemar_p']}")
    else:
        for f in frames:
            sym = f["ticker"].iloc[0]
            print(f"\n=== {sym} ({f['category'].iloc[0]}) ===")
            p_oos, folds = walk_forward(f, PRICE_FEATURES, args.lookback, args)
            s_oos, _ = walk_forward(f, PRICE_FEATURES + SENTIMENT_FEATURES,
                                    args.lookback, args)
            n_oos, _ = (walk_forward(f, PRICE_FEATURES + NOISE_FEATURES, args.lookback, args)
                        if not args.no_control else (pd.DataFrame(), 0))
            if p_oos.empty:
                print("  not enough data")
                continue
            r = evaluate(p_oos, s_oos, sym)
            r["category"] = f["category"].iloc[0]
            results.append(r)
            preds += [p_oos.assign(model="price_only"), s_oos.assign(model="price_plus_sentiment")]
            if not n_oos.empty:
                rc = evaluate(p_oos, n_oos, f"{sym}_noise_control")
                rc["category"] = f["category"].iloc[0]
                rc["model_b"] = "price_plus_noise"
                results.append(rc)
                preds.append(n_oos.assign(model="price_plus_noise"))
            print(f"  majority baseline    {r['majority_baseline']*100:.2f}%")
            print(f"  price_only           {r['price_only']*100:.2f}%")
            print(f"  price_plus_sentiment {r['price_plus_sentiment']*100:.2f}%")
            print(f"  difference {r['difference_pp']:+.2f}pp  "
                  f"95% CI [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}]  "
                  f"McNemar p={r['mcnemar_p']}"
                  + ("  SIGNIFICANT" if r["significant"] else ""))

        # Per-ticker tests are underpowered at ~558 predictions each; pooling
        # the predictions is the test with enough power to matter.
        all_p = pd.concat([d for d in preds if (d["model"] == "price_only").all()])
        all_s = pd.concat([d for d in preds if (d["model"] == "price_plus_sentiment").all()])
        r = evaluate(all_p.drop(columns="model"), all_s.drop(columns="model"),
                     "ALL TICKERS POOLED")
        if r:
            results.append(r)
            print(f"\n=== Pooled across tickers (n={r['n']:,}) ===")
            print(f"  price_only           {r['price_only']*100:.2f}%")
            print(f"  price_plus_sentiment {r['price_plus_sentiment']*100:.2f}%")
            print(f"  difference {r['difference_pp']:+.2f}pp  "
                  f"95% CI [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}]  "
                  f"McNemar p={r['mcnemar_p']}")
            print("  " + ("Significant at p<0.05." if r["significant"]
                          else "Not significant — no detectable effect."))

        noise_preds = [d for d in preds if (d["model"] == "price_plus_noise").all()]
        if noise_preds:
            all_n = pd.concat(noise_preds)
            rc = evaluate(all_p.drop(columns="model"), all_n.drop(columns="model"),
                          "ALL TICKERS POOLED (noise control)")
            if rc:
                rc["model_b"] = "price_plus_noise"
                results.append(rc)
                print(f"\n=== Noise control, pooled (n={rc['n']:,}) ===")
                print(f"  price_plus_noise     {rc['price_plus_sentiment']*100:.2f}%")
                print(f"  difference {rc['difference_pp']:+.2f}pp  "
                      f"95% CI [{rc['ci_low_pp']:+.2f}, {rc['ci_high_pp']:+.2f}]  "
                      f"McNemar p={rc['mcnemar_p']}")
                print("\n  Interpretation:")
                print(f"    sentiment vs price_only : {r['difference_pp']:+.2f}pp")
                print(f"    noise     vs price_only : {rc['difference_pp']:+.2f}pp")
                if abs(r["difference_pp"] - rc["difference_pp"]) < 1.5:
                    print("    -> similar. The penalty is dimensionality, not sentiment:")
                    print("       five uninformative features would have done the same.")
                elif r["difference_pp"] < rc["difference_pp"] - 1.5:
                    print("    -> sentiment is WORSE than random noise. Something in the")
                    print("       sentiment features is actively misleading the model.")
                else:
                    print("    -> sentiment beats random noise, so it carries some signal,")
                    print("       even if not enough to beat price_only outright.")

    res = pd.DataFrame(results)
    rp = os.path.join(REPORT_DIR, "direction_results.csv")
    res.to_csv(rp, index=False)
    pd.concat(preds, ignore_index=True).to_csv(
        os.path.join(REPORT_DIR, "direction_predictions.csv"), index=False)

    print(f"\n=== Summary ===\n{res.to_string(index=False)}")
    print(f"\nSaved -> {rp}")
    print("\nA difference whose CI spans zero is not an effect, whatever its sign.")


if __name__ == "__main__":
    main()