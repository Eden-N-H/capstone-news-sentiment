"""
Walk-forward LSTM: does news sentiment improve next-day return prediction?

DESIGN
------
The question is comparative, so every ticker is modelled twice under identical
conditions — once on price features alone, once with sentiment added. A
sentiment model scoring 55% directional accuracy means nothing on its own; it
means something only against the price-only model on the same folds.

Three baselines are also reported, because an LSTM that cannot beat them is not
predicting anything:
  * zero      — always predict no change. Hard to beat on RMSE, since daily
                returns are mostly noise around zero.
  * previous  — predict yesterday's return (momentum).
  * majority  — always predict the more common direction. This is the number
                that matters: roughly 53% of days are up, so 53% directional
                accuracy is worthless.

WALK-FORWARD
------------
Train on an expanding window, predict the next block, step forward, retrain.
Out-of-sample predictions from every fold are concatenated and scored together.
The data is never shuffled and the scaler is fitted on each training fold only —
fitting it on the whole series would leak the test period's mean and variance
backwards into training, which inflates results and is easy to miss.

TARGET
------
Next-day simple return, with direction taken from the sign of the prediction.
Predicting the price level instead produces impressive R-squared because the
model learns to echo yesterday's price; returns are the honest target and are
what "fluctuations" means.

NO-NEWS DAYS
------------
Filled with the mean of the previous three days that had news — past values
only, so no future information enters the feature. Days still empty after that
(the start of a series) become 0.

Requirements:
    pip install torch pandas numpy scikit-learn

Run:
    python model_lstm.py
    python model_lstm.py --only NVDA GOOG
    python model_lstm.py --epochs 60 --lookback 30
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from config import END_DATE_STR, REPORT_DIR, START_DATE_STR, select_stocks, ticker_dir

PRICE_FEATURES = ["return_1d", "return_5d", "volatility_5d", "volume_z", "close_ma_ratio"]
SENTIMENT_FEATURES = ["sentiment_filled", "sentiment_std_filled", "article_count_log",
                      "pct_positive_filled", "pct_negative_filled"]

LOOKBACK = 20
HIDDEN = 32
LAYERS = 1
DROPOUT = 0.2
EPOCHS = 40
PATIENCE = 8
LR = 1e-3
BATCH = 32

# Walk-forward: train on everything up to a point, predict the next block.
INITIAL_TRAIN_FRAC = 0.55
STEP_DAYS = 63          # one quarter
VAL_FRAC = 0.15         # tail of each training window, for early stopping

SEED = 42


def fill_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling 3-day fill for days with no news.

    .shift(1) before .rolling() is what makes this causal: without it, day t's
    own value would be included in its own fill, and on a day that has no value
    pandas would still reach forward when the window is centred. Shifting first
    means the fill only ever sees days strictly before t.
    """
    out = df.copy()
    for col, target in [("sentiment_mean", "sentiment_filled"),
                        ("sentiment_std", "sentiment_std_filled"),
                        ("pct_positive", "pct_positive_filled"),
                        ("pct_negative", "pct_negative_filled")]:
        s = out[col] if col in out.columns else pd.Series(np.nan, index=out.index)
        prior_mean = s.shift(1).rolling(3, min_periods=1).mean()
        out[target] = s.fillna(prior_mean).fillna(0.0)

    out["article_count_log"] = np.log1p(out.get("article_count", 0))
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Price features plus the filled sentiment columns, all causal."""
    out = df.sort_values("date").reset_index(drop=True).copy()
    price = "adj_close" if "adj_close" in out.columns else "close"

    out["return_1d"] = out[price].pct_change()
    out["return_5d"] = out[price].pct_change(5)
    out["volatility_5d"] = out["return_1d"].rolling(5).std()
    out["close_ma_ratio"] = out[price] / out[price].rolling(20).mean() - 1

    vol = out["volume"].replace(0, np.nan)
    out["volume_z"] = ((np.log(vol) - np.log(vol).rolling(20).mean())
                       / np.log(vol).rolling(20).std())

    out = fill_sentiment(out)

    # Target: next day's return. Known only after the fact, which is the point.
    out["target"] = out["return_1d"].shift(-1)
    return out


def make_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    """Sliding windows. Sample i uses rows [i, i+lookback) to predict y[i+lookback-1]."""
    if len(X) <= lookback:
        return np.empty((0, lookback, X.shape[1])), np.empty((0,))
    xs = np.stack([X[i:i + lookback] for i in range(len(X) - lookback + 1)])
    ys = y[lookback - 1:]
    return xs, ys


class LSTMRegressor:
    """Thin wrapper so the walk-forward loop stays readable."""

    def __init__(self, n_features, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT,
                 lr=LR, device=None):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.device = device or self._pick_device()

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                                    dropout=dropout if layers > 1 else 0.0)
                self.drop = nn.Dropout(dropout)
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(self.drop(out[:, -1])).squeeze(-1)

        self.net = Net().to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

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
        Xtr = torch.tensor(Xtr, dtype=torch.float32, device=self.device)
        ytr = torch.tensor(ytr, dtype=torch.float32, device=self.device)
        Xva = torch.tensor(Xva, dtype=torch.float32, device=self.device)
        yva = torch.tensor(yva, dtype=torch.float32, device=self.device)

        best, best_state, bad = float("inf"), None, 0
        n = len(Xtr)

        for _ in range(epochs):
            self.net.train()
            perm = torch.randperm(n, device=self.device)   # shuffles samples, not time
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                self.opt.zero_grad()
                loss = self.loss_fn(self.net(Xtr[idx]), ytr[idx])
                loss.backward()
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

    def predict(self, X):
        torch = self.torch
        self.net.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32, device=self.device)
            return self.net(t).cpu().numpy()


def metrics(y_true, y_pred, name):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # Flat days are excluded from directional accuracy: a zero return has no
    # direction to get right, and including them silently rewards predicting 0.
    moved = y_true != 0
    direction = (float(np.mean(np.sign(y_pred[moved]) == np.sign(y_true[moved])))
                 if moved.sum() else float("nan"))
    return {"model": name, "n": int(len(y_true)), "rmse": rmse, "mae": mae,
            "directional_accuracy": direction}


def walk_forward(df, feature_cols, lookback, args, seed=SEED):
    """
    Expanding-window walk-forward. Returns (out-of-sample frame, n_folds).

    Each fold fits its own scaler on training rows only.
    """
    import torch
    from sklearn.preprocessing import StandardScaler

    data = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
    if len(data) < lookback + 100:
        return pd.DataFrame(), 0

    X_all = data[feature_cols].to_numpy(dtype=np.float64)
    y_all = data["target"].to_numpy(dtype=np.float64)
    dates = data["date"].to_numpy()

    start = int(len(data) * INITIAL_TRAIN_FRAC)
    preds, trues, pdates = [], [], []
    folds = 0

    for test_start in range(start, len(data), args.step):
        test_end = min(test_start + args.step, len(data))
        if test_end - test_start < 5:
            break

        torch.manual_seed(seed + folds)
        np.random.seed(seed + folds)

        scaler = StandardScaler().fit(X_all[:test_start])
        Xs = scaler.transform(X_all)

        # Test sequences need the lookback days preceding the block, which are
        # training rows — allowed, since they are strictly in the past.
        Xtr_seq, ytr_seq = make_sequences(Xs[:test_start], y_all[:test_start], lookback)
        if len(Xtr_seq) < 50:
            continue

        cut = int(len(Xtr_seq) * (1 - VAL_FRAC))
        Xtr, ytr = Xtr_seq[:cut], ytr_seq[:cut]
        Xva, yva = Xtr_seq[cut:], ytr_seq[cut:]

        te_lo = max(0, test_start - lookback + 1)
        Xte_seq, yte_seq = make_sequences(Xs[te_lo:test_end], y_all[te_lo:test_end], lookback)
        keep = len(Xte_seq) - (test_end - test_start)
        Xte, yte = Xte_seq[max(keep, 0):], yte_seq[max(keep, 0):]
        if not len(Xte):
            continue

        model = LSTMRegressor(len(feature_cols), hidden=args.hidden, lr=args.lr)
        model.fit(Xtr, ytr, Xva, yva, epochs=args.epochs, patience=args.patience)

        preds.append(model.predict(Xte))
        trues.append(yte)
        pdates.append(dates[test_end - len(Xte):test_end])
        folds += 1

    if not preds:
        return pd.DataFrame(), 0

    return pd.DataFrame({
        "date": np.concatenate(pdates),
        "y_true": np.concatenate(trues),
        "y_pred": np.concatenate(preds),
    }), folds


def baselines(oos, data):
    """Naive references scored on exactly the same out-of-sample rows."""
    y = oos["y_true"].to_numpy()
    rows = [metrics(y, np.zeros_like(y), "baseline_zero")]

    prev = pd.Series(y).shift(1).fillna(0).to_numpy()
    rows.append(metrics(y, prev, "baseline_previous"))

    up = float((y > 0).mean())
    majority = np.full_like(y, 1.0 if up >= 0.5 else -1.0)
    m = metrics(y, majority, "baseline_majority")
    m["rmse"] = float("nan")     # a constant sign is not a return prediction
    m["mae"] = float("nan")
    rows.append(m)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Walk-forward LSTM with and without sentiment.")
    ap.add_argument("--only", nargs="*", help="Restrict to these tickers")
    ap.add_argument("--lookback", type=int, default=LOOKBACK)
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--step", type=int, default=STEP_DAYS,
                    help="Walk-forward block size in trading days (default 63)")
    args = ap.parse_args()

    stocks = select_stocks(args.only)
    results, all_preds = [], []

    for stock in stocks:
        path = os.path.join(ticker_dir(stock.symbol),
                            f"{stock.symbol}_daily_{START_DATE_STR}_{END_DATE_STR}.csv")
        if not os.path.exists(path):
            print(f"{stock.symbol}: no daily file — run build_daily_dataset.py first")
            continue

        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        feats = build_features(df)

        print(f"\n=== {stock.symbol} ({stock.category}) ===")
        coverage = float((df["article_count"] > 0).mean()) * 100
        print(f"  news coverage {coverage:.1f}% of sessions")

        for name, cols in [("price_only", PRICE_FEATURES),
                           ("price_plus_sentiment", PRICE_FEATURES + SENTIMENT_FEATURES)]:
            oos, folds = walk_forward(feats, cols, args.lookback, args)
            if oos.empty:
                print(f"  {name}: not enough data")
                continue

            m = metrics(oos["y_true"], oos["y_pred"], name)
            m.update({"ticker": stock.symbol, "category": stock.category,
                      "folds": folds, "news_coverage_pct": round(coverage, 1)})
            results.append(m)

            oos = oos.assign(ticker=stock.symbol, model=name)
            all_preds.append(oos)

            print(f"  {name:22} RMSE {m['rmse']:.5f}  "
                  f"dir.acc {m['directional_accuracy']*100:5.2f}%  ({folds} folds)")

        if all_preds:
            for b in baselines(all_preds[-1], feats):
                b.update({"ticker": stock.symbol, "category": stock.category,
                          "folds": 0, "news_coverage_pct": round(coverage, 1)})
                results.append(b)
                acc = b["directional_accuracy"]
                print(f"  {b['model']:22} "
                      + (f"RMSE {b['rmse']:.5f}  " if b["rmse"] == b["rmse"] else " " * 17)
                      + f"dir.acc {acc*100:5.2f}%")

    if not results:
        return

    res = pd.DataFrame(results)[
        ["ticker", "category", "model", "n", "folds", "news_coverage_pct",
         "rmse", "mae", "directional_accuracy"]]
    rpath = os.path.join(REPORT_DIR, "model_results.csv")
    res.to_csv(rpath, index=False)

    ppath = os.path.join(REPORT_DIR, "model_predictions.csv")
    pd.concat(all_preds, ignore_index=True).to_csv(ppath, index=False)

    print("\n=== Results ===")
    print(res.to_string(index=False))

    # The comparison the research question actually asks for.
    print("\n=== Does sentiment help? ===")
    wide = res[res.model.isin(["price_only", "price_plus_sentiment"])].pivot(
        index="ticker", columns="model", values=["rmse", "directional_accuracy"])
    for ticker in wide.index:
        r0 = wide.loc[ticker, ("rmse", "price_only")]
        r1 = wide.loc[ticker, ("rmse", "price_plus_sentiment")]
        d0 = wide.loc[ticker, ("directional_accuracy", "price_only")] * 100
        d1 = wide.loc[ticker, ("directional_accuracy", "price_plus_sentiment")] * 100
        print(f"  {ticker:5} RMSE {r0:.5f} -> {r1:.5f} ({(r1-r0)/r0*100:+.1f}%)   "
              f"dir.acc {d0:5.2f}% -> {d1:5.2f}% ({d1-d0:+.2f}pp)")

    print(f"\nSaved -> {rpath}")
    print(f"Saved -> {ppath}")
    print("\nRead directional accuracy against baseline_majority, not against 50%.")


if __name__ == "__main__":
    main()
