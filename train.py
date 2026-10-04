"""
Train a short-horizon direction model on collected order book features.

  python train.py --symbol btcusdt
  python train.py --symbol btcusdt --sweep      # accuracy vs horizon curve

Rules enforced here on purpose:
  - Chronological splits only. Never shuffle time series.
  - An embargo gap between splits, because a row's label looks HORIZON seconds
    into the future and would otherwise leak across the boundary.
  - Raw price levels (bid/ask/mid/microprice) are dropped. They are
    non-stationary; a tree will happily memorize price regions and post a
    fake 90% that collapses the moment BTC leaves the training range.
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from collect import forward_index

DATA = Path("data")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)

# non-stationary / leaky columns that must never be features
DROP = {"ts", "label", "fwd_ret_bps", "bid", "ask", "mid", "microprice"}


def feature_cols(df):
    return [c for c in df.columns if c not in DROP]


def split(df, horizon, train=0.70, val=0.15):
    """Chronological split with an embargo of `horizon` seconds at each seam."""
    ts = df["ts"].to_numpy()
    t0, t1 = ts[0], ts[-1]
    span = t1 - t0
    a = t0 + span * train
    b = t0 + span * (train + val)

    tr = df[ts <= a - horizon]
    va = df[(ts >= a) & (ts <= b - horizon)]
    te = df[ts >= b]
    return tr, va, te


def fit(tr, va, feats, seed=0):
    ymap = {-1: 0, 0: 1, 1: 2}
    dtr = lgb.Dataset(tr[feats], tr["label"].map(ymap))
    dva = lgb.Dataset(va[feats], va["label"].map(ymap), reference=dtr)

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": seed,
    }
    return lgb.train(
        params, dtr, num_boost_round=2000, valid_sets=[dva],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )


def evaluate(model, te, feats, horizon=10.0, name="test"):
    inv = {0: -1, 1: 0, 2: 1}
    proba = model.predict(te[feats], num_iteration=model.best_iteration)
    pred = np.vectorize(inv.get)(proba.argmax(1))
    y = te["label"].to_numpy()

    acc = accuracy_score(y, pred)
    base = pd.Series(y).value_counts(normalize=True).max()   # always-guess-majority

    # the number that actually matters: mean forward return conditional on signal
    fwd = te["fwd_ret_bps"].to_numpy()
    edge_up = fwd[pred == 1].mean() if (pred == 1).any() else 0.0
    edge_dn = fwd[pred == -1].mean() if (pred == -1).any() else 0.0

    print(f"\n--- {name} ({len(te):,} rows) ---")
    print(f"accuracy        {acc:.4f}   (majority baseline {base:.4f})")
    print(f"lift over base  {acc - base:+.4f}")
    print(f"mean fwd_ret | pred=UP    {edge_up:+.3f} bps  (n={(pred==1).sum():,})")
    print(f"mean fwd_ret | pred=DOWN  {edge_dn:+.3f} bps  (n={(pred==-1).sum():,})")
    print(f"gross edge per trade      {(edge_up - edge_dn)/2:+.3f} bps")

    # Significance. Overlapping windows make naive t-stats far too generous, so
    # also report one on a non-overlapping subsample (trades >= horizon apart).
    traded = pred != 0
    pnl = (pred * fwd)[traded]
    t_all = pnl.mean() / pnl.std() * np.sqrt(len(pnl)) if len(pnl) > 1 else 0.0

    ts_tr = te["ts"].to_numpy()[traded]
    keep, last = [], -np.inf
    for k, tt in enumerate(ts_tr):
        if tt - last >= horizon:
            keep.append(k); last = tt
    pnl_ind = pnl[keep]
    t_ind = (pnl_ind.mean() / pnl_ind.std() * np.sqrt(len(pnl_ind))
             if len(pnl_ind) > 1 and pnl_ind.std() else 0.0)
    print(f"t-stat (overlapping)      {t_all:.2f}   n={len(pnl):,}")
    print(f"t-stat (non-overlapping)  {t_ind:.2f}   n={len(pnl_ind):,}   <- trust this one")
    print("\nconfusion (rows=true down/flat/up):")
    print(confusion_matrix(y, pred, labels=[-1, 0, 1]))

    return {
        "accuracy": float(acc), "baseline": float(base),
        "edge_up_bps": float(edge_up), "edge_down_bps": float(edge_dn),
        "gross_edge_bps": float((edge_up - edge_dn) / 2),
        "t_stat": float(t_ind),
        "n_test": int(len(te)),
    }


def relabel(df, horizon):
    """Re-derive labels at a different horizon without re-collecting."""
    ts = df["ts"].to_numpy()
    mid = df["mid"].to_numpy()
    j, ok = forward_index(ts, horizon)
    fwd = np.where(ok, (mid[j] / mid - 1.0) * 1e4, np.nan)
    thr = np.maximum(df["spread_bps"].to_numpy() / 2.0, 0.5)
    out = df.copy()
    out["fwd_ret_bps"] = fwd
    out["label"] = np.select([fwd > thr, fwd < -thr], [1, -1], default=0)
    out.loc[~ok, "label"] = np.nan
    return out.dropna(subset=["label", "fwd_ret_bps"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="btcusdt")
    ap.add_argument("--horizon", type=float, default=10.0)
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()

    path = DATA / f"labeled_{a.symbol}.parquet"
    if not path.exists():
        raise SystemExit(f"missing {path} -- run: python collect.py --symbol {a.symbol} --label")
    df = pd.read_parquet(path).sort_values("ts").reset_index(drop=True)
    print(f"loaded {len(df):,} rows, {(df.ts.iloc[-1]-df.ts.iloc[0])/3600:.1f} hours")

    horizons = [1, 2, 5, 10, 30, 60] if a.sweep else [a.horizon]
    results = {}

    for h in horizons:
        d = relabel(df, h)      # parquet labels are fixed at build time; always rederive
        feats = feature_cols(d)
        tr, va, te = split(d, h)
        if min(len(tr), len(va), len(te)) < 1000:
            print(f"horizon {h}s: not enough data yet, skipping")
            continue
        m = fit(tr, va, feats)
        r = evaluate(m, te, feats, horizon=h, name=f"horizon={h}s")
        results[str(h)] = r
        if not a.sweep:
            m.save_model(MODELS / f"{a.symbol}_h{int(h)}.txt")
            (MODELS / f"{a.symbol}_h{int(h)}_features.json").write_text(json.dumps(feats))
            imp = pd.Series(
                m.feature_importance("gain"), index=feats
            ).sort_values(ascending=False)
            print("\ntop features by gain:")
            print(imp.head(12).round(0))

    if a.sweep:
        s = pd.DataFrame(results).T
        s.index.name = "horizon_s"
        print("\n=== signal decay ===")
        print(s[["gross_edge_bps", "t_stat", "n_test", "accuracy"]].round(4))
        s.to_csv(MODELS / f"{a.symbol}_sweep.csv")
        print(f"\nsaved -> models/{a.symbol}_sweep.csv  (this table is your headline chart)")


if __name__ == "__main__":
    main()
