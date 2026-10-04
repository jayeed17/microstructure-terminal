"""
Expanding-window walk-forward: for each calendar day D after the first 3,
train on all rows before D (embargoed), evaluate on D only.

  python walkforward.py btcusdt
"""

import sys, time, numpy as np, pandas as pd
from train import relabel, feature_cols, fit

sym, H, CONF, SKIP = sys.argv[1], 1, 0.6, 3
VAL_FRAC = 0.15 / 0.85            # train.py's 70:15 train:val ratio, applied to pre-D data

df = pd.read_parquet(f"data/labeled_{sym}.parquet").sort_values("ts").reset_index(drop=True)
d = relabel(df, H)
feats = feature_cols(d)
ts = d["ts"].to_numpy()
date = pd.to_datetime(ts, unit="s", utc=True).normalize()
days = sorted(date.unique())
print(f"{sym}: {len(d):,} rows, days {days[0].date()} .. {days[-1].date()} ({len(days)}), "
      f"evaluating {len(days) - SKIP}", flush=True)

out = []
for D in days[SKIP:]:
    t0 = time.time()
    d0 = D.timestamp(); d1 = d0 + 86400
    pre_start = ts[0]
    a = d0 - (d0 - pre_start) * VAL_FRAC                 # train/val seam
    tr = d[ts <= a - H]
    va = d[(ts >= a) & (ts <= d0 - H)]
    te = d[(ts >= d0) & (ts < d1)]
    m = fit(tr, va, feats)
    p = m.predict(te[feats], num_iteration=m.best_iteration)
    sig = np.where(p[:, 2] > CONF, 1.0, np.where(p[:, 0] > CONF, -1.0, 0.0))
    traded = sig != 0
    g = (sig * te["fwd_ret_bps"].to_numpy())[traded]
    sp = te["spread_bps"].to_numpy()[traded]
    row = {"date": D.date(), "train_rows": len(tr), "val_rows": len(va), "eval_rows": len(te),
           "eval_hours": (te.ts.max() - te.ts.min()) / 3600, "best_iter": m.best_iteration,
           "n_trades": int(traded.sum()),
           "gross_bps": g.mean() if len(g) else np.nan,
           "spread_bps": sp.mean() if len(g) else np.nan,
           "breakeven_fee_bps": (g - sp).mean() if len(g) else np.nan,
           "hit_rate": (g > 0).mean() if len(g) else np.nan}
    out.append(row)
    pd.DataFrame(out).round(4).to_csv(f"models/{sym}_walkforward.csv", index=False)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()},
          f"[{time.time() - t0:.0f}s]", flush=True)
print("DONE", flush=True)
