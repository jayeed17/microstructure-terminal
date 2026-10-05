"""
Latency sensitivity on the walk-forward: signal at t, order lands at t+L.
Entry at the book state as-of t+L, exit as-of t+L+H. Same fits as walkforward.py.

  python latency.py btcusdt
"""

import sys, time, numpy as np, pandas as pd
from collect import forward_index
from train import relabel, feature_cols, fit

sym, H, CONF, SKIP = sys.argv[1], 1, 0.6, 3
VAL_FRAC = 0.15 / 0.85
LATS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]

df = pd.read_parquet(f"data/labeled_{sym}.parquet").sort_values("ts").reset_index(drop=True)
# delayed entry/exit on the full book timeline, attached before relabel drops rows
ts_all, mid, spr = df["ts"].to_numpy(), df["mid"].to_numpy(), df["spread_bps"].to_numpy()
lat_cols = []
for L in LATS:
    if L == 0:
        e, ok_e = np.arange(len(df)), np.ones(len(df), bool)   # L=0 matches relabel exactly
    else:
        e, ok_e = forward_index(ts_all, L)
    x, ok_x = forward_index(ts_all, L + H)
    ok = ok_e & ok_x
    df[f"ret_L{L}"] = np.where(ok, (mid[x] / mid[e] - 1.0) * 1e4, np.nan)
    df[f"spr_L{L}"] = spr[e]
    lat_cols += [f"ret_L{L}", f"spr_L{L}"]

d = relabel(df, H)
feats = [c for c in feature_cols(d) if c not in lat_cols]
ts = d["ts"].to_numpy()
date = pd.to_datetime(ts, unit="s", utc=True).normalize()
days = sorted(date.unique())

rows = []
for D in days[SKIP:]:
    t0 = time.time()
    d0 = D.timestamp(); d1 = d0 + 86400
    a = d0 - (d0 - ts[0]) * VAL_FRAC
    tr = d[ts <= a - H]
    va = d[(ts >= a) & (ts <= d0 - H)]
    te = d[(ts >= d0) & (ts < d1)]
    m = fit(tr, va, feats)
    p = m.predict(te[feats], num_iteration=m.best_iteration)
    sig = np.where(p[:, 2] > CONF, 1.0, np.where(p[:, 0] > CONF, -1.0, 0.0))
    for L in LATS:
        r = te[f"ret_L{L}"].to_numpy()
        traded = (sig != 0) & ~np.isnan(r)
        g = sig[traded] * r[traded]
        sp = te[f"spr_L{L}"].to_numpy()[traded]
        rows.append({"date": D.date(), "latency_ms": int(L * 1000), "n_trades": int(traded.sum()),
                     "gross_sum": g.sum(), "breakeven_sum": (g - sp).sum()})
    print(D.date(), f"[{time.time() - t0:.0f}s]", flush=True)

daily = pd.DataFrame(rows)
daily["gross_bps"] = daily.gross_sum / daily.n_trades
daily["breakeven_fee_bps"] = daily.breakeven_sum / daily.n_trades
agg = daily.groupby("latency_ms").agg(n_trades=("n_trades", "sum"), gross_sum=("gross_sum", "sum"),
                                      breakeven_sum=("breakeven_sum", "sum"),
                                      days_positive=("gross_bps", lambda s: int((s > 0).sum())),
                                      days=("gross_bps", "size"),
                                      daily_breakeven_mean=("breakeven_fee_bps", "mean"),
                                      daily_breakeven_std=("breakeven_fee_bps", "std"))
agg["gross_bps"] = agg.gross_sum / agg.n_trades           # trade-weighted over all days
agg["breakeven_fee_bps"] = agg.breakeven_sum / agg.n_trades
agg = agg.drop(columns=["gross_sum", "breakeven_sum"]).reset_index()
agg = agg[["latency_ms", "n_trades", "gross_bps", "breakeven_fee_bps", "days_positive", "days",
           "daily_breakeven_mean", "daily_breakeven_std"]]
agg.round(4).to_csv(f"models/{sym}_latency.csv", index=False)
daily.drop(columns=["gross_sum", "breakeven_sum"]).round(4).to_csv(
    f"models/{sym}_latency_daily.csv", index=False)
print(agg.round(3).to_string(index=False))
print("DONE", flush=True)
