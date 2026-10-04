"""
Cost-aware backtest. This is the file that decides whether the project is honest.

  python backtest.py --symbol btcusdt --horizon 10 --fee-bps 10

--fee-bps is your ROUND-TRIP taker fee in basis points. Look up your actual
Binance.US tier before quoting a number in a writeup; retail taker fees there
are high enough that most short-horizon signals die on contact. That result is
worth reporting, not hiding.

Assumptions, stated plainly:
  - Enter by crossing the spread (pay half-spread in), exit the same way.
  - Hold exactly `horizon` seconds, no stops, no sizing.
  - Forward returns are rederived for the model's horizon, not read from the
    parquet (whose label column is fixed at collection time).
  - No market impact. Fine for small size; not fine if you scale it.
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from train import relabel

DATA = Path("data")
MODELS = Path("models")


def run(symbol, horizon, fee_bps, conf, train_frac=0.85):
    model = lgb.Booster(model_file=str(MODELS / f"{symbol}_h{int(horizon)}.txt"))
    feats = json.loads((MODELS / f"{symbol}_h{int(horizon)}_features.json").read_text())

    df = pd.read_parquet(DATA / f"labeled_{symbol}.parquet").sort_values("ts").reset_index(drop=True)
    # The parquet's fwd_ret_bps is frozen at build_labels' horizon (default 10s).
    # Scoring an h-second model against it measures the wrong outcome, so always
    # rederive the forward return for THIS model's horizon.
    df = relabel(df, horizon)
    ts = df["ts"].to_numpy()
    cut = ts[0] + (ts[-1] - ts[0]) * train_frac
    te = df[ts >= cut].reset_index(drop=True)          # out-of-sample only
    if len(te) < 500:
        raise SystemExit("not enough out-of-sample rows -- collect more data")

    proba = model.predict(te[feats])
    p_dn, p_up = proba[:, 0], proba[:, 2]

    sig = np.zeros(len(te))
    sig[p_up > conf] = 1.0
    sig[p_dn > conf] = -1.0

    gross = sig * te["fwd_ret_bps"].to_numpy()
    cost = np.where(sig != 0, fee_bps + te["spread_bps"].to_numpy(), 0.0)  # half-spread in + out
    net = gross - cost

    traded = sig != 0
    n = int(traded.sum())
    if n == 0:
        print(f"conf={conf}: no trades triggered")
        return None

    # non-overlapping equivalent: scale to independent bets per horizon window
    hours = (te["ts"].iloc[-1] - te["ts"].iloc[0]) / 3600
    net_t = net[traded]
    sharpe = net_t.mean() / net_t.std() * np.sqrt(3600 / horizon * hours) if net_t.std() else 0.0

    print(f"\nconf > {conf:.2f}")
    print(f"  trades              {n:,}  ({n/len(te)*100:.1f}% of ticks, {hours:.1f}h OOS)")
    print(f"  gross edge          {gross[traded].mean():+.3f} bps/trade")
    print(f"  avg cost            {cost[traded].mean():.3f} bps/trade")
    print(f"  NET edge            {net_t.mean():+.3f} bps/trade")
    print(f"  hit rate (gross>0)  {(gross[traded] > 0).mean():.3f}")
    print(f"  cum net             {net_t.sum():+,.0f} bps")
    print(f"  approx Sharpe       {sharpe:.2f}")
    return {"conf": conf, "n": n, "gross": gross[traded].mean(),
            "cost": cost[traded].mean(), "net": net_t.mean(), "sharpe": sharpe}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="btcusdt")
    ap.add_argument("--horizon", type=float, default=10.0)
    ap.add_argument("--fee-bps", type=float, default=10.0)
    a = ap.parse_args()

    rows = [r for c in (0.40, 0.50, 0.60, 0.70, 0.80)
            if (r := run(a.symbol, a.horizon, a.fee_bps, c))]

    if rows:
        out = pd.DataFrame(rows).round(3)
        print("\n=== edge vs selectivity ===")
        print(out.to_string(index=False))
        out.to_csv(MODELS / f"{a.symbol}_backtest.csv", index=False)
        best = out.loc[out.net.idxmax()]
        print(f"\nbreakeven fee: {out.gross.max():.2f} bps gross vs {a.fee_bps} bps assumed")
        if best.net <= 0:
            print("NET EDGE IS NEGATIVE AT EVERY THRESHOLD.")
            print("Report this. 'Signal exists gross, dies against costs' is the finding.")


if __name__ == "__main__":
    main()
