"""
Binance.US L2 order book collector + microstructure features.

  pip install websockets pandas pyarrow numpy

  python collect.py                 # stream & write data/raw_*.parquet
  python collect.py --symbol ethusdt
  python collect.py --label         # build data/labeled.parquet from raw chunks

No API key needed. Public market data only. Works from US IPs (binance.com does not).
"""

import argparse
import asyncio
import json
import signal
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import websockets

DATA = Path("data")
DATA.mkdir(exist_ok=True)
FLUSH_EVERY = 5000          # rows per parquet chunk
HORIZON_S = 10.0            # forward window for labels
VOL_WINDOW = 300            # ticks (~30s at 100ms) for realized vol
GAP_TOL_S = 10.0            # silence longer than this = outage, not a quiet book


# ---------------------------------------------------------------- features

def book_features(bids, asks):
    """bids/asks: [[price_str, qty_str], ...] sorted best-first."""
    b = [(float(p), float(q)) for p, q in bids]
    a = [(float(p), float(q)) for p, q in asks]
    bp, bq = b[0]
    ap, aq = a[0]
    mid = (bp + ap) / 2.0
    micro = (bp * aq + ap * bq) / (bq + aq)

    f = {
        "bid": bp, "ask": ap, "bid_qty": bq, "ask_qty": aq,
        "mid": mid,
        "microprice": micro,
        "micro_dev_bps": (micro - mid) / mid * 1e4,
        "spread_bps": (ap - bp) / mid * 1e4,
    }

    # queue imbalance at increasing depth
    for n in (1, 5, 20):
        bv = sum(q for _, q in b[:n])
        av = sum(q for _, q in a[:n])
        f[f"imb_{n}"] = (bv - av) / (bv + av) if (bv + av) else 0.0
        f[f"depth_{n}"] = bv + av

    # distance-weighted book pressure (near-touch liquidity counts more)
    bw = sum(q / (1.0 + (mid - p) / mid * 1e4) for p, q in b)
    aw = sum(q / (1.0 + (p - mid) / mid * 1e4) for p, q in a)
    f["press_w"] = (bw - aw) / (bw + aw) if (bw + aw) else 0.0

    return f, (bp, bq, ap, aq)


def ofi(prev, cur):
    """Order Flow Imbalance (Cont, Kukanov & Stoikov 2014). prev/cur = (bp,bq,ap,aq)."""
    pbp, pbq, pap, paq = prev
    bp, bq, ap, aq = cur
    e = 0.0
    if bp > pbp:
        e += bq
    elif bp == pbp:
        e += bq - pbq
    else:
        e -= pbq
    if ap < pap:
        e -= aq
    elif ap == pap:
        e -= aq - paq
    else:
        e += paq
    return e


# ---------------------------------------------------------------- collector

class Collector:
    def __init__(self, symbol):
        self.symbol = symbol.lower()
        self.url = (
            f"wss://stream.binance.us:9443/stream?streams="
            f"{self.symbol}@depth20@100ms/{self.symbol}@aggTrade"
        )
        self.rows = []
        self.trades = deque()          # (ts, signed_qty, notional)
        self.mids = deque(maxlen=VOL_WINDOW)
        self.prev_top = None
        self.ofi_run = deque(maxlen=50)
        self.running = True
        self.n = 0

    def on_trade(self, d):
        ts = d["T"] / 1000.0
        qty = float(d["q"])
        px = float(d["p"])
        # m=True -> buyer is maker -> aggressor was the seller
        sgn = -1.0 if d["m"] else 1.0
        self.trades.append((ts, sgn * qty, sgn * qty * px))

    def _trade_flow(self, now):
        while self.trades and now - self.trades[0][0] > 10.0:
            self.trades.popleft()
        out = {}
        for w in (1.0, 5.0, 10.0):
            sel = [t for t in self.trades if now - t[0] <= w]
            out[f"tflow_{int(w)}s"] = sum(t[1] for t in sel)
            out[f"tnotional_{int(w)}s"] = sum(t[2] for t in sel)
            out[f"tcount_{int(w)}s"] = len(sel)
        return out

    def on_depth(self, d):
        bids, asks = d.get("bids"), d.get("asks")
        if not bids or not asks:
            return
        now = time.time()
        f, top = book_features(bids, asks)

        f["ofi"] = ofi(self.prev_top, top) if self.prev_top else 0.0
        self.prev_top = top
        self.ofi_run.append(f["ofi"])
        f["ofi_cum50"] = float(np.sum(self.ofi_run))

        self.mids.append(f["mid"])
        if len(self.mids) > 20:
            r = np.diff(np.log(np.asarray(self.mids)))
            f["rvol_bps"] = float(r.std() * 1e4)
            f["mom_bps"] = (self.mids[-1] / self.mids[0] - 1.0) * 1e4
        else:
            f["rvol_bps"] = np.nan
            f["mom_bps"] = np.nan

        f.update(self._trade_flow(now))
        f["ts"] = now
        self.rows.append(f)
        self.n += 1

        if len(self.rows) >= FLUSH_EVERY:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        df = pd.DataFrame(self.rows)
        path = DATA / f"raw_{self.symbol}_{int(time.time())}.parquet"
        df.to_parquet(path, index=False)
        print(f"  wrote {len(df):,} rows -> {path.name}  (total {self.n:,})", flush=True)
        self.rows = []

    async def run(self):
        backoff = 1
        while self.running:
            try:
                async with websockets.connect(self.url, ping_interval=20) as ws:
                    print(f"connected: {self.symbol}  {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
                    backoff = 1
                    # rolling windows must not straddle a disconnect
                    self.mids.clear(); self.ofi_run.clear(); self.trades.clear()
                    self.prev_top = None
                    async for msg in ws:
                        if not self.running:
                            break
                        m = json.loads(msg)
                        stream, d = m.get("stream", ""), m.get("data", {})
                        if "@depth" in stream:
                            self.on_depth(d)
                        elif "aggTrade" in stream:
                            self.on_trade(d)
            except Exception as e:
                if not self.running:
                    break
                print(f"reconnecting in {backoff}s ({type(e).__name__}: {e})", flush=True)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
        self.flush()


# ---------------------------------------------------------------- labeling

def forward_index(ts, horizon, tol=GAP_TOL_S):
    """As-of index of the book state at t + horizon.

    depth20 only pushes when the book changes, so a quiet 5-7s stretch is not
    missing data -- the price simply didn't move. Use the last snapshot at or
    before t+h. A row is invalid only if its window runs past the end of the
    data, or crosses a silence longer than `tol` (a real outage / restart).
    """
    n = len(ts)
    i = np.arange(n)
    j = np.searchsorted(ts, ts + horizon, side="right") - 1
    ok = (ts + horizon) <= ts[-1]
    big = np.concatenate([[0], np.cumsum(np.diff(ts) > tol)])
    k = np.minimum(j + 1, n - 1)          # include the gap that spans t+h
    ok &= big[k] == big[i]
    return j, ok


def build_labels(symbol, horizon=HORIZON_S):
    """Forward mid return over `horizon` seconds, with a spread-aware deadband.

    A move only counts as up/down if it would have covered half the spread --
    i.e. if it was actually tradeable. Otherwise it's 'flat'. This is what stops
    the model from learning to predict noise.
    """
    files = sorted(DATA.glob(f"raw_{symbol}_*.parquet"))
    if not files:
        raise SystemExit(f"no raw chunks for {symbol} in {DATA}/")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values("ts").reset_index(drop=True)

    ts = df["ts"].to_numpy()
    mid = df["mid"].to_numpy()
    j, ok = forward_index(ts, horizon)

    fwd = np.where(ok, (mid[j] / mid - 1.0) * 1e4, np.nan)
    df["fwd_ret_bps"] = fwd
    n_gap = int((~ok).sum())
    if n_gap:
        print(f"dropped {n_gap:,} rows (window crossed an outage or ran past the end)")

    thr = np.maximum(df["spread_bps"].to_numpy() / 2.0, 0.5)
    df["label"] = np.select([fwd > thr, fwd < -thr], [1, -1], default=0)
    df.loc[~ok, "label"] = np.nan

    df = df.dropna(subset=["label", "rvol_bps"]).reset_index(drop=True)
    out = DATA / f"labeled_{symbol}.parquet"
    df.to_parquet(out, index=False)

    print(f"\n{len(df):,} labeled rows -> {out}")
    print(f"span: {(ts[-1]-ts[0])/3600:.1f} hours")
    print("\nclass balance:")
    print(df["label"].map({-1: "down", 0: "flat", 1: "up"}).value_counts(normalize=True))
    print(f"\nmedian spread: {df.spread_bps.median():.2f} bps")
    print(f"|fwd_ret| mean: {df.fwd_ret_bps.abs().mean():.2f} bps")
    print("\ntop feature correlations with fwd_ret_bps:")
    feats = [c for c in df.columns if c not in ("ts", "label", "fwd_ret_bps", "bid", "ask", "mid", "microprice")]
    print(df[feats + ["fwd_ret_bps"]].corr()["fwd_ret_bps"].drop("fwd_ret_bps")
          .abs().sort_values(ascending=False).head(10))
    return df


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="btcusdt")
    ap.add_argument("--label", action="store_true")
    ap.add_argument("--horizon", type=float, default=HORIZON_S)
    a = ap.parse_args()

    if a.label:
        build_labels(a.symbol.lower(), a.horizon)
        return

    c = Collector(a.symbol)

    def stop(*_):
        print("\nstopping, flushing...")
        c.running = False

    signal.signal(signal.SIGINT, stop)
    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, stop)
        except (ValueError, OSError):
            pass
    if hasattr(signal, "SIGBREAK"):          # Windows console close / Ctrl+Break
        signal.signal(signal.SIGBREAK, stop)
    asyncio.run(c.run())


if __name__ == "__main__":
    main()
