"""
Live microstructure terminal.

  streamlit run app.py

Streams the Binance.US book in a background thread, computes the same features
collect.py writes to disk, and -- if a trained model exists in models/ -- scores
the live book. Read-only. There is no order placement anywhere in this repo.
"""

import asyncio
import json
import threading
import time
from collections import deque
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import websockets

from collect import book_features, ofi

MODELS = Path("models")
st.set_page_config(page_title="Microstructure Terminal", layout="wide")


class LiveState:
    def __init__(self, symbol):
        self.symbol = symbol
        self.lock = threading.Lock()
        self.book = None            # (bids, asks)
        self.feat = None
        self.hist = deque(maxlen=600)
        self.trades = deque()
        self.mids = deque(maxlen=300)
        self.prev_top = None
        self.ofi_run = deque(maxlen=50)
        self.connected = False

    def url(self):
        s = self.symbol.lower()
        return (f"wss://stream.binance.us:9443/stream?streams="
                f"{s}@depth20@100ms/{s}@aggTrade")

    def _flow(self, now):
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
            f["rvol_bps"] = f["mom_bps"] = 0.0
        f.update(self._flow(now))
        f["ts"] = now
        with self.lock:
            self.book = (bids[:15], asks[:15])
            self.feat = f
            self.hist.append(f)

    def on_trade(self, d):
        sgn = -1.0 if d["m"] else 1.0
        q, p = float(d["q"]), float(d["p"])
        self.trades.append((d["T"] / 1000.0, sgn * q, sgn * q * p))

    async def _run(self):
        backoff = 1
        while True:
            try:
                async with websockets.connect(self.url(), ping_interval=20) as ws:
                    self.connected = True
                    backoff = 1
                    async for msg in ws:
                        m = json.loads(msg)
                        stream, d = m.get("stream", ""), m.get("data", {})
                        if "@depth" in stream:
                            self.on_depth(d)
                        elif "aggTrade" in stream:
                            self.on_trade(d)
            except Exception:
                self.connected = False
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    def start(self):
        threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True).start()
        return self


@st.cache_resource
def get_state(symbol):
    return LiveState(symbol).start()


@st.cache_resource
def get_model(symbol, horizon):
    import lightgbm as lgb
    mp = MODELS / f"{symbol}_h{int(horizon)}.txt"
    fp = MODELS / f"{symbol}_h{int(horizon)}_features.json"
    if not (mp.exists() and fp.exists()):
        return None, None
    return lgb.Booster(model_file=str(mp)), json.loads(fp.read_text())


# ---------------------------------------------------------------- ui

with st.sidebar:
    st.header("terminal")
    symbol = st.selectbox("symbol", ["btcusdt", "ethusdt", "solusdt", "btcusd"])
    horizon = st.selectbox("model horizon (s)", [1, 2, 5, 10, 30, 60], index=3)
    st.caption("binance.us public market data · read-only · no trading")

state = get_state(symbol)
model, feats = get_model(symbol, horizon)


@st.fragment(run_every=0.5)
def render():
    with state.lock:
        f = dict(state.feat) if state.feat else None
        book = state.book
        hist = list(state.hist)

    if not f:
        st.info("connecting to binance.us…")
        return

    c = st.columns(5)
    c[0].metric("mid", f"{f['mid']:,.2f}")
    c[1].metric("spread", f"{f['spread_bps']:.2f} bps")
    c[2].metric("imbalance L1", f"{f['imb_1']:+.3f}")
    c[3].metric("OFI (50 tick)", f"{f['ofi_cum50']:+,.2f}")
    c[4].metric("realized vol", f"{f['rvol_bps']:.2f} bps")

    left, right = st.columns([1, 2])

    with left:
        st.subheader("book")
        bids, asks = book
        lad = pd.concat([
            pd.DataFrame({"px": [float(p) for p, _ in asks],
                          "qty": [float(q) for _, q in asks], "side": "ask"}),
            pd.DataFrame({"px": [float(p) for p, _ in bids],
                          "qty": [float(q) for _, q in bids], "side": "bid"}),
        ])
        st.altair_chart(
            alt.Chart(lad).mark_bar().encode(
                x=alt.X("qty:Q", title="size"),
                y=alt.Y("px:O", sort="descending", title=None,
                        axis=alt.Axis(format=",.2f")),
                color=alt.Color("side:N",
                                scale=alt.Scale(domain=["bid", "ask"],
                                                range=["#16a34a", "#dc2626"]),
                                legend=None),
            ).properties(height=430),
            use_container_width=True,
        )

    with right:
        h = pd.DataFrame(hist)
        h["t"] = pd.to_datetime(h["ts"], unit="s")

        st.subheader("mid")
        st.altair_chart(
            alt.Chart(h).mark_line(color="#2563eb").encode(
                x=alt.X("t:T", title=None),
                y=alt.Y("mid:Q", scale=alt.Scale(zero=False), title=None),
            ).properties(height=170),
            use_container_width=True,
        )

        st.subheader("flow")
        m = h.melt(id_vars="t", value_vars=["imb_5", "press_w"],
                   var_name="feature", value_name="v")
        st.altair_chart(
            alt.Chart(m).mark_line().encode(
                x=alt.X("t:T", title=None),
                y=alt.Y("v:Q", title=None),
                color=alt.Color("feature:N", legend=alt.Legend(orient="bottom")),
            ).properties(height=170),
            use_container_width=True,
        )

    st.subheader(f"model · {horizon}s ahead")
    if model is None:
        st.warning(f"no model at models/{symbol}_h{horizon}.txt — "
                   f"run: python train.py --symbol {symbol} --horizon {horizon}")
        return

    row = pd.DataFrame([{k: f.get(k, 0.0) for k in feats}])[feats]
    p_dn, p_flat, p_up = model.predict(row)[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("P(down)", f"{p_dn:.1%}")
    m2.metric("P(flat)", f"{p_flat:.1%}")
    m3.metric("P(up)", f"{p_up:.1%}")
    st.progress(float(p_up), text=f"directional lean  {p_up - p_dn:+.1%}")
    st.caption(
        "Probabilities are gross of fees and spread. Check backtest.py for the "
        "net number before reading anything into these."
    )


render()
