"""
Microstructure terminal.

  streamlit run app.py

Three tabs:
  Findings  -- results from models/*.csv and docs/*.png. No network, no data/.
  Replay    -- plays a raw chunk from data/sample/ through the committed h1 model.
  Live      -- streams the Binance.US book and scores it. Only connects while open.

Read-only. There is no order placement anywhere in this repo.
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

from collect import book_features, ofi

ROOT = Path(__file__).resolve().parent
MODELS, DOCS, SAMPLE = ROOT / "models", ROOT / "docs", ROOT / "data" / "sample"
CONF = 0.6                      # walk-forward signal threshold (walkforward.py)
SYMBOLS = {"btcusdt": "BTC", "ethusdt": "ETH"}
st.set_page_config(page_title="Microstructure Terminal", layout="wide")


# ---------------------------------------------------------------- shared

@st.cache_resource
def get_model(symbol, horizon=1):
    mp = MODELS / f"{symbol}_h{int(horizon)}.txt"
    fp = MODELS / f"{symbol}_h{int(horizon)}_features.json"
    if not (mp.exists() and fp.exists()):
        return None, None
    import lightgbm as lgb
    return lgb.Booster(model_file=str(mp)), json.loads(fp.read_text())


def show_metrics(f):
    c = st.columns(5)
    c[0].metric("mid", f"{f['mid']:,.2f}")
    c[1].metric("spread", f"{f['spread_bps']:.3f} bps")
    c[2].metric("imbalance L1", f"{f['imb_1']:+.3f}")
    c[3].metric("OFI (50 tick)", f"{f['ofi_cum50']:+,.3f}")
    c[4].metric("realized vol", f"{f['rvol_bps']:.3f} bps")


def ladder_chart(lad, height=300):
    return alt.Chart(lad).mark_bar().encode(
        x=alt.X("qty:Q", title="size"),
        y=alt.Y("level:O", sort=None, title=None),
        color=alt.Color("side:N", scale=alt.Scale(domain=["bid", "ask"],
                                                  range=["#16a34a", "#dc2626"]),
                        legend=None),
        tooltip=["side", "level", alt.Tooltip("qty:Q", format=",.4f")],
    ).properties(height=height)


def show_history(h):
    st.markdown("**mid**")
    st.altair_chart(
        alt.Chart(h).mark_line(color="#2563eb").encode(
            x=alt.X("t:T", title=None, axis=alt.Axis(format="%H:%M:%S")),
            y=alt.Y("mid:Q", scale=alt.Scale(zero=False), title=None),
        ).properties(height=150),
        width="stretch",
    )
    st.markdown("**flow**")
    m = h.melt(id_vars="t", value_vars=["imb_5", "press_w"], var_name="feature", value_name="v")
    st.altair_chart(
        alt.Chart(m).mark_line().encode(
            x=alt.X("t:T", title=None, axis=alt.Axis(format="%H:%M:%S")),
            y=alt.Y("v:Q", title=None),
            color=alt.Color("feature:N", legend=alt.Legend(orient="bottom")),
        ).properties(height=150),
        width="stretch",
    )


def show_model(probs, symbol, horizon=1):
    st.markdown(f"**model · {horizon}s ahead**")
    if probs is None:
        st.warning(f"No model at models/{symbol}_h{horizon}.txt — "
                   f"run: python train.py --symbol {symbol} --horizon {horizon}")
        return
    p_dn, p_flat, p_up = probs
    m = st.columns(4)
    m[0].metric("P(down)", f"{p_dn:.1%}")
    m[1].metric("P(flat)", f"{p_flat:.1%}")
    m[2].metric("P(up)", f"{p_up:.1%}")
    sig = "UP" if p_up > CONF else "DOWN" if p_dn > CONF else "—"
    m[3].metric(f"signal (conf {CONF})", sig)
    st.progress(float(p_up), text=f"directional lean  {p_up - p_dn:+.1%}")
    st.caption("Probabilities are gross of fees and spread. See the Findings tab "
               "for what the signal is worth net of costs.")


def show_features(f, feats):
    if not feats:
        return
    st.markdown("**feature panel**")
    rows = [{"feature": k, "value": f.get(k, 0.0)} for k in feats]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 height=min(35 * len(rows) + 38, 400),
                 column_config={"value": st.column_config.NumberColumn(format="%.4f")})


# ---------------------------------------------------------------- findings

def findings():
    st.markdown(
        "### Order flow imbalance predicts 1–2s crypto price moves — "
        "but a retail participant can't capture it\n"
        "The signal is real and replicates across BTC and ETH and across every "
        "walk-forward day. It is also roughly an order of magnitude too small to clear "
        "retail taker fees, and most of its breakeven margin is gone within 250ms of "
        "latency. Data: Binance.US L2 book + trades, 21 Sep – 4 Oct 2026."
    )

    st.subheader("1. Signal decay by horizon")
    try:
        dec = pd.DataFrame({
            name: pd.read_csv(MODELS / f"{sym}_sweep.csv").set_index("horizon_s")["gross_edge_bps"]
            for sym, name in SYMBOLS.items()
        }).T.round(3)
        dec.columns = [f"{h}s" for h in dec.columns]
        dec.index.name = "gross edge (bps)"
        st.dataframe(dec, width="stretch")
    except Exception as e:
        st.warning(f"Decay table unavailable: {e}")
    show_png("decay.png", "Gross edge vs horizon")

    st.subheader("2. Walk-forward robustness (horizon 1s, confidence 0.6)")
    try:
        st.dataframe(walkforward_summary(), width="stretch")
        st.caption("Expanding-window walk-forward, retrained daily, scored on the held-out "
                   "next day. Aggregates are trade-weighted across days.")
    except Exception as e:
        st.warning(f"Walk-forward table unavailable: {e}")
    show_png("daily.png", "Daily walk-forward breakeven fee")

    st.subheader("3. Latency sensitivity (BTC)")
    try:
        st.dataframe(latency_table(), hide_index=True, width="stretch")
        st.caption("Signal at t, order arrives at t+L. The 50ms row is an artifact: with a "
                   "193ms median snapshot gap, sub-100ms entry almost always resolves to the "
                   "L=0 snapshot.")
    except Exception as e:
        st.warning(f"Latency table unavailable: {e}")
    show_png("latency.png", "Breakeven fee vs latency")

    st.subheader("Conclusion")
    st.markdown(
        "Order book imbalance predicts BTC and ETH mid-price direction at 1–2 second "
        "horizons with a hit rate near 0.88 and gross edge near 1 bps, positive on 11 of 11 "
        "walk-forward days on both assets. It is not capturable by a retail participant:\n\n"
        "1. **Fees.** Breakeven is ~0.9–1.0 bps round-trip against retail taker fees an "
        "order of magnitude larger.\n"
        "2. **Latency.** Most of the breakeven margin is consumed within 250ms, before a "
        "retail order could reach the venue.\n\n"
        "The edge clears costs only near zero fees and well under 100ms latency — a "
        "colocated market maker, not someone on home internet."
    )


def show_png(name, caption):
    p = DOCS / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.warning(f"docs/{name} missing — run: python docs/charts.py")


@st.cache_data
def walkforward_summary():
    cols = {}
    for sym, name in SYMBOLS.items():
        d = pd.read_csv(MODELS / f"{sym}_walkforward.csv")
        w = d["n_trades"]
        wm = lambda c: (d[c] * w).sum() / w.sum()
        be = d["breakeven_fee_bps"]
        cols[name] = {
            "Trades": f"{w.sum():,}",
            "Gross edge": f"{wm('gross_bps'):.3f} bps",
            "Spread paid": f"{wm('spread_bps'):.3f} bps",
            "Breakeven fee": f"{wm('breakeven_fee_bps'):.3f} bps",
            "Hit rate": f"{wm('hit_rate'):.3f}",
            "Days with positive edge": f"{(d['gross_bps'] > 0).sum()}/{len(d)}",
            "Daily breakeven (mean ± sd)": f"{be.mean():.3f} ± {be.std():.3f}",
        }
    return pd.DataFrame(cols)


@st.cache_data
def latency_table():
    lat = pd.read_csv(MODELS / "btcusdt_latency.csv")
    daily = pd.read_csv(MODELS / "btcusdt_latency_daily.csv")
    pos = daily.groupby("latency_ms")["breakeven_fee_bps"].agg(lambda s: f"{(s > 0).sum()}/{len(s)}")
    return pd.DataFrame({
        "L (ms)": lat["latency_ms"],
        "gross (bps)": lat["gross_bps"].round(3),
        "breakeven (bps)": lat["breakeven_fee_bps"].round(3),
        "days with positive breakeven": lat["latency_ms"].map(pos),
    })


# ---------------------------------------------------------------- replay

@st.cache_data
def sample_files(symbol):
    """{label: path}. File names carry the flush (end) time, so label by first row."""
    out = {}
    for p in sorted(SAMPLE.glob(f"raw_{symbol}_*.parquet")):
        ts = pd.read_parquet(p, columns=["ts"])["ts"]
        t0, t1 = pd.to_datetime(ts.min(), unit="s"), pd.to_datetime(ts.max(), unit="s")
        out[f"{t0:%Y-%m-%d %H:%M}–{t1:%H:%M} UTC"] = str(p)
    return out


@st.cache_data(max_entries=4)
def load_replay(path, symbol):
    d = pd.read_parquet(path).sort_values("ts").reset_index(drop=True)
    model, feats = get_model(symbol)
    if model is not None:
        p = model.predict(d[feats])
        d["p_dn"], d["p_flat"], d["p_up"] = p[:, 0], p[:, 1], p[:, 2]
    return d


def depth_ladder(f):
    """Cumulative bid/ask size at L1/L5/L20, rebuilt from depth_n and imb_n.
    Raw chunks store aggregates, not per-level prices, so this is the most the
    sample can show."""
    rows = []
    for n in (20, 5, 1):
        dep, imb = f[f"depth_{n}"], f[f"imb_{n}"]
        rows.append({"level": f"ask ≤L{n}", "qty": dep * (1 - imb) / 2, "side": "ask"})
    for n in (1, 5, 20):
        dep, imb = f[f"depth_{n}"], f[f"imb_{n}"]
        rows.append({"level": f"bid ≤L{n}", "qty": dep * (1 + imb) / 2, "side": "bid"})
    return pd.DataFrame(rows)


def replay():
    c = st.columns([1, 2, 2])
    symbol = c[0].selectbox("symbol", list(SYMBOLS), key="rp_symbol")
    files = sample_files(symbol)
    if not files:
        st.error("No sample chunks found in data/sample/.")
        return
    label = c[1].selectbox("chunk", list(files), key="rp_file")
    speed = c[2].select_slider("speed", [1, 2, 5, 10, 25, 50, 100], value=5,
                               format_func=lambda x: f"{x}×", key="rp_speed")

    path = Path(files[label])
    try:
        d = load_replay(str(path), symbol)
    except Exception as e:
        st.error(f"Could not load {path.name}: {e}")
        return
    model, feats = get_model(symbol)

    ss = st.session_state
    if ss.get("rp_loaded") != str(path):
        ss.rp_loaded, ss.rp_t, ss.rp_wall = str(path), float(d["ts"].iloc[0]), None

    b = st.columns([1, 1, 6])
    playing = b[0].toggle("play", value=True, key="rp_play")
    if b[1].button("restart", key="rp_restart"):
        ss.rp_t, ss.rp_wall = float(d["ts"].iloc[0]), None

    st.caption(
        "Replays a raw collector chunk (2026-10-03 UTC) at the chosen speed through the "
        "committed 1s model. Assuming that model was trained on the current labeled files, "
        "this day falls in train.py's held-out test split (last 15%), so these are "
        "out-of-sample predictions — but they are a demo, not the evaluation; the "
        "walk-forward in Findings is."
    )

    @st.fragment(run_every=0.5 if playing else None)
    def tick():
        ts = d["ts"].to_numpy()
        now = time.monotonic()
        if playing and ss.rp_wall is not None:
            ss.rp_t += (now - ss.rp_wall) * speed
        ss.rp_wall = now if playing else None
        i = int(np.searchsorted(ts, ss.rp_t, side="right")) - 1
        i = min(max(i, 0), len(d) - 1)
        if ss.rp_t >= ts[-1]:
            st.info("End of chunk — press restart or pick another chunk.")
        f = d.iloc[i].to_dict()

        clock = pd.to_datetime(f["ts"], unit="s").strftime("%H:%M:%S.%f")[:-3]
        st.markdown(f"**replay clock** {clock} UTC · row {i + 1:,}/{len(d):,} · {speed}×")
        show_metrics(f)
        left, right = st.columns([1, 2])
        with left:
            st.markdown("**depth ladder** (cumulative size)")
            st.altair_chart(ladder_chart(depth_ladder(f)), width="stretch")
            st.caption("Raw chunks store per-snapshot aggregates, not per-level prices, "
                       "so the ladder shows cumulative bid/ask size to L1/L5/L20.")
        with right:
            h = d.iloc[max(0, i - 600):i + 1].copy()
            h["t"] = pd.to_datetime(h["ts"], unit="s")
            show_history(h)
        probs = (f["p_dn"], f["p_flat"], f["p_up"]) if model is not None else None
        show_model(probs, symbol)
        show_features(f, feats)

    tick()


# ---------------------------------------------------------------- live

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
        self.started = time.time()
        self.attempts = 0
        self.error = None

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
        import websockets
        backoff = 1
        while True:
            self.attempts += 1
            try:
                async with websockets.connect(self.url(), ping_interval=20,
                                              open_timeout=10) as ws:
                    self.connected, self.error = True, None
                    backoff = 1
                    async for msg in ws:
                        m = json.loads(msg)
                        stream, d = m.get("stream", ""), m.get("data", {})
                        if "@depth" in stream:
                            self.on_depth(d)
                        elif "aggTrade" in stream:
                            self.on_trade(d)
            except Exception as e:
                self.connected = False
                self.error = f"{type(e).__name__}: {e}"
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    def start(self):
        threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True).start()
        return self


@st.cache_resource
def get_state(symbol):
    return LiveState(symbol).start()


UNREACHABLE = "Binance.US is not reachable from this host — see the Replay tab."


def live():
    symbol = st.selectbox("symbol", ["btcusdt", "ethusdt", "solusdt", "btcusd"], key="lv_symbol")
    st.caption("Binance.US public market data · read-only · no trading. "
               "Streams only while this tab is open.")
    try:
        state = get_state(symbol)
    except Exception as e:
        st.error(UNREACHABLE)
        st.caption(f"{type(e).__name__}: {e}")
        return
    model, feats = get_model(symbol)

    @st.fragment(run_every=0.5)
    def tick():
        try:
            with state.lock:
                f = dict(state.feat) if state.feat else None
                book = state.book
                hist = list(state.hist)

            if not f:
                waited = time.time() - state.started
                if state.error or waited > 15:
                    st.error(UNREACHABLE)
                    st.caption(f"Last error: {state.error or 'no data after 15s'} · "
                               f"attempts: {state.attempts} · retrying in the background")
                else:
                    st.info("connecting to binance.us…")
                return
            if not state.connected:
                st.warning(f"Connection lost — showing last data, reconnecting. "
                           f"({state.error})")

            show_metrics(f)
            left, right = st.columns([1, 2])
            with left:
                st.markdown("**book**")
                bids, asks = book
                lad = pd.DataFrame(
                    [{"level": f"{float(p):,.2f}", "qty": float(q), "side": "ask"}
                     for p, q in reversed(asks)]
                    + [{"level": f"{float(p):,.2f}", "qty": float(q), "side": "bid"}
                       for p, q in bids])
                st.altair_chart(ladder_chart(lad, height=430), width="stretch")
            with right:
                h = pd.DataFrame(hist)
                h["t"] = pd.to_datetime(h["ts"], unit="s")
                show_history(h)
            probs = None
            if model is not None:
                row = pd.DataFrame([{k: f.get(k, 0.0) for k in feats}])[feats]
                probs = tuple(model.predict(row)[0])
            show_model(probs, symbol)
            show_features(f, feats)
        except Exception as e:
            st.error(f"Live view error: {type(e).__name__}: {e}")

    tick()


# ---------------------------------------------------------------- page

st.title("Microstructure Terminal")
t_find, t_replay, t_live = st.tabs(["Findings", "Replay", "Live"], default="Findings",
                                   key="tab", on_change="rerun")
with t_find:
    if t_find.open is not False:
        findings()
with t_replay:
    if t_replay.open:
        replay()
with t_live:
    if t_live.open:
        live()
