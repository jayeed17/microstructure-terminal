"""
Microstructure terminal.

  streamlit run app.py

Three tabs:
  Findings  -- results from models/*.csv and docs/*.png. No network, no data/.
  Replay    -- plays a raw chunk from data/sample/ through the committed h1 model.
  Live      -- streams the Binance.US book and scores it. Stops 30s after the last viewer leaves.

Read-only. There is no order placement anywhere in this repo.
"""

import asyncio
import html
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
LIVE_IDLE_S = 30                # live stream closes this long after the last viewer tick
FONT = '"Times New Roman", Times, Georgia, sans-serif'
st.set_page_config(page_title="Microstructure Terminal", layout="wide")

# Colors are left to the Streamlit theme so light and dark both work. Only neutral
# rgba greys are set here.
CSS = f"""
<style>
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp button,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricLabel"], .stApp [data-testid="stCaptionContainer"],
.stApp [role="tab"], .stApp [role="option"] {{
  font-family: {FONT} !important;
}}
.stApp code, .stApp pre {{ font-family: ui-monospace, Menlo, Consolas, monospace !important; }}
.stApp {{ font-variant-numeric: lining-nums tabular-nums; }}
[data-testid="stMainBlockContainer"] {{ max-width: 1120px; padding-top: 3rem; }}
.stApp h1 {{ font-size: 2.1rem; font-weight: 600; letter-spacing: -0.01em; padding-bottom: 0; }}
.stApp h2 {{
  font-size: 1.3rem; font-weight: 600; margin-top: 2.75rem; padding-top: 1.25rem;
  border-top: 1px solid rgba(128, 128, 128, 0.35);
}}
.stApp p, .stApp li {{ line-height: 1.6; }}
.lead {{ font-size: 1.2rem; line-height: 1.55; max-width: 46rem; margin-bottom: 0.9rem; }}
.lead strong {{ font-weight: 600; }}
.prose {{ max-width: 46rem; }}
.lbl {{
  font-size: 0.8rem; letter-spacing: 0.08em; text-transform: uppercase;
  opacity: 0.7; margin: 0.25rem 0 0.35rem;
}}
[data-testid="stMetricValue"] {{ font-size: 1.9rem; }}
.tbl {{ overflow-x: auto; margin: 0.5rem 0 1rem; }}
.tbl table {{ border-collapse: collapse; width: auto; min-width: 50%; font-size: 0.95rem; border: none; }}
.tbl th, .tbl td {{
  padding: 0.35rem 1.1rem 0.35rem 0.6rem; border: none;
  border-bottom: 1px solid rgba(128, 128, 128, 0.25); white-space: nowrap;
}}
.tbl thead th {{ font-weight: 600; border-bottom: 1px solid rgba(128, 128, 128, 0.6); }}
.tbl .n {{ text-align: right; }}
.tbl .l {{ text-align: left; }}
.stApp [data-testid="stImage"] img {{ border-radius: 4px; max-width: 46rem; }}
@media (max-width: 640px) {{
  [data-testid="stMainBlockContainer"] {{ padding-top: 3.5rem; }}
  .stApp h1 {{ font-size: 1.7rem; }}
  .lead {{ font-size: 1.08rem; }}
  .tbl table {{ font-size: 0.82rem; }}
  .tbl th, .tbl td {{ padding: 0.3rem 0.4rem; white-space: normal; }}
  [data-testid="stMetricValue"] {{ font-size: 1.4rem; }}
}}
</style>
"""


# ---------------------------------------------------------------- shared

@st.cache_resource(show_spinner="Loading model…")
def get_model(symbol, horizon=1):
    mp = MODELS / f"{symbol}_h{int(horizon)}.txt"
    fp = MODELS / f"{symbol}_h{int(horizon)}_features.json"
    if not (mp.exists() and fp.exists()):
        return None, None
    import lightgbm as lgb
    return lgb.Booster(model_file=str(mp)), json.loads(fp.read_text())


def label(text):
    st.markdown(f'<div class="lbl">{html.escape(text)}</div>', unsafe_allow_html=True)


def prose(md):
    st.markdown(f'<div class="prose">\n\n{md}\n\n</div>', unsafe_allow_html=True)


def table(df, label_cols=1):
    """Static HTML table. The first `label_cols` columns are left-aligned text,
    the rest right-aligned numbers in tabular figures."""
    cls = lambda i: "l" if i < label_cols else "n"
    head = "".join(f'<th class="{cls(i)}">{html.escape(str(c))}</th>' for i, c in enumerate(df.columns))
    body = "".join(
        "<tr>" + "".join(f'<td class="{cls(i)}">{html.escape(str(v))}</td>' for i, v in enumerate(row)) + "</tr>"
        for row in df.itertuples(index=False))
    st.markdown(f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>',
                unsafe_allow_html=True)


def chart(c):
    return (c.configure_axis(labelFont=FONT, titleFont=FONT, labelFontSize=12)
             .configure_legend(labelFont=FONT, titleFont=FONT, labelFontSize=12))


def show_metrics(f):
    c = st.columns(5)
    c[0].metric("mid", f"{f['mid']:,.2f}")
    c[1].metric("spread", f"{f['spread_bps']:.3f} bps")
    c[2].metric("imbalance L1", f"{f['imb_1']:+.3f}")
    c[3].metric("OFI (50 tick)", f"{f['ofi_cum50']:+,.3f}")
    c[4].metric("realized vol", f"{f['rvol_bps']:.3f} bps")


def ladder_chart(lad, height=300):
    return chart(alt.Chart(lad).mark_bar().encode(
        x=alt.X("qty:Q", title="size"),
        y=alt.Y("level:O", sort=None, title=None),
        color=alt.Color("side:N", scale=alt.Scale(domain=["bid", "ask"],
                                                  range=["#16a34a", "#dc2626"]),
                        legend=None),
        tooltip=["side", "level", alt.Tooltip("qty:Q", format=",.4f")],
    ).properties(height=height))


def show_history(h):
    label("mid")
    st.altair_chart(chart(
        alt.Chart(h).mark_line(color="#2563eb").encode(
            x=alt.X("t:T", title=None, axis=alt.Axis(format="%H:%M:%S")),
            y=alt.Y("mid:Q", scale=alt.Scale(zero=False), title=None),
        ).properties(height=150)),
        width="stretch",
    )
    label("flow")
    m = h.melt(id_vars="t", value_vars=["imb_5", "press_w"], var_name="feature", value_name="v")
    st.altair_chart(chart(
        alt.Chart(m).mark_line().encode(
            x=alt.X("t:T", title=None, axis=alt.Axis(format="%H:%M:%S")),
            y=alt.Y("v:Q", title=None),
            color=alt.Color("feature:N", legend=alt.Legend(orient="bottom")),
        ).properties(height=150)),
        width="stretch",
    )


def show_model(probs, symbol, horizon=1):
    label(f"model, {horizon}s ahead")
    if probs is None:
        st.warning(f"No model at models/{symbol}_h{horizon}.txt. "
                   f"Run python train.py --symbol {symbol} --horizon {horizon}.")
        return
    p_dn, p_flat, p_up = probs
    m = st.columns(4)
    m[0].metric("P(down)", f"{p_dn:.1%}")
    m[1].metric("P(flat)", f"{p_flat:.1%}")
    m[2].metric("P(up)", f"{p_up:.1%}")
    sig = "up" if p_up > CONF else "down" if p_dn > CONF else "none"
    m[3].metric(f"signal (conf {CONF})", sig)
    st.progress(float(p_up), text=f"directional lean {p_up - p_dn:+.1%}")
    st.caption("Probabilities are gross of fees and spread. The Findings tab shows "
               "what the signal is worth after costs.")


def show_features(f, feats):
    if not feats:
        return
    label("feature panel")
    n = -(-len(feats) // 3)
    for col, part in zip(st.columns(3), (feats[i:i + n] for i in range(0, len(feats), n))):
        with col:
            table(pd.DataFrame({"feature": part, "value": [f"{f.get(k, 0.0):.4f}" for k in part]}))


# ---------------------------------------------------------------- findings

def findings():
    st.markdown(
        '<div class="lead"><strong>Order flow imbalance predicts crypto price moves '
        "1 to 2 seconds ahead. A retail participant can't capture it.</strong></div>",
        unsafe_allow_html=True)
    prose(
        "The signal is real. It replicates across BTC and ETH and on every walk-forward "
        "day. It is also roughly an order of magnitude too small to clear retail taker "
        "fees, and most of its breakeven margin is gone within 250ms of latency.\n\n"
        "Data is the Binance.US L2 book and trades, 21 Sep to 4 Oct 2026."
    )

    st.markdown("## 1. Signal decay by horizon")
    try:
        dec = pd.DataFrame({
            name: pd.read_csv(MODELS / f"{sym}_sweep.csv").set_index("horizon_s")["gross_edge_bps"]
            for sym, name in SYMBOLS.items()
        }).T
        out = pd.DataFrame({"gross edge (bps)": dec.index})
        for h in dec.columns:
            out[f"{h}s"] = [f"{v:.3f}" for v in dec[h]]
        table(out)
    except Exception as e:
        st.warning(f"Decay table unavailable. {e}")
    prose("Edge is concentrated at 1 to 2 seconds and decays monotonically on BTC. ETH "
          "flattens near 0.3 bps beyond 10s. Overlapping windows inflate significance at "
          "long horizons, so that difference is not something to build on.")
    show_png("decay.png", "Gross edge vs horizon")

    st.markdown("## 2. Walk-forward robustness (horizon 1s, confidence 0.6)")
    try:
        table(walkforward_summary())
        st.caption("Expanding-window walk-forward, retrained daily, scored on the held-out "
                   "next day. Aggregates are trade-weighted across days.")
    except Exception as e:
        st.warning(f"Walk-forward table unavailable. {e}")
    prose("The edge is positive every day on both assets, with no significant trend. ETH's "
          "median spread is 0.333 bps but spread paid was 0.074 bps, so the model trades "
          "disproportionately when spreads are tight. That is a selection effect, not a "
          "cost saving. Neither symbol's breakeven comes close to a realistic taker fee.")
    show_png("daily.png", "Daily walk-forward breakeven fee")

    st.markdown("## 3. Latency sensitivity (BTC)")
    try:
        table(latency_table(), label_cols=0)
    except Exception as e:
        st.warning(f"Latency table unavailable. {e}")
    prose("The signal fires at t and the order arrives at t+L. The 50ms row is an artifact. "
          "The median snapshot gap is 193ms, so sub-100ms entry almost always resolves to "
          "the L=0 snapshot.")
    show_png("latency.png", "Breakeven fee vs latency")

    st.markdown("## Conclusion")
    prose(
        "Order book imbalance predicts BTC and ETH mid-price direction 1 to 2 seconds "
        "ahead. The hit rate is near 0.88 and gross edge is near 1 bps. It was positive "
        "on 11 of 11 walk-forward days on both assets. A retail participant can't "
        "capture it, for two reasons.\n\n"
        "1. **Fees.** Breakeven is about 0.9 to 1.0 bps round-trip. Retail taker fees "
        "are an order of magnitude larger.\n"
        "2. **Latency.** Most of the breakeven margin is gone within 250ms, before a "
        "retail order could reach the venue.\n\n"
        "The edge clears costs only near zero fees and well under 100ms latency. That is "
        "a colocated market maker, not someone on home internet."
    )


def show_png(name, caption):
    p = DOCS / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.warning(f"docs/{name} is missing. Run python docs/charts.py.")


@st.cache_data
def walkforward_summary():
    rows = {}
    for sym, name in SYMBOLS.items():
        d = pd.read_csv(MODELS / f"{sym}_walkforward.csv")
        w = d["n_trades"]
        wm = lambda c: (d[c] * w).sum() / w.sum()
        be = d["breakeven_fee_bps"]
        rows[name] = [
            f"{w.sum():,}",
            f"{wm('gross_bps'):.3f} bps",
            f"{wm('spread_bps'):.3f} bps",
            f"{wm('breakeven_fee_bps'):.3f} bps",
            f"{wm('hit_rate'):.3f}",
            f"{(d['gross_bps'] > 0).sum()}/{len(d)}",
            f"{be.mean():.3f} ± {be.std():.3f}",
        ]
    out = pd.DataFrame({"": ["Trades", "Gross edge", "Spread paid", "Breakeven fee", "Hit rate",
                             "Days with positive edge", "Daily breakeven (mean ± sd)"]})
    for name, vals in rows.items():
        out[name] = vals
    return out


@st.cache_data
def latency_table():
    lat = pd.read_csv(MODELS / "btcusdt_latency.csv")
    daily = pd.read_csv(MODELS / "btcusdt_latency_daily.csv")
    pos = daily.groupby("latency_ms")["breakeven_fee_bps"].agg(lambda s: f"{(s > 0).sum()}/{len(s)}")
    return pd.DataFrame({
        "L (ms)": lat["latency_ms"],
        "gross (bps)": lat["gross_bps"].map("{:.3f}".format),
        "breakeven (bps)": lat["breakeven_fee_bps"].map("{:.3f}".format),
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
        out[f"{t0:%Y-%m-%d %H:%M} to {t1:%H:%M} UTC"] = str(p)
    return out


@st.cache_data(max_entries=4, show_spinner="Loading chunk…")
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
    ss = st.session_state
    ss.setdefault("rp_speed", 5)
    ss.setdefault("rp_play", True)
    c = st.columns([1, 2, 2])
    symbol = c[0].selectbox("symbol", list(SYMBOLS), key="rp_symbol",
                            on_change=lambda: ss.pop("rp_file", None))
    files = sample_files(symbol)
    if not files:
        st.error("No sample chunks found in data/sample/.")
        return
    chunk = c[1].selectbox("chunk", list(files), key="rp_file")
    speed = c[2].select_slider("speed", [1, 2, 5, 10, 25, 50, 100],
                               format_func=lambda x: f"{x}×", key="rp_speed")

    path = Path(files[chunk])
    try:
        d = load_replay(str(path), symbol)
    except Exception as e:
        st.error(f"Could not load {path.name}. {e}")
        return
    model, feats = get_model(symbol)

    if ss.get("rp_loaded") != str(path):
        ss.rp_loaded, ss.rp_t, ss.rp_wall = str(path), float(d["ts"].iloc[0]), None

    b = st.columns([1, 1, 6])
    playing = b[0].toggle("play", key="rp_play")
    if b[1].button("restart", key="rp_restart"):
        ss.rp_t, ss.rp_wall = float(d["ts"].iloc[0]), None

    st.caption(
        "Plays one raw collector chunk from 2026-10-03 UTC through the committed 1s model "
        "at the chosen speed. If that model was trained on the current labeled files, this "
        "day is in train.py's held-out test split (the last 15%), so these predictions are "
        "out of sample. This is a demo. The evaluation is the walk-forward in Findings."
    )

    @st.fragment(run_every=0.5 if playing else None)
    def tick():
        ts = d["ts"].to_numpy()
        now = time.monotonic()
        if playing and ss.rp_wall is not None:
            # Cap the step so time spent on another tab doesn't fast-forward the replay.
            ss.rp_t += min(now - ss.rp_wall, 1.0) * speed
        ss.rp_wall = now if playing else None
        i = int(np.searchsorted(ts, ss.rp_t, side="right")) - 1
        i = min(max(i, 0), len(d) - 1)
        if ss.rp_t >= ts[-1]:
            st.info("End of chunk. Press restart or pick another chunk.")
        f = d.iloc[i].to_dict()

        clock = pd.to_datetime(f["ts"], unit="s").strftime("%H:%M:%S.%f")[:-3]
        st.markdown(f"**replay clock** {clock} UTC · row {i + 1:,}/{len(d):,} · {speed}×")
        show_metrics(f)
        left, right = st.columns([1, 2])
        with left:
            label("depth ladder (cumulative size)")
            st.altair_chart(ladder_chart(depth_ladder(f)), width="stretch")
            st.caption("Raw chunks store per-snapshot aggregates, not per-level prices. "
                       "The ladder shows cumulative bid and ask size to L1, L5 and L20.")
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
        self.alive = True
        self.started = self.last_seen = time.time()
        self.attempts = 0
        self.error = None

    def url(self):
        s = self.symbol.lower()
        return (f"wss://stream.binance.us:9443/stream?streams="
                f"{s}@depth20@100ms/{s}@aggTrade")

    def idle(self):
        return time.time() - self.last_seen > LIVE_IDLE_S

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
        while not self.idle():
            self.attempts += 1
            try:
                async with websockets.connect(self.url(), ping_interval=20,
                                              open_timeout=10) as ws:
                    self.connected, self.error = True, None
                    backoff = 1
                    async for msg in ws:
                        if self.idle():
                            break
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
        self.connected = self.alive = False

    def start(self):
        threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True).start()
        return self


@st.cache_resource
def live_registry():
    return {"lock": threading.Lock(), "states": {}}


def get_state(symbol):
    """One stream per symbol, shared by all viewers. A stream nobody has polled for
    LIVE_IDLE_S closes itself; the next viewer starts a fresh one."""
    reg = live_registry()
    with reg["lock"]:
        s = reg["states"].get(symbol)
        if s is None or not s.alive:
            s = reg["states"][symbol] = LiveState(symbol).start()
        s.last_seen = time.time()
    return s


UNREACHABLE = "Binance.US is not reachable from this host. See the Replay tab."


def live():
    symbol = st.selectbox("symbol", list(SYMBOLS), key="lv_symbol")
    st.caption("Public Binance.US market data. Read-only, no trading. Connects when this "
               f"tab opens and disconnects {LIVE_IDLE_S} seconds after the last viewer leaves.")
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
            state.last_seen = time.time()
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
                    st.info("Connecting to binance.us…")
                return
            if not state.connected:
                st.warning(f"Connection lost. Showing the last data while reconnecting. "
                           f"({state.error})")

            show_metrics(f)
            left, right = st.columns([1, 2])
            with left:
                label("book")
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
            st.error(f"Live view error. {type(e).__name__}: {e}")

    tick()


# ---------------------------------------------------------------- page

# Closed tabs don't render their widgets, and Streamlit drops state for widgets that
# weren't rendered. Reassigning the keys each run keeps Replay and Live settings
# across tab switches.
for k in ("rp_symbol", "rp_file", "rp_speed", "rp_play", "lv_symbol"):
    if k in st.session_state:
        st.session_state[k] = st.session_state[k]

st.html(CSS)
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
