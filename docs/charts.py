"""
README charts. Reads the committed CSVs in models/ only -- no analysis is recomputed.

  python docs/charts.py
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parent.parent
MODELS, OUT = ROOT / "models", ROOT / "docs"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {"BTC": "#2a78d6", "ETH": "#eb6834"}      # fixed categorical order
SYMS = {"BTC": "btcusdt", "ETH": "ethusdt"}
MEDIAN_SNAPSHOT_GAP_MS = 193                        # BTC labeled data, see README "Data"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False,
})


def end_label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                va="center", color=INK, fontsize=9, fontweight="bold")
    ax.plot([], [], color=color)


def decay():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, sym in SYMS.items():
        s = pd.read_csv(MODELS / f"{sym}_sweep.csv")
        ax.plot(s.horizon_s, s.gross_edge_bps, marker="o", color=COLOR[name], label=name)
        end_label(ax, s.horizon_s.iloc[-1], s.gross_edge_bps.iloc[-1], name, COLOR[name])
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 30, 60], ["1s", "2s", "5s", "10s", "30s", "60s"])
    ax.minorticks_off()
    ax.set_xlim(0.85, 85)
    ax.set_ylim(0, None)
    ax.set_xlabel("prediction horizon (log scale)")
    ax.set_ylabel("gross edge per trade (bps)")
    ax.set_title("Signal decay by horizon", loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "decay.png", dpi=150)


def latency():
    s = pd.read_csv(MODELS / "btcusdt_latency.csv")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(s.latency_ms, s.breakeven_fee_bps, marker="o", color=COLOR["BTC"])
    for _, r in s.iterrows():
        if r.latency_ms in (0, 50):
            continue
        ax.annotate(f"{r.breakeven_fee_bps:.2f}", (r.latency_ms, r.breakeven_fee_bps),
                    xytext=(7, 6), textcoords="offset points", ha="left", fontsize=9, color=INK2)
    r50 = s[s.latency_ms == 50].iloc[0]
    ax.plot(50, r50.breakeven_fee_bps, "o", ms=13, mfc="none", mec=INK2, mew=1.2)
    ax.annotate(f"50 ms: not measurable\n(median snapshot gap {MEDIAN_SNAPSHOT_GAP_MS} ms;\n"
                "entry resolves to the L=0 snapshot)",
                (50, r50.breakeven_fee_bps), xytext=(150, r50.breakeven_fee_bps - 0.02),
                fontsize=9, color=INK2, va="top",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xticks(s.latency_ms, [f"{int(v)}" for v in s.latency_ms])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("order arrival latency L (ms)")
    ax.set_ylabel("breakeven fee (bps)")
    ax.set_title("BTC breakeven fee vs latency (horizon 1s, conf 0.6)", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "latency.png", dpi=150)


def daily():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, sym in SYMS.items():
        w = pd.read_csv(MODELS / f"{sym}_walkforward.csv", parse_dates=["date"])
        ax.plot(w.date, w.breakeven_fee_bps, marker="o", color=COLOR[name], label=name)
        end_label(ax, w.date.iloc[-1], w.breakeven_fee_bps.iloc[-1], name, COLOR[name])
    ax.axhline(0, color=INK, lw=1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.set_ylim(-0.1, 1.5)
    ax.set_ylabel("breakeven fee (bps)")
    ax.set_title("Walk-forward daily breakeven fee (horizon 1s, conf 0.6)", loc="left")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "daily.png", dpi=150)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    decay(); latency(); daily()
    print("wrote docs/decay.png docs/latency.png docs/daily.png")
