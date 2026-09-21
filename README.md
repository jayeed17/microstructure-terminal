# Crypto Market Microstructure Terminal

Live L2 order book analytics and short-horizon direction modeling on Binance.US
public market data.

**Research question:** does order flow imbalance predict price movement at the
1–60 second horizon, and does any of that edge survive transaction costs?

Read-only. No API keys, no order placement anywhere in this repo.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

```bash
# 1. collect — run this on the always-on box, leave it for 48h+
python collect.py --symbol btcusdt

# 2. label — forward returns with a spread-aware deadband
python collect.py --symbol btcusdt --label

# 3. model — chronological splits, embargoed seams
python train.py --symbol btcusdt --horizon 10
python train.py --symbol btcusdt --sweep      # signal decay curve

# 4. costs — the honest part
python backtest.py --symbol btcusdt --horizon 10 --fee-bps 10

# 5. terminal
streamlit run app.py
```

Run steps 1–2 before anything else. There is no shortcut around collecting data.

## Features

Computed on every 100ms book snapshot (20 levels each side):

| group | features |
|---|---|
| queue imbalance | `imb_1`, `imb_5`, `imb_20`, `depth_*` |
| order flow | `ofi` (Cont/Kukanov/Stoikov), `ofi_cum50` |
| price pressure | `microprice`, `micro_dev_bps`, `press_w` |
| spread | `spread_bps` |
| trade flow | signed volume / notional / count over 1s, 5s, 10s |
| volatility | `rvol_bps`, `mom_bps` over trailing 300 ticks |

**Label.** Forward mid return over `horizon` seconds, bucketed to
down / flat / up with a deadband of half the prevailing spread. A move only
counts as directional if it would have been large enough to trade on.

## Methodology guardrails

These are deliberate and worth defending in a writeup:

- **Chronological splits with an embargo.** Every label looks `horizon` seconds
  forward, so the gap between train/val/test must be at least that long or the
  split leaks. Shuffled splits on this data produce fake accuracy.
- **Raw price levels are dropped as features.** `mid`, `bid`, `ask` and
  `microprice` are non-stationary. A tree given them memorizes price regions and
  posts an impressive number that evaporates out of sample.
- **Results are reported net of costs.** Gross edge on order book data is real
  and well documented. Net edge after crossing the spread and paying taker fees
  usually is not, at retail fee tiers. Both numbers go in the writeup.

## Interpreting the output

The headline artifact is the signal decay table from `--sweep`: accuracy and
gross edge as a function of prediction horizon. Expect edge to be largest at
1–5s and to erode toward zero by 60s.

`backtest.py` prints net edge per trade across confidence thresholds. If it is
negative at every threshold, that is the result. Write it up as
*"signal is statistically real at 10s but ~X bps of gross edge does not clear
~Y bps of round-trip cost"* — that sentence is more credible than any equity
curve a student project can produce.

## Notes

- `binance.com` returns 403 from US IPs. `binance.us` is the correct host.
- Check your actual Binance.US fee tier before setting `--fee-bps`. Retail
  taker fees there are high; do not quote a placeholder in a report.
- Storage: ~10 snapshots/sec ≈ 850k rows/day/symbol, roughly 60–80 MB/day as
  parquet. Fine to keep weeks of it.
