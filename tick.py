"""
tick.py  -  ONE run of the adaptive bot. Designed for GitHub Actions.

Runs entirely in the cloud with NO exchange keys: it pulls real prices from
Kraken (which, unlike Binance, doesn't block data-center IPs) and PAPER-TRADES
them - simulating fills internally. Still fake money, still adaptive; every
trade is recorded so report.py can show it in STATUS.md.

Each run: load memory (state.json) -> fetch real candles -> re-learn on schedule
(validate on unseen data) -> trade only with a validated edge, else hold cash ->
drawdown kill switch -> record trade + equity -> save.

Test offline first:  python tick.py --dry-run
This is a simulation. No real money is involved and profit is not guaranteed.
"""

import argparse
import json
import os
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd

STATE_FILE = "state.json"
LOG_FILE = "trades_log.csv"
TRADES_FILE = "trades.csv"
EQUITY_FILE = "equity_history.csv"

PARAM_SPACE = {
    "fast": [8, 12, 20, 30], "slow": [40, 60, 100, 150], "rsi_period": [14],
    "rsi_low": [40, 45, 50], "rsi_high": [70, 80, 90], "trend": [100, 200],
    "stop_loss": [0.02, 0.03, 0.05], "take_profit": [0.03, 0.05, 0.08, 0.12],
}
PPY = {"1m": 525600, "5m": 105120, "15m": 35040, "1h": 8760, "4h": 2190, "1d": 365}


# ------------------------- indicators & signals -------------------------
def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def sma(s, p): return s.rolling(p).mean()

def rsi(s, p=14):
    d = s.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    ag = up.ewm(alpha=1/p, adjust=False).mean(); al = dn.ewm(alpha=1/p, adjust=False).mean()
    return (100 - 100/(1 + ag/al.replace(0, np.nan))).fillna(50.0)

def compute_signals(df, p):
    c = df["close"]
    long_ok = (ema(c, p["fast"]) > ema(c, p["slow"])) & \
              (rsi(c, p["rsi_period"]).between(p["rsi_low"], p["rsi_high"])) & \
              (c > sma(c, p["trend"]))
    exit_sig = ema(c, p["fast"]) < ema(c, p["slow"])
    return long_ok.fillna(False).to_numpy(), exit_sig.fillna(False).to_numpy()


# ------------------------- backtest & metrics -------------------------
def backtest(df, p, fee=0.001, slip=0.0005, cash0=10_000.0):
    long_ok, exit_sig = compute_signals(df, p)
    close, high, low = df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy()
    warmup = max(p["slow"], p["trend"], p["rsi_period"]) + 2
    cash, qty, basis, entry, pos = cash0, 0.0, 0.0, 0.0, False
    eq, trades = np.empty(len(df)), []
    sl, tp = p["stop_loss"], p["take_profit"]
    for i in range(len(df)):
        px = close[i]
        if pos:
            hit, fill = False, px
            if low[i] <= entry*(1-sl): hit, fill = True, entry*(1-sl)
            elif high[i] >= entry*(1+tp): hit, fill = True, entry*(1+tp)
            elif exit_sig[i]: hit = True
            if hit:
                cash = qty*fill*(1-slip)*(1-fee); trades.append(cash-basis)
                qty, pos = 0.0, False
        elif i >= warmup and long_ok[i]:
            entry = px*(1+slip); qty = cash*(1-fee)/entry; basis = cash; cash, pos = 0.0, True
        eq[i] = cash + qty*px
    if pos:
        cash = qty*close[-1]*(1-slip)*(1-fee); trades.append(cash-basis); eq[-1] = cash
    return eq, trades

def metrics(eq, trades, ppy):
    eq = np.asarray(eq, float); r = np.diff(eq)/eq[:-1]
    t = np.asarray(trades, float); n = len(t)
    win = t[t > 0]; loss = t[t < 0]; gl = -loss.sum()
    return {"total_return": eq[-1]/eq[0]-1,
            "sharpe": (r.mean()/r.std()*np.sqrt(ppy)) if r.std() > 0 else 0.0,
            "win_rate": len(win)/n if n else 0.0,
            "profit_factor": (win.sum()/gl) if gl > 0 else (float("inf") if n else 0.0),
            "n_trades": n}

def sample(space, rng):
    while True:
        p = {k: rng.choice(v) for k, v in space.items()}
        if p["fast"] < p["slow"]: return p

def optimize(df, ppy, samples, rng):
    best, score = None, -1e18
    for _ in range(samples):
        p = sample(PARAM_SPACE, rng)
        m = metrics(*backtest(df, p), ppy)
        s = m["sharpe"] if m["n_trades"] >= 8 else -1e9
        if s > score: best, score = {"params": p, "metrics": m}, s
    return best


# ------------------------- data -------------------------
def synthetic(n=800, seed=None):
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for i in range(1, n): r[i] = 0.8*r[i-1] + rng.normal(0, 0.006)
    c = 30000*np.exp(np.cumsum(r)); o = np.concatenate([[c[0]], c[:-1]])
    w = np.abs(rng.normal(0, 0.004, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({"open": o, "high": np.maximum(o, c)*(1+w),
                         "low": np.minimum(o, c)*(1-w), "close": c, "volume": 1.0}, index=idx)

def fetch_data(symbol, timeframe):
    """Real OHLCV from Kraken (public, no keys, not geoblocked on cloud IPs)."""
    import ccxt
    ex = ccxt.kraken({"enableRateLimit": True})
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe)   # Kraken returns up to ~720 bars
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df["ts"], unit="ms")
    return df[["open", "high", "low", "close", "volume"]].astype(float)


# ------------------------- state, logging, records -------------------------
def load_state():
    s = json.load(open(STATE_FILE)) if os.path.exists(STATE_FILE) else {}
    s.setdefault("params", None); s.setdefault("in_position", False)
    s.setdefault("entry", 0.0); s.setdefault("qty", 0.0)
    s.setdefault("entry_time", None); s.setdefault("equity_est", 10_000.0)
    s.setdefault("start_equity", 10_000.0); s.setdefault("peak", 10_000.0)
    s.setdefault("halted", False); s.setdefault("last_relearn", None)
    s.setdefault("last_price", 0.0)
    return s

def save_state(s): json.dump(s, open(STATE_FILE, "w"), indent=2)

def log(event, detail):
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a") as f:
        if new: f.write("time,event,detail\n")
        f.write(f"{datetime.now(timezone.utc).isoformat()},{event},{detail}\n")
    print(f"{event}: {detail}")

def record_trade(row):
    new = not os.path.exists(TRADES_FILE)
    with open(TRADES_FILE, "a") as f:
        if new: f.write("entry_time,exit_time,entry_price,exit_price,qty,pnl_usdt,pnl_pct,reason\n")
        f.write(",".join(str(x) for x in row) + "\n")

def record_equity(now, equity, price, in_pos):
    new = not os.path.exists(EQUITY_FILE)
    with open(EQUITY_FILE, "a") as f:
        if new: f.write("time,equity,price,in_position\n")
        f.write(f"{now.isoformat()},{equity:.2f},{price:.2f},{int(in_pos)}\n")

def close_position(s, price, reason, now):
    pnl = s["qty"] * (price - s["entry"])
    pnl_pct = (price / s["entry"] - 1) * 100 if s["entry"] else 0.0
    s["equity_est"] += pnl
    record_trade([s.get("entry_time"), now.isoformat(), round(s["entry"], 2),
                  round(price, 2), s["qty"], round(pnl, 2), round(pnl_pct, 2), reason])
    log("sell", f"@~{price:.2f} ({reason}) pnl {pnl_pct:+.2f}% (${pnl:+,.2f})")
    s["in_position"], s["qty"], s["entry"], s["entry_time"] = False, 0.0, 0.0, None


# ------------------------- one tick -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USD")     # Kraken uses USD pairs
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--quote", type=float, default=1000.0)
    ap.add_argument("--relearn-hours", type=int, default=48)
    ap.add_argument("--lookback", type=int, default=480)
    ap.add_argument("--oos", type=int, default=120)
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--min-pf", type=float, default=1.05)
    ap.add_argument("--max-drawdown", type=float, default=0.20)
    ap.add_argument("--dry-run", action="store_true", help="offline synthetic data")
    args = ap.parse_args()

    s = load_state()
    ppy = PPY.get(args.timeframe, 8760)

    df = synthetic(n=args.lookback + args.oos + 20) if args.dry_run \
        else fetch_data(args.symbol, args.timeframe)

    # fit windows to whatever history we actually got
    lookback, oos = args.lookback, args.oos
    if len(df) < lookback + oos + 5:
        oos = max(60, len(df) // 5)
        lookback = max(120, len(df) - oos - 5)

    price = float(df["close"].iloc[-1])
    now = datetime.now(timezone.utc)

    # ---- re-learn on schedule (the adaptive part) ----
    due = s["last_relearn"] is None or \
        (now - datetime.fromisoformat(s["last_relearn"])).total_seconds() >= args.relearn_hours*3600
    if due and len(df) >= lookback + oos:
        rng = random.Random(int(now.timestamp()))
        best = optimize(df.iloc[-(lookback+oos):-oos], ppy, args.samples, rng)
        v = metrics(*backtest(df.iloc[-oos:], best["params"]), ppy) if best else None
        armed = bool(v and v["total_return"] > 0 and v["n_trades"] >= 8
                     and v["profit_factor"] >= args.min_pf)
        s["last_relearn"] = now.isoformat()
        if armed:
            s["params"] = best["params"]
            log("relearn", f"edge validated oos={v['total_return']*100:+.1f}% -> armed")
        else:
            if s["in_position"]:
                close_position(s, price, "relearn-exit", now)
            s["params"] = None
            log("relearn", f"no edge (oos={v['total_return']*100:.1f}%) -> CASH" if v else "no data -> CASH")

    # ---- drawdown kill switch ----
    mtm = s["equity_est"] + (s["qty"]*(price-s["entry"]) if s["in_position"] else 0.0)
    s["peak"] = max(s["peak"], mtm)
    if not s["halted"] and mtm <= s["peak"]*(1-args.max_drawdown):
        s["halted"] = True
        if s["in_position"]:
            close_position(s, price, "kill-switch", now)
        log("HALT", f"drawdown limit hit at ${mtm:,.0f}; no new entries")

    # ---- act on the latest candle (paper fills) ----
    if s["params"] and not s["halted"]:
        long_ok, exit_sig = compute_signals(df, s["params"])
        i = len(df)-1; p = s["params"]
        if s["in_position"]:
            hit_sl = price <= s["entry"]*(1-p["stop_loss"])
            hit_tp = price >= s["entry"]*(1+p["take_profit"])
            if hit_sl or hit_tp or exit_sig[i]:
                close_position(s, price, "stop" if hit_sl else "target" if hit_tp else "trend", now)
        elif long_ok[i]:
            s["qty"] = round(args.quote/price, 6)
            s["entry"], s["in_position"], s["entry_time"] = price, True, now.isoformat()
            log("buy", f"{s['qty']} @~{price:.2f} (~${args.quote:.0f})")
        else:
            log("hold", f"armed, no entry signal (px {price:.2f})")
    else:
        log("cash", f"no validated edge or halted (px {price:.2f})")

    # ---- record equity snapshot + save ----
    s["last_price"] = price
    final_mtm = s["equity_est"] + (s["qty"]*(price-s["entry"]) if s["in_position"] else 0.0)
    record_equity(now, final_mtm, price, s["in_position"])
    save_state(s)


if __name__ == "__main__":
    main()
