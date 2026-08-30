"""
tick.py  -  ONE run of the adaptive bot. Designed for GitHub Actions.

Keyless PAPER trading on real Kraken data, now with ORDER-BOOK fills:
when it trades it pulls the live order book and "walks" it (buys into the asks,
sells into the bids) so every fill includes real spread + slippage. It logs the
slippage so you can see what execution costs you. Still fake money.

Each run: load memory -> fetch candles + (at trade time) the live order book ->
re-learn on schedule -> trade only with a validated edge -> drawdown kill switch
-> record trade/fill/equity -> save.

Test offline:  python tick.py --dry-run
Simulation only. No real money. Profit not guaranteed.
"""

import argparse
import json
import os
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd

STATE_FILE, LOG_FILE = "state.json", "trades_log.csv"
TRADES_FILE, EQUITY_FILE, FILLS_FILE = "trades.csv", "equity_history.csv", "fills.csv"

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


# ------------------------- order-book fills -------------------------
def fill_buy(asks, quote_amount):
    """Walk the asks spending `quote_amount` USD -> (avg_price, qty)."""
    spent, qty, last = 0.0, 0.0, (asks[-1][0] if asks else 0.0)
    for price, amount in asks:
        last = price
        remaining = quote_amount - spent
        if remaining <= 0: break
        cost = price * amount
        if cost >= remaining:
            qty += remaining / price; spent += remaining
            return spent / qty, qty
        qty += amount; spent += cost
    remaining = quote_amount - spent           # book too thin: fill rest at worst price
    if remaining > 0 and last > 0:
        qty += remaining / last; spent += remaining
    return (spent / qty if qty else last), qty

def fill_sell(bids, qty_to_sell):
    """Walk the bids selling `qty_to_sell` -> (avg_price, usd_received)."""
    got, sold, last = 0.0, 0.0, (bids[-1][0] if bids else 0.0)
    for price, amount in bids:
        last = price
        take = min(amount, qty_to_sell - sold)
        got += take * price; sold += take
        if sold >= qty_to_sell:
            return got / sold, got
    rem = qty_to_sell - sold
    if rem > 0 and last > 0:
        got += rem * last; sold += rem
    return (got / sold if sold else last), got

def synth_book(price, levels=50, spread=0.0002, size=0.5):
    asks = [[price*(1+spread*(k+1)), size] for k in range(levels)]
    bids = [[price*(1-spread*(k+1)), size] for k in range(levels)]
    return {"asks": asks, "bids": bids}


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

def make_exchange():
    import ccxt
    return ccxt.kraken({"enableRateLimit": True})

def fetch_data(ex, symbol, timeframe):
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df["ts"], unit="ms")
    return df[["open", "high", "low", "close", "volume"]].astype(float)

def fetch_book(ex, symbol, price):
    try:
        b = ex.fetch_order_book(symbol, limit=50)
        if b and b.get("asks") and b.get("bids"):
            return b
    except Exception as e:
        print("order book fetch failed, using synthetic:", e)
    return synth_book(price)


# ------------------------- state, logging, records -------------------------
def load_state():
    s = json.load(open(STATE_FILE)) if os.path.exists(STATE_FILE) else {}
    for k, v in {"params": None, "in_position": False, "entry": 0.0, "qty": 0.0,
                 "cost_basis": 0.0, "entry_time": None, "equity_est": 10_000.0,
                 "start_equity": 10_000.0, "peak": 10_000.0, "halted": False,
                 "last_relearn": None, "last_price": 0.0}.items():
        s.setdefault(k, v)
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

def record_fill(now, side, mid, fill, usd):
    slip = (fill/mid - 1)*100 if side == "buy" else (mid/fill - 1)*100  # +% = worse than mid
    new = not os.path.exists(FILLS_FILE)
    with open(FILLS_FILE, "a") as f:
        if new: f.write("time,side,mid,fill,slip_pct,usd\n")
        f.write(f"{now.isoformat()},{side},{mid:.2f},{fill:.2f},{slip:.4f},{usd:.2f}\n")
    return slip

def record_equity(now, equity, price, in_pos):
    new = not os.path.exists(EQUITY_FILE)
    with open(EQUITY_FILE, "a") as f:
        if new: f.write("time,equity,price,in_position\n")
        f.write(f"{now.isoformat()},{equity:.2f},{price:.2f},{int(in_pos)}\n")

def open_position(s, book, quote, fee, now):
    mid = (book["bids"][0][0] + book["asks"][0][0]) / 2
    avg, qty = fill_buy(book["asks"], quote)
    qty *= (1 - fee)                                   # fee taken in base
    s["entry"], s["qty"], s["cost_basis"] = avg, round(qty, 8), quote
    s["in_position"], s["entry_time"] = True, now.isoformat()
    slip = record_fill(now, "buy", mid, avg, quote)
    log("buy", f"{s['qty']} @~{avg:.2f} (mid {mid:.2f}, slip {slip:+.3f}%, ~${quote:.0f})")

def close_position(s, book, reason, fee, now):
    mid = (book["bids"][0][0] + book["asks"][0][0]) / 2
    avg, proceeds = fill_sell(book["bids"], s["qty"])
    proceeds *= (1 - fee)
    pnl = proceeds - s["cost_basis"]
    pnl_pct = (proceeds / s["cost_basis"] - 1) * 100 if s["cost_basis"] else 0.0
    s["equity_est"] += pnl
    slip = record_fill(now, "sell", mid, avg, proceeds)
    record_trade([s.get("entry_time"), now.isoformat(), round(s["entry"], 2),
                  round(avg, 2), s["qty"], round(pnl, 2), round(pnl_pct, 2), reason])
    log("sell", f"@~{avg:.2f} (mid {mid:.2f}, slip {slip:+.3f}%) pnl {pnl_pct:+.2f}% (${pnl:+,.2f}) [{reason}]")
    s["in_position"], s["qty"], s["entry"], s["cost_basis"], s["entry_time"] = False, 0.0, 0.0, 0.0, None


# ------------------------- one tick -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USD")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--quote", type=float, default=1000.0)
    ap.add_argument("--fee", type=float, default=0.001, help="taker fee per side")
    ap.add_argument("--relearn-hours", type=int, default=48)
    ap.add_argument("--lookback", type=int, default=480)
    ap.add_argument("--oos", type=int, default=120)
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--min-pf", type=float, default=1.05)
    ap.add_argument("--max-drawdown", type=float, default=0.20)
    ap.add_argument("--dry-run", action="store_true", help="offline synthetic data + book")
    args = ap.parse_args()

    s = load_state()
    ppy = PPY.get(args.timeframe, 8760)

    if args.dry_run:
        ex, df = None, synthetic(n=args.lookback + args.oos + 20)
    else:
        ex = make_exchange()
        df = fetch_data(ex, args.symbol, args.timeframe)

    lookback, oos = args.lookback, args.oos
    if len(df) < lookback + oos + 5:
        oos = max(60, len(df) // 5); lookback = max(120, len(df) - oos - 5)

    price = float(df["close"].iloc[-1])
    now = datetime.now(timezone.utc)
    book = synth_book(price) if args.dry_run else fetch_book(ex, args.symbol, price)

    # ---- re-learn on schedule ----
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
                close_position(s, book, "relearn-exit", args.fee, now)
            s["params"] = None
            log("relearn", f"no edge (oos={v['total_return']*100:.1f}%) -> CASH" if v else "no data -> CASH")

    # ---- drawdown kill switch ----
    mtm = s["equity_est"] + (s["qty"]*price - s["cost_basis"] if s["in_position"] else 0.0)
    s["peak"] = max(s["peak"], mtm)
    if not s["halted"] and mtm <= s["peak"]*(1-args.max_drawdown):
        s["halted"] = True
        if s["in_position"]:
            close_position(s, book, "kill-switch", args.fee, now)
        log("HALT", f"drawdown limit hit at ${mtm:,.0f}; no new entries")

    # ---- act on the latest candle (order-book paper fills) ----
    if s["params"] and not s["halted"]:
        long_ok, exit_sig = compute_signals(df, s["params"])
        i = len(df)-1; p = s["params"]
        if s["in_position"]:
            hit_sl = price <= s["entry"]*(1-p["stop_loss"])
            hit_tp = price >= s["entry"]*(1+p["take_profit"])
            if hit_sl or hit_tp or exit_sig[i]:
                close_position(s, book, "stop" if hit_sl else "target" if hit_tp else "trend", args.fee, now)
        elif long_ok[i]:
            open_position(s, book, args.quote, args.fee, now)
        else:
            log("hold", f"armed, no entry signal (px {price:.2f})")
    else:
        log("cash", f"no validated edge or halted (px {price:.2f})")

    # ---- record equity snapshot + save ----
    s["last_price"] = price
    final_mtm = s["equity_est"] + (s["qty"]*price - s["cost_basis"] if s["in_position"] else 0.0)
    record_equity(now, final_mtm, price, s["in_position"])
    save_state(s)


if __name__ == "__main__":
    main()
