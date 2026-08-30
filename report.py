"""
report.py  -  build STATUS.md (a glanceable dashboard) + equity.png from the
files tick.py writes. Run right after tick.py in the workflow. GitHub renders
STATUS.md on your phone, so you just open that one file to see how it's going.
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATE_FILE, TRADES_FILE, EQUITY_FILE, OUT, CHART = \
    "state.json", "trades.csv", "equity_history.csv", "STATUS.md", "equity.png"


def money(x):
    return f"${x:,.2f}"


def load_state():
    return json.load(open(STATE_FILE)) if os.path.exists(STATE_FILE) else {}


def make_chart():
    if not os.path.exists(EQUITY_FILE):
        return False
    df = pd.read_csv(EQUITY_FILE, parse_dates=["time"])
    if df.empty:
        return False
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(df["time"], df["equity"], color="#1f6feb", lw=1.8, marker="o", ms=2.5)
    ax.axhline(df["equity"].iloc[0], color="#8b949e", ls="--", lw=1, label="start")
    ax.fill_between(df["time"], df["equity"].iloc[0], df["equity"],
                    where=df["equity"] >= df["equity"].iloc[0], color="#2da44e", alpha=0.12)
    ax.fill_between(df["time"], df["equity"].iloc[0], df["equity"],
                    where=df["equity"] < df["equity"].iloc[0], color="#cf222e", alpha=0.12)
    ax.set_title("Equity over time (testnet, estimated)")
    ax.set_ylabel("Equity ($)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(CHART, dpi=130)
    return True


def trades_table():
    if not os.path.exists(TRADES_FILE):
        return "_No closed trades yet._", None
    df = pd.read_csv(TRADES_FILE)
    if df.empty:
        return "_No closed trades yet._", df
    recent = df.tail(10).iloc[::-1]
    rows = ["| Exit time | Entry | Exit | P/L % | P/L $ | Why |",
            "|---|---|---|---|---|---|"]
    for _, r in recent.iterrows():
        exit_t = str(r["exit_time"])[:16].replace("T", " ")
        emoji = "🟢" if r["pnl_pct"] >= 0 else "🔴"
        rows.append(f"| {exit_t} | {r['entry_price']:.2f} | {r['exit_price']:.2f} | "
                    f"{emoji} {r['pnl_pct']:+.2f}% | {r['pnl_usdt']:+.2f} | {r['reason']} |")
    return "\n".join(rows), df


def main():
    s = load_state()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    start = s.get("start_equity", 10_000.0)
    equity_est = s.get("equity_est", start)
    in_pos = s.get("in_position", False)
    halted = s.get("halted", False)
    price = s.get("last_price", 0.0)
    entry = s.get("entry", 0.0)
    qty = s.get("qty", 0.0)
    params = s.get("params")

    unreal = qty * (price - entry) if in_pos else 0.0
    unreal_pct = (price / entry - 1) * 100 if (in_pos and entry) else 0.0
    equity_now = equity_est + unreal
    total_pct = (equity_now / start - 1) * 100 if start else 0.0

    if halted:
        status = "⛔ **HALTED** (drawdown kill switch tripped — no new entries)"
    elif in_pos:
        status = "🟢 **In a trade**"
    elif params:
        status = "🟡 **Armed, waiting** for an entry signal"
    else:
        status = "💵 **In cash** — no validated edge right now"

    make_chart()
    table, tdf = trades_table()

    # realized summary
    if tdf is not None and not tdf.empty:
        wins = (tdf["pnl_pct"] > 0).sum()
        n = len(tdf)
        realized = tdf["pnl_usdt"].sum()
        summary = (f"- Closed trades: **{n}**  |  wins: **{wins}** "
                   f"({wins/n*100:.0f}%)  |  realized P/L: **{money(realized)}**")
    else:
        summary = "- Closed trades: **0**"

    arrow = "🟢" if total_pct >= 0 else "🔴"
    lines = [
        f"# Bot status  —  updated {now}",
        "",
        status,
        "",
        "## Overall",
        f"- Equity now: **{money(equity_now)}**  (started {money(start)})",
        f"- Total since start: {arrow} **{total_pct:+.2f}%**  _(testnet, estimated)_",
        summary,
        "",
    ]

    if in_pos:
        deployed = qty * entry
        pl_arrow = "🟢" if unreal_pct >= 0 else "🔴"
        lines += [
            "## Open position",
            f"- Size: **{qty} units**  (~{money(deployed)} deployed)",
            f"- Entry price: **{entry:,.2f}**",
            f"- Current price: **{price:,.2f}**",
            f"- Unrealized P/L: {pl_arrow} **{unreal_pct:+.2f}%**  ({money(unreal)})",
            "",
        ]

    if params:
        lines += ["## Currently trading these settings",
                  "```json", json.dumps(params, indent=2), "```", ""]

    lines += [
        "## Recent trades",
        table,
        "",
        "## Equity chart",
        f"![equity]({CHART})",
        "",
        "---",
        "_Binance **testnet** (fake money). Equity is an internal estimate. "
        "This bot adapts but does **not** guarantee profit — it's a learning run._",
    ]

    open(OUT, "w").write("\n".join(lines))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
