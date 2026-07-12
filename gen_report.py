"""Generate web/public/backtest.json for the GitHub Pages dashboard.

Runs the committed-fixture backtest and emits per-strategy summaries, equity
curves, and the full per-trade blotter. To stay honest, the per-trade loop here
is a faithful replay of `mind_shot.backtest.simulate` and every run is
cross-checked against the official `BacktestResult` — if the replay ever drifts
from the validated engine, this script raises instead of shipping wrong numbers.

Run from the repo root:  python gen_report.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `mind_shot` imports from anywhere

from mind_shot import config
from mind_shot.backtest import (
    PLAYBOOK_TEST_START_S,
    _load_fixture,
    _signal_arrays,
    run,
)
from mind_shot.models import H, L, O, T
from mind_shot.strategies import STRATEGIES, ExitStyle, compute_series

OUT = Path(__file__).resolve().parent / "web" / "public" / "backtest.json"

# Same account model as mind_shot.backtest.simulate defaults / run().
WALLET0, LEVERAGE, ALLOC, FEE, MMR = 100.0, 10.0, 0.15, 0.0005, 0.005


def simulate_with_trades(candles, strat, series):
    """Replay simulate()'s exact logic, capturing each trade and the equity curve."""
    opens = [c[O] for c in candles]
    highs = [c[H] for c in candles]
    lows = [c[L] for c in candles]
    times = [c[T] for c in candles]
    atr = series["atr"]
    el, es, xl, xs = _signal_arrays(strat, series)
    tp_atr = strat.tp_atr if strat.exit_style is ExitStyle.BRACKET else None
    sl_atr = strat.sl_atr
    n = len(candles)

    wallet = peak = WALLET0
    max_dd = 0.0
    pos = 0
    entry = sl = tp = liq = notional = 0.0
    entry_ts = 0
    trades, wins, l_tr, s_tr = 0, 0, 0, 0
    log = []
    equity = [round(wallet, 2)]

    for i in range(n - 1):
        nx = i + 1
        if pos != 0:
            exited, exit_px = False, 0.0
            if pos == 1:
                down = sl if sl > liq else liq
                if lows[nx] <= down:
                    exit_px, exited = down, True
                elif tp is not None and highs[nx] >= tp:
                    exit_px, exited = tp, True
                elif xl[i] or es[i]:
                    exit_px, exited = opens[nx], True
            else:
                up = sl if sl < liq else liq
                if highs[nx] >= up:
                    exit_px, exited = up, True
                elif tp is not None and lows[nx] <= tp:
                    exit_px, exited = tp, True
                elif xs[i] or el[i]:
                    exit_px, exited = opens[nx], True
            if exited:
                gross = (exit_px / entry - 1.0) if pos == 1 else (entry / exit_px - 1.0)
                pnl = notional * (gross - 2 * FEE)
                wallet = max(0.0, wallet + pnl)
                trades += 1
                wins += 1 if pnl > 0 else 0
                l_tr += 1 if pos == 1 else 0
                s_tr += 1 if pos == -1 else 0
                log.append({
                    "side": "long" if pos == 1 else "short",
                    "entry": round(entry, 4),
                    "exit": round(exit_px, 4),
                    "entry_ts": entry_ts,
                    "exit_ts": times[nx],
                    "pnl_usd": round(pnl, 2),
                    "ret_pct": round((gross - 2 * FEE) * LEVERAGE * 100, 1),
                    "wallet_after": round(wallet, 2),
                    "win": pnl > 0,
                })
                equity.append(round(wallet, 2))
                pos = 0
                peak = max(peak, wallet)
                if peak > 0:
                    max_dd = min(max_dd, (wallet - peak) / peak)
                if wallet < 5.0:
                    break
        if pos == 0 and wallet > 5.0:
            if times[nx] < PLAYBOOK_TEST_START_S:
                continue
            if atr[i] is None:
                continue
            if el[i] or es[i]:
                pos = 1 if el[i] else -1
                entry = opens[nx]
                entry_ts = times[nx]
                notional = ALLOC * wallet * LEVERAGE
                if pos == 1:
                    liq = entry * (1 - wallet / notional + MMR)
                    sl = entry - sl_atr * atr[i]
                    tp = entry + tp_atr * atr[i] if tp_atr else None
                else:
                    liq = entry * (1 + wallet / notional - MMR)
                    sl = entry + sl_atr * atr[i]
                    tp = entry - tp_atr * atr[i] if tp_atr else None

    return {"trades": trades, "wins": wins, "l_tr": l_tr, "s_tr": s_tr,
            "wallet": round(wallet, 2), "max_dd": round(100.0 * max_dd, 1),
            "log": log, "equity": equity}


def main() -> None:
    official = {r.strategy_id: r for r in run(verbose=False)}
    data = {a: _load_fixture(a) for a in {s.asset for s in STRATEGIES}}

    results, all_trades = [], []
    for strat in STRATEGIES:
        candles = data[strat.asset]
        rep = simulate_with_trades(candles, strat, compute_series(candles))
        off = official[strat.id]
        # Fidelity gate — the blotter MUST reproduce the validated aggregates.
        assert (rep["trades"], rep["wins"], rep["l_tr"], rep["s_tr"], rep["wallet"]) == \
               (off.trades, off.wins, off.long_trades, off.short_trades, off.final_wallet), \
               f"trade replay diverged from official BacktestResult for {strat.id}"

        results.append({
            "strategy_id": off.strategy_id, "name": off.name, "asset": off.asset,
            "trades": off.trades, "wins": off.wins,
            "long_trades": off.long_trades, "short_trades": off.short_trades,
            "win_rate": round(off.win_rate, 1), "expected_win_rate": round(off.expected_win_rate, 1),
            "final_wallet": round(off.final_wallet, 2), "return_pct": round(off.return_pct, 1),
            "max_drawdown_pct": round(off.max_drawdown_pct, 1), "passes": off.passes,
            "equity": rep["equity"],
        })
        for t in rep["log"]:
            all_trades.append({"strategy_id": off.strategy_id, "name": off.name, "asset": off.asset, **t})

    total_trades = sum(r["trades"] for r in results)
    total_wins = sum(r["wins"] for r in results)
    doc = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {
            "account_usd": config.ACCOUNT_USD, "leverage": config.LEVERAGE,
            "alloc_pct": config.ALLOC_PCT, "margin": "cross",
            "round_trip_pct": 0.10, "window": "6-month playbook window · 4h close",
        },
        "summary": {
            "strategies": len(results),
            "passing": sum(1 for r in results if r["passes"]),
            "avg_win_rate": round(sum(r["win_rate"] for r in results) / len(results), 1),
            "overall_win_rate": round(100.0 * total_wins / total_trades, 1) if total_trades else 0,
            "total_trades": total_trades,
            "avg_return_pct": round(sum(r["return_pct"] for r in results) / len(results), 1),
            "best_return_pct": max((r["return_pct"] for r in results), default=0),
            "worst_drawdown_pct": min((r["max_drawdown_pct"] for r in results), default=0),
        },
        "results": results,
        "trades": all_trades,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"wrote {OUT}  ({doc['summary']['passing']}/{len(results)} pass, "
          f"{total_trades} trades, {total_wins} wins - replay matches engine)")


if __name__ == "__main__":
    main()
