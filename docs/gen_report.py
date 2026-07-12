"""Generate docs/backtest.json from the committed playbook fixtures.

Run from repo root:  python docs/gen_report.py
The static dashboard (docs/index.html) reads the JSON this writes — so the
GitHub Pages report always reflects the real backtest, never hand-typed numbers.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `mind_shot` imports when run from anywhere

from mind_shot import config
from mind_shot.backtest import run

OUT = Path(__file__).resolve().parent / "backtest.json"


def main() -> None:
    results = run(verbose=False)
    rows = [
        {
            "strategy_id": r.strategy_id,
            "name": r.name,
            "asset": r.asset,
            "trades": r.trades,
            "wins": r.wins,
            "long_trades": r.long_trades,
            "short_trades": r.short_trades,
            "win_rate": round(r.win_rate, 1),
            "expected_win_rate": round(r.expected_win_rate, 1),
            "final_wallet": round(r.final_wallet, 2),
            "return_pct": round(r.return_pct, 1),
            "max_drawdown_pct": round(r.max_drawdown_pct, 1),
            "passes": r.passes,
        }
        for r in results
    ]
    passing = sum(1 for r in rows if r["passes"])
    total_trades = sum(r["trades"] for r in rows)
    doc = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {
            "account_usd": config.ACCOUNT_USD,
            "leverage": config.LEVERAGE,
            "alloc_pct": config.ALLOC_PCT,
            "margin": "cross",
            "round_trip_pct": 0.10,
            "window": "6-month playbook window · 4h close",
        },
        "summary": {
            "strategies": len(rows),
            "passing": passing,
            "avg_win_rate": round(sum(r["win_rate"] for r in rows) / len(rows), 1) if rows else 0,
            "total_trades": total_trades,
            "avg_return_pct": round(sum(r["return_pct"] for r in rows) / len(rows), 1) if rows else 0,
            "best_return_pct": max((r["return_pct"] for r in rows), default=0),
            "worst_drawdown_pct": min((r["max_drawdown_pct"] for r in rows), default=0),
        },
        "results": rows,
    }
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"wrote {OUT}  ({passing}/{len(rows)} strategies pass, {total_trades} trades)")


if __name__ == "__main__":
    main()
