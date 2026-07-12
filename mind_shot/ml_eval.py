"""
Prequential evaluation of the self-learning ML — the honest proof.

Replays every playbook trade chronologically per strategy (same fills as
``mind_shot.backtest``; totals are asserted against the official results) and,
at each entry, asks each learner for P(win) BEFORE revealing the outcome, then
lets it learn (test-then-train). Reported per learner:

  log-loss (primary — proper scoring rule), Brier, accuracy@0.5

Learners compared:
  v2        — the full calibrated ensemble in ``mind_shot.ml``
  v1        — the previous model, reimplemented verbatim (Laplace buckets,
              geometric mean, ×2 stop-loss penalty)
  base-rate — cumulative Laplace win rate per strategy (no context)

Also simulates the ``ML_MIN_CONF`` veto gate for each learner: how many trades
it would have blocked and what those trades actually did.

Run:  python -m mind_shot.ml_eval
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from . import ml
from .backtest import PLAYBOOK_TEST_START_S, _load_fixture, _signal_arrays, run
from .models import H, L, O, T
from .strategies import STRATEGIES, ExitStyle, compute_series

WALLET0, LEVERAGE, ALLOC, FEE, MMR = 100.0, 10.0, 0.15, 0.0005, 0.005
GATE = 0.40  # config.ML_MIN_CONF default


# ── trade replay (mirrors backtest.simulate; fidelity asserted in main) ──────
def replay_trades(strat, candles, series) -> List[Dict[str, Any]]:
    opens = [c[O] for c in candles]
    highs = [c[H] for c in candles]
    lows = [c[L] for c in candles]
    times = [c[T] for c in candles]
    atr = series["atr"]
    el, es, xl, xs = _signal_arrays(strat, series)
    tp_atr = strat.tp_atr if strat.exit_style is ExitStyle.BRACKET else None
    sl_atr = strat.sl_atr
    n = len(candles)

    wallet = WALLET0
    pos = 0
    entry = sl = tp = liq = notional = 0.0
    sig_i = 0
    out: List[Dict[str, Any]] = []
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
                out.append({"sig_i": sig_i, "side": "long" if pos == 1 else "short",
                            "won": pnl > 0, "pnl": pnl, "t": times[nx]})
                pos = 0
                if wallet < 5.0:
                    break
        if pos == 0 and wallet > 5.0:
            if times[nx] < PLAYBOOK_TEST_START_S or atr[i] is None:
                continue
            if el[i] or es[i]:
                pos = 1 if el[i] else -1
                sig_i = i
                entry = opens[nx]
                notional = ALLOC * wallet * LEVERAGE
                if pos == 1:
                    liq = entry * (1 - wallet / notional + MMR)
                    sl = entry - sl_atr * atr[i]
                    tp = entry + tp_atr * atr[i] if tp_atr else None
                else:
                    liq = entry * (1 + wallet / notional - MMR)
                    sl = entry + sl_atr * atr[i]
                    tp = entry - tp_atr * atr[i] if tp_atr else None
    return out


# ── v1 model, reimplemented verbatim for comparison ─────────────────────────
class V1Model:
    PENALTY = 2

    def __init__(self) -> None:
        self.dims: Dict[str, Dict[str, List[float]]] = {d: {} for d in ("vol", "rsi", "ses", "strat")}

    def predict(self, snap: Dict[str, Any]) -> float:
        p = 1.0
        for dim, key in (("vol", "vol_b"), ("rsi", "rsi_b"), ("ses", "ses_b"), ("strat", "strat_b")):
            wl = self.dims[dim].get(str(snap[key]), [0, 0])
            p *= (wl[0] + 1) / (wl[0] + wl[1] + 2)
        return p ** 0.25

    def update(self, snap: Dict[str, Any], won: bool) -> None:
        delta = 1 if won else self.PENALTY
        idx = 0 if won else 1
        for dim, key in (("vol", "vol_b"), ("rsi", "rsi_b"), ("ses", "ses_b"), ("strat", "strat_b")):
            self.dims[dim].setdefault(str(snap[key]), [0, 0])[idx] += delta


class BaseRate:
    def __init__(self) -> None:
        self.w = 0
        self.l = 0

    def predict(self, _snap: Dict[str, Any]) -> float:
        return (self.w + 1) / (self.w + self.l + 2)

    def update(self, _snap: Dict[str, Any], won: bool) -> None:
        self.w += 1 if won else 0
        self.l += 0 if won else 1


def _score(name: str, rows: List[Tuple[float, bool]]) -> Dict[str, Any]:
    n = len(rows)
    ll = sum(-math.log(min(max(p if w else 1 - p, 1e-9), 1.0)) for p, w in rows) / n
    br = sum((p - (1.0 if w else 0.0)) ** 2 for p, w in rows) / n
    ac = sum(1 for p, w in rows if (p >= 0.5) == w) / n
    gated = [(p, w) for p, w in rows if p < GATE]
    return {"name": name, "n": n, "logloss": ll, "brier": br, "acc": ac,
            "gated": len(gated), "gated_wins": sum(1 for _, w in gated if w)}


def main() -> None:
    official = {r.strategy_id: r for r in run(verbose=False)}
    data = {a: _load_fixture(a) for a in {s.asset for s in STRATEGIES}}

    ml_nodes = {s.id: ml.empty_ml() for s in STRATEGIES}
    v1s = {s.id: V1Model() for s in STRATEGIES}
    bases = {s.id: BaseRate() for s in STRATEGIES}
    rows: Dict[str, List[Tuple[float, bool]]] = {"v2": [], "v1": [], "base-rate": []}

    events = []
    for strat in STRATEGIES:
        candles = data[strat.asset]
        series = compute_series(candles)
        trades = replay_trades(strat, candles, series)
        off = official[strat.id]
        assert (len(trades), sum(1 for t in trades if t["won"])) == (off.trades, off.wins), \
            f"replay diverged from official backtest for {strat.id}"
        for tr in trades:
            snap = ml.build_snapshot(series, tr["sig_i"], strat.id, candles[tr["sig_i"]][T], side=tr["side"])
            events.append((tr["t"], strat.id, snap, tr["won"]))
    events.sort(key=lambda e: e[0])  # global chronology, like live

    for _t, sid, snap, won in events:
        rows["v2"].append((ml.predict(ml_nodes[sid], snap), won))
        rows["v1"].append((v1s[sid].predict(snap), won))
        rows["base-rate"].append((bases[sid].predict(snap), won))
        ml.update(ml_nodes[sid], snap, won)
        v1s[sid].update(snap, won)
        bases[sid].update(snap, won)

    print(f"Prequential evaluation - {len(events)} playbook trades, test-then-train, global chronology")
    print(f"{'model':<10} {'logloss':>8} {'brier':>7} {'acc':>6}   gate<{GATE:.2f}: blocked (of which wins)")
    print("-" * 78)
    scores = {name: _score(name, r) for name, r in rows.items()}
    for s in scores.values():
        print(f"{s['name']:<10} {s['logloss']:>8.4f} {s['brier']:>7.4f} {s['acc']:>5.1%}   "
              f"{s['gated']:>3} trades ({s['gated_wins']} wins)")
    print("-" * 78)
    v2, v1 = scores["v2"], scores["v1"]
    verdict = "v2 BEATS v1" if v2["logloss"] < v1["logloss"] else "v2 DOES NOT beat v1"
    print(f"{verdict} on log-loss: {v2['logloss']:.4f} vs {v1['logloss']:.4f} "
          f"({(v1['logloss'] - v2['logloss']) / v1['logloss'] * 100:+.1f}%)")
    for sid, node in ml_nodes.items():
        s = ml.summary(node, sid)
        print(f"  {sid:<22} weights={s['weights']}  cal=(a={s['calibration']['a']}, b={s['calibration']['b']})  "
              f"drift={s['drift_events']}")


if __name__ == "__main__":
    main()
