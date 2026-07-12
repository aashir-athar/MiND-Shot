"""
The self-learning second opinion — v2.

A calibrated online ensemble that learns from every closed trade and carries an
honest, measurable record of whether it is actually helping. It never invents a
signal — it only scores one (and may veto it when ``ML_GATING_ENABLED``).

Five experts predict P(win) at entry time; their predictions are frozen into the
trade's ML snapshot so learning at close time is strictly prequential (predict
first, learn after — no hindsight):

  • BASE   — the strategy's own decayed win rate, shrunk toward a tempered prior
             built from its backtest win rate. The "safe" expert.
  • BUCKETS — hierarchical Bayesian context buckets (volatility / RSI / session /
             strategy) with empirical-Bayes shrinkage toward the base rate and
             exponential forgetting, combined in log-odds space.
  • FTRL   — an FTRL-Proximal online logistic regression (McMahan et al., 2013)
             over bucket one-hots plus scaled numeric context features, predicting
             a residual on the base-rate logit. L1 keeps it sparse; per-coordinate
             learning rates keep it stable on tiny samples.
  • POOL   — the cross-strategy context memory (``__global.ml_shared``): every
             strategy's closed trade teaches it, and its context evidence votes
             in log-odds deltas applied to the predicting strategy's own base
             rate, so knowledge transfers across the five strategies. Asleep
             when no shared node is passed.
  • TRAINED — the weekly walk-forward directional model (``ml_trainer.py``),
             made side-aware: P(favorable move) = P(up) for longs, 1−P(up) for
             shorts. Absent (asleep) when no model / not stamped at entry.

An exponentially-weighted (Hedge / multiplicative-weights) ensemble mixes the
awake experts in log-odds space. Hedge carries the classic regret guarantee:
cumulative log-loss is never much worse than the single best expert in
hindsight, so the ensemble can only be beaten by its own best component, and
only by a vanishing margin. A floor on the weights keeps every expert alive so
a regime change can resurrect a previously-bad expert.

The mixed logit passes through an online Platt calibration layer (slope/offset
learned by regularised online gradient descent) so the published number is a
*probability*, not a score. A Page–Hinkley detector watches the per-trade
log-loss stream; on drift it accelerates forgetting and softens the ensemble
weights toward uniform.

Honesty ledger: every closed trade updates prequential log-loss / Brier /
accuracy for the model AND for the base-rate baseline, plus 10-bin reliability
counts — so "is the ML adding value?" is a number in ``state.json``, not a
marketing claim.

v1 nodes are migrated in place, never wiped: raw totals are preserved, the ×2
stop-loss penalty that v1 baked into its loss counts is divided back out, and
the original v1 dimensions are kept under ``legacy_v1`` for audit.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from . import config, indicators as ind
from .models import C, H, L, V, Candle
from .strategies import STRATEGIES

_STRAT_INDEX = {s.id: i for i, s in enumerate(STRATEGIES)}
_STRAT_BY_ID = {s.id: s for s in STRATEGIES}

ML_VERSION = 2

# ── tunables (fixed, documented; not env-driven so behaviour is reproducible) ─
HALF_LIFE_TRADES = 120       # decayed counts lose half their weight every N closed trades
PRIOR_STRENGTH = 6.0         # pseudo-trades behind the tempered backtest prior
BUCKET_PRIOR_M = 6.0         # pseudo-trades shrinking each bucket toward the base rate
BUCKET_SHRINK_M = 10.0       # evidence needed before a bucket's log-odds vote counts fully
BUCKET_LOGIT_CAP = 2.5       # cap on the summed bucket evidence (log-odds units)
TEMPER_TOWARD = 0.60         # backtest win rates are in-sample; temper priors toward this
FTRL_ALPHA = 0.08            # per-coordinate learning-rate scale
FTRL_BETA = 1.0
FTRL_L1 = 0.02               # sparsity — small data must not grow big weights
FTRL_L2 = 0.5
FTRL_LOGIT_CAP = 2.0         # cap on the FTRL residual (log-odds units)
HEDGE_ETA = 0.15             # multiplicative-weights learning rate
HEDGE_FLOOR = 0.03           # minimum share per expert (keeps everyone alive)
HEDGE_LOSS_CLIP = 4.0        # clip per-trade log-loss fed to Hedge
CAL_LR = 0.02                # online Platt calibration learning rate
CAL_REG = 0.01               # pull toward the identity transform (a=1, b=0)
PH_DELTA = 0.02              # Page–Hinkley tolerance
PH_LAMBDA = 3.0              # Page–Hinkley alarm threshold
PH_MIN_N = 20                # trades before drift can fire
DRIFT_EXTRA_DECAY = 0.85     # counts multiplier applied on drift
PROB_FLOOR, PROB_CEIL = 0.02, 0.98
POOL_LOGIT_CAP = 2.0         # cap on the cross-strategy pool's log-odds vote
EXPERTS = ("base", "buckets", "ftrl", "pool", "trained")
# Start base-heavy: the safe expert carries a fresh node; the context experts and
# the (historically weak) trained model must EARN weight through realised log-loss.
# ``pool`` is the cross-strategy context expert — five strategies share one context
# memory, so it starts with more evidence behind it than the per-strategy pair.
HEDGE_INIT = {"base": 0.45, "buckets": 0.15, "ftrl": 0.15, "pool": 0.20, "trained": 0.05}

_DECAY = 0.5 ** (1.0 / HALF_LIFE_TRADES)


# ── small math helpers ───────────────────────────────────────────────────────
def _sigmoid(z: float) -> float:
    if z > 30:
        return 1.0
    if z < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _finite(v: Any, fallback: float) -> float:
    """A finite float, or the fallback. NaN/Inf/garbage from a corrupt node or a
    degenerate feature must never propagate — a bare NaN in state.json breaks
    strict JSON parsers downstream (the Electron host's JSON.parse)."""
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else fallback


def _logloss(p: float, won: bool) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return -math.log(p if won else 1.0 - p)


def laplace(wins: float, losses: float) -> float:
    """Kept from v1 — used by the engine's ml_summary back-compat block."""
    return (wins + 1) / (wins + losses + 2)


# ── bucketing (v1-compatible definitions) ────────────────────────────────────
def vol_bucket(v: float) -> int:
    return 0 if v < 0.7 else (2 if v > 1.3 else 1)


def rsi_bucket(r: float) -> int:
    return 0 if r < 30 else 1 if r < 50 else 2 if r < 70 else 3


def session_bucket(hour: int) -> int:
    return 0 if hour < 7 else 1 if hour < 13 else 2 if hour < 21 else 3


def strategy_bucket(strategy_id: str) -> int:
    return _STRAT_INDEX.get(strategy_id, 0)


# ── node lifecycle / migration ───────────────────────────────────────────────
def empty_ml() -> Dict[str, Any]:
    """A fresh v2 node. ``state.ensure_strategy`` delegates here."""
    return {
        "version": ML_VERSION,
        # raw lifetime totals (never decayed — display / audit)
        "total_trades": 0, "wins": 0, "gross_profit": 0.0, "gross_loss": 0.0,
        # decayed evidence (what the models actually learn from)
        "dw": 0.0, "dl": 0.0,                       # decayed wins / losses (strategy level)
        "dims": {d: {} for d in ("vol", "rsi", "ses", "strat")},  # bucket → [dw, dl]
        "ftrl": {"z": {}, "n": {}},                 # FTRL-Proximal accumulators per feature
        "hedge": dict(HEDGE_INIT),
        "cal": {"a": 1.0, "b": 0.0, "n": 0},
        "metrics": {
            "n": 0, "logloss": 0.0, "brier": 0.0, "acc": 0.0, "base_logloss": 0.0,
            "bins": [[0, 0.0, 0.0] for _ in range(10)],   # [n, Σpred, Σwon] per decile
        },
        "drift": {"n": 0, "mean": 0.0, "cum": 0.0, "min_cum": 0.0, "events": 0, "last_at": None},
    }


def empty_shared() -> Dict[str, Any]:
    """The cross-strategy context memory, stored once in ``__global.ml_shared``.

    Every strategy's closed trade teaches it; every strategy's prediction can
    read it. It stores only strategy-agnostic evidence — context buckets
    (vol/RSI/session) and an FTRL residual model — never the ``strat`` dimension,
    so what it learns ("high funding kills mean-reversion entries") transfers."""
    return {
        "version": 1,
        "dw": 0.0, "dl": 0.0,
        "dims": {d: {} for d in ("vol", "rsi", "ses")},
        "ftrl": {"z": {}, "n": {}},
    }


def breakeven_p(strategy_id: str) -> Optional[float]:
    """Win probability needed to break even given the strategy's fixed R:R —
    ``sl/(sl+tp)`` in ATR units (fees excluded; they nudge it slightly higher).
    ``None`` for revert strategies, whose payoff is not fixed."""
    s = _STRAT_BY_ID.get(strategy_id)
    if not s or s.tp_atr is None or (s.sl_atr + s.tp_atr) <= 0:
        return None
    return s.sl_atr / (s.sl_atr + s.tp_atr)


def _prior_p(strategy_id: str) -> float:
    """Tempered prior win rate: average of the (in-sample) backtest number and a
    conservative live expectation, so a fresh node neither hypes nor sandbags."""
    s = _STRAT_BY_ID.get(strategy_id)
    bt = (s.backtest_win_rate / 100.0) if (s and s.backtest_win_rate) else TEMPER_TOWARD
    return _clamp(0.5 * bt + 0.5 * TEMPER_TOWARD, 0.50, 0.80)


def _ensure_v2(ml: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate a v1 node in place (idempotent). Raw totals are preserved; v1's
    ×2 stop-loss penalty is divided back out of the per-bucket loss counts so
    the migrated evidence is an honest win/loss record; the original v1 dims are
    retained under ``legacy_v1`` for audit (append-only, never wiped).

    The migration is build-then-swap: the v2 node is assembled completely from
    READS of the old node, and the old dict is only replaced at the very end —
    a corrupt field can therefore never half-destroy the original evidence. If
    the build itself fails, the raw node is preserved under ``legacy_raw``."""
    if isinstance(ml.get("version"), (int, float)) and ml["version"] >= ML_VERSION:
        hedge = ml.get("hedge")
        if isinstance(hedge, dict):        # heal experts added after this node was written
            healed = False
            for e in EXPERTS:              # (e.g. "pool") so they mix at their init share
                if e not in hedge:
                    hedge[e] = HEDGE_INIT.get(e, 0.1)
                    healed = True
            if healed:                     # keep the display invariant Σw = 1
                total = sum(max(0.0, _finite(w, 0.0)) for w in hedge.values()) or 1.0
                for e in hedge:
                    hedge[e] = max(0.0, _finite(hedge[e], 0.0)) / total
        return ml
    fresh = empty_ml()
    try:
        penalty = max(1.0, float(config.ML_PENALTY_SL))
        legacy: Dict[str, Any] = {}
        for dim in ("vol", "rsi", "ses", "strat"):
            old = ml.get(dim)
            if isinstance(old, dict) and old:
                legacy[dim] = old
                for bucket, wl in old.items():
                    try:
                        w, l = float(wl[0]), float(wl[1]) / penalty
                    except (TypeError, ValueError, IndexError, KeyError):
                        continue
                    if not (math.isfinite(w) and math.isfinite(l)) or w < 0 or l < 0:
                        continue
                    fresh["dims"][dim][str(bucket)] = [w, l]
        total = max(0, int(_finite(ml.get("total_trades", 0), 0)))
        wins = min(total, max(0, int(_finite(ml.get("wins", 0), 0))))
        fresh["total_trades"], fresh["wins"] = total, wins
        fresh["gross_profit"] = _finite(ml.get("gross_profit", 0.0), 0.0)
        fresh["gross_loss"] = _finite(ml.get("gross_loss", 0.0), 0.0)
        fresh["dw"], fresh["dl"] = float(wins), float(total - wins)
        if legacy:
            fresh["legacy_v1"] = legacy
    except Exception:  # noqa: BLE001 — a hopeless node still keeps its raw data
        fresh = empty_ml()
        fresh["total_trades"] = max(0, int(_finite(ml.get("total_trades", 0), 0)))
        fresh["wins"] = min(fresh["total_trades"], max(0, int(_finite(ml.get("wins", 0), 0))))
        fresh["legacy_raw"] = dict(ml)
    ml.clear()
    ml.update(fresh)
    return ml


# ── snapshot (entry-time context, stored on the trade) ───────────────────────
_STRETCH_SCALE = {"rsi2": 10.0, "stoch_k": 10.0}   # 0-100 oscillators; σ-unit sources scale 1.0


def build_snapshot(
    series: Dict[str, List[Optional[float]]],
    i: int,
    strategy_id: str,
    time_s: int,
    side: Optional[str] = None,
    trained_prob: Optional[float] = None,
    funding: Optional[float] = None,
) -> Dict[str, Any]:
    """Entry-time context: v1-compatible bucket ints plus scaled numeric features
    and the side-aware trained-model probability, all frozen for prequential
    learning at close time.

    v3 features target the strategies' documented failure mode ("a real trend
    breaking out of the range produces a cluster of losses"): higher-timeframe
    trend (5-day / 20-day return) and its interaction with trade side, the
    funding rate at entry, and how far the oscillator is stretched past its
    trigger. All fixed-scaled and clamped; absent data simply omits the key
    (the FTRL model is sparse, and missing features carry no weight)."""
    atr = series.get("atr", [])
    vol_regime = 1.0
    if i >= 50 and i < len(atr) and atr[i] and atr[i - 50]:
        vol_regime = atr[i] / atr[i - 50]
    rsi_arr = series.get("rsi14", [])
    rsi_v = rsi_arr[i] if (i < len(rsi_arr) and rsi_arr[i] is not None) else 50.0
    dt = datetime.fromtimestamp(time_s, tz=timezone.utc)
    hour, dow = dt.hour, dt.weekday()

    adx_arr = series.get("adx", [])
    adx_v = adx_arr[i] if (i < len(adx_arr) and adx_arr[i] is not None) else 25.0
    closes = series.get("closes", [])
    atr_pct = 2.0
    if i < len(atr) and i < len(closes) and atr[i] and closes[i]:
        atr_pct = atr[i] / closes[i] * 100.0

    side_v = 1.0 if side == "long" else (-1.0 if side == "short" else 0.0)
    x = {
        "rsi": _clamp((rsi_v - 50.0) / 50.0, -3, 3),
        "adx": _clamp(adx_v / 25.0 - 1.0, -3, 3),
        "atrp": _clamp(atr_pct / 2.0 - 1.0, -3, 3),
        "volr": _clamp(vol_regime - 1.0, -1.5, 1.5),
        "hs": math.sin(2 * math.pi * hour / 24), "hc": math.cos(2 * math.pi * hour / 24),
        "ds": math.sin(2 * math.pi * dow / 7), "dc": math.cos(2 * math.pi * dow / 7),
        "side": side_v,
        "side_rsi": _clamp(side_v * (rsi_v - 50.0) / 50.0, -3, 3),
    }
    # Higher-timeframe trend (6 bars/day on 4h): 5-day and 20-day returns, plus
    # the side interaction — fading INTO a strong trend is the known loss cluster.
    if i >= 30 and i < len(closes) and closes[i] and closes[i - 30]:
        t5 = _clamp((closes[i] / closes[i - 30] - 1.0) * 100.0 / 5.0, -3, 3)
        x["t5d"] = t5
        x["side_t5"] = _clamp(side_v * t5, -3, 3)
    if i >= 120 and i < len(closes) and closes[i] and closes[i - 120]:
        t20 = _clamp((closes[i] / closes[i - 120] - 1.0) * 100.0 / 15.0, -3, 3)
        x["t20d"] = t20
        x["side_t20"] = _clamp(side_v * t20, -3, 3)
    if funding is not None and isinstance(funding, (int, float)) and math.isfinite(funding):
        x["fund"] = _clamp(funding / 0.05, -3, 3)
        x["side_fund"] = _clamp(side_v * funding / 0.05, -3, 3)
    # Stretch: how far past its trigger the strategy's own oscillator sits.
    strat = _STRAT_BY_ID.get(strategy_id)
    if strat is not None and side in ("long", "short"):
        src = series.get(strat.source, [])
        val = src[i] if (i < len(src) and src[i] is not None) else None
        if val is not None:
            scale = _STRETCH_SCALE.get(strat.source, 1.0)
            beyond = (strat.long_below - val) if side == "long" else (val - strat.short_above)
            x["stretch"] = _clamp(beyond / scale, -1, 3)
    p_tr: Optional[float] = None
    if trained_prob is not None and side in ("long", "short"):
        p_tr = trained_prob if side == "long" else 1.0 - trained_prob
        p_tr = _clamp(p_tr, 0.0, 1.0)
    return {
        "v": 2,
        "vol_b": vol_bucket(vol_regime), "rsi_b": rsi_bucket(rsi_v),
        "ses_b": session_bucket(hour), "strat_b": strategy_bucket(strategy_id),
        "x": x, "p_tr": p_tr, "side": side, "sid": strategy_id, "t": time_s,
    }


def _ftrl_features(snap: Dict[str, Any], shared: bool = False) -> Dict[str, float]:
    """Sparse named features. The shared (cross-strategy) variant drops the
    ``strat`` one-hot — pooled knowledge must be strategy-agnostic."""
    feats: Dict[str, float] = {}
    dims = (("vol", "vol_b"), ("rsi", "rsi_b"), ("ses", "ses_b"))
    if not shared:
        dims = dims + (("strat", "strat_b"),)
    for dim, key in dims:
        b = snap.get(key)
        if b is not None:
            feats[f"{dim}={b}"] = 1.0
    for name, val in (snap.get("x") or {}).items():
        if isinstance(val, (int, float)) and math.isfinite(val):
            feats[name] = float(val)
    return feats


# ── expert predictions ───────────────────────────────────────────────────────
def _base_p(ml: Dict[str, Any], strategy_id: str) -> float:
    prior = _prior_p(strategy_id)
    dw = max(0.0, _finite(ml.get("dw", 0.0), 0.0))
    dl = max(0.0, _finite(ml.get("dl", 0.0), 0.0))
    return (dw + prior * PRIOR_STRENGTH) / (dw + dl + PRIOR_STRENGTH)


def _buckets_p(ml: Dict[str, Any], snap: Dict[str, Any], base: float) -> float:
    """Naive-Bayes-style log-odds combination with per-bucket empirical-Bayes
    shrinkage: each dimension votes (logit(p_bucket) − logit(base)), weighted by
    how much decayed evidence sits in that bucket."""
    base_logit = _logit(base)
    total = 0.0
    for dim, key in (("vol", "vol_b"), ("rsi", "rsi_b"), ("ses", "ses_b"), ("strat", "strat_b")):
        b = snap.get(key)
        if b is None:
            continue
        wl = ml.get("dims", {}).get(dim, {}).get(str(b))
        if not wl:
            continue
        w = max(0.0, _finite(wl[0], 0.0))
        l = max(0.0, _finite(wl[1], 0.0))
        n = w + l
        if n <= 0:
            continue
        p_b = (w + base * BUCKET_PRIOR_M) / (n + BUCKET_PRIOR_M)
        total += (n / (n + BUCKET_SHRINK_M)) * (_logit(p_b) - base_logit)
    return _sigmoid(base_logit + _clamp(total, -BUCKET_LOGIT_CAP, BUCKET_LOGIT_CAP))


def _ftrl_weight(z: float, n: float) -> float:
    """FTRL-Proximal closed-form weight for one coordinate."""
    if abs(z) <= FTRL_L1:
        return 0.0
    sign = -1.0 if z < 0 else 1.0
    return -(z - sign * FTRL_L1) / ((FTRL_BETA + math.sqrt(n)) / FTRL_ALPHA + FTRL_L2)


def _ftrl_p(ml: Dict[str, Any], snap: Dict[str, Any], base: float) -> float:
    f = ml.get("ftrl", {})
    zs, ns = f.get("z", {}), f.get("n", {})
    resid = sum(_ftrl_weight(_finite(zs.get(k, 0.0), 0.0), max(0.0, _finite(ns.get(k, 0.0), 0.0))) * v
                for k, v in _ftrl_features(snap).items())
    return _sigmoid(_logit(base) + _clamp(_finite(resid, 0.0), -FTRL_LOGIT_CAP, FTRL_LOGIT_CAP))


def _shared_base(shared: Dict[str, Any]) -> float:
    s_dw = max(0.0, _finite(shared.get("dw", 0.0), 0.0))
    s_dl = max(0.0, _finite(shared.get("dl", 0.0), 0.0))
    return (s_dw + TEMPER_TOWARD * PRIOR_STRENGTH) / (s_dw + s_dl + PRIOR_STRENGTH)


def _shared_bucket_delta(shared: Dict[str, Any], snap: Dict[str, Any], sb_logit: float) -> float:
    total = 0.0
    for dim, key in (("vol", "vol_b"), ("rsi", "rsi_b"), ("ses", "ses_b")):
        b = snap.get(key)
        if b is None:
            continue
        try:
            wl = shared.get("dims", {}).get(dim, {}).get(str(b))
            w = max(0.0, _finite(wl[0], 0.0)) if wl else 0.0
            l = max(0.0, _finite(wl[1], 0.0)) if wl else 0.0
        except (TypeError, IndexError, KeyError, AttributeError):
            continue                        # one corrupt bucket must not mute the rest
        n = w + l
        if n <= 0:
            continue
        base_p = _sigmoid(sb_logit)
        p_b = (w + base_p * BUCKET_PRIOR_M) / (n + BUCKET_PRIOR_M)
        total += (n / (n + BUCKET_SHRINK_M)) * (_logit(p_b) - sb_logit)
    return total


def _shared_ftrl_resid(shared: Dict[str, Any], feats: Dict[str, float]) -> float:
    f = shared.get("ftrl", {}) if isinstance(shared.get("ftrl"), dict) else {}
    zs, ns = f.get("z", {}), f.get("n", {})
    if not isinstance(zs, dict) or not isinstance(ns, dict):
        return 0.0
    return sum(_ftrl_weight(_finite(zs.get(k, 0.0), 0.0), max(0.0, _finite(ns.get(k, 0.0), 0.0))) * v
               for k, v in feats.items())


def _pool_p(shared: Dict[str, Any], snap: Dict[str, Any], base: float) -> Optional[float]:
    """Cross-strategy pool expert: the SHARED context evidence votes in log-odds
    DELTAS applied to THIS strategy's base rate — so what one strategy's trades
    teach ("this context loses") transfers to the others even though their base
    win rates differ. The shared FTRL is trained as a boosting residual ON TOP
    of the bucket deltas (see ``_shared_update``), so the two terms never
    double-count the same context. Returns ``None`` (asleep) when no shared
    node exists; a corrupt shared node puts the expert to sleep rather than
    degrading all five strategies."""
    if not isinstance(shared, dict) or not isinstance(shared.get("dims"), dict):
        return None
    try:
        sb_logit = _logit(_shared_base(shared))
        total = _shared_bucket_delta(shared, snap, sb_logit)
        total += _shared_ftrl_resid(shared, _ftrl_features(snap, shared=True))
        return _sigmoid(_logit(base) + _clamp(_finite(total, 0.0), -POOL_LOGIT_CAP, POOL_LOGIT_CAP))
    except Exception:  # noqa: BLE001 — blast radius of a bad shared node is ALL strategies
        return None


def _expert_preds(ml: Dict[str, Any], snap: Dict[str, Any],
                  shared: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Predictions from every awake expert (each guarded finite-in-(0,1)).
    ``trained`` sleeps when unstamped; ``pool`` sleeps without a shared node."""
    base = _finite(_base_p(ml, snap.get("sid", "")), _prior_p(snap.get("sid", "")))
    preds = {
        "base": base,
        "buckets": _finite(_buckets_p(ml, snap, base), base),
        "ftrl": _finite(_ftrl_p(ml, snap, base), base),
    }
    p_pool = _pool_p(shared, snap, base) if shared is not None else None
    if p_pool is not None:
        preds["pool"] = _finite(p_pool, base)
    p_tr = snap.get("p_tr")
    if isinstance(p_tr, (int, float)) and math.isfinite(p_tr) and 0.0 <= p_tr <= 1.0:
        preds["trained"] = _clamp(float(p_tr), PROB_FLOOR, PROB_CEIL)
    return preds


def _combine(ml: Dict[str, Any], preds: Dict[str, float]) -> float:
    """Hedge-weighted log-opinion pool over awake experts, then calibration.
    A missing weight defaults to the expert's INIT share (consistent with
    ``_hedge_update``) so a newly-introduced expert is never silently muted."""
    hedge = ml.get("hedge", {})
    def w(e):
        return max(0.0, _finite(hedge.get(e, HEDGE_INIT.get(e, 0.1)), HEDGE_INIT.get(e, 0.1)))
    wsum = sum(w(e) for e in preds) or 1.0
    mixed_logit = sum(w(e) * _logit(p) for e, p in preds.items()) / wsum
    cal = ml.get("cal", {"a": 1.0, "b": 0.0})
    a = _clamp(_finite(cal.get("a", 1.0), 1.0), 0.25, 4.0)
    b = _clamp(_finite(cal.get("b", 0.0), 0.0), -1.5, 1.5)
    return _clamp(_sigmoid(a * mixed_logit + b), PROB_FLOOR, PROB_CEIL)


def predict(ml: Dict[str, Any], snap: Dict[str, Any],
            shared: Optional[Dict[str, Any]] = None) -> float:
    """Calibrated P(this trade closes as a win).

    Freezes the per-expert predictions into ``snap["_pred"]`` so that when the
    trade closes, ``update`` can score exactly what was believed at entry —
    strictly prequential, no hindsight. Pass the global ``ml_shared`` node to
    wake the cross-strategy pool expert. Never raises; worst case returns the
    strategy prior."""
    try:
        _ensure_v2(ml)
        preds = _expert_preds(ml, snap, shared)
        p = _combine(ml, preds)
        if isinstance(snap, dict):
            snap["_pred"] = {"experts": {k: round(v, 6) for k, v in preds.items()},
                             "p": round(p, 6)}
        return p
    except Exception:  # noqa: BLE001 — a scoring bug must never block the engine
        return _prior_p(snap.get("sid", "") if isinstance(snap, dict) else "")


# ── learning ─────────────────────────────────────────────────────────────────
def _decay_counts(ml: Dict[str, Any], factor: float = _DECAY) -> None:
    ml["dw"] = ml.get("dw", 0.0) * factor
    ml["dl"] = ml.get("dl", 0.0) * factor
    for dim in ml.get("dims", {}).values():
        for wl in dim.values():
            wl[0] *= factor
            wl[1] *= factor


def _hedge_update(ml: Dict[str, Any], experts: Dict[str, float], won: bool) -> None:
    """Multiplicative-weights update over the AWAKE experts only (specialists
    framework): sleeping experts keep exactly their prior share, so an absent
    expert can neither earn nor ratchet up weight through renormalisation."""
    hedge = ml["hedge"]
    awake = [e for e in experts if e in EXPERTS]
    if not awake:
        return
    awake_mass = sum(hedge.get(e, HEDGE_INIT.get(e, 0.1)) for e in awake) or 1.0
    for e in awake:
        loss = min(_logloss(experts[e], won), HEDGE_LOSS_CLIP)
        hedge[e] = hedge.get(e, HEDGE_INIT.get(e, 0.1)) * math.exp(-HEDGE_ETA * loss)
    new_mass = sum(hedge[e] for e in awake) or 1.0
    scale = awake_mass / new_mass                 # awake mass is conserved
    for e in awake:
        hedge[e] *= scale
    floor = HEDGE_FLOOR * awake_mass              # floor is a share of the awake pool
    floor_total = 0.0
    for e in awake:                               # keep every awake expert alive
        if hedge[e] < floor:
            floor_total += floor - hedge[e]
            hedge[e] = floor
    if floor_total > 0:
        over = {e: hedge[e] for e in awake if hedge[e] > floor}
        over_sum = sum(over.values()) or 1.0
        for e in over:
            hedge[e] -= floor_total * over[e] / over_sum


def _ftrl_update(ml: Dict[str, Any], snap: Dict[str, Any], won: bool, base: float) -> None:
    f = ml["ftrl"]
    zs, ns = f["z"], f["n"]
    feats = _ftrl_features(snap)
    # gradient of log-loss at the *current* weights (standard FTRL bookkeeping)
    resid = sum(_ftrl_weight(zs.get(k, 0.0), ns.get(k, 0.0)) * v for k, v in feats.items())
    p = _sigmoid(_logit(base) + _clamp(resid, -FTRL_LOGIT_CAP, FTRL_LOGIT_CAP))
    g_scalar = p - (1.0 if won else 0.0)
    for k, v in feats.items():
        g = g_scalar * v
        n_old = ns.get(k, 0.0)
        w_old = _ftrl_weight(zs.get(k, 0.0), n_old)
        sigma = (math.sqrt(n_old + g * g) - math.sqrt(n_old)) / FTRL_ALPHA
        zs[k] = zs.get(k, 0.0) + g - sigma * w_old
        ns[k] = n_old + g * g


def _shared_update(shared: Dict[str, Any], snap: Dict[str, Any], won: bool) -> None:
    """Teach the cross-strategy pool: decay, bucket counts, and the shared FTRL
    trained as a BOOSTING RESIDUAL — its gradient is taken at the margin that
    already includes the bucket deltas, so it learns only what the buckets
    miss and the two terms never double-count the same context."""
    if not isinstance(shared, dict) or "dims" not in shared:
        return
    sb_logit = _logit(_shared_base(shared))
    f = shared.setdefault("ftrl", {"z": {}, "n": {}})
    zs, ns = f.setdefault("z", {}), f.setdefault("n", {})
    feats = _ftrl_features(snap, shared=True)
    margin = sb_logit + _shared_bucket_delta(shared, snap, sb_logit) + _shared_ftrl_resid(shared, feats)
    p = _sigmoid(_clamp(_finite(margin, sb_logit), sb_logit - POOL_LOGIT_CAP, sb_logit + POOL_LOGIT_CAP))
    g_scalar = p - (1.0 if won else 0.0)
    for k, v in feats.items():
        g = g_scalar * v
        n_old = max(0.0, _finite(ns.get(k, 0.0), 0.0))
        w_old = _ftrl_weight(_finite(zs.get(k, 0.0), 0.0), n_old)
        sigma = (math.sqrt(n_old + g * g) - math.sqrt(n_old)) / FTRL_ALPHA
        zs[k] = _finite(zs.get(k, 0.0), 0.0) + g - sigma * w_old
        ns[k] = n_old + g * g
    shared["dw"] = s_dw * _DECAY
    shared["dl"] = s_dl * _DECAY
    for dim in shared["dims"].values():
        for wl in dim.values():
            wl[0] *= _DECAY
            wl[1] *= _DECAY
    inc_w, inc_l = (1.0, 0.0) if won else (0.0, 1.0)
    shared["dw"] += inc_w
    shared["dl"] += inc_l
    for dim, key in (("vol", "vol_b"), ("rsi", "rsi_b"), ("ses", "ses_b")):
        b = snap.get(key)
        if not isinstance(b, int):
            continue
        wl = shared["dims"].setdefault(dim, {}).setdefault(str(b), [0.0, 0.0])
        wl[0] += inc_w
        wl[1] += inc_l


def _cal_update(ml: Dict[str, Any], mixed_logit: float, won: bool) -> None:
    cal = ml["cal"]
    a, b = cal.get("a", 1.0), cal.get("b", 0.0)
    p = _sigmoid(a * mixed_logit + b)
    err = p - (1.0 if won else 0.0)
    a -= CAL_LR * (err * mixed_logit + CAL_REG * (a - 1.0))
    b -= CAL_LR * (err + CAL_REG * b)
    cal["a"], cal["b"] = _clamp(a, 0.25, 4.0), _clamp(b, -1.5, 1.5)
    cal["n"] = cal.get("n", 0) + 1


def _metrics_update(ml: Dict[str, Any], p: float, p_base: float, won: bool) -> None:
    m = ml["metrics"]
    y = 1.0 if won else 0.0
    m["n"] += 1
    m["logloss"] += _logloss(p, won)
    m["brier"] += (p - y) ** 2
    m["acc"] += 1.0 if ((p >= 0.5) == won) else 0.0
    m["base_logloss"] += _logloss(p_base, won)
    b = min(int(p * 10), 9)
    m["bins"][b][0] += 1
    m["bins"][b][1] += p
    m["bins"][b][2] += y


def _drift_update(ml: Dict[str, Any], loss: float, when: Optional[int]) -> bool:
    """Page–Hinkley on the per-trade log-loss stream. Returns True on alarm."""
    d = ml["drift"]
    d["n"] += 1
    n = d["n"]
    d["mean"] += (loss - d["mean"]) / n
    d["cum"] += loss - d["mean"] - PH_DELTA
    d["min_cum"] = min(d["min_cum"], d["cum"])
    if n >= PH_MIN_N and (d["cum"] - d["min_cum"]) > PH_LAMBDA:
        d["events"] += 1
        d["last_at"] = when
        d["cum"] = d["min_cum"] = 0.0
        d["mean"] = loss
        d["n"] = 1
        return True
    return False


def update(
    ml: Dict[str, Any],
    snap: Dict[str, Any],
    won: bool,
    gross_profit: float = 0.0,
    gross_loss: float = 0.0,
    shared: Optional[Dict[str, Any]] = None,
) -> None:
    """Fold one closed trade back into every layer of the learner.

    The raw lifetime totals are booked FIRST, before anything that could throw,
    so a learning-layer bug can neither lose a close nor double-count it. Trades
    opened before the v2 upgrade carry a v1 snapshot with no frozen entry-time
    predictions; they still teach the buckets/FTRL/counts, but are skipped by
    the Hedge / calibration / metrics layers (no honest prequential sample
    exists for them). Never raises."""
    if not isinstance(ml, dict):
        return
    # 1) The close itself — exactly once, unconditionally.
    ml["total_trades"] = int(_finite(ml.get("total_trades", 0), 0)) + 1
    ml["wins"] = int(_finite(ml.get("wins", 0), 0)) + (1 if won else 0)
    ml["gross_profit"] = _finite(ml.get("gross_profit", 0.0), 0.0) + _finite(gross_profit, 0.0)
    ml["gross_loss"] = _finite(ml.get("gross_loss", 0.0), 0.0) + _finite(gross_loss, 0.0)
    # 2) Learning — best effort; a bug here must never surface into the engine.
    try:
        _ensure_v2(ml)
        snap = snap if isinstance(snap, dict) else {}
        sid = snap.get("sid", "")
        base_before = _base_p(ml, sid)

        pred = snap.get("_pred") or {}
        raw_experts = pred.get("experts") or {}
        experts = {e: _finite(p, -1.0) for e, p in raw_experts.items() if e in EXPERTS}
        experts = {e: p for e, p in experts.items() if 0.0 <= p <= 1.0}
        if experts:
            # prequential scoring first (with the entry-time beliefs)…
            hedge = ml["hedge"]
            wsum = sum(max(0.0, _finite(hedge.get(e, 0.0), 0.0)) for e in experts) or 1.0
            mixed_logit = sum(max(0.0, _finite(hedge.get(e, 0.0), 0.0)) * _logit(p)
                              for e, p in experts.items()) / wsum
            p_final = _finite(pred.get("p"), -1.0)
            if not (0.0 <= p_final <= 1.0):
                p_final = _sigmoid(mixed_logit)
            _metrics_update(ml, p_final, _finite(experts.get("base", base_before), base_before), won)
            drifted = _drift_update(ml, _logloss(p_final, won), snap.get("t"))
            # …then learning.
            _hedge_update(ml, experts, won)
            _cal_update(ml, mixed_logit, won)
            if drifted:
                _decay_counts(ml, DRIFT_EXTRA_DECAY)
                for e in experts:                   # soften only the AWAKE experts —
                    if e in ml["hedge"]:            # sleeping weights stay frozen
                        ml["hedge"][e] = 0.75 * ml["hedge"][e] + 0.25 * HEDGE_INIT.get(e, 0.1)

        _ftrl_update(ml, snap, won, base_before)
        if shared is not None:
            try:
                _shared_update(shared, snap, won)
            except Exception:  # noqa: BLE001 — pooled learning is additive, never fatal
                pass
        _decay_counts(ml)
        inc_w, inc_l = (1.0, 0.0) if won else (0.0, 1.0)
        ml["dw"] += inc_w
        ml["dl"] += inc_l
        for dim, key in (("vol", "vol_b"), ("rsi", "rsi_b"), ("ses", "ses_b"), ("strat", "strat_b")):
            b = snap.get(key)
            if not isinstance(b, int):
                continue
            wl = ml["dims"][dim].setdefault(str(b), [0.0, 0.0])
            wl[0] += inc_w
            wl[1] += inc_l
    except Exception:  # noqa: BLE001 — totals are already booked; skip the learning
        pass


# ── reporting ────────────────────────────────────────────────────────────────
def summary(ml: Dict[str, Any], strategy_id: str = "") -> Dict[str, Any]:
    """Honest per-strategy scoreboard for the dashboard blob."""
    _ensure_v2(ml)
    m = ml.get("metrics", {})
    n = m.get("n", 0)
    out: Dict[str, Any] = {
        "version": ML_VERSION,
        "p_win": round(_base_p(ml, strategy_id), 4),
        "n_eff": round(ml.get("dw", 0.0) + ml.get("dl", 0.0), 2),
        "weights": {e: round(w, 4) for e, w in ml.get("hedge", {}).items()},
        "calibration": {"a": round(ml.get("cal", {}).get("a", 1.0), 4),
                        "b": round(ml.get("cal", {}).get("b", 0.0), 4)},
        "drift_events": ml.get("drift", {}).get("events", 0),
        "scored_trades": n,
    }
    if n:
        out["logloss"] = round(m["logloss"] / n, 4)
        out["brier"] = round(m["brier"] / n, 4)
        out["accuracy"] = round(m["acc"] / n, 4)
        out["baseline_logloss"] = round(m["base_logloss"] / n, 4)
        out["edge_vs_baseline"] = round((m["base_logloss"] - m["logloss"]) / n, 4)
    return out


# ── Trained logistic model (matches ml_trainer.py features exactly) ──────────
def apply_trained_model(model: Optional[Dict[str, Any]], candles: Sequence[Candle], i: int, asset: str) -> Optional[float]:
    """Probability of an up-move over the trainer's horizon, or ``None``.

    Recomputes the same 12 features ``ml_trainer.build_features`` used, then applies
    the asset-specific standardised logistic weights.
    """
    if not model or i < 50:
        return None
    # No cross-asset fallback: BTC weights + BTC standardisation applied to ETH
    # features would be a confidently-wrong number. Absent model → no opinion.
    m = model.get(asset.lower())
    if not isinstance(m, dict) or "weights" not in m:
        return None
    closes = [c[C] for c in candles]
    highs = [c[H] for c in candles]
    lows = [c[L] for c in candles]
    vols = [c[V] for c in candles]
    if i < 12 or i >= len(closes):
        return None
    atr = ind.atr(highs, lows, closes, 14)
    rsi14 = ind.rsi(closes, 14)
    ema_f = ind.ema(closes, 12)
    ema_s = ind.ema(closes, 26)
    v_sma = ind.sma(vols, 20)
    v_std = ind.rolling_std(vols, 20, ddof=0)
    h50 = ind.sma(highs, 50)
    l50 = ind.sma(lows, 50)
    if any(arr[i] is None for arr in (atr, rsi14, ema_f, ema_s, v_sma, v_std, h50, l50)):
        return None
    if closes[i] <= 0 or v_sma[i] <= 0 or v_std[i] <= 0 or ema_s[i] == 0:
        return None
    feats = [
        (closes[i] / closes[i - 1] - 1) * 100,
        (closes[i] / closes[i - 3] - 1) * 100,
        (closes[i] / closes[i - 6] - 1) * 100,
        (closes[i] / closes[i - 12] - 1) * 100,
        (rsi14[i] - 50) / 50,
        atr[i] / closes[i] * 100,
        (ema_f[i] - ema_s[i]) / ema_s[i] * 100,
        (vols[i] - v_sma[i]) / v_std[i],
        math.sin(2 * math.pi * datetime.fromtimestamp(candles[i][0], tz=timezone.utc).hour / 24),
        math.cos(2 * math.pi * datetime.fromtimestamp(candles[i][0], tz=timezone.utc).hour / 24),
        (closes[i] - h50[i]) / closes[i] * 100,
        (closes[i] - l50[i]) / closes[i] * 100,
    ]
    means, stds, weights = m.get("means"), m.get("stds"), m.get("weights")
    bias = m.get("bias", 0.0)
    if not all(isinstance(v, list) and len(v) >= len(feats) for v in (means, stds, weights)):
        return None
    if any(stds[j] == 0 for j in range(len(feats))):
        return None
    z = bias + sum(weights[j] * (feats[j] - means[j]) / stds[j] for j in range(len(feats)))
    return _sigmoid(z)


__all__ = [
    "ML_VERSION", "EXPERTS",
    "vol_bucket", "rsi_bucket", "session_bucket", "strategy_bucket", "laplace",
    "empty_ml", "empty_shared", "breakeven_p",
    "build_snapshot", "predict", "update", "summary", "apply_trained_model",
]
