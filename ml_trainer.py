#!/usr/bin/env python3
"""
MiND-Shot ML Trainer v2 — walk-forward directional model (pure stdlib).

Trains the weekly "trained" expert: a logistic regression predicting direction
(up vs down over the next 4 bars) on 4h BTC/ETH candles. The engine applies it
via ``mind_shot.ml.apply_trained_model`` with the exact same 12 features, and
the online ensemble weighs it against the other experts by its live log-loss.

What v2 fixes over v1 (which trained on a single 70/30 split of at most 720
Kraken bars and once shipped an ETH model at 47% out-of-sample — worse than a
coin flip):

  • Data — merges the committed 4h fixtures (the playbook window) with live
    Kraken history, deduplicated by timestamp: several times more bars.
  • Validation — expanding-window walk-forward cross-validation (K chronological
    folds, train on everything before each fold, test on the fold), reporting
    mean out-of-sample accuracy / log-loss / Brier — not one lucky split.
  • Optimisation — mini-batch SGD with momentum, learning-rate decay, proper
    seeded shuffling, and early stopping on a validation tail.
  • Champion/challenger — the new weights are PUBLISHED only if their mean
    walk-forward log-loss beats the naive base-rate baseline; otherwise the
    previous published weights are kept and the failed attempt is recorded.
    A model worse than "always predict the base rate" never ships.
  • Persistence — merges into ``state/trained_model.json`` (never wipes other
    keys) and appends a capped metrics history for auditability.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mind_shot import config, indicators as ind
from mind_shot.history import load_history

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

MODEL_FILE = config.TRAINED_MODEL_FILE
FEATURES_V = 1                # must match mind_shot.ml.apply_trained_model exactly
HORIZON_BARS = 4
N_FOLDS = 5
SEED = 42
HISTORY_CAP = 24

KRAKEN_PAIR = {"BTC": "XBTUSDT", "ETH": "ETHUSDT"}


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


# ── data ─────────────────────────────────────────────────────────────────────
def fetch_kraken(pair: str, interval: int = 240) -> List[Tuple]:
    """Most recent ~720 bars from Kraken's public OHLC endpoint (best effort)."""
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read().decode())
    if d.get("error"):
        raise RuntimeError(f"Kraken: {d['error']}")
    res = d["result"]
    key = next(k for k in res if k != "last")
    return [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[6]))
            for r in res[key]]


def load_candles(asset: str) -> Tuple[List[Tuple], Dict[str, Any]]:
    """Committed fixtures + live Kraken, merged and deduplicated by timestamp.

    Kraken's final row is the still-forming candle — it is dropped so no label
    is ever computed from an unfinished bar. Merge seams (fixture→live venue
    switch, or a widening gap once the 720-bar Kraken window drifts past the
    fixture end) are detected and logged so a silent data hole can't masquerade
    as a clean training set."""
    candles = {c[0]: c for c in load_history(asset)}   # deep committed history ∪ fixtures
    n_fix = len(candles)
    n_live = 0
    try:
        live = fetch_kraken(KRAKEN_PAIR[asset])[:-1]     # drop the forming candle
        for c in live:
            candles[c[0]] = c
        n_live = len(live)
    except Exception as err:  # noqa: BLE001 — offline/CI runs train on fixtures alone
        log(f"  Kraken unavailable ({err}) — training on fixtures only")
    merged = [candles[t] for t in sorted(candles)]
    gaps = sum(1 for a, b in zip(merged, merged[1:]) if b[0] - a[0] > 2 * 4 * 3600)
    if gaps:
        log(f"  WARNING: {gaps} gap(s) > 8h in the merged {asset} series — "
            f"indicator windows spanning a gap are noisier")
    span_days = (merged[-1][0] - merged[0][0]) / 86400 if merged else 0
    return merged, {"fixture_bars": n_fix, "live_bars": n_live,
                    "merged_bars": len(merged), "span_days": round(span_days), "gaps_gt_8h": gaps}


# ── features (identical to mind_shot.ml.apply_trained_model) ─────────────────
def build_features(candles: Sequence[Tuple], horizon_bars: int = HORIZON_BARS) -> Tuple[List[List[float]], List[int]]:
    n = len(candles)
    if n < 100:
        return [], []
    closes = [c[4] for c in candles]
    highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]
    vols = [c[5] for c in candles]
    times = [c[0] for c in candles]

    atr = ind.atr(highs, lows, closes, 14)
    rsi14 = ind.rsi(closes, 14)
    ema_f = ind.ema(closes, 12)
    ema_s = ind.ema(closes, 26)
    v_sma = ind.sma(vols, 20)
    v_std = ind.rolling_std(vols, 20, ddof=0)
    h50 = ind.sma(highs, 50)
    l50 = ind.sma(lows, 50)

    X: List[List[float]] = []
    y: List[int] = []
    for i in range(50, n - horizon_bars):
        if any(a[i] is None for a in (atr, rsi14, ema_f, ema_s, v_sma, v_std, h50, l50)):
            continue
        if closes[i] <= 0 or v_sma[i] <= 0 or v_std[i] <= 0 or ema_s[i] == 0:
            continue
        hr = datetime.fromtimestamp(times[i], tz=timezone.utc).hour
        X.append([
            (closes[i] / closes[i - 1] - 1) * 100,
            (closes[i] / closes[i - 3] - 1) * 100,
            (closes[i] / closes[i - 6] - 1) * 100,
            (closes[i] / closes[i - 12] - 1) * 100,
            (rsi14[i] - 50) / 50,
            atr[i] / closes[i] * 100,
            (ema_f[i] - ema_s[i]) / ema_s[i] * 100,
            (vols[i] - v_sma[i]) / v_std[i],
            math.sin(2 * math.pi * hr / 24),
            math.cos(2 * math.pi * hr / 24),
            (closes[i] - h50[i]) / closes[i] * 100,
            (closes[i] - l50[i]) / closes[i] * 100,
        ])
        y.append(1 if closes[i + horizon_bars] > closes[i] else 0)
    return X, y


# ── logistic regression (mini-batch SGD, momentum, early stopping) ───────────
def sigmoid(z: float) -> float:
    if z > 30:
        return 1.0
    if z < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def standardize_fit(X: List[List[float]]) -> Tuple[List[float], List[float]]:
    k = len(X[0])
    means = [sum(r[j] for r in X) / len(X) for j in range(k)]
    stds = [math.sqrt(sum((r[j] - means[j]) ** 2 for r in X) / len(X)) or 1.0 for j in range(k)]
    return means, stds


def standardize_apply(X: List[List[float]], means: List[float], stds: List[float]) -> List[List[float]]:
    return [[(r[j] - means[j]) / stds[j] for j in range(len(r))] for r in X]


def logloss_of(probs: List[float], y: List[int]) -> float:
    return sum(-math.log(min(max(p if t else 1 - p, 1e-9), 1.0)) for p, t in zip(probs, y)) / max(len(y), 1)


def train_logistic(
    X: List[List[float]], y: List[int], *,
    epochs: int = 80, lr0: float = 0.08, l2: float = 1e-3,
    momentum: float = 0.9, batch: int = 32, patience: int = 8, seed: int = SEED,
) -> Tuple[List[float], float]:
    """Mini-batch SGD with momentum + lr decay, early-stopped on a chronological
    validation tail (last 15% of the training window — never future data)."""
    k = len(X[0])
    split = max(int(len(X) * 0.85), 1)
    # Purge the label horizon at the boundary: the last HORIZON_BARS training
    # labels are computed from closes INSIDE the validation window.
    purged = max(split - HORIZON_BARS, 1)
    Xt, yt, Xv, yv = X[:purged], y[:purged], X[split:], y[split:]
    base = sum(yt) / len(yt) if yt else 0.5
    w = [0.0] * k
    b = math.log(max(base, 1e-6) / max(1 - base, 1e-6))   # warm-start at base rate
    vw = [0.0] * k
    vb = 0.0
    rng = random.Random(seed)
    best = (float("inf"), list(w), b)
    stale = 0
    idx = list(range(len(Xt)))
    for epoch in range(epochs):
        rng.shuffle(idx)
        lr = lr0 / (1.0 + 0.05 * epoch)
        for s in range(0, len(idx), batch):
            gws = [0.0] * k
            gb = 0.0
            chunk = idx[s:s + batch]
            for i in chunk:
                p = sigmoid(sum(w[j] * Xt[i][j] for j in range(k)) + b)
                e = p - yt[i]
                for j in range(k):
                    gws[j] += e * Xt[i][j]
                gb += e
            m = len(chunk)
            for j in range(k):
                vw[j] = momentum * vw[j] - lr * (gws[j] / m + l2 * w[j])
                w[j] += vw[j]
            vb = momentum * vb - lr * (gb / m)
            b += vb
        if Xv:
            val = logloss_of([sigmoid(sum(w[j] * r[j] for j in range(k)) + b) for r in Xv], yv)
            if val < best[0] - 1e-5:
                best = (val, list(w), b)
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
    return (best[1], best[2]) if Xv else (w, b)


def predict_batch(X: List[List[float]], w: List[float], b: float) -> List[float]:
    k = len(w)
    return [sigmoid(sum(w[j] * r[j] for j in range(k)) + b) for r in X]


# ── walk-forward evaluation ──────────────────────────────────────────────────
def walk_forward(X: List[List[float]], y: List[int]) -> List[Dict[str, float]]:
    """Expanding-window CV: K chronological folds, train strictly on the past."""
    n = len(X)
    block = n // (N_FOLDS + 1)
    folds = []
    for f in range(1, N_FOLDS + 1):
        tr_end = block * f
        te_end = min(block * (f + 1), n)
        # Purge the label horizon: the last HORIZON_BARS train labels are
        # computed from closes inside the test fold — drop them (embargo).
        Xtr, ytr = X[:tr_end - HORIZON_BARS], y[:tr_end - HORIZON_BARS]
        Xte, yte = X[tr_end:te_end], y[tr_end:te_end]
        if len(Xtr) < 80 or len(Xte) < 20:
            continue
        means, stds = standardize_fit(Xtr)
        w, b = train_logistic(standardize_apply(Xtr, means, stds), ytr, seed=SEED + f)
        probs = predict_batch(standardize_apply(Xte, means, stds), w, b)
        base = sum(ytr) / len(ytr)
        acc = sum(1 for p, t in zip(probs, yte) if (p >= 0.5) == bool(t)) / len(yte)
        base_acc = max(sum(yte), len(yte) - sum(yte)) / len(yte)
        folds.append({
            "train": len(Xtr), "test": len(Xte),
            "acc": round(acc * 100, 2),
            "logloss": round(logloss_of(probs, yte), 4),
            "brier": round(sum((p - t) ** 2 for p, t in zip(probs, yte)) / len(yte), 4),
            "baseline_acc": round(base_acc * 100, 2),
            "baseline_logloss": round(logloss_of([base] * len(yte), yte), 4),
        })
    return folds


def train_asset(asset: str) -> Optional[Dict[str, Any]]:
    log(f"\n-- Training {asset} --")
    candles, data_info = load_candles(asset)
    log(f"  Data: {data_info['merged_bars']} bars over {data_info['span_days']} days "
        f"({data_info['fixture_bars']} fixture + {data_info['live_bars']} live)")
    X, y = build_features(candles)
    if len(X) < 200:
        log(f"  Not enough samples: {len(X)}")
        return None
    folds = walk_forward(X, y)
    if not folds:
        log("  Walk-forward produced no valid folds")
        return None
    mean = lambda k: sum(f[k] for f in folds) / len(folds)  # noqa: E731
    oos_acc, oos_ll, oos_br = mean("acc"), mean("logloss"), mean("brier")
    base_acc, base_ll = mean("baseline_acc"), mean("baseline_logloss")
    log(f"  Walk-forward ({len(folds)} folds): acc {oos_acc:.2f}% (base {base_acc:.2f}%) · "
        f"logloss {oos_ll:.4f} (base {base_ll:.4f}) · brier {oos_br:.4f}")

    beats_baseline = oos_ll < base_ll
    log(f"  Challenger {'BEATS' if beats_baseline else 'DOES NOT BEAT'} the base-rate baseline")

    # final weights: fit on all data (validation numbers above stay the honest report)
    means, stds = standardize_fit(X)
    w, b = train_logistic(standardize_apply(X, means, stds), y)
    return {
        "weights": w, "bias": b, "means": means, "stds": stds,
        "train_samples": len(X), "horizon_bars": HORIZON_BARS, "features_v": FEATURES_V,
        "folds": folds,
        "oos_acc": round(oos_acc, 2), "oos_logloss": round(oos_ll, 4), "oos_brier": round(oos_br, 4),
        "baseline_acc": round(base_acc, 2), "baseline_logloss": round(base_ll, 4),
        "edge": round(oos_acc - base_acc, 2),
        "beats_baseline": beats_baseline,
        "data": data_info,
    }


def main() -> None:
    log("MiND-Shot ML Trainer v2 - expanding-window walk-forward logistic on 4h data")
    log("Honesty mandate: metrics below are out-of-sample means, and a challenger that")
    log("cannot beat the base-rate baseline is recorded but NOT published.")
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    previous: Dict[str, Any] = {}
    if MODEL_FILE.exists():
        try:
            previous = json.loads(MODEL_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous = {}

    result = dict(previous)  # merge, never wipe
    result["trained_at"] = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    result["trainer_version"] = 2
    history = result.get("history") or []
    run_record: Dict[str, Any] = {"at": result["trained_at"]}

    for asset in ("BTC", "ETH"):
        key = asset.lower()
        try:
            challenger = train_asset(asset)
        except Exception as err:  # noqa: BLE001
            log(f"{asset} training failed: {err}")
            run_record[key] = {"error": str(err)}
            continue
        if challenger is None:
            run_record[key] = {"error": "insufficient data"}
            continue
        run_record[key] = {k: challenger[k] for k in
                           ("oos_acc", "oos_logloss", "baseline_logloss", "edge", "beats_baseline", "train_samples")}
        incumbent = previous.get(key) if isinstance(previous.get(key), dict) else None
        if challenger["beats_baseline"]:
            challenger["published"] = True
            result[key] = challenger
            log(f"  {asset}: published new weights")
        elif incumbent and incumbent.get("weights") and incumbent.get("beats_baseline"):
            # Keep a champion only if IT proved its own edge (v2 discipline). The
            # rejected challenger's numbers are recorded, never served as champion's.
            incumbent["challenger_rejected"] = run_record[key]
            result[key] = incumbent
            log(f"  {asset}: challenger rejected - keeping the proven incumbent")
        else:
            # No model — incumbent either doesn't exist or never proved an edge
            # (e.g. the v1 single-split model). Serving nothing is the honest
            # move: the engine's trained expert simply sleeps.
            retired = {k: v for k, v in (incumbent or {}).items()
                       if k not in ("weights", "bias", "means", "stds")}
            retired.update({"published": False,
                            "retired_reason": "no model beats the base-rate baseline",
                            "last_challenger": run_record[key]})
            result[key] = retired
            log(f"  {asset}: NO model published - nothing beats the base-rate baseline")
        served = result[key]
        # Engine-blob back-compat keys always describe the SERVED model (or None).
        result[f"{key}_oos_wr"] = served.get("oos_acc") if served.get("weights") else None
        result[f"{key}_samples"] = served.get("train_samples") if served.get("weights") else None

    history.append(run_record)
    result["history"] = history[-HISTORY_CAP:]
    _atomic_write(MODEL_FILE, json.dumps(result, indent=2))
    log(f"\nModel saved to {MODEL_FILE}")

    sys.stdout.write("<<<MINDSHOT_TRAINING>>>")
    sys.stdout.write(json.dumps({"ok": True, "stats": result}, default=str))
    sys.stdout.write("<<</MINDSHOT_TRAINING>>>\n")
    sys.stdout.flush()


def _atomic_write(path: Path, text: str) -> None:
    """Write via temp file + rename — a crash mid-write must never wipe the
    champion weights or the training history (append-only policy)."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".model-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


if __name__ == "__main__":
    main()
