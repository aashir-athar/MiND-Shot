"""
Engine orchestration — one poll cycle, v2.

Pipeline (order matters and is deliberate):

  1. Load state + the weekly trained model.
  2. Fetch market context FIRST (best-effort) — it powers the funding-pause risk
     gate at entry time, not just the dashboard. A context outage silently
     disables that one gate; it never blocks trading.
  3. Fetch candles once per active (asset, timeframe) pair and compute series.
  4. Run every strategy: advance its open trade (prompt intrabar SL/TP
     detection), then look for a fresh entry behind the risk gates and the ML
     veto. Every closed trade is folded back into the self-learning ML.
  5. Persist state IMMEDIATELY — trade lifecycle must never depend on the
     best-effort enrichment below succeeding.
  6. Fetch whale flow (best-effort), compute verdicts and analytics, and emit
     the dashboard blob — the same schema as v1 so the Electron host and any
     webhook consumers keep working unchanged.

Risk gates enforced at entry (first blocking reason wins):
  daily loss limit · max concurrent trades · post-SL cool-down ·
  funding-pause threshold (new in v2 — previously documented but never enforced).
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from . import config, context, intelligence, market, ml, notifier, whale
from .market import TF_MIN
from .models import C, T
from .state import (
    GLOBAL_KEY,
    ensure_global,
    ensure_strategy,
    load_state,
    load_trained_model,
    save_state,
)
from .strategies import STRATEGIES, Strategy, active_pairs, compute_series
from .trading import Trade, manage_trade, open_trade

log = logging.getLogger("mind_shot.engine")

WARMUP_BARS = 60          # bars of history before a signal is trusted
KLINE_LIMIT = 720         # Kraken's hard cap per OHLC request
PRICE_SAMPLE_BARS = 48    # closes shipped to the dashboard sparkline

Pair = Tuple[str, str]    # (asset, timeframe)


# ── risk gates ───────────────────────────────────────────────────────────────
def _risk_block(
    strategy: Strategy,
    state: Dict[str, Any],
    gs: Dict[str, Any],
    funding: Optional[float] = None,
) -> Optional[str]:
    """Return a human-readable reason to suppress a fresh entry, or ``None``.

    Order: daily loss limit → concurrency cap → post-SL cool-down → funding pause.
    """
    today_r = intelligence.aggregate_periods(gs["daily_r"])["today"]
    if today_r <= gs.get("daily_loss_limit_r", -3.0):
        return f"daily loss limit ({today_r:.2f}R)"

    open_count = sum(
        1 for k, v in state.items()
        if k != GLOBAL_KEY and isinstance(v, dict) and v.get("active_trade")
    )
    max_open = gs.get("max_concurrent_trades", 4)
    if open_count >= max_open:
        return f"{open_count} trades open (max {max_open})"

    last_sl = gs.get("sl_cooldowns", {}).get(strategy.id, 0)
    if last_sl > 0:
        cd_min = gs.get("cooldown_minutes_after_sl", 240)
        elapsed = (datetime.now(tz=timezone.utc).timestamp() - last_sl) / 60
        if elapsed < cd_min:
            return f"cool-down after SL ({int(cd_min - elapsed)}min left)"

    threshold = gs.get("funding_pause_threshold")
    if funding is not None and threshold and abs(funding) > threshold:
        return f"funding {funding:+.3f}% beyond ±{threshold:.2f}% (market overheated)"
    return None


# ── close bookkeeping ────────────────────────────────────────────────────────
def _record_close(gs: Dict[str, Any], trade: Trade, info: Dict[str, Any]) -> None:
    """Streaks, cool-downs, daily R, heatmap, and the capped journal."""
    won = info["won"]
    if won:
        gs["cur_win_streak"] += 1
        gs["cur_loss_streak"] = 0
        gs["max_win_streak"] = max(gs["max_win_streak"], gs["cur_win_streak"])
    else:
        gs["cur_loss_streak"] += 1
        gs["cur_win_streak"] = 0
        gs["max_loss_streak"] = max(gs["max_loss_streak"], gs["cur_loss_streak"])
        gs.setdefault("sl_cooldowns", {})[trade.strategy_id] = datetime.now(tz=timezone.utc).timestamp()
    intelligence.update_period_stats(gs["daily_r"], trade.opened_bar, info["pnl_r"])
    intelligence.update_heatmap(gs["heatmap"], trade.opened_bar, won)
    gs["journal"].append({
        "strategy": trade.strategy_id, "asset": trade.asset, "tf": trade.tf, "side": trade.side,
        "entry": trade.entry, "exit": info["exit_price"], "kind": info["kind"],
        "pnl_r": info["pnl_r"], "won": won, "opened_at": trade.opened_bar, "closed_at": trade.last_bar,
    })
    if len(gs["journal"]) > config.JOURNAL_CAP:
        gs["journal"] = gs["journal"][-config.JOURNAL_CAP:]


# ── per-strategy processing ──────────────────────────────────────────────────
def _base_result(strategy: Strategy, candles: List) -> Dict[str, Any]:
    n = len(candles)
    sample = candles[-PRICE_SAMPLE_BARS:] if n >= PRICE_SAMPLE_BARS else candles
    return {
        "strategy": strategy.id, "strategy_name": strategy.name,
        "asset": strategy.asset, "tf": strategy.timeframe,
        "events": [], "new_entry": None, "active_trade": None, "trade_closed": False,
        "distance_to_signal": None,
        "candle_close_at": candles[-1][T] * 1000 + TF_MIN[strategy.timeframe] * 60 * 1000,
        "last_close": candles[-1][C],
        "price_24h": [round(c[C], 6) for c in sample],
    }


def _advance_open_trade(
    strategy: Strategy,
    st: Dict[str, Any],
    candles: List,
    series: Dict[str, List],
    gs: Dict[str, Any],
    res: Dict[str, Any],
) -> None:
    """Walk the open trade over any new bars; alert, close, and teach the ML."""
    trade = Trade.from_dict(st["active_trade"])
    events, closed, info = manage_trade(trade, strategy, candles, series)
    for ev in events:
        res["events"].append(ev)
        payload, text = notifier.event_alert(ev, trade, strategy)
        notifier.deliver(payload, text)
    if closed and info:
        ml.update(st["ml"], trade.ml_snap, info["won"], info["gross_profit"], info["gross_loss"],
                  shared=gs.setdefault("ml_shared", ml.empty_shared()))
        _record_close(gs, trade, info)
        st["active_trade"] = None
        res["trade_closed"] = True
        res["closed_outcome"] = {"won": info["won"], "pnl_r": info["pnl_r"], "kind": info["kind"]}
        log.info("[%s] CLOSED %s: %s %.3fR", strategy.id, info["kind"].upper(),
                 "WIN" if info["won"] else "LOSS", info["pnl_r"])
    else:
        trade.update_live(candles[-1][C], config.LEVERAGE, int(datetime.now(tz=timezone.utc).timestamp()))
        st["active_trade"] = trade.to_dict()
        res["active_trade"] = st["active_trade"]


def _maybe_enter(
    strategy: Strategy,
    st: Dict[str, Any],
    candles: List,
    series: Dict[str, List],
    state: Dict[str, Any],
    gs: Dict[str, Any],
    res: Dict[str, Any],
    trained_model: Optional[Dict[str, Any]],
    funding: Optional[float],
) -> None:
    """Evaluate the last CLOSED bar for a fresh signal, behind every gate."""
    confirm_idx = len(candles) - 2
    side = strategy.entry(series, confirm_idx)
    if side is None:
        st["last_signal_bar"] = candles[confirm_idx][T]
        return

    try:
        trained_prob = ml.apply_trained_model(trained_model, candles, confirm_idx, strategy.asset)
    except Exception:  # noqa: BLE001 — advisory input only
        trained_prob = None
    snap = ml.build_snapshot(series, confirm_idx, strategy.id, candles[confirm_idx][T],
                             side=side.value, trained_prob=trained_prob, funding=funding)
    ml_conf = ml.predict(st["ml"], snap, shared=gs.setdefault("ml_shared", ml.empty_shared()))

    blocked = _risk_block(strategy, state, gs, funding)
    ml_trades = st["ml"].get("total_trades", 0)
    if blocked:
        res["blocked_by"] = blocked
        log.info("[%s] %s blocked: %s", strategy.id, side.value, blocked)
    elif config.ML_GATING_ENABLED and ml_trades >= config.ML_MIN_TRADES and ml_conf < config.ML_MIN_CONF:
        res["blocked_by"] = f"ML conf {ml_conf*100:.1f}% < {config.ML_MIN_CONF*100:.0f}%"
        log.info("[%s] %s blocked by ML (%.1f%%)", strategy.id, side.value, ml_conf * 100)
    else:
        trade = open_trade(strategy, candles, confirm_idx, series, snap)
        if trade is not None:
            st["active_trade"] = trade.to_dict()
            res["active_trade"] = st["active_trade"]
            breakeven = ml.breakeven_p(strategy.id)
            res["new_entry"] = {"side": trade.side, "entry": trade.entry, "ml_conf": round(ml_conf, 4),
                                "breakeven": round(breakeven, 4) if breakeven else None,
                                "ev_ok": (ml_conf > breakeven) if breakeven else None}
            payload, text = notifier.entry_alert(trade, strategy, ml_conf,
                                                 adx=series["adx"][confirm_idx], breakeven=breakeven)
            notifier.deliver(payload, text)
            log.info("[%s] NEW %s @ %s  ml=%.1f%%", strategy.id, trade.side.upper(),
                     notifier.fmt(trade.entry), ml_conf * 100)
    st["last_signal_bar"] = candles[confirm_idx][T]


def process_strategy(
    strategy: Strategy,
    candles: List,
    series: Dict[str, List],
    state: Dict[str, Any],
    gs: Dict[str, Any],
    results: List[Dict[str, Any]],
    ml_summary: Dict[str, Any],
    trained_model: Optional[Dict[str, Any]] = None,
    funding: Optional[float] = None,
) -> None:
    st = ensure_strategy(state, strategy.id)
    res = _base_result(strategy, candles)
    confirm_idx = len(candles) - 2
    if confirm_idx >= 0:
        res["distance_to_signal"] = intelligence.distance_to_signal(strategy, series, confirm_idx)

    if st["active_trade"]:
        _advance_open_trade(strategy, st, candles, series, gs, res)

    fresh_bar = confirm_idx >= WARMUP_BARS and candles[confirm_idx][T] > st["last_signal_bar"]
    if st["active_trade"] is None and fresh_bar:
        _maybe_enter(strategy, st, candles, series, state, gs, res, trained_model, funding)

    mlst = st["ml"]
    if mlst.get("total_trades", 0) > 0:
        v2 = ml.summary(mlst, strategy.id)
        ml_summary[strategy.id] = {
            "total_trades": mlst["total_trades"], "wins": mlst["wins"],
            "wr": mlst["wins"] / mlst["total_trades"],
            "conf": v2["p_win"],           # calibrated-prior P(win), replaces raw laplace
            "v2": v2,
        }
    results.append(res)


# ── the poll ─────────────────────────────────────────────────────────────────
def _fetch_pairs(pairs: List[Pair]) -> Tuple[Dict[Pair, List], Dict[Pair, Dict]]:
    pair_candles: Dict[Pair, List] = {}
    pair_series: Dict[Pair, Dict] = {}
    for pair in pairs:
        try:
            candles = market.fetch_klines(pair[0], pair[1], limit=KLINE_LIMIT)
            pair_candles[pair] = candles
            pair_series[pair] = compute_series(candles)
        except Exception as err:  # noqa: BLE001 — one dead feed must not sink the poll
            log.error("fetch %s %s failed: %s", pair[0], pair[1], err)
    return pair_candles, pair_series


def run_one_poll() -> Dict[str, Any]:
    started = time.monotonic()
    state = load_state()
    gs = ensure_global(state)
    trained_model = load_trained_model()

    # Context BEFORE trading: it feeds the funding-pause gate. Outage → gate off.
    try:
        market_ctx = context.fetch_market_context()
    except Exception as err:  # noqa: BLE001
        log.warning("market context unavailable (funding gate off this poll): %s", err)
        market_ctx = {}

    pair_candles, pair_series = _fetch_pairs(active_pairs())

    results: List[Dict[str, Any]] = []
    ml_summary: Dict[str, Any] = {}
    for strategy in STRATEGIES:
        candles = pair_candles.get(strategy.pair)
        series = pair_series.get(strategy.pair)
        if not candles or not series:
            results.append({"strategy": strategy.id, "asset": strategy.asset, "tf": strategy.timeframe,
                            "error": "data unavailable", "events": [], "price_24h": []})
            continue
        funding = market_ctx.get(f"{strategy.asset.lower()}_funding")
        try:
            process_strategy(strategy, candles, series, state, gs, results, ml_summary,
                             trained_model, funding)
        except Exception as err:  # noqa: BLE001 — one bad strategy must not sink the poll
            log.exception("strategy %s errored: %s", strategy.id, err)

    # Persist trade-lifecycle state NOW — it must never depend on the best-effort
    # enrichment below succeeding (a whale-flow hiccup must not roll back trades).
    save_state(state)

    try:
        whale_ctx = whale.fetch_whale_flow()
    except Exception as err:  # noqa: BLE001
        log.warning("whale flow unavailable: %s", err)
        whale_ctx = {"BTC": {}, "ETH": {}, "whale_alerts": [], "net_signal": {}}

    verdicts: Dict[str, Any] = {}
    hour = datetime.now(tz=timezone.utc).hour
    for r in results:
        if r.get("error"):
            continue
        asset = r["asset"]
        candles = pair_candles.get((asset, r["tf"]))
        ml_conf = (ml_summary.get(r["strategy"]) or {}).get("conf")
        whale_sig = (whale_ctx.get("net_signal") or {}).get(asset)
        funding = market_ctx.get(f"{asset.lower()}_funding")
        try:
            trained_prob = ml.apply_trained_model(trained_model, candles, len(candles) - 2, asset) if candles else None
        except Exception:  # noqa: BLE001 — advisory input; a corrupt model must not sink the blob
            trained_prob = None
        verdicts[r["strategy"]] = intelligence.compute_trade_verdict(
            asset, ml_conf, whale_sig, funding, None, hour, trained_prob
        )

    weekly = intelligence.compute_weekly_summary(gs)
    correlation = intelligence.compute_correlation(results)
    strat_stats = intelligence.compute_strategy_stats(gs["journal"], [s.id for s in STRATEGIES])

    blob = _build_blob(results, ml_summary, market_ctx, whale_ctx, strat_stats, verdicts,
                       weekly, correlation, trained_model, gs)
    opened = sum(1 for r in results if r.get("new_entry"))
    closed = sum(1 for r in results if r.get("trade_closed"))
    blocked = sum(1 for r in results if r.get("blocked_by"))
    errors = sum(1 for r in results if r.get("error"))
    log.info("Poll done in %.1fs — %d opened · %d closed · %d blocked · %d feed errors",
             time.monotonic() - started, opened, closed, blocked, errors)
    if config.OUTPUT_JSON:
        sys.stdout.write("<<<MINDSHOT_JSON>>>")
        sys.stdout.write(json.dumps(blob))
        sys.stdout.write("<<</MINDSHOT_JSON>>>\n")
        sys.stdout.flush()
    return blob


def _build_blob(results, ml_summary, market_ctx, whale_ctx, strat_stats, verdicts,
                weekly, correlation, trained_model, gs) -> Dict[str, Any]:
    return {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "polls": results,
        "ml_summary": ml_summary,
        "market": market_ctx,
        "whale": whale_ctx,
        "strategy_stats": strat_stats,
        "tp_accuracy": strat_stats,  # kept for dashboard back-compat
        "verdicts": verdicts,
        "weekly_summary": weekly,
        "correlation": correlation,
        "trained_model_summary": {
            "btc_oos_wr": (trained_model or {}).get("btc_oos_wr"),
            "eth_oos_wr": (trained_model or {}).get("eth_oos_wr"),
            "trained_at": (trained_model or {}).get("trained_at"),
        } if trained_model else None,
        "risk_controls": {
            "daily_loss_limit_r": gs.get("daily_loss_limit_r"),
            "max_concurrent_trades": gs.get("max_concurrent_trades"),
            "cooldown_minutes_after_sl": gs.get("cooldown_minutes_after_sl"),
            "funding_pause_threshold": gs.get("funding_pause_threshold"),
            "risk_per_trade_pct": gs.get("risk_per_trade_pct"),
            "paper_mode": gs.get("paper_mode"),
            "sl_cooldowns": gs.get("sl_cooldowns", {}),
        },
        "periods": intelligence.aggregate_periods(gs["daily_r"]),
        "streaks": {
            "cur_win": gs["cur_win_streak"], "cur_loss": gs["cur_loss_streak"],
            "max_win": gs["max_win_streak"], "max_loss": gs["max_loss_streak"],
        },
        "heatmap": gs["heatmap"],
        "daily_r": gs["daily_r"],
        "journal": gs["journal"][-30:],
        "account_size": gs.get("account_size", config.ACCOUNT_USD),
        "leverage": config.LEVERAGE,
        "strategies": [
            {"id": s.id, "name": s.name, "asset": s.asset, "tf": s.timeframe,
             "backtest_win_rate": s.backtest_win_rate, "description": s.description}
            for s in STRATEGIES
        ],
    }


def main() -> None:
    config.configure_logging()
    if config.TEST_ALERT:
        payload, text = notifier.sample_alert()
        ok = notifier.deliver(payload, text)
        log.info("Sample alert: %s (%s)", "delivered" if ok else "not delivered", config.delivery_label())
        return
    log.info("Delivery: %s", config.delivery_label())
    log.info("MiND-Shot Engine %s — %s", _version(), datetime.now(tz=timezone.utc).isoformat())
    log.info("Strategies: %d  ·  pairs: %s  ·  polls/run: %d  ·  interval: %ds",
             len(STRATEGIES), active_pairs(), config.POLLS_PER_RUN, config.POLL_INTERVAL_SEC)
    for poll_num in range(1, config.POLLS_PER_RUN + 1):
        log.info("── Poll %d/%d @ %s UTC ──", poll_num, config.POLLS_PER_RUN,
                 datetime.now(tz=timezone.utc).strftime("%H:%M:%S"))
        try:
            run_one_poll()
        except Exception as err:  # noqa: BLE001
            log.exception("poll error: %s", err)
        if poll_num < config.POLLS_PER_RUN:
            time.sleep(config.POLL_INTERVAL_SEC)
    log.info("Done.")


def _version() -> str:
    from . import __version__
    return __version__


__all__ = ["run_one_poll", "main", "process_strategy"]
