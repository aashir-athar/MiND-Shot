"""
Delivery — webhook (Make.com / n8n / Pipedream) or the direct Telegram Bot API,
plus the HTML alert formatting.

If ``WEBHOOK_URL`` is set it wins; otherwise a ``TG_TOKEN`` + ``TG_CHAT_ID`` pair
posts straight to Telegram. With neither, :func:`deliver` is a no-op (dry run).

Alerts are sized from ``ACCOUNT_USD`` / ``ALLOC_PCT`` / ``LEVERAGE`` so every
message shows the real dollar risk and reward for the configured account.
"""
from __future__ import annotations

import html
import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, Tuple

from . import config
from .models import Side
from .strategies import STRATEGY_BY_ID, Strategy
from .trading import Trade

log = logging.getLogger("mind_shot.notifier")

_TIMEOUT = 15.0
_RULE = "━━━━━━━━━━━━━━━━━━━━━━━"


def deliver(payload: Dict[str, Any], fallback_text: str) -> bool:
    """Send ``payload`` to the configured channel. Never raises."""
    if config.WEBHOOK_URL:
        return _post_json(config.WEBHOOK_URL, payload)
    if config.TG_TOKEN and config.TG_CHAT_ID:
        url = f"https://api.telegram.org/bot{config.TG_TOKEN}/sendMessage"
        body = {
            "chat_id": config.TG_CHAT_ID,
            "text": fallback_text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        return _post_json(url, body)
    log.debug("dry-run: %s", payload.get("type"))
    return False


def _post_json(url: str, body: Dict[str, Any]) -> bool:
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as err:
        log.warning("delivery failed: %s", err)
        return False


def fmt(price: float) -> str:
    return f"{price:,.2f}" if price >= 1000 else f"{price:,.4f}"


def _esc(s: object) -> str:
    """Escape data before it lands inside an HTML-parse-mode Telegram message.

    Only ``<`` ``>`` ``&`` need escaping; leave quotes alone. Prevents free-text
    like a strategy description ("%K(14) <20 / >80") from being read as a bogus
    HTML tag. Wrap DATA values only — never the template's own <b>/<i>/<code> tags.
    """
    return html.escape(str(s), quote=False)


def _sizing() -> Tuple[float, float]:
    """(margin, notional) in USD for the configured account."""
    margin = config.ACCOUNT_USD * config.ALLOC_PCT / 100.0
    return margin, margin * config.LEVERAGE


def _leg(entry: float, level: float) -> Tuple[float, float, float]:
    """(price_move_%, usd_on_notional, %_of_account) between entry and a level."""
    _, notional = _sizing()
    move_pct = abs(level - entry) / entry * 100.0 if entry else 0.0
    usd = notional * move_pct / 100.0
    acct_pct = (usd / config.ACCOUNT_USD * 100.0) if config.ACCOUNT_USD else 0.0
    return move_pct, usd, acct_pct


def entry_alert(trade: Trade, strategy: Strategy, ml_conf: float, adx: float | None = None) -> Tuple[Dict[str, Any], str]:
    """Build the (payload, html_text) pair for a new entry."""
    lev = config.LEVERAGE
    margin, notional = _sizing()
    is_long = trade.side == Side.LONG.value
    dot = "🟢" if is_long else "🔴"
    side_word = "LONG" if is_long else "SHORT"

    sl_move, sl_usd, sl_acct = _leg(trade.entry, trade.sl)
    wr = f"🏆 <i>{strategy.backtest_win_rate:.0f}% backtest</i>" if strategy.backtest_win_rate else ""
    adx_str = f"   ·   📐 ADX <b>{adx:.0f}</b> (range ✓)" if adx is not None else ""

    lines = [
        f"{dot} <b>{side_word}</b>   ·   <code>{_esc(trade.asset)}/USD</code>   ·   <b>{_esc(trade.tf)}</b>   ·   ⚡<code>{lev}×</code>",
        _RULE,
        f"🧩 <b>{_esc(strategy.name)}</b>   {wr}".rstrip(),
        f"🧠 ML <b>{ml_conf * 100:.0f}%</b>{adx_str}",
        "",
        f"📍 <b>Entry</b>    <code>{fmt(trade.entry)}</code>",
        f"🛡 <b>Stop</b>     <code>{fmt(trade.sl)}</code>   <b>−{sl_move:.2f}%</b>   ·   −${sl_usd:.2f}",
    ]

    if trade.tp is not None:
        tp_move, tp_usd, tp_acct = _leg(trade.entry, trade.tp)
        lines.append(f"🎯 <b>Target</b>   <code>{fmt(trade.tp)}</code>   <b>+{tp_move:.2f}%</b>   ·   +${tp_usd:.2f}")
        lines += [
            _RULE,
            f"💰 <b>Size</b>   ${margin:.0f} margin → ${notional:.0f} notional  ({config.ALLOC_PCT:.0f}% · {lev}×)",
            f"📊 <b>If hit</b>   🎯 +${tp_usd:.2f} (+{tp_acct:.1f}%)    🛡 −${sl_usd:.2f} (−{sl_acct:.1f}%)",
        ]
    else:
        anchor = "VWAP" if strategy.mean_price_key == "vwap" else "the mean"
        tgt = fmt(trade.target) if trade.target else "—"
        lines.append(f"🎯 <b>Exit</b>     ride back to {anchor} ≈ <code>{tgt}</code>  <i>(dynamic)</i>")
        lines += [
            _RULE,
            f"💰 <b>Size</b>   ${margin:.0f} margin → ${notional:.0f} notional  ({config.ALLOC_PCT:.0f}% · {lev}×)",
            f"📊 <b>Risk</b>   🛡 −${sl_usd:.2f} (−{sl_acct:.1f}%) if stopped  ·  profit taken at {anchor}",
        ]

    lines += [
        "",
        f"💡 <i>{_esc(strategy.description)}</i>",
        "⚠️ <i>Educational, not financial advice — manage your own risk.</i>",
    ]
    text = "\n".join(lines)

    payload = {
        "type": "entry",
        "side": trade.side.upper(),
        "asset": trade.asset,
        "tf": trade.tf,
        "strategy": strategy.id,
        "strategy_name": strategy.name,
        "preset": strategy.name,        # back-compat alias for v1 webhook mappings
        "ml_conf": round(ml_conf * 100, 1),
        "leverage": lev,
        "entry": trade.entry,
        "sl": trade.sl,
        "tp": trade.tp,
        "tp1": trade.tp,                # back-compat: single TP exposed as tp1
        "tp2": None, "tp3": None, "tp4": None,
        "target": trade.target,
        "risk_usd": round(sl_usd, 2),
        "margin_usd": round(margin, 2),
        "notional_usd": round(notional, 2),
        "adx": round(adx, 1) if adx is not None else None,
        "text": text,
    }
    return payload, text


def event_alert(event: Dict[str, Any], trade: Trade, strategy: Strategy) -> Tuple[Dict[str, Any], str]:
    """Build the (payload, html_text) pair for a TP / SL / exit event."""
    lev = config.LEVERAGE
    _, notional = _sizing()
    kind = event["type"]
    price = event["price"]
    sign = 1 if trade.side == Side.LONG.value else -1
    move = (price - trade.entry) / trade.entry * 100 * sign        # signed price move
    usd = notional * move / 100.0
    acct = (usd / config.ACCOUNT_USD * 100.0) if config.ACCOUNT_USD else 0.0
    margin_pct = move * lev                                         # % on the margin at lev

    emoji = "🛡" if kind == "sl" else ("🎯" if kind == "tp" else "✅")
    label = {"sl": "STOP HIT", "tp": "TAKE-PROFIT HIT", "exit": "EXIT · reverted to mean"}.get(kind, kind.upper())
    money = f"{'+' if usd >= 0 else '−'}${abs(usd):.2f}"

    text = (
        f"{emoji} <b>{label}</b>   ·   <code>{_esc(trade.asset)}/USD</code>  <b>{_esc(trade.tf)}</b>\n"
        f"{_RULE}\n"
        f"🧩 <i>{_esc(strategy.name)}</i>\n"
        f"💰 <b>{money}</b>   ({acct:+.1f}% acct  ·  {margin_pct:+.0f}% on margin @{lev}×)\n"
        f"📍 @ <code>{fmt(price)}</code>"
    )
    payload = {
        "type": "event",
        "event": kind,
        "asset": trade.asset,
        "tf": trade.tf,
        "strategy": strategy.id,
        "side": trade.side.upper(),     # back-compat aliases
        "entry": trade.entry,
        "leverage": lev,
        "price": price,
        "pnl_usd": round(usd, 2),
        "pct": round(move, 2),
        "pct_leveraged": round(margin_pct, 2),
        "text": text,
    }
    return payload, text


def sample_alert() -> Tuple[Dict[str, Any], str]:
    """A realistic, clearly-labelled SAMPLE entry — shows exactly what a real
    signal looks like and doubles as a delivery test."""
    strat = STRATEGY_BY_ID["vwap_bracket_eth"]
    entry, atr = 1800.0, 31.0
    trade = Trade(
        strategy_id=strat.id, asset=strat.asset, tf=strat.timeframe, side="long",
        entry=entry, init_sl=entry - 1.5 * atr, sl=entry - 1.5 * atr, tp=entry + 0.75 * atr,
        exit_style="bracket", opened_bar=0, last_bar=0, ml_snap={},
    )
    _, text = entry_alert(trade, strat, 0.58, adx=18.0)
    banner = (
        "🧪 <b>SAMPLE ALERT</b> — this is exactly what a real signal looks like.\n"
        "Not a live trade. Real alerts fire only on a 4h close with ADX&lt;25.\n"
        f"{_RULE}\n"
    )
    text = banner + text
    return {"type": "test", "text": text}, text


__all__ = ["deliver", "fmt", "entry_alert", "event_alert", "sample_alert"]
