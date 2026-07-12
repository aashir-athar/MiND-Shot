"""
Trade lifecycle for the five strategies.

Two exit regimes, exactly as backtested:
  • BRACKET — a fixed take-profit and stop in ATR units (VWAP/RSI-2/Stochastic).
  • REVERT  — exit when the signal returns to its mean, with a hard ATR stop
              (VWAP-to-VWAP, Z-score-to-mean).

Price-level checks (stop / take-profit) run intrabar against every bar's high/low,
with the pessimistic rule that the STOP fills first if a bar touches both — and
they run against the still-forming bar too, so a breached stop is detected on the
next poll (minutes) instead of at bar close (hours). The fill is booked at the
level itself, exactly as the backtest books it, so live results stay comparable.
The revert exit only fires on a *closed* bar — never the still-forming one — so
the live engine and the backtest agree on what counts as a signal.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .models import C, H, L, O, T, Candle, ExitStyle, Side
from .strategies import Strategy


@dataclass
class Trade:
    strategy_id: str
    asset: str
    tf: str
    side: str                 # "long" | "short"
    entry: float
    init_sl: float
    sl: float
    tp: Optional[float]
    exit_style: str           # ExitStyle value
    opened_bar: int
    last_bar: int
    ml_snap: Dict[str, int] = field(default_factory=dict)
    target: Optional[float] = None        # live exit anchor for REVERT (display)
    live_pnl_pct: float = 0.0
    live_pnl_leveraged: float = 0.0
    duration_sec: int = 0

    # ── serialisation (state.json round-trips through plain dicts) ──
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Trade":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def side_enum(self) -> Side:
        return Side(self.side)

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry - self.init_sl)

    def update_live(self, last_close: float, leverage: float, now_bar_time: int) -> None:
        sign = self.side_enum.sign
        self.live_pnl_pct = round((last_close - self.entry) / self.entry * 100 * sign, 3)
        self.live_pnl_leveraged = round(self.live_pnl_pct * leverage, 2)
        self.duration_sec = max(0, int(now_bar_time - self.opened_bar))


def open_trade(
    strategy: Strategy,
    candles: Sequence[Candle],
    confirm_idx: int,
    series: Dict[str, List[Optional[float]]],
    ml_snap: Dict[str, int],
) -> Optional[Trade]:
    """Build a Trade from a signal confirmed on the closed bar ``confirm_idx``.

    Matches the backtest exactly: the signal and the ATR stop/target distance come
    from the closed bar, but the fill is the OPEN of the next bar (``confirm_idx+1``,
    the just-opened bar — known, not look-ahead). Management then starts on the bar
    after the entry bar.
    """
    side = strategy.entry(series, confirm_idx)
    if side is None:
        return None
    atr_val = series["atr"][confirm_idx]
    if atr_val is None or atr_val <= 0:
        return None
    exec_idx = confirm_idx + 1
    if exec_idx >= len(candles):
        return None  # no next bar to fill against yet
    entry = candles[exec_idx][O]
    sign = side.sign
    sl = entry - sign * strategy.sl_atr * atr_val
    tp = entry + sign * strategy.tp_atr * atr_val if (strategy.tp_atr is not None) else None
    target = series[strategy.mean_price_key][confirm_idx] if strategy.mean_price_key is not None else None
    return Trade(
        strategy_id=strategy.id,
        asset=strategy.asset,
        tf=strategy.timeframe,
        side=side.value,
        entry=entry,
        init_sl=sl,
        sl=sl,
        tp=tp,
        exit_style=strategy.exit_style.value,
        opened_bar=candles[exec_idx][T],
        last_bar=candles[exec_idx][T],
        ml_snap=ml_snap,
        target=target,
    )


def _level_hit(trade: Trade, candle: Candle) -> Optional[Tuple[str, float]]:
    """Stop / take-profit check for one bar. Stop is evaluated first (pessimistic)."""
    high, low = candle[H], candle[L]
    if trade.side == Side.LONG.value:
        if low <= trade.sl:
            return ("sl", trade.sl)
        if trade.tp is not None and high >= trade.tp:
            return ("tp", trade.tp)
    else:
        if high >= trade.sl:
            return ("sl", trade.sl)
        if trade.tp is not None and low <= trade.tp:
            return ("tp", trade.tp)
    return None


def manage_trade(
    trade: Trade,
    strategy: Strategy,
    candles: Sequence[Candle],
    series: Dict[str, List[Optional[float]]],
) -> Tuple[List[Dict[str, Any]], bool, Optional[Dict[str, Any]]]:
    """Advance an open trade over any bars newer than ``trade.last_bar``.

    Returns ``(events, closed, close_info)``. ``events`` is a list of
    ``{"type": "sl"|"tp"|"exit", "price": float, "time": int}``. When ``closed``
    is True, ``close_info`` carries the realised outcome (won, pnl_r, exit price).
    """
    n = len(candles)
    last_closed = n - 2  # the final candle is still forming
    events: List[Dict[str, Any]] = []

    start = next((i for i in range(n) if candles[i][T] > trade.last_bar), None)
    if start is None:
        return events, False, None
    is_revert = strategy.exit_style is ExitStyle.REVERT

    for i in range(start, n):
        candle = candles[i]
        forming = i == n - 1

        # Price levels (stop / take-profit) are checked on EVERY bar, including the
        # still-forming one: its high/low already contain any breach, and a real
        # stop order would have filled at that level intrabar. Booking at the level
        # keeps the fill identical to what the backtest books for the same bar —
        # only the detection latency improves (next poll vs bar close).
        hit = _level_hit(trade, candle)
        if hit is not None:
            kind, price = hit
            events.append({"type": kind, "price": price, "time": candle[T]})
            if not forming:
                trade.last_bar = candle[T]
            return events, True, _close_info(trade, kind, price)

        if not forming:
            # Signal-based (revert) exits still need a CLOSED bar — the forming
            # bar's oscillator value is not final, and the backtest only ever
            # evaluates closed bars.
            if is_revert and i <= last_closed and strategy.should_exit(series, i, trade.side_enum):
                exit_idx = i + 1  # fill at next bar's open (exists: i <= n-2)
                price = candles[exit_idx][O]
                events.append({"type": "exit", "price": price, "time": candles[exit_idx][T]})
                trade.last_bar = candle[T]
                return events, True, _close_info(trade, "exit", price)

        if is_revert and strategy.mean_price_key is not None and series[strategy.mean_price_key][i] is not None:
            trade.target = series[strategy.mean_price_key][i]
        if not forming:
            trade.last_bar = candle[T]

    return events, False, None


def _close_info(trade: Trade, kind: str, exit_price: float) -> Dict[str, Any]:
    risk = trade.risk_per_unit
    sign = trade.side_enum.sign
    pnl_r = ((exit_price - trade.entry) * sign / risk) if risk > 0 else 0.0
    won = pnl_r > 0
    return {
        "kind": kind,
        "exit_price": exit_price,
        "pnl_r": round(pnl_r, 3),
        "won": won,
        "gross_profit": pnl_r if pnl_r > 0 else 0.0,
        "gross_loss": abs(pnl_r) if pnl_r < 0 else 0.0,
    }


__all__ = ["Trade", "open_trade", "manage_trade"]
