"""Every alert we send with parse_mode=HTML must be valid Telegram HTML.

Regression guard for the "Unsupported start tag" 400 — a strategy description
like "%K(14) <20 / >80" was injected raw into <i>…</i>, so Telegram read "<20"
as a bogus opening tag. All data fields are now escaped at the render boundary;
this test proves it for every strategy through both alert builders.
"""
import re

from mind_shot.notifier import entry_alert, event_alert
from mind_shot.strategies import STRATEGIES

# A "<" that starts a real tag is followed by a letter (<b>) or "/" (</b>).
# A "<" followed by anything else (digit, space, "=") is what breaks Telegram.
_BAD = re.compile(r"<(?![a-zA-Z/])")


def _trade(strategy):
    from mind_shot.trading import Trade
    e = 100.0
    return Trade(
        strategy_id=strategy.id, asset=strategy.asset, tf=strategy.timeframe,
        side="long", entry=e, init_sl=e - 2, sl=e - 2, tp=e + 1,
        exit_style="bracket", opened_bar=0, last_bar=0, ml_snap={}, target=e,
    )


def test_all_alerts_are_valid_telegram_html():
    for strat in STRATEGIES:
        tr = _trade(strat)
        _, entry_txt = entry_alert(tr, strat, 0.6, adx=18.0)
        _, event_txt = event_alert({"type": "tp", "price": 101.0}, tr, strat)
        for txt in (entry_txt, event_txt):
            bad = _BAD.search(txt)
            assert bad is None, (
                f"{strat.id}: raw '<' at offset {bad.start()} would break Telegram HTML:\n"
                f"  …{txt[max(0, bad.start()-15):bad.start()+15]!r}…"
            )


if __name__ == "__main__":
    test_all_alerts_are_valid_telegram_html()
    print(f"OK — {len(STRATEGIES)} strategies × 2 alerts all render as valid Telegram HTML")
