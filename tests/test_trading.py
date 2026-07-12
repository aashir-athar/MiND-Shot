"""Unit tests for the trade lifecycle (open / manage / exit)."""
import unittest

from mind_shot.strategies import STRATEGY_BY_ID
from mind_shot.trading import Trade, manage_trade, open_trade


class TestOpenTrade(unittest.TestCase):
    def test_bracket_levels(self):
        strat = STRATEGY_BY_ID["vwap_bracket_eth"]  # sl 1.5×ATR, tp 0.75×ATR
        # Signal on closed bar 1; fill at the OPEN of bar 2 (= 100).
        candles = [(0, 100, 100, 100, 100, 1), (1, 100, 100, 100, 100, 1), (2, 100, 100, 100, 100, 1)]
        series = {"adx": [None, 20.0, 20.0], "vwap_z": [None, -2.5, -2.5], "atr": [None, 10.0, 10.0]}
        trade = open_trade(strat, candles, 1, series, {"vol_b": 1})
        self.assertIsNotNone(trade)
        self.assertEqual(trade.side, "long")
        self.assertAlmostEqual(trade.entry, 100.0)
        self.assertAlmostEqual(trade.sl, 85.0)        # 100 - 1.5*10
        self.assertAlmostEqual(trade.tp, 107.5)       # 100 + 0.75*10

    def test_revert_has_no_tp_and_sets_target(self):
        strat = STRATEGY_BY_ID["zscore_revert_btc"]   # sl 3×ATR, exit at mean
        candles = [(0, 100, 100, 100, 100, 1), (1, 100, 100, 100, 100, 1), (2, 100, 100, 100, 100, 1)]
        series = {"adx": [None, 20.0, 20.0], "zscore": [None, -2.0, -2.0],
                  "atr": [None, 5.0, 5.0], "sma20": [None, 110.0, 110.0]}
        trade = open_trade(strat, candles, 1, series, {})
        self.assertIsNotNone(trade)
        self.assertIsNone(trade.tp)
        self.assertAlmostEqual(trade.sl, 85.0)        # 100 - 3*5
        self.assertAlmostEqual(trade.target, 110.0)


def _trade(side="long", entry=100.0, sl=85.0, tp=107.5, style="bracket", strat="vwap_bracket_eth"):
    return Trade(strategy_id=strat, asset="ETH", tf="4h", side=side, entry=entry,
                 init_sl=sl, sl=sl, tp=tp, exit_style=style, opened_bar=0, last_bar=0, ml_snap={})


class TestManageBracket(unittest.TestCase):
    strat = STRATEGY_BY_ID["vwap_bracket_eth"]

    def test_take_profit_hit(self):
        trade = _trade()
        candles = [(0, 100, 100, 100, 100, 1),
                   (1, 100, 108, 99, 105, 1),    # closed bar: high 108 >= tp 107.5
                   (2, 105, 105, 105, 105, 1)]
        events, closed, info = manage_trade(trade, self.strat, candles, {})
        self.assertTrue(closed)
        self.assertEqual(events[0]["type"], "tp")
        self.assertTrue(info["won"])
        self.assertAlmostEqual(info["pnl_r"], 0.5, places=3)

    def test_stop_loss_hit(self):
        trade = _trade()
        candles = [(0, 100, 100, 100, 100, 1),
                   (1, 100, 101, 84, 90, 1),     # low 84 <= sl 85
                   (2, 90, 90, 90, 90, 1)]
        events, closed, info = manage_trade(trade, self.strat, candles, {})
        self.assertTrue(closed)
        self.assertEqual(events[0]["type"], "sl")
        self.assertFalse(info["won"])
        self.assertAlmostEqual(info["pnl_r"], -1.0, places=3)

    def test_stop_fills_first_when_both_touched(self):
        trade = _trade()
        candles = [(0, 100, 100, 100, 100, 1),
                   (1, 100, 110, 84, 100, 1),    # closed bar touches BOTH tp and sl → stop wins
                   (2, 100, 100, 100, 100, 1)]   # forming
        events, closed, info = manage_trade(trade, self.strat, candles, {})
        self.assertTrue(closed)
        self.assertEqual(events[0]["type"], "sl")

    def test_stays_open_when_untouched(self):
        trade = _trade()
        candles = [(0, 100, 100, 100, 100, 1),
                   (1, 100, 103, 98, 101, 1),    # closed, untouched
                   (2, 101, 101, 101, 101, 1)]   # forming
        events, closed, info = manage_trade(trade, self.strat, candles, {})
        self.assertFalse(closed)
        self.assertEqual(events, [])


class TestFormingBar(unittest.TestCase):
    """v2: price levels are detected promptly on the still-forming bar."""
    strat = STRATEGY_BY_ID["vwap_bracket_eth"]

    def test_stop_detected_on_forming_bar(self):
        trade = _trade()
        candles = [(0, 100, 100, 100, 100, 1),
                   (1, 100, 103, 99, 101, 1),    # closed, untouched
                   (2, 101, 101, 84, 86, 1)]     # FORMING: low 84 already through sl 85
        events, closed, info = manage_trade(trade, self.strat, candles, {})
        self.assertTrue(closed, "stop breached on the forming bar must close promptly")
        self.assertEqual(events[0]["type"], "sl")
        self.assertAlmostEqual(info["exit_price"], 85.0)   # booked at the level, like the backtest

    def test_revert_signal_ignored_on_forming_bar(self):
        strat = STRATEGY_BY_ID["zscore_revert_btc"]
        trade = _trade(style="revert", tp=None, sl=85.0, strat="zscore_revert_btc")
        candles = [(0, 100, 100, 100, 100, 1),
                   (1, 100, 103, 99, 101, 1),    # closed; z still extended → no exit
                   (2, 101, 106, 101, 105, 1)]   # FORMING; z reverted but bar not final
        series = {"adx": [None, 20.0, 20.0], "zscore": [None, -1.5, 0.5],
                  "sma20": [None, 104.0, 104.0]}
        events, closed, info = manage_trade(trade, strat, candles, series)
        self.assertFalse(closed, "signal exits must wait for the bar to close")


class TestManageRevert(unittest.TestCase):
    strat = STRATEGY_BY_ID["zscore_revert_btc"]

    def test_revert_exit_on_closed_bar(self):
        trade = _trade(style="revert", tp=None, sl=85.0, strat="zscore_revert_btc")
        candles = [(0, 100, 100, 100, 100, 1),
                   (1, 100, 106, 99, 105, 1),    # closed; not stopped; z reverts >= 0
                   (2, 105, 105, 105, 105, 1)]   # forming
        series = {"adx": [None, 20.0, 20.0], "zscore": [None, 0.2, 0.3],
                  "sma20": [None, 104.0, 104.0]}
        events, closed, info = manage_trade(trade, self.strat, candles, series)
        self.assertTrue(closed)
        self.assertEqual(events[0]["type"], "exit")
        self.assertTrue(info["won"])

    def test_revert_hard_stop_still_applies(self):
        trade = _trade(style="revert", tp=None, sl=85.0, strat="zscore_revert_btc")
        candles = [(0, 100, 100, 100, 100, 1),
                   (1, 100, 101, 80, 84, 1),     # closed; low 80 <= sl 85
                   (2, 84, 84, 84, 84, 1)]       # forming
        series = {"adx": [None, 20.0, 20.0], "zscore": [None, -1.5, -1.5], "sma20": [None, 104.0, 104.0]}
        events, closed, info = manage_trade(trade, self.strat, candles, series)
        self.assertTrue(closed)
        self.assertEqual(events[0]["type"], "sl")
        self.assertFalse(info["won"])


if __name__ == "__main__":
    unittest.main()
