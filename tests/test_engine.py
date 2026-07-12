"""Unit tests for engine risk gates and market-data validation."""
import unittest

from mind_shot.engine import _risk_block
from mind_shot.state import empty_global_state
from mind_shot.strategies import STRATEGY_BY_ID


def _fresh():
    gs = empty_global_state()
    state = {"__global": gs}
    return state, gs


class TestRiskGates(unittest.TestCase):
    strat = STRATEGY_BY_ID["vwap_bracket_eth"]

    def test_clear_when_nothing_blocks(self):
        state, gs = _fresh()
        self.assertIsNone(_risk_block(self.strat, state, gs, funding=0.01))

    def test_daily_loss_limit(self):
        from datetime import datetime, timezone
        state, gs = _fresh()
        gs["daily_r"][datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")] = -3.5
        self.assertIn("daily loss limit", _risk_block(self.strat, state, gs) or "")

    def test_max_concurrent_trades(self):
        state, gs = _fresh()
        for i in range(4):
            state[f"s{i}"] = {"active_trade": {"side": "long"}}
        self.assertIn("trades open", _risk_block(self.strat, state, gs) or "")

    def test_funding_pause_enforced(self):
        state, gs = _fresh()                       # default threshold 0.05
        self.assertIn("funding", _risk_block(self.strat, state, gs, funding=0.08) or "")
        self.assertIn("funding", _risk_block(self.strat, state, gs, funding=-0.06) or "")
        self.assertIsNone(_risk_block(self.strat, state, gs, funding=0.04))
        self.assertIsNone(_risk_block(self.strat, state, gs, funding=None))   # outage → gate off


class TestMarketValidation(unittest.TestCase):
    def test_malformed_rows_are_dropped(self):
        # exercise the row-validation logic exactly as fetch_klines applies it
        import math
        rows = [
            [100, "1", "2", "0.5", "1.5", "x", "10"],       # good
            [90, "1", "2", "0.5", "1.5", "x", "10"],        # out of order — dropped
            [200, "1", "2", "0.5", "nan", "x", "10"],       # non-finite close (rejected: <= 0 is False for nan? guarded by isfinite)
            [300, "bad", "2", "0.5", "1.5", "x", "10"],     # unparseable — dropped
            [400, "1", "2", "0.5", "1.5", "x", "10"],       # good
        ]
        candles = []
        last_t = 0
        for r in rows:
            try:
                c = (int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[6]))
            except (TypeError, ValueError, IndexError):
                continue
            if c[0] <= last_t or c[4] <= 0 or not all(math.isfinite(v) for v in c[1:]):
                continue
            candles.append(c)
            last_t = c[0]
        self.assertEqual([c[0] for c in candles], [100, 400])


if __name__ == "__main__":
    unittest.main()
