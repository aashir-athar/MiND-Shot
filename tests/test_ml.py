"""Unit tests for the v2 self-learning ML (mind_shot.ml) and the trainer core.

Every mathematical property the ensemble relies on gets a direct check:
migration safety, prior sanity, bucket/FTRL learning direction, Hedge weight
dynamics (including the sleeping-experts invariant), calibration bounds, drift
detection, crash-safety on malformed input, JSON round-tripping, determinism.
"""
import copy
import json
import math
import unittest

from mind_shot import ml

SID = "vwap_bracket_eth"          # a real strategy id (prior comes from its backtest WR)


def snap_for(vol_b=1, rsi_b=1, ses_b=1, strat_b=0, side="long", p_tr=None, t=1_700_000_000):
    """Hand-built v2 snapshot (bypasses build_snapshot so tests control buckets)."""
    return {
        "v": 2, "vol_b": vol_b, "rsi_b": rsi_b, "ses_b": ses_b, "strat_b": strat_b,
        "x": {"rsi": 0.1, "adx": -0.2, "atrp": 0.0, "volr": 0.0,
              "hs": 0.5, "hc": 0.5, "ds": 0.0, "dc": 1.0,
              "side": 1.0 if side == "long" else -1.0,
              "side_rsi": 0.1 if side == "long" else -0.1},
        "p_tr": p_tr, "side": side, "sid": SID, "t": t,
    }


def teach(node, snaps_outcomes):
    """predict-then-update each (snap, won) pair — the live call pattern."""
    for snap, won in snaps_outcomes:
        ml.predict(node, snap)
        ml.update(node, snap, won)


class TestMigration(unittest.TestCase):
    def test_v1_node_migrates_preserving_totals(self):
        v1 = {"vol": {"1": [3, 6]}, "rsi": {"2": [1, 4]}, "ses": {}, "strat": {"0": [3, 6]},
              "total_trades": 6, "wins": 3, "gross_profit": 1.125, "gross_loss": 3.0}
        node = copy.deepcopy(v1)
        p = ml.predict(node, snap_for())
        self.assertEqual(node["version"], ml.ML_VERSION)
        self.assertEqual(node["total_trades"], 6)
        self.assertEqual(node["wins"], 3)
        self.assertAlmostEqual(node["gross_profit"], 1.125)
        # v1's ×2 SL penalty divided back out of per-bucket loss counts
        self.assertEqual(node["dims"]["vol"]["1"], [3.0, 3.0])
        # original v1 dims preserved for audit — never wiped
        self.assertEqual(node["legacy_v1"]["vol"], {"1": [3, 6]})
        self.assertTrue(0.0 < p < 1.0)

    def test_migration_is_idempotent(self):
        node = {"vol": {"1": [2, 4]}, "total_trades": 3, "wins": 1}
        ml.predict(node, snap_for())
        once = json.dumps(node, sort_keys=True)
        ml.predict(node, snap_for())
        self.assertEqual(once, json.dumps(node, sort_keys=True))


class TestPrediction(unittest.TestCase):
    def test_fresh_node_predicts_near_prior(self):
        node = ml.empty_ml()
        p = ml.predict(node, snap_for())
        self.assertTrue(0.5 <= p <= 0.8, f"fresh prediction {p} outside sane prior band")

    def test_prediction_always_in_bounds(self):
        node = ml.empty_ml()
        for won in [True] * 50 + [False] * 50 + [True, False] * 25:
            ml.predict(node, snap_for())
            ml.update(node, snap_for(), won)
            p = ml.predict(node, snap_for())
            self.assertTrue(0.0 < p < 1.0)

    def test_predict_freezes_expert_beliefs_into_snap(self):
        node = ml.empty_ml()
        snap = snap_for(p_tr=0.9)
        ml.predict(node, snap)
        self.assertIn("_pred", snap)
        self.assertIn("base", snap["_pred"]["experts"])
        self.assertIn("trained", snap["_pred"]["experts"])
        self.assertTrue(0 <= snap["_pred"]["p"] <= 1)

    def test_predict_never_raises_on_garbage(self):
        for bad_node, bad_snap in [({}, {}), (ml.empty_ml(), {"x": "junk"}),
                                   ({"version": 2}, None), (ml.empty_ml(), {"vol_b": "NaN"})]:
            p = ml.predict(bad_node, bad_snap)
            self.assertTrue(0.0 <= p <= 1.0)


class TestLearning(unittest.TestCase):
    def test_buckets_learn_context_direction(self):
        node = ml.empty_ml()
        lessons = []
        for _ in range(40):
            lessons.append((snap_for(vol_b=2), False))   # high-vol context always loses
            lessons.append((snap_for(vol_b=0), True))    # low-vol context always wins
        teach(node, lessons)
        p_bad = ml.predict(node, snap_for(vol_b=2))
        p_good = ml.predict(node, snap_for(vol_b=0))
        self.assertGreater(p_good, p_bad + 0.10, f"context not learned: {p_good} vs {p_bad}")

    def test_ftrl_learns_numeric_feature(self):
        node = ml.empty_ml()
        lessons = []
        for _ in range(60):
            lessons.append((snap_for(side="long"), True))
            lessons.append((snap_for(side="short"), False))
        teach(node, lessons)
        self.assertGreater(ml.predict(node, snap_for(side="long")),
                           ml.predict(node, snap_for(side="short")))

    def test_decay_forgets_old_regime(self):
        node = ml.empty_ml()
        teach(node, [(snap_for(vol_b=2), False)] * 30)    # old regime: vol 2 loses
        p_after_bad = ml.predict(node, snap_for(vol_b=2))
        teach(node, [(snap_for(vol_b=2), True)] * 60)     # new regime: vol 2 wins
        p_after_good = ml.predict(node, snap_for(vol_b=2))
        self.assertGreater(p_after_good, p_after_bad + 0.15)


class TestHedge(unittest.TestCase):
    def test_accurate_expert_earns_weight(self):
        node = ml.empty_ml()
        lessons = []
        for k in range(80):                # trained expert is always right, alternating outcomes
            won = (k % 2 == 0)
            lessons.append((snap_for(p_tr=0.95 if won else 0.05), won))
        teach(node, lessons)
        self.assertGreater(node["hedge"]["trained"], ml.HEDGE_INIT["trained"] * 2,
                           f"accurate expert failed to earn weight: {node['hedge']}")

    def test_sleeping_expert_weight_is_frozen(self):
        node = ml.empty_ml()
        teach(node, [(snap_for(p_tr=None), (k % 3 != 0)) for k in range(60)])
        self.assertAlmostEqual(node["hedge"]["trained"], ml.HEDGE_INIT["trained"], places=9,
                               msg="asleep expert's weight must not move")

    def test_weights_stay_normalised(self):
        node = ml.empty_ml()
        teach(node, [(snap_for(p_tr=0.7), (k % 4 != 0)) for k in range(50)])
        self.assertAlmostEqual(sum(node["hedge"].values()), 1.0, places=6)
        for w in node["hedge"].values():
            self.assertGreater(w, 0.0)


class TestCalibrationAndDrift(unittest.TestCase):
    def test_calibration_parameters_stay_bounded(self):
        node = ml.empty_ml()
        teach(node, [(snap_for(), False)] * 120)          # brutal: everything loses
        self.assertTrue(0.25 <= node["cal"]["a"] <= 4.0)
        self.assertTrue(-1.5 <= node["cal"]["b"] <= 1.5)

    def test_drift_fires_on_regime_break(self):
        node = ml.empty_ml()
        teach(node, [(snap_for(), True)] * 60)            # long winning regime…
        teach(node, [(snap_for(), False)] * 40)           # …then everything breaks
        self.assertGreater(node["drift"]["events"], 0, "drift never detected")

    def test_no_false_drift_on_stationary_stream(self):
        node = ml.empty_ml()
        # steady 75% win rate, deterministic pattern
        teach(node, [(snap_for(), (k % 4 != 3)) for k in range(120)])
        self.assertEqual(node["drift"]["events"], 0)


class TestRobustness(unittest.TestCase):
    def test_v1_snapshot_still_teaches(self):
        node = ml.empty_ml()
        v1_snap = {"vol_b": 1, "rsi_b": 2, "ses_b": 0, "strat_b": 3}   # pre-upgrade open trade
        ml.update(node, v1_snap, True)
        self.assertEqual(node["total_trades"], 1)
        self.assertEqual(node["wins"], 1)
        self.assertEqual(node["metrics"]["n"], 0)          # no entry-time beliefs → not scored

    def test_update_never_loses_the_close(self):
        node = ml.empty_ml()
        for bad in [None, {}, {"x": object()}, {"vol_b": float("nan")}]:
            ml.update(node, bad, True)
        self.assertEqual(node["total_trades"], 4)
        self.assertEqual(node["wins"], 4)

    def test_state_is_json_serialisable_and_deterministic(self):
        def build():
            node = ml.empty_ml()
            teach(node, [(snap_for(vol_b=k % 3, p_tr=0.6), (k % 5 != 0)) for k in range(70)])
            return node
        a, b = build(), build()
        ja, jb = json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True)
        self.assertEqual(ja, jb, "identical trade streams must produce identical state")
        self.assertEqual(json.loads(ja), a)

    def test_metrics_ledger_is_honest(self):
        node = ml.empty_ml()
        teach(node, [(snap_for(), (k % 4 != 3)) for k in range(40)])
        s = ml.summary(node, SID)
        self.assertEqual(s["scored_trades"], 40)
        self.assertIn("logloss", s)
        self.assertIn("edge_vs_baseline", s)
        n_binned = sum(b[0] for b in node["metrics"]["bins"])
        self.assertEqual(n_binned, 40)


class TestReviewRegressions(unittest.TestCase):
    """Each test pins a defect found by the adversarial review."""

    def test_corrupt_v1_node_never_loses_data(self):
        node = {"vol": "not-a-dict-value", "rsi": {"1": "garbage"}, "total_trades": 9, "wins": 5,
                "strat": {"0": [2, 4]}}
        p = ml.predict(node, snap_for())
        self.assertTrue(0.0 <= p <= 1.0)
        self.assertEqual(node["version"], ml.ML_VERSION)
        self.assertEqual(node["total_trades"], 9)          # totals survive
        self.assertEqual(node["dims"]["strat"]["0"], [2.0, 2.0])   # good field migrated
        kept = node.get("legacy_v1", {})
        self.assertIn("rsi", kept)                          # corrupt originals preserved for audit

    def test_nan_in_node_never_reaches_state_json(self):
        node = ml.empty_ml()
        node["dw"] = float("nan")
        node["ftrl"]["z"]["rsi"] = float("inf")
        snap = snap_for()
        p = ml.predict(node, snap)
        self.assertTrue(math.isfinite(p) and 0.0 < p < 1.0)
        ml.update(node, snap, True)
        # strict JSON (allow_nan=False) is what a JS JSON.parse can read
        json.dumps(snap, allow_nan=False)
        for key in ("hedge", "cal", "metrics", "drift"):
            json.dumps(node[key], allow_nan=False)

    def test_update_books_close_exactly_once_even_when_learning_breaks(self):
        node = ml.empty_ml()
        node["hedge"] = "poisoned"                          # learning layers will throw
        snap = snap_for()
        snap["_pred"] = {"experts": {"base": 0.7}, "p": 0.7}
        ml.update(node, snap, True)
        self.assertEqual(node["total_trades"], 1)           # not 0, not 2
        self.assertEqual(node["wins"], 1)

    def test_no_cross_asset_model_fallback(self):
        model = {"btc": {"weights": [0.1] * 12, "bias": 0.0,
                         "means": [0.0] * 12, "stds": [1.0] * 12}}
        candles = []
        px = 100.0
        for i in range(120):
            px *= 1.001
            candles.append((1_700_000_000 + i * 14400, px, px * 1.002, px * 0.998, px, 10.0))
        self.assertIsNone(ml.apply_trained_model(model, candles, 110, "ETH"),
                          "ETH must never be scored with BTC weights")

    def test_drift_softening_keeps_sleeping_weight_frozen(self):
        node = ml.empty_ml()
        teach(node, [(snap_for(), True)] * 60)              # winning regime (trained asleep)
        teach(node, [(snap_for(), False)] * 40)             # break → drift fires
        self.assertGreater(node["drift"]["events"], 0)
        self.assertAlmostEqual(node["hedge"]["trained"], ml.HEDGE_INIT["trained"], places=9)


class TestTrainerCore(unittest.TestCase):
    def test_logistic_learns_separable_data(self):
        import ml_trainer as t
        X = [[1.0, 0.5], [0.9, 0.4], [1.1, 0.6], [0.8, 0.7]] * 40
        y = [1, 1, 1, 1] * 40
        X += [[-1.0, -0.5], [-0.9, -0.6], [-1.1, -0.4], [-0.8, -0.7]] * 40
        y += [0, 0, 0, 0] * 40
        w, b = t.train_logistic(X, y, epochs=30)
        probs = t.predict_batch(X, w, b)
        acc = sum(1 for p, tgt in zip(probs, y) if (p >= 0.5) == bool(tgt)) / len(y)
        self.assertGreater(acc, 0.95)

    def test_features_match_engine_apply(self):
        """Trainer features must stay index-identical to ml.apply_trained_model."""
        import ml_trainer as t
        candles = []
        px = 100.0
        for i in range(200):
            px *= 1.0 + (0.003 if i % 7 < 4 else -0.002)
            candles.append((1_700_000_000 + i * 14400, px * 0.999, px * 1.004, px * 0.996, px, 50.0 + i % 9))
        X, y = t.build_features(candles)
        self.assertTrue(X and len(X[0]) == 12)
        model = {"btc": {"weights": [0.1] * 12, "bias": 0.0,
                         "means": [0.0] * 12, "stds": [1.0] * 12}}
        p = ml.apply_trained_model(model, candles, len(candles) - 6, "BTC")
        self.assertIsNotNone(p)
        self.assertTrue(0.0 <= p <= 1.0)


if __name__ == "__main__":
    unittest.main()
