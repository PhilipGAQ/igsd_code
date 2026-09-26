"""Lightweight checks for the released method; no model or GPU required."""

import ast
import importlib.util
import pathlib
import unittest
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
PLANNER = ROOT / "verl/trainer/ppo/igsd_m2.py"


def _load_eligibility():
    path = ROOT / "verl/trainer/ppo/igsd_candidate.py"
    spec = importlib.util.spec_from_file_location("igsd_candidate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.is_eligible_disagreement


def _load_gate():
    """Extract the numerical gate without importing GPU/runtime dependencies."""
    tree = ast.parse(PLANNER.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_token_gate_and_row_normalize"
    )
    env = {"np": np, "Any": Any, "tuple": tuple, "dict": dict, "float": float}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(PLANNER), "exec"), env)
    return env[function.name]


class MethodContractTest(unittest.TestCase):
    def test_disagreement_has_no_positive_internal_signal_screen(self):
        eligible = _load_eligibility()
        self.assertTrue(eligible(11, 22, -0.5))
        self.assertTrue(eligible(11, 22, 0.5))
        self.assertFalse(eligible(11, 11, 0.5))
        self.assertFalse(eligible(11, 22, float("nan")))
        self.assertFalse(eligible(11, 22, 0.0))

    def test_gate_matches_positive_tanh_and_keeps_rejected_positions_zero(self):
        gate = _load_gate()
        gains = np.asarray([-1.0, 0.0, 0.2, 0.5, float("nan")])
        weights, _ = gate(
            gains,
            np.ones(5, dtype=bool),
            {
                "igsd_token_gate_mode": "rectified_sigmoid",
                "igsd_token_gate_beta": 5.0,
                "igsd_token_gate_normalization": "none",
                "igsd_token_invalid_fallback_weight": 0.0,
            },
        )
        np.testing.assert_allclose(weights[:2], np.zeros(2), atol=1e-7)
        np.testing.assert_allclose(weights[2:4], np.tanh(2.5 * gains[2:4]), atol=1e-6)
        self.assertEqual(weights[4], 0.0)

    def test_launcher_and_planner_have_no_directional_prefilter(self):
        script = (ROOT / "scripts/train_igsd.sh").read_text(encoding="utf-8")
        source = PLANNER.read_text(encoding="utf-8")
        for forbidden in ("positive_log_odds", "direction_rejected", "random_routing_priority"):
            self.assertNotIn(forbidden, script)
            self.assertNotIn(forbidden, source)
        self.assertIn("algorithm.igsd_sampled_pair_gate_source=environment_ig", script)
        self.assertIn("algorithm.igsd_distill_mode=candidate_pair_jsd", script)

    def test_release_rejects_legacy_distillation_configuration(self):
        config_path = ROOT / "verl/utils/config.py"
        tree = ast.parse(config_path.read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_validate_igsd_config"
        )
        namespace = {"DictConfig": dict, "math": __import__("math")}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(config_path), "exec"), namespace)
        with self.assertRaisesRegex(ValueError, "This release requires"):
            namespace[function.name]({"enable_igsd": True, "igsd_distill_mode": "topk_jsd"})


if __name__ == "__main__":
    unittest.main()
