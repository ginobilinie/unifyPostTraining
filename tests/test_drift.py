from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unify_post_training.drift import compute_drift_report
from unify_post_training.states import StateRecord


class DriftTests(unittest.TestCase):
    def test_identical_states_have_low_drift(self) -> None:
        states = [
            StateRecord("base", "p1", "t1", 0, "prompt", "prompt answer", [0.0, 1.0]),
            StateRecord("base", "p1", "t1", 1, "prompt", "prompt answer next", [0.5, 1.0]),
        ]
        report = compute_drift_report(states, states)
        self.assertEqual(report.lexical_jaccard, 0.0)
        self.assertAlmostEqual(report.centroid_l2 or 0.0, 0.0)
        self.assertAlmostEqual(report.mmd_rbf or 0.0, 0.0)

    def test_shifted_embeddings_have_positive_centroid_distance(self) -> None:
        left = [StateRecord("base", "p", "l", 0, "p", "alpha beta", [0.0, 0.0])]
        right = [StateRecord("post", "p", "r", 0, "p", "gamma delta", [1.0, 1.0])]
        report = compute_drift_report(left, right)
        self.assertGreater(report.lexical_jaccard, 0.0)
        self.assertGreater(report.centroid_l2 or 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
