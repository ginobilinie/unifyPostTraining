from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unify_post_training.metrics import TaskScore, compute_forgetting_report, pearson_correlation


class MetricsTests(unittest.TestCase):
    def test_forgetting_report(self) -> None:
        scores = [
            TaskScore("base", "mmlu", "test", 0.8),
            TaskScore("post", "mmlu", "test", 0.6),
            TaskScore("base", "gsm8k", "test", 0.5),
            TaskScore("post", "gsm8k", "test", 0.45),
        ]
        report = compute_forgetting_report(scores)
        self.assertAlmostEqual(report.mean_forgetting, 0.125)
        self.assertAlmostEqual(report.mean_retention_ratio or 0.0, (0.75 + 0.9) / 2)

    def test_pearson_correlation(self) -> None:
        self.assertAlmostEqual(pearson_correlation([1, 2, 3], [2, 4, 6]) or 0.0, 1.0)


if __name__ == "__main__":
    unittest.main()
