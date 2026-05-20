#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys


def main() -> None:
    if importlib.util.find_spec("verl") is None:
        raise SystemExit(
            "verl is not installed in this environment. "
            "Install verl before running the RL phase. The recommended order is: "
            "first validate SFT/KD/OPD state-drift behavior, then add a verl GRPO/PPO run "
            "as the on-policy RL point in the same report schema."
        )
    print(
        "verl is installed. Add the concrete GRPO/PPO config here once the Phase 1 "
        "SFT validation produces stable eval and rollout artifacts."
    )


if __name__ == "__main__":
    sys.exit(main())
