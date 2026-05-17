# unifyPostTraining

`unifyPostTraining` is a research scaffold for studying post-training through a
state distribution lens. The central claim is that SFT, RL, and on-policy
distillation differ less by token-level objective than by the state visitation
distribution they train on.

## Research Question

Does catastrophic forgetting correlate more strongly with state distribution
drift than with the nominal post-training objective?

The project is organized around four hypotheses:

- `H1`: forgetting increases with state distribution drift.
- `H2`: on-policy training retains more base-model capability than off-policy
  training under comparable supervision.
- `H3`: state source matters more than signal source.
- `H4`: OPD students can outperform degraded teachers by learning on their own
  trajectory space.

## Repository Layout

- `docs/concept.tex`: paper draft for the state distribution view.
- `docs/experimental_design.md`: experimental protocol.
- `configs/experiment_matrix.json`: method/task/metric matrix.
- `src/unify_post_training/`: reusable analysis code.
- `scripts/`: command-line entry points.
- `tests/`: smoke tests for metrics and drift estimators.
- `data/`: local generated samples and score tables.
- `reports/`: local analysis outputs.

## JSONL State Format

State drift analysis expects one JSON object per state:

```json
{
  "model": "base",
  "prompt_id": "gsm8k-0001",
  "trajectory_id": "base:gsm8k-0001:0",
  "step": 12,
  "prompt": "Natalia sold clips...",
  "prefix": "Natalia sold 48 clips in April...",
  "embedding": [0.1, -0.2, 0.3]
}
```

Only `prefix` is required for lexical drift. `embedding` is required for MMD,
Wasserstein proxy, and centroid distance.

## Quick Start

Run the local smoke tests:

```bash
python3 -m py_compile src/unify_post_training/*.py scripts/*.py
python3 -m unittest discover -s tests
```

Create a synthetic toy dataset:

```bash
python3 scripts/make_toy_states.py --out-dir data/toy
```

Measure drift between base and post-trained state files:

```bash
python3 scripts/measure_state_drift.py \
  --left data/toy/base_states.jsonl \
  --right data/toy/post_states.jsonl \
  --out reports/toy_state_drift.json
```

Analyze forgetting vs drift:

```bash
python3 scripts/analyze_forgetting.py \
  --scores data/toy/scores.jsonl \
  --drift reports/toy_state_drift.json \
  --out reports/toy_forgetting.json
```

## Experimental Spine

The intended progression is:

1. Train or load `base`, `SFT`, `offline KD`, `OPD`, and `RL` checkpoints.
2. Evaluate target task improvement and retention tasks for each checkpoint.
3. Roll out each checkpoint on a fixed prompt suite and collect intermediate
   states.
4. Embed prefixes with a frozen encoder.
5. Estimate drift from base to each post-trained model.
6. Plot drift against forgetting and test whether on-policy methods lie on a
   better retention frontier.

This repo currently provides the measurement and orchestration layer. Model
training adapters should be added once the target base model and training stack
are fixed.
