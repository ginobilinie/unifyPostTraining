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

## Phase 1: Qwen2.5-0.5B-Instruct SFT Validation

This is the first executable test of the state-distribution hypothesis:

1. Evaluate the base model on GSM8K.
2. SFT the base model with TRL/PEFT LoRA by default.
3. Evaluate the SFT adapter.
4. Collect base and SFT rollout states on the same prompts.
5. Measure state drift.
6. Write one report containing forgetting and drift.

Run a small but useful first pass:

```bash
MODEL_PATH=/data/dong/explore/hf_models/Qwen2.5-0.5B-Instruct \
PYTHON_BIN=/home/dong/anaconda3/envs/qwen_sft/bin/python \
TRAINER=trl \
TRAIN_LIMIT=1000 \
EVAL_LIMIT=200 \
ROLLOUT_LIMIT=100 \
bash scripts/run_phase1_validation.sh
```

Run a stronger SFT stress test intended to create more state drift and expose
possible retention loss:

```bash
MODEL_PATH=/data/dong/explore/hf_models/Qwen2.5-0.5B-Instruct \
PYTHON_BIN=/home/dong/anaconda3/envs/qwen_sft/bin/python \
TRAINER=trl \
OUT_ROOT=outputs/phase1_qwen2_5_0_5b_strong_sft \
REPORT_ROOT=reports/phase1_qwen2_5_0_5b_strong_sft \
TRAIN_LIMIT=7473 \
EVAL_LIMIT=500 \
GSM8K_EVAL_BATCH_SIZE=8 \
GSM8K_MAX_NEW_TOKENS=192 \
RETENTION_LIMIT=200 \
MMLU_LIMIT=500 \
ROLLOUT_LIMIT=200 \
EPOCHS=5 \
BATCH_SIZE=8 \
GRAD_ACCUM=2 \
LR=5e-4 \
LORA_RANK=32 \
LORA_ALPHA=64 \
bash scripts/run_phase1_validation.sh
```

For follow-up runs on the same base model and same eval limits, skip the base
evals and base rollout collection by reusing an existing report directory:

```bash
MODEL_PATH=/data/dong/explore/hf_models/Qwen3-0.6B-Base \
PYTHON_BIN=/home/dong/anaconda3/envs/qwen_sft/bin/python \
TRAINER=trl \
REUSE_BASE=1 \
BASE_REPORT_ROOT=reports/phase1_qwen3_0_6b_base_strong_sft \
OUT_ROOT=outputs/phase1_qwen3_0_6b_base_medium_sft \
REPORT_ROOT=reports/phase1_qwen3_0_6b_base_medium_sft \
TRAIN_LIMIT=7473 \
EVAL_LIMIT=500 \
GSM8K_EVAL_BATCH_SIZE=8 \
GSM8K_MAX_NEW_TOKENS=192 \
RETENTION_LIMIT=200 \
MMLU_LIMIT=500 \
ROLLOUT_LIMIT=200 \
EPOCHS=2 \
BATCH_SIZE=8 \
GRAD_ACCUM=2 \
LR=2e-4 \
LORA_RANK=32 \
LORA_ALPHA=64 \
bash scripts/run_phase1_validation.sh
```

For base models such as `Qwen3-0.6B-Base`, prefer plain causal-LM formatting
instead of chat templates:

```bash
MODEL_PATH=/data/dong/explore/hf_models/Qwen3-0.6B-Base \
PYTHON_BIN=/home/dong/anaconda3/envs/qwen_sft/bin/python \
TRAINER=trl \
PROMPT_FORMAT=plain \
OUT_ROOT=outputs/phase1_qwen3_0_6b_base_plain_medium_sft \
REPORT_ROOT=reports/phase1_qwen3_0_6b_base_plain_medium_sft \
TRAIN_LIMIT=7473 \
EVAL_LIMIT=500 \
GSM8K_EVAL_BATCH_SIZE=8 \
GSM8K_MAX_NEW_TOKENS=192 \
RETENTION_LIMIT=200 \
MMLU_LIMIT=500 \
ROLLOUT_LIMIT=200 \
EPOCHS=2 \
BATCH_SIZE=8 \
GRAD_ACCUM=2 \
LR=2e-4 \
LORA_RANK=32 \
LORA_ALPHA=64 \
bash scripts/run_phase1_validation.sh
```

For a quick smoke test:

```bash
PYTHON_BIN=/home/dong/anaconda3/envs/qwen_sft/bin/python \
TRAINER=trl \
TRAIN_LIMIT=8 \
EVAL_LIMIT=4 \
ROLLOUT_LIMIT=4 \
EPOCHS=1 \
bash scripts/run_phase1_validation.sh
```

Main outputs:

- `outputs/phase1_qwen2_5_0_5b/sft_lora/adapter.pt`
- `reports/phase1_qwen2_5_0_5b/base_eval/metrics.json`
- `reports/phase1_qwen2_5_0_5b/sft_eval/metrics.json`
- `reports/phase1_qwen2_5_0_5b/state_drift.json`
- `reports/phase1_qwen2_5_0_5b/phase1_report.json`

The current RL placeholder is `scripts/run_verl_rl_placeholder.py`. Use `verl`
for the RL phase after SFT/KD/OPD validation is stable on this same report
schema.
