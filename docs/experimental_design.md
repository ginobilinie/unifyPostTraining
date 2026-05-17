# Experimental Design: Validating the State Distribution Hypothesis

## Goal

Validate the hypothesis that catastrophic forgetting and generalization in
post-training are governed primarily by the state distribution used during
training, rather than by the objective in isolation.

## Hypotheses

- `H1 State Drift`: forgetting correlates with divergence between base and
  post-training state distributions.
- `H2 On-Policy Advantage`: on-policy training reduces forgetting under
  comparable supervision.
- `H3 Signal vs State`: state source explains retention better than whether the
  signal is labels, rewards, or teacher logits.
- `H4 Teacher Independence`: OPD can outperform a teacher that has high target
  performance but poor retention.

## Methods

| Method | State Source | Signal | Policy Type |
| --- | --- | --- | --- |
| SFT | expert dataset | ground-truth tokens | off-policy |
| Offline KD | teacher rollouts | teacher logits | off-policy |
| OPD | student rollouts | teacher logits | on-policy |
| RL | current policy rollouts | reward | on-policy |

## Tasks

Target capability:

- mathematical reasoning such as GSM8K or MATH,
- code generation such as HumanEval or MBPP,
- or instruction-following with a synthetic instruction dataset.

Retention capabilities:

- MMLU for general knowledge,
- GSM8K for reasoning,
- HumanEval or MBPP for code,
- TruthfulQA for factuality,
- perplexity on a pretraining-like held-out corpus.

## Metrics

Target performance measures post-training improvement on the new capability.

Forgetting:

```text
forgetting = score_base - score_post
```

Retention ratio:

```text
retention_ratio = score_post / score_base
```

Pareto efficiency is measured by target improvement versus aggregate retention.

## State Sampling

For each checkpoint:

1. Use a fixed prompt suite.
2. Generate trajectories with fixed decoding settings.
3. Collect intermediate prefix states `s_t = (x, y_<t)`.
4. Save states as JSONL records with `prompt_id`, `trajectory_id`, `step`,
   `prompt`, `prefix`, and optional `embedding`.

## Drift Estimators

The current scaffold supports:

- lexical Jaccard distance over prefix token sets,
- embedding centroid distance,
- Gaussian-kernel MMD,
- a sliced one-dimensional Wasserstein proxy.

Planned extensions:

- discriminator separability,
- token-level KL along shared trajectories,
- layer-representation drift using the base model as encoder.

## Controlled Ablations

### Same Signal, Different States

Compare offline KD and OPD. Both use teacher supervision, but offline KD trains
on teacher states while OPD trains on student states.

### On-Policy Ratio Sweep

Sample mixed states:

```text
s ~ alpha d_student + (1 - alpha) d_teacher
alpha in {0, 0.25, 0.5, 0.75, 1}
```

Retention should improve monotonically or near-monotonically as `alpha`
increases if state source is the main driver.

### Teacher Degradation

1. Train a high-target, low-retention SFT teacher.
2. Distill into students using offline KD and OPD.
3. Compare target score and retention.

Expected pattern:

| Model | Target Task | Retention |
| --- | --- | --- |
| Teacher | high | low |
| KD student | similar | low |
| OPD student | high | higher |

## Critical Validation Plot

The most important figure is a scatter plot:

- x-axis: state distribution drift from base,
- y-axis: forgetting relative to base.

A strong positive correlation supports the claim that distribution discontinuity
is a primary driver of forgetting.
