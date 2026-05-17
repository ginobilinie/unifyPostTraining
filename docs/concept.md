# Post-Training is About States, Not Tokens:
# A State Distribution View of SFT, RL, and On-Policy Distillation

Anonymous Authors

## Abstract

Large language model post-training methods such as supervised fine-tuning,
reinforcement learning, and distillation are often analyzed through objective
functions such as forward or reverse KL divergence. We propose a complementary
perspective: the primary factor governing generalization and catastrophic
forgetting is not the optimization objective alone, but the state distribution
induced during training.

We formalize post-training as a process of shaping the state visitation
distribution of an autoregressive policy. Under this view, SFT corresponds to
off-policy state fitting, while RL and on-policy distillation operate under
on-policy state distributions, leading to more local and stable updates. We
empirically test whether forgetting correlates with state distribution drift,
whether on-policy training improves retention under comparable supervision, and
whether OPD can surpass a degraded teacher by learning on student-induced
states.

## Introduction

Post-training has become a critical component in modern LLM pipelines,
including supervised fine-tuning, reinforcement learning from feedback or
verifiable rewards, and distillation.

Several phenomena remain insufficiently explained:

- Why does SFT often lead to catastrophic forgetting?
- Why do RL-based methods remain stable even with weak KL regularization?
- Why does on-policy distillation behave similarly to RL?
- Why can students outperform their teachers?

**Core Thesis:** post-training is fundamentally about *state distribution
shaping*, not merely token-level supervision.

## Preliminaries

An autoregressive model defines:

$$
p_\theta(y_t \mid x, y_{<t}).
$$

We define the state as:

$$
s_t = (x, y_{<t}).
$$

The token $y_t$ is an action, the prefix state $s_t$ is the context, and a full
completion is a trajectory through prefix states.

Let $d^\pi(s)$ denote the state visitation distribution induced by policy
$\pi$. We distinguish dataset states $d_{\text{expert}}$ from model-induced
states $d_{\pi_\theta}$.

## A State Distribution View of Post-Training

### SFT as Off-Policy State Fitting

SFT optimizes:

$$
\mathbb{E}_{s \sim d_{\text{expert}}} [\log p_\theta(y \mid s)].
$$

Training states do not come from the current policy. This induces exposure
bias, covariate shift, and distribution mismatch.

**Hypothesis:** catastrophic forgetting arises from large divergence between
$d_{\text{expert}}$ and $d_{\pi_\theta}$.

### RL as On-Policy Local Improvement

RL optimizes:

$$
\mathbb{E}_{s \sim d_{\pi_\theta}} [r(s)].
$$

States are sampled from the current policy, which encourages local updates and
reduced distribution discontinuity.

### OPD as Teacher-Guided On-Policy Learning

OPD performs:

$$
\mathbb{E}_{s \sim d_{\pi_S}}
\left[\mathrm{KL}(\pi_T(\cdot \mid s) \| \pi_S(\cdot \mid s))\right].
$$

The teacher provides the signal, but the student controls the state
distribution. OPD therefore resembles policy improvement and DAgger-style
learning.

## Why Students Can Surpass Teachers

Traditional distillation implicitly assumes teacher-induced states are the right
training support. We argue instead that the student state distribution
determines learning quality. Teacher errors do not fully transfer when the
student learns over its own trajectory space.

## A Unified Framework

We unify post-training as:

$$
d_{t+1}(s) = \mathcal{T}(d_t(s), \text{signal}).
$$

| Method | State Source | Signal |
| --- | --- | --- |
| SFT | dataset | labels |
| RL | policy | reward |
| OPD | policy | teacher logits |
| DAgger | learner | expert actions |
| Self-play | policy | environment |

*Table: Unified view of post-training.*

## Experimental Design

We compare SFT, offline KD, OPD, and RL on target capability improvement,
retention, and state drift. State distribution drift is approximated with
embedding distribution distance, MMD, discriminator separability, and
token-level KL along sampled trajectories.

## Expected Results

We expect SFT and offline KD to show larger state drift and forgetting, OPD and
RL to retain more capabilities, and forgetting to correlate strongly with state
distribution drift.

## Conclusion

We propose a state distribution perspective for understanding post-training.
The key insight is:

> Post-training is not only about predicting tokens, but about shaping which
> states the model visits.
