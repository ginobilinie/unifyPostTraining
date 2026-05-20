#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/dong/explore/hf_models/Qwen3-0.6B-Base}"
BASE_REPORT_ROOT="${BASE_REPORT_ROOT:-reports/phase1_qwen3_0_6b_base_plain_medium_sft}"
OUT_ROOT="${OUT_ROOT:-outputs/rl_qwen3_0_6b_base}"
REPORT_ROOT="${REPORT_ROOT:-reports/rl_qwen3_0_6b_base}"
PYTHON_BIN="${PYTHON_BIN:-/home/dong/anaconda3/envs/qwen_sft/bin/python}"
TRAIN_FILE="${TRAIN_FILE:-data/gsm8k/main_train.jsonl}"
TEST_FILE="${TEST_FILE:-data/gsm8k/main_test.jsonl}"
TRUTHFULQA_FILE="${TRUTHFULQA_FILE:-data/truthfulqa/validation_mc1.jsonl}"
MMLU_FILE="${MMLU_FILE:-data/mmlu/test_selected.jsonl}"
PROMPT_FORMAT="${PROMPT_FORMAT:-plain}"
RL_LIMIT="${RL_LIMIT:-1000}"
RL_UPDATES="${RL_UPDATES:-200}"
RL_PROMPTS_PER_UPDATE="${RL_PROMPTS_PER_UPDATE:-4}"
RL_GENERATIONS_PER_PROMPT="${RL_GENERATIONS_PER_PROMPT:-4}"
RL_MAX_NEW_TOKENS="${RL_MAX_NEW_TOKENS:-192}"
RL_TEMPERATURE="${RL_TEMPERATURE:-0.7}"
RL_TOP_P="${RL_TOP_P:-0.95}"
RL_KL_BETA="${RL_KL_BETA:-0.02}"
LR="${LR:-5e-5}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
EVAL_LIMIT="${EVAL_LIMIT:-500}"
GSM8K_EVAL_BATCH_SIZE="${GSM8K_EVAL_BATCH_SIZE:-8}"
GSM8K_MAX_NEW_TOKENS="${GSM8K_MAX_NEW_TOKENS:-192}"
RETENTION_LIMIT="${RETENTION_LIMIT:-200}"
MMLU_LIMIT="${MMLU_LIMIT:-500}"
ROLLOUT_LIMIT="${ROLLOUT_LIMIT:-200}"
DRIFT_MAX_STATES="${DRIFT_MAX_STATES:-1000}"
DRY_RUN="${DRY_RUN:-0}"

copy_base_artifact() {
  local src="$1"
  local dst="$2"
  if [ ! -e "$src" ]; then
    echo "Missing reusable base artifact: $src" >&2
    exit 1
  fi
  rm -rf "$dst"
  mkdir -p "$(dirname "$dst")"
  cp -a "$src" "$dst"
}

mkdir -p "$OUT_ROOT" "$REPORT_ROOT"

echo "RL config:"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  BASE_REPORT_ROOT=$BASE_REPORT_ROOT"
echo "  OUT_ROOT=$OUT_ROOT"
echo "  REPORT_ROOT=$REPORT_ROOT"
echo "  PROMPT_FORMAT=$PROMPT_FORMAT"
echo "  RL_LIMIT=$RL_LIMIT"
echo "  RL_UPDATES=$RL_UPDATES"
echo "  RL_PROMPTS_PER_UPDATE=$RL_PROMPTS_PER_UPDATE"
echo "  RL_GENERATIONS_PER_PROMPT=$RL_GENERATIONS_PER_PROMPT"
echo "  RL_MAX_NEW_TOKENS=$RL_MAX_NEW_TOKENS"
echo "  RL_TEMPERATURE=$RL_TEMPERATURE"
echo "  RL_KL_BETA=$RL_KL_BETA"
echo "  LR=$LR"
echo "  LORA_RANK=$LORA_RANK"
echo "  LORA_ALPHA=$LORA_ALPHA"
echo "  DRY_RUN=$DRY_RUN"

if [ "$DRY_RUN" = "1" ] || [ "$DRY_RUN" = "true" ]; then
  echo "Dry run requested; exiting before model loading."
  exit 0
fi

copy_base_artifact "$BASE_REPORT_ROOT/base_eval" "$REPORT_ROOT/base_eval"
copy_base_artifact "$BASE_REPORT_ROOT/base_truthfulqa_eval" "$REPORT_ROOT/base_truthfulqa_eval"
copy_base_artifact "$BASE_REPORT_ROOT/base_mmlu_eval" "$REPORT_ROOT/base_mmlu_eval"
copy_base_artifact "$BASE_REPORT_ROOT/base_states.jsonl" "$REPORT_ROOT/base_states.jsonl"

"$PYTHON_BIN" scripts/train_rl_grpo_lora.py \
  --model-path "$MODEL_PATH" \
  --train-file "$TRAIN_FILE" \
  --output-dir "$OUT_ROOT/rl_lora" \
  --prompt-format "$PROMPT_FORMAT" \
  --limit "$RL_LIMIT" \
  --updates "$RL_UPDATES" \
  --prompts-per-update "$RL_PROMPTS_PER_UPDATE" \
  --generations-per-prompt "$RL_GENERATIONS_PER_PROMPT" \
  --max-new-tokens "$RL_MAX_NEW_TOKENS" \
  --temperature "$RL_TEMPERATURE" \
  --top-p "$RL_TOP_P" \
  --kl-beta "$RL_KL_BETA" \
  --lr "$LR" \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT"

"$PYTHON_BIN" scripts/evaluate_gsm8k.py \
  --model-path "$MODEL_PATH" \
  --adapter-dir "$OUT_ROOT/rl_lora" \
  --test-file "$TEST_FILE" \
  --out-dir "$REPORT_ROOT/rl_eval" \
  --model-name rl \
  --limit "$EVAL_LIMIT" \
  --batch-size "$GSM8K_EVAL_BATCH_SIZE" \
  --max-new-tokens "$GSM8K_MAX_NEW_TOKENS" \
  --prompt-format "$PROMPT_FORMAT"

"$PYTHON_BIN" scripts/evaluate_truthfulqa_mc.py \
  --model-path "$MODEL_PATH" \
  --adapter-dir "$OUT_ROOT/rl_lora" \
  --test-file "$TRUTHFULQA_FILE" \
  --out-dir "$REPORT_ROOT/rl_truthfulqa_eval" \
  --model-name rl \
  --limit "$RETENTION_LIMIT"

"$PYTHON_BIN" scripts/evaluate_mmlu_mc.py \
  --model-path "$MODEL_PATH" \
  --adapter-dir "$OUT_ROOT/rl_lora" \
  --test-file "$MMLU_FILE" \
  --out-dir "$REPORT_ROOT/rl_mmlu_eval" \
  --model-name rl \
  --limit "$MMLU_LIMIT"

"$PYTHON_BIN" scripts/collect_rollouts.py \
  --model-path "$MODEL_PATH" \
  --adapter-dir "$OUT_ROOT/rl_lora" \
  --prompts-file "$TEST_FILE" \
  --out "$REPORT_ROOT/rl_states.jsonl" \
  --model-name rl \
  --limit "$ROLLOUT_LIMIT" \
  --prompt-format "$PROMPT_FORMAT"

"$PYTHON_BIN" scripts/measure_state_drift.py \
  --left "$REPORT_ROOT/base_states.jsonl" \
  --right "$REPORT_ROOT/rl_states.jsonl" \
  --out "$REPORT_ROOT/state_drift.json" \
  --max-states "$DRIFT_MAX_STATES"

"$PYTHON_BIN" scripts/merge_scores.py \
  --base-metrics "$REPORT_ROOT/base_eval/metrics.json" \
  --post-metrics "$REPORT_ROOT/rl_eval/metrics.json" \
  --out "$REPORT_ROOT/scores.jsonl" \
  --role target \
  --post-model rl \
  --base-retention-metrics "$REPORT_ROOT/base_truthfulqa_eval/metrics.json" \
  --post-retention-metrics "$REPORT_ROOT/rl_truthfulqa_eval/metrics.json" \
  --retention-task truthfulqa_mc1 \
  --retention-pair "$REPORT_ROOT/base_mmlu_eval/metrics.json" "$REPORT_ROOT/rl_mmlu_eval/metrics.json" mmlu_selected

"$PYTHON_BIN" scripts/analyze_forgetting.py \
  --scores "$REPORT_ROOT/scores.jsonl" \
  --drift "$REPORT_ROOT/state_drift.json" \
  --out "$REPORT_ROOT/rl_report.json" \
  --post-model rl

echo "RL report: $REPORT_ROOT/rl_report.json"
