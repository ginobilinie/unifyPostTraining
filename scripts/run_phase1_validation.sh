#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/dong/explore/hf_models/Qwen2.5-0.5B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-data/gsm8k/main_train_sft.jsonl}"
TEST_FILE="${TEST_FILE:-data/gsm8k/main_test.jsonl}"
TRUTHFULQA_FILE="${TRUTHFULQA_FILE:-data/truthfulqa/validation_mc1.jsonl}"
MMLU_FILE="${MMLU_FILE:-data/mmlu/test_selected.jsonl}"
OUT_ROOT="${OUT_ROOT:-outputs/phase1_qwen2_5_0_5b}"
REPORT_ROOT="${REPORT_ROOT:-reports/phase1_qwen2_5_0_5b}"
BASE_REPORT_ROOT="${BASE_REPORT_ROOT:-$REPORT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/home/dong/anaconda3/envs/qwen_sft/bin/python}"
TRAINER="${TRAINER:-trl}"
REUSE_BASE="${REUSE_BASE:-0}"
TRAIN_LIMIT="${TRAIN_LIMIT:-1000}"
EVAL_LIMIT="${EVAL_LIMIT:-200}"
GSM8K_EVAL_BATCH_SIZE="${GSM8K_EVAL_BATCH_SIZE:-1}"
GSM8K_MAX_NEW_TOKENS="${GSM8K_MAX_NEW_TOKENS:-256}"
ROLLOUT_LIMIT="${ROLLOUT_LIMIT:-100}"
RETENTION_LIMIT="${RETENTION_LIMIT:-200}"
MMLU_LIMIT="${MMLU_LIMIT:-500}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
PROMPT_FORMAT="${PROMPT_FORMAT:-chat}"
LR="${LR:-2e-4}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
ASSISTANT_ONLY_LOSS="${ASSISTANT_ONLY_LOSS:-0}"
DRIFT_MAX_STATES="${DRIFT_MAX_STATES:-1000}"
DRY_RUN="${DRY_RUN:-0}"

copy_base_artifact() {
  local src="$1"
  local dst="$2"
  if [ ! -d "$src" ]; then
    echo "Missing reusable base artifact directory: $src" >&2
    exit 1
  fi
  rm -rf "$dst"
  mkdir -p "$(dirname "$dst")"
  cp -a "$src" "$dst"
}

mkdir -p "$OUT_ROOT" "$REPORT_ROOT"

echo "Phase 1 config:"
echo "  PYTHON_BIN=$PYTHON_BIN"
echo "  TRAINER=$TRAINER"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  TRAIN_LIMIT=$TRAIN_LIMIT"
echo "  EVAL_LIMIT=$EVAL_LIMIT"
echo "  GSM8K_EVAL_BATCH_SIZE=$GSM8K_EVAL_BATCH_SIZE"
echo "  GSM8K_MAX_NEW_TOKENS=$GSM8K_MAX_NEW_TOKENS"
echo "  ROLLOUT_LIMIT=$ROLLOUT_LIMIT"
echo "  RETENTION_LIMIT=$RETENTION_LIMIT"
echo "  MMLU_LIMIT=$MMLU_LIMIT"
echo "  EPOCHS=$EPOCHS"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  GRAD_ACCUM=$GRAD_ACCUM"
echo "  EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM))"
echo "  MAX_LENGTH=$MAX_LENGTH"
echo "  PROMPT_FORMAT=$PROMPT_FORMAT"
echo "  LR=$LR"
echo "  LORA_RANK=$LORA_RANK"
echo "  LORA_ALPHA=$LORA_ALPHA"
echo "  LORA_DROPOUT=$LORA_DROPOUT"
echo "  ASSISTANT_ONLY_LOSS=$ASSISTANT_ONLY_LOSS"
echo "  DRIFT_MAX_STATES=$DRIFT_MAX_STATES"
echo "  OUT_ROOT=$OUT_ROOT"
echo "  REPORT_ROOT=$REPORT_ROOT"
echo "  REUSE_BASE=$REUSE_BASE"
echo "  BASE_REPORT_ROOT=$BASE_REPORT_ROOT"
echo "  DRY_RUN=$DRY_RUN"

if [ "$DRY_RUN" = "1" ] || [ "$DRY_RUN" = "true" ]; then
  echo "Dry run requested; exiting before model loading."
  exit 0
fi

if [ "$REUSE_BASE" = "1" ] || [ "$REUSE_BASE" = "true" ]; then
  echo "Reusing base artifacts from $BASE_REPORT_ROOT"
  copy_base_artifact "$BASE_REPORT_ROOT/base_eval" "$REPORT_ROOT/base_eval"
  if [ -d "$BASE_REPORT_ROOT/base_truthfulqa_eval" ]; then
    copy_base_artifact "$BASE_REPORT_ROOT/base_truthfulqa_eval" "$REPORT_ROOT/base_truthfulqa_eval"
  fi
  if [ -d "$BASE_REPORT_ROOT/base_mmlu_eval" ]; then
    copy_base_artifact "$BASE_REPORT_ROOT/base_mmlu_eval" "$REPORT_ROOT/base_mmlu_eval"
  fi
  if [ -f "$BASE_REPORT_ROOT/base_states.jsonl" ]; then
    cp "$BASE_REPORT_ROOT/base_states.jsonl" "$REPORT_ROOT/base_states.jsonl"
  else
    echo "Missing reusable base states: $BASE_REPORT_ROOT/base_states.jsonl" >&2
    exit 1
  fi
else
  "$PYTHON_BIN" scripts/evaluate_gsm8k.py \
    --model-path "$MODEL_PATH" \
    --test-file "$TEST_FILE" \
    --out-dir "$REPORT_ROOT/base_eval" \
    --model-name base \
    --limit "$EVAL_LIMIT" \
    --batch-size "$GSM8K_EVAL_BATCH_SIZE" \
    --max-new-tokens "$GSM8K_MAX_NEW_TOKENS" \
    --prompt-format "$PROMPT_FORMAT"

  if [ -f "$TRUTHFULQA_FILE" ]; then
    "$PYTHON_BIN" scripts/evaluate_truthfulqa_mc.py \
      --model-path "$MODEL_PATH" \
      --test-file "$TRUTHFULQA_FILE" \
      --out-dir "$REPORT_ROOT/base_truthfulqa_eval" \
      --model-name base \
      --limit "$RETENTION_LIMIT"
  else
    echo "Skipping TruthfulQA retention eval because $TRUTHFULQA_FILE does not exist."
  fi

  if [ -f "$MMLU_FILE" ]; then
    "$PYTHON_BIN" scripts/evaluate_mmlu_mc.py \
      --model-path "$MODEL_PATH" \
      --test-file "$MMLU_FILE" \
      --out-dir "$REPORT_ROOT/base_mmlu_eval" \
      --model-name base \
      --limit "$MMLU_LIMIT"
  else
    echo "Skipping MMLU retention eval because $MMLU_FILE does not exist."
  fi
fi

if [ "$TRAINER" = "trl" ]; then
  TRL_EXTRA_ARGS=()
  if [ "$ASSISTANT_ONLY_LOSS" = "1" ] || [ "$ASSISTANT_ONLY_LOSS" = "true" ]; then
    TRL_EXTRA_ARGS+=(--assistant-only-loss)
  fi
  "$PYTHON_BIN" scripts/train_sft_trl.py \
    --model-path "$MODEL_PATH" \
    --train-file "$TRAIN_FILE" \
    --output-dir "$OUT_ROOT/sft_lora" \
    --limit "$TRAIN_LIMIT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --max-length "$MAX_LENGTH" \
    --lr "$LR" \
    --lora-rank "$LORA_RANK" \
    --lora-alpha "$LORA_ALPHA" \
    --lora-dropout "$LORA_DROPOUT" \
    --prompt-format "$PROMPT_FORMAT" \
    "${TRL_EXTRA_ARGS[@]}"
else
  "$PYTHON_BIN" scripts/train_sft_lora.py \
    --model-path "$MODEL_PATH" \
    --train-file "$TRAIN_FILE" \
    --output-dir "$OUT_ROOT/sft_lora" \
    --limit "$TRAIN_LIMIT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --max-length "$MAX_LENGTH" \
    --lr "$LR" \
    --lora-rank "$LORA_RANK" \
    --lora-alpha "$LORA_ALPHA" \
    --lora-dropout "$LORA_DROPOUT" \
    --prompt-format "$PROMPT_FORMAT"
fi

"$PYTHON_BIN" scripts/evaluate_gsm8k.py \
  --model-path "$MODEL_PATH" \
  --adapter-dir "$OUT_ROOT/sft_lora" \
  --test-file "$TEST_FILE" \
  --out-dir "$REPORT_ROOT/sft_eval" \
  --model-name sft \
  --limit "$EVAL_LIMIT" \
  --batch-size "$GSM8K_EVAL_BATCH_SIZE" \
  --max-new-tokens "$GSM8K_MAX_NEW_TOKENS" \
  --prompt-format "$PROMPT_FORMAT"

if [ -f "$TRUTHFULQA_FILE" ]; then
  "$PYTHON_BIN" scripts/evaluate_truthfulqa_mc.py \
    --model-path "$MODEL_PATH" \
    --adapter-dir "$OUT_ROOT/sft_lora" \
    --test-file "$TRUTHFULQA_FILE" \
    --out-dir "$REPORT_ROOT/sft_truthfulqa_eval" \
    --model-name sft \
    --limit "$RETENTION_LIMIT"
fi

if [ -f "$MMLU_FILE" ]; then
  "$PYTHON_BIN" scripts/evaluate_mmlu_mc.py \
    --model-path "$MODEL_PATH" \
    --adapter-dir "$OUT_ROOT/sft_lora" \
    --test-file "$MMLU_FILE" \
    --out-dir "$REPORT_ROOT/sft_mmlu_eval" \
    --model-name sft \
    --limit "$MMLU_LIMIT"
fi

if [ "$REUSE_BASE" = "1" ] || [ "$REUSE_BASE" = "true" ]; then
  echo "Skipping base rollout collection because REUSE_BASE=$REUSE_BASE"
else
  "$PYTHON_BIN" scripts/collect_rollouts.py \
    --model-path "$MODEL_PATH" \
    --prompts-file "$TEST_FILE" \
    --out "$REPORT_ROOT/base_states.jsonl" \
    --model-name base \
    --limit "$ROLLOUT_LIMIT" \
    --prompt-format "$PROMPT_FORMAT"
fi

"$PYTHON_BIN" scripts/collect_rollouts.py \
  --model-path "$MODEL_PATH" \
  --adapter-dir "$OUT_ROOT/sft_lora" \
  --prompts-file "$TEST_FILE" \
  --out "$REPORT_ROOT/sft_states.jsonl" \
  --model-name sft \
  --limit "$ROLLOUT_LIMIT" \
  --prompt-format "$PROMPT_FORMAT"

"$PYTHON_BIN" scripts/measure_state_drift.py \
  --left "$REPORT_ROOT/base_states.jsonl" \
  --right "$REPORT_ROOT/sft_states.jsonl" \
  --out "$REPORT_ROOT/state_drift.json" \
  --max-states "$DRIFT_MAX_STATES"

"$PYTHON_BIN" scripts/merge_scores.py \
  --base-metrics "$REPORT_ROOT/base_eval/metrics.json" \
  --post-metrics "$REPORT_ROOT/sft_eval/metrics.json" \
  --out "$REPORT_ROOT/scores.jsonl" \
  --role target \
  --base-retention-metrics "$REPORT_ROOT/base_truthfulqa_eval/metrics.json" \
  --post-retention-metrics "$REPORT_ROOT/sft_truthfulqa_eval/metrics.json" \
  --retention-task truthfulqa_mc1 \
  --retention-pair "$REPORT_ROOT/base_mmlu_eval/metrics.json" "$REPORT_ROOT/sft_mmlu_eval/metrics.json" mmlu_selected

"$PYTHON_BIN" scripts/analyze_forgetting.py \
  --scores "$REPORT_ROOT/scores.jsonl" \
  --drift "$REPORT_ROOT/state_drift.json" \
  --out "$REPORT_ROOT/phase1_report.json" \
  --post-model sft

echo "Phase 1 report: $REPORT_ROOT/phase1_report.json"
