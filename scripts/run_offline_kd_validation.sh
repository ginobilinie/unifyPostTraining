#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/dong/explore/hf_models/Qwen3-0.6B-Base}"
TEACHER_ADAPTER_DIR="${TEACHER_ADAPTER_DIR:-outputs/phase1_qwen3_0_6b_base_stress_sft/sft_lora}"
BASE_REPORT_ROOT="${BASE_REPORT_ROOT:-reports/phase1_qwen3_0_6b_base_stress_sft}"
OUT_ROOT="${OUT_ROOT:-outputs/offline_kd_qwen3_0_6b_base_from_stress_teacher}"
REPORT_ROOT="${REPORT_ROOT:-reports/offline_kd_qwen3_0_6b_base_from_stress_teacher}"
PYTHON_BIN="${PYTHON_BIN:-/home/dong/anaconda3/envs/qwen_sft/bin/python}"
TRAIN_FILE="${TRAIN_FILE:-data/gsm8k/main_train.jsonl}"
KD_TRAIN_FILE="${KD_TRAIN_FILE:-$OUT_ROOT/teacher_continuations.jsonl}"
TEST_FILE="${TEST_FILE:-data/gsm8k/main_test.jsonl}"
TRUTHFULQA_FILE="${TRUTHFULQA_FILE:-data/truthfulqa/validation_mc1.jsonl}"
MMLU_FILE="${MMLU_FILE:-data/mmlu/test_selected.jsonl}"
PROMPT_FORMAT="${PROMPT_FORMAT:-plain}"
KD_LIMIT="${KD_LIMIT:-1000}"
KD_GEN_BATCH_SIZE="${KD_GEN_BATCH_SIZE:-8}"
KD_MAX_NEW_TOKENS="${KD_MAX_NEW_TOKENS:-192}"
KD_TEACHER_TEMPERATURE="${KD_TEACHER_TEMPERATURE:-0.3}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
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

echo "Offline KD config:"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  TEACHER_ADAPTER_DIR=$TEACHER_ADAPTER_DIR"
echo "  BASE_REPORT_ROOT=$BASE_REPORT_ROOT"
echo "  OUT_ROOT=$OUT_ROOT"
echo "  REPORT_ROOT=$REPORT_ROOT"
echo "  KD_TRAIN_FILE=$KD_TRAIN_FILE"
echo "  PROMPT_FORMAT=$PROMPT_FORMAT"
echo "  KD_LIMIT=$KD_LIMIT"
echo "  KD_TEACHER_TEMPERATURE=$KD_TEACHER_TEMPERATURE"
echo "  EPOCHS=$EPOCHS"
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

"$PYTHON_BIN" scripts/generate_teacher_continuations.py \
  --model-path "$MODEL_PATH" \
  --teacher-adapter-dir "$TEACHER_ADAPTER_DIR" \
  --prompts-file "$TRAIN_FILE" \
  --out "$KD_TRAIN_FILE" \
  --prompt-format "$PROMPT_FORMAT" \
  --limit "$KD_LIMIT" \
  --batch-size "$KD_GEN_BATCH_SIZE" \
  --max-new-tokens "$KD_MAX_NEW_TOKENS" \
  --temperature "$KD_TEACHER_TEMPERATURE"

"$PYTHON_BIN" scripts/train_sft_trl.py \
  --model-path "$MODEL_PATH" \
  --train-file "$KD_TRAIN_FILE" \
  --output-dir "$OUT_ROOT/kd_lora" \
  --prompt-format "$PROMPT_FORMAT" \
  --limit "$KD_LIMIT" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum "$GRAD_ACCUM" \
  --lr "$LR" \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT"

"$PYTHON_BIN" scripts/evaluate_gsm8k.py \
  --model-path "$MODEL_PATH" \
  --adapter-dir "$OUT_ROOT/kd_lora" \
  --test-file "$TEST_FILE" \
  --out-dir "$REPORT_ROOT/kd_eval" \
  --model-name offline_kd \
  --limit "$EVAL_LIMIT" \
  --batch-size "$GSM8K_EVAL_BATCH_SIZE" \
  --max-new-tokens "$GSM8K_MAX_NEW_TOKENS" \
  --prompt-format "$PROMPT_FORMAT"

"$PYTHON_BIN" scripts/evaluate_truthfulqa_mc.py \
  --model-path "$MODEL_PATH" \
  --adapter-dir "$OUT_ROOT/kd_lora" \
  --test-file "$TRUTHFULQA_FILE" \
  --out-dir "$REPORT_ROOT/kd_truthfulqa_eval" \
  --model-name offline_kd \
  --limit "$RETENTION_LIMIT"

"$PYTHON_BIN" scripts/evaluate_mmlu_mc.py \
  --model-path "$MODEL_PATH" \
  --adapter-dir "$OUT_ROOT/kd_lora" \
  --test-file "$MMLU_FILE" \
  --out-dir "$REPORT_ROOT/kd_mmlu_eval" \
  --model-name offline_kd \
  --limit "$MMLU_LIMIT"

"$PYTHON_BIN" scripts/collect_rollouts.py \
  --model-path "$MODEL_PATH" \
  --adapter-dir "$OUT_ROOT/kd_lora" \
  --prompts-file "$TEST_FILE" \
  --out "$REPORT_ROOT/kd_states.jsonl" \
  --model-name offline_kd \
  --limit "$ROLLOUT_LIMIT" \
  --prompt-format "$PROMPT_FORMAT"

"$PYTHON_BIN" scripts/measure_state_drift.py \
  --left "$REPORT_ROOT/base_states.jsonl" \
  --right "$REPORT_ROOT/kd_states.jsonl" \
  --out "$REPORT_ROOT/state_drift.json" \
  --max-states "$DRIFT_MAX_STATES"

"$PYTHON_BIN" scripts/merge_scores.py \
  --base-metrics "$REPORT_ROOT/base_eval/metrics.json" \
  --post-metrics "$REPORT_ROOT/kd_eval/metrics.json" \
  --out "$REPORT_ROOT/scores.jsonl" \
  --role target \
  --post-model offline_kd \
  --base-retention-metrics "$REPORT_ROOT/base_truthfulqa_eval/metrics.json" \
  --post-retention-metrics "$REPORT_ROOT/kd_truthfulqa_eval/metrics.json" \
  --retention-task truthfulqa_mc1 \
  --retention-pair "$REPORT_ROOT/base_mmlu_eval/metrics.json" "$REPORT_ROOT/kd_mmlu_eval/metrics.json" mmlu_selected

"$PYTHON_BIN" scripts/analyze_forgetting.py \
  --scores "$REPORT_ROOT/scores.jsonl" \
  --drift "$REPORT_ROOT/state_drift.json" \
  --out "$REPORT_ROOT/kd_report.json" \
  --post-model offline_kd

echo "Offline KD report: $REPORT_ROOT/kd_report.json"
