#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/dong/explore/hf_models/Qwen3-0.6B-Base}"
TEACHER_ADAPTER_DIR="${TEACHER_ADAPTER_DIR:-outputs/phase1_qwen3_0_6b_base_plain_medium_sft/sft_lora}"
BASE_REPORT_ROOT="${BASE_REPORT_ROOT:-reports/phase1_qwen3_0_6b_base_plain_medium_sft}"
OUT_ROOT="${OUT_ROOT:-outputs/opd_qwen3_0_6b_base}"
REPORT_ROOT="${REPORT_ROOT:-reports/opd_qwen3_0_6b_base}"
PYTHON_BIN="${PYTHON_BIN:-/home/dong/anaconda3/envs/qwen_sft/bin/python}"
TRAIN_FILE="${TRAIN_FILE:-data/gsm8k/main_train.jsonl}"
TEST_FILE="${TEST_FILE:-data/gsm8k/main_test.jsonl}"
TRUTHFULQA_FILE="${TRUTHFULQA_FILE:-data/truthfulqa/validation_mc1.jsonl}"
MMLU_FILE="${MMLU_FILE:-data/mmlu/test_selected.jsonl}"
PROMPT_FORMAT="${PROMPT_FORMAT:-plain}"
OPD_LIMIT="${OPD_LIMIT:-1000}"
OPD_UPDATES="${OPD_UPDATES:-200}"
OPD_PROMPTS_PER_UPDATE="${OPD_PROMPTS_PER_UPDATE:-4}"
OPD_STATE_BATCH_SIZE="${OPD_STATE_BATCH_SIZE:-8}"
OPD_MAX_STATES_PER_UPDATE="${OPD_MAX_STATES_PER_UPDATE:-32}"
OPD_ROLLOUT_MAX_NEW_TOKENS="${OPD_ROLLOUT_MAX_NEW_TOKENS:-96}"
OPD_STATE_STRIDE="${OPD_STATE_STRIDE:-8}"
OPD_OBJECTIVE="${OPD_OBJECTIVE:-teacher_continuation_ce}"
OPD_TEACHER_CONTINUATION_MAX_NEW_TOKENS="${OPD_TEACHER_CONTINUATION_MAX_NEW_TOKENS:-64}"
OPD_TEACHER_TEMPERATURE="${OPD_TEACHER_TEMPERATURE:-0.0}"
OPD_ROLLOUT_TEMPERATURE="${OPD_ROLLOUT_TEMPERATURE:-0.7}"
OPD_ROLLOUT_TOP_P="${OPD_ROLLOUT_TOP_P:-0.95}"
LR="${LR:-2e-4}"
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

echo "OPD config:"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  TEACHER_ADAPTER_DIR=$TEACHER_ADAPTER_DIR"
echo "  BASE_REPORT_ROOT=$BASE_REPORT_ROOT"
echo "  OUT_ROOT=$OUT_ROOT"
echo "  REPORT_ROOT=$REPORT_ROOT"
echo "  PROMPT_FORMAT=$PROMPT_FORMAT"
echo "  OPD_LIMIT=$OPD_LIMIT"
echo "  OPD_UPDATES=$OPD_UPDATES"
echo "  OPD_PROMPTS_PER_UPDATE=$OPD_PROMPTS_PER_UPDATE"
echo "  OPD_STATE_BATCH_SIZE=$OPD_STATE_BATCH_SIZE"
echo "  OPD_MAX_STATES_PER_UPDATE=$OPD_MAX_STATES_PER_UPDATE"
echo "  OPD_ROLLOUT_MAX_NEW_TOKENS=$OPD_ROLLOUT_MAX_NEW_TOKENS"
echo "  OPD_OBJECTIVE=$OPD_OBJECTIVE"
echo "  OPD_TEACHER_CONTINUATION_MAX_NEW_TOKENS=$OPD_TEACHER_CONTINUATION_MAX_NEW_TOKENS"
echo "  OPD_TEACHER_TEMPERATURE=$OPD_TEACHER_TEMPERATURE"
echo "  OPD_ROLLOUT_TEMPERATURE=$OPD_ROLLOUT_TEMPERATURE"
echo "  OPD_ROLLOUT_TOP_P=$OPD_ROLLOUT_TOP_P"
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

"$PYTHON_BIN" scripts/train_opd_lora.py \
  --model-path "$MODEL_PATH" \
  --teacher-adapter-dir "$TEACHER_ADAPTER_DIR" \
  --prompts-file "$TRAIN_FILE" \
  --output-dir "$OUT_ROOT/opd_lora" \
  --prompt-format "$PROMPT_FORMAT" \
  --limit "$OPD_LIMIT" \
  --updates "$OPD_UPDATES" \
  --prompts-per-update "$OPD_PROMPTS_PER_UPDATE" \
  --state-batch-size "$OPD_STATE_BATCH_SIZE" \
  --max-states-per-update "$OPD_MAX_STATES_PER_UPDATE" \
  --rollout-max-new-tokens "$OPD_ROLLOUT_MAX_NEW_TOKENS" \
  --state-stride "$OPD_STATE_STRIDE" \
  --objective "$OPD_OBJECTIVE" \
  --teacher-continuation-max-new-tokens "$OPD_TEACHER_CONTINUATION_MAX_NEW_TOKENS" \
  --teacher-temperature "$OPD_TEACHER_TEMPERATURE" \
  --temperature "$OPD_ROLLOUT_TEMPERATURE" \
  --top-p "$OPD_ROLLOUT_TOP_P" \
  --lr "$LR" \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT"

"$PYTHON_BIN" scripts/evaluate_gsm8k.py \
  --model-path "$MODEL_PATH" \
  --adapter-dir "$OUT_ROOT/opd_lora" \
  --test-file "$TEST_FILE" \
  --out-dir "$REPORT_ROOT/opd_eval" \
  --model-name opd \
  --limit "$EVAL_LIMIT" \
  --batch-size "$GSM8K_EVAL_BATCH_SIZE" \
  --max-new-tokens "$GSM8K_MAX_NEW_TOKENS" \
  --prompt-format "$PROMPT_FORMAT"

"$PYTHON_BIN" scripts/evaluate_truthfulqa_mc.py \
  --model-path "$MODEL_PATH" \
  --adapter-dir "$OUT_ROOT/opd_lora" \
  --test-file "$TRUTHFULQA_FILE" \
  --out-dir "$REPORT_ROOT/opd_truthfulqa_eval" \
  --model-name opd \
  --limit "$RETENTION_LIMIT"

"$PYTHON_BIN" scripts/evaluate_mmlu_mc.py \
  --model-path "$MODEL_PATH" \
  --adapter-dir "$OUT_ROOT/opd_lora" \
  --test-file "$MMLU_FILE" \
  --out-dir "$REPORT_ROOT/opd_mmlu_eval" \
  --model-name opd \
  --limit "$MMLU_LIMIT"

"$PYTHON_BIN" scripts/collect_rollouts.py \
  --model-path "$MODEL_PATH" \
  --adapter-dir "$OUT_ROOT/opd_lora" \
  --prompts-file "$TEST_FILE" \
  --out "$REPORT_ROOT/opd_states.jsonl" \
  --model-name opd \
  --limit "$ROLLOUT_LIMIT" \
  --prompt-format "$PROMPT_FORMAT"

"$PYTHON_BIN" scripts/measure_state_drift.py \
  --left "$REPORT_ROOT/base_states.jsonl" \
  --right "$REPORT_ROOT/opd_states.jsonl" \
  --out "$REPORT_ROOT/state_drift.json" \
  --max-states "$DRIFT_MAX_STATES"

"$PYTHON_BIN" scripts/merge_scores.py \
  --base-metrics "$REPORT_ROOT/base_eval/metrics.json" \
  --post-metrics "$REPORT_ROOT/opd_eval/metrics.json" \
  --out "$REPORT_ROOT/scores.jsonl" \
  --role target \
  --post-model opd \
  --base-retention-metrics "$REPORT_ROOT/base_truthfulqa_eval/metrics.json" \
  --post-retention-metrics "$REPORT_ROOT/opd_truthfulqa_eval/metrics.json" \
  --retention-task truthfulqa_mc1 \
  --retention-pair "$REPORT_ROOT/base_mmlu_eval/metrics.json" "$REPORT_ROOT/opd_mmlu_eval/metrics.json" mmlu_selected

"$PYTHON_BIN" scripts/analyze_forgetting.py \
  --scores "$REPORT_ROOT/scores.jsonl" \
  --drift "$REPORT_ROOT/state_drift.json" \
  --out "$REPORT_ROOT/opd_report.json" \
  --post-model opd

echo "OPD report: $REPORT_ROOT/opd_report.json"
