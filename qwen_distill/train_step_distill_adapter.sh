#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/share/home/group6/Project/group6-image-edit}
METADATA=${METADATA:-${ROOT}/outputs/qwen_step_distill/step_distill_train.json}
OUT=${OUT:-${ROOT}/outputs/qwen_step_distill/train_base4_to_base40}
GPU=${GPU:-0}

cd "${ROOT}"
PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" \
python -m keepedit.qwen_distill.train_adapter \
  --metadata_json "${METADATA}" \
  --output_dir "${OUT}" \
  --hidden "${HIDDEN:-128}" \
  --image_size "${IMAGE_SIZE:-512}" \
  --batch_size "${BATCH_SIZE:-8}" \
  --epochs "${EPOCHS:-688}" \
  --learning_rate "${LR:-3e-5}" \
  --val_count "${VAL_COUNT:-10}" \
  --eval_every "${EVAL_EVERY:-5}" \
  --sample_every "${SAMPLE_EVERY:-5}" \
  --max_sample_rows "${MAX_SAMPLE_ROWS:-1}" \
  --log_step_every "${LOG_STEP_EVERY:-1}"
