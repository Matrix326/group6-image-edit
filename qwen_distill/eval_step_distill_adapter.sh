#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/share/home/group6/Project/group6-image-edit}
STUDENT_JSON=${STUDENT_JSON:-${ROOT}/outputs/qwen_step_distill/cache_dev60/qwen_distill_lora_speed_results.json}
CKPT=${CKPT:-${ROOT}/checkpoints/base4_to_base40/step_distill_adapter_best.pt}
OUT=${OUT:-${ROOT}/outputs/qwen_step_distill/eval_base4_adapter_dev60}

cd "${ROOT}"
PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" \
python -m keepedit.qwen_distill.eval_adapter \
  --student_results_json "${STUDENT_JSON}" \
  --checkpoint "${CKPT}" \
  --variant_name "${VARIANT_NAME:-qwen_4step_adapter}" \
  --output_dir "${OUT}" \
  --hidden "${HIDDEN:-128}" \
  --image_size "${IMAGE_SIZE:-512}" \
  --grid_rows "${GRID_ROWS:-20}"
