#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/share/home/group6/Project/group6-image-edit}
CACHE_JSON=${CACHE_JSON:-${ROOT}/outputs/qwen_step_distill/cache_dev60/qwen_distill_lora_speed_results.json}
OUT=${OUT:-${ROOT}/outputs/qwen_step_distill/step_distill_train.json}

cd "${ROOT}"
PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" \
python -m keepedit.qwen_distill.export_step_distill_metadata \
  --cache_results_json "${CACHE_JSON}" \
  --output_json "${OUT}" \
  --student_variant "${STUDENT_VARIANT:-base4}" \
  --teacher_variant "${TEACHER_VARIANT:-base40}"
