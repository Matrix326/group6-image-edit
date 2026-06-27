#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/share/home/group6/Project/group6-image-edit}
QWEN_ROOT=${QWEN_ROOT:-/share/home/group6/Project/qwen-image-edit-baseline}
MANIFEST=${MANIFEST:-/share/home/group6/our_project/artifacts/qwen_distill_dev60_final/magicbrush_dev60_keepedit.json}
OUT=${OUT:-${ROOT}/outputs/qwen_image_edit_baseline_dev60}
GPU=${GPU:-0}

cd "${ROOT}"
PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${GPU}" \
python -m keepedit.qwen_distill.cache_qwen_variants \
  --manifest "${MANIFEST}" \
  --output_dir "${OUT}" \
  --project_root "${QWEN_ROOT}" \
  --model_path models/Qwen-Image-Edit-2511 \
  --max_samples "${MAX_SAMPLES:-60}" \
  --variants "${VARIANTS:-base40:40:4.0:none}" \
  --seed "${SEED:-42}" \
  --resume
