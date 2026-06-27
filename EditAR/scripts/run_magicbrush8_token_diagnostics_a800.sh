#!/bin/bash
set -euo pipefail

cd "${REPO_DIR:-/path/to/EditAR}"

PYTHON="${PYTHON:-/path/to/EditAR/.venv/bin/python}"
BASE_CKPT=${BASE_CKPT:-checkpoints/editar/editar_release/editar_release.pt}
VQ_CKPT=${VQ_CKPT:-pretrained_models/vq_ds16_t2i.pt}
MAGICBRUSH_PATH=${MAGICBRUSH_PATH:-/path/to/edit-data/MagicBrush_HF}
OUT_DIR=${OUT_DIR:-outputs/diagnostics/magicbrush8_baseline}
CUDA_DEVICES=${CUDA_DEVICES:-0}
MAX_SAMPLES=${MAX_SAMPLES:-8}
SEED=${SEED:-0}
DIAGNOSTIC_MODE=${DIAGNOSTIC_MODE:-both}

mkdir -p "$(dirname "${OUT_DIR}")" outputs/logs/token_diagnostics

echo "[$(date '+%F %T')] MagicBrush token diagnostics -> ${OUT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON}" tools/diagnose_magicbrush_tokens.py \
  --gpt-ckpt "${BASE_CKPT}" \
  --vq-ckpt "${VQ_CKPT}" \
  --magicbrush-path "${MAGICBRUSH_PATH}" \
  --output-dir "${OUT_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --diagnostic-mode "${DIAGNOSTIC_MODE}" \
  --seed "${SEED}" \
  --image-size 512 \
  --gpt-model GPT-XL \
  --gpt-mode joint_cls_emb \
  --cfg-scale 1.0 \
  --top-k 1000 \
  --top-p 1.0 \
  --temperature 1.0 \
  --topk-save 64 \
  --mixed-precision bf16

echo "[$(date '+%F %T')] done"
