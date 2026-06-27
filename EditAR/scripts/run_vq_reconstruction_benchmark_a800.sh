#!/bin/bash
set -euo pipefail

cd "${REPO_DIR:-/path/to/EditAR}"

PYTHON=${PYTHON:-/path/to/EditAR/.venv/bin/python}
VQ_CKPT=${VQ_CKPT:-pretrained_models/vq_ds16_t2i.pt}
HF_DATASET=${HF_DATASET:-/path/to/edit-data/MagicBrush_HF}
HF_SPLIT=${HF_SPLIT:-dev}
MAX_SAMPLES=${MAX_SAMPLES:-64}
BATCH_SIZE=${BATCH_SIZE:-8}
IMAGE_SIZE=${IMAGE_SIZE:-512}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/vq_reconstruction_benchmark/magicbrush_dev_${MAX_SAMPLES}}
LOG_DIR=${LOG_DIR:-outputs/logs/vq_reconstruction_benchmark}

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

run_field() {
  local field="$1"
  local gpu="$2"
  local out_dir="${OUTPUT_ROOT}/${field}"
  local log_path="${LOG_DIR}/${field}.log"

  mkdir -p "${out_dir}"
  echo "[$(date '+%F %T')] start VQ reconstruction for ${field} on CUDA ${gpu}" | tee "${log_path}"

  (
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" tools/vq_reconstruct_benchmark.py \
      --hf-dataset "${HF_DATASET}" \
      --hf-split "${HF_SPLIT}" \
      --hf-image-field "${field}" \
      --vq-ckpt "${VQ_CKPT}" \
      --vq-model VQ-16 \
      --codebook-size 16384 \
      --codebook-embed-dim 8 \
      --image-size "${IMAGE_SIZE}" \
      --batch-size "${BATCH_SIZE}" \
      --max-samples "${MAX_SAMPLES}" \
      --output-dir "${out_dir}" \
      --save-tokens

    echo "[$(date '+%F %T')] done VQ reconstruction for ${field}" | tee -a "${log_path}"
  ) >> "${log_path}" 2>&1 &

  echo "$!" > "${LOG_DIR}/${field}.pid"
}

run_field "source_img" "5"
run_field "target_img" "6"

wait

"${PYTHON}" - <<'PY'
import json
from pathlib import Path

root = Path("outputs/vq_reconstruction_benchmark")
latest = sorted(root.glob("magicbrush_dev_*"), key=lambda p: p.stat().st_mtime)[-1]
rows = {}
for field in ["source_img", "target_img"]:
    summary_path = latest / field / "summary.json"
    rows[field] = json.loads(summary_path.read_text()) if summary_path.exists() else {"status": "missing"}
out = latest / "summary.json"
out.write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
PY

echo "[$(date '+%F %T')] all VQ reconstruction benchmark jobs finished: ${OUTPUT_ROOT}"
