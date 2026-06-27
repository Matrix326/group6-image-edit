#!/usr/bin/env bash
set -euo pipefail

cd "${REPO_DIR:-/path/to/EditAR}"

PYTHON="${PYTHON:-/path/to/EditAR/.venv/bin/python}"
BASE_CKPT="${BASE_CKPT:-checkpoints/editar/editar_release/editar_release.pt}"
VQ_CKPT="${VQ_CKPT:-pretrained_models/vq_ds16_t2i.pt}"
MAGICBRUSH_PATH="${MAGICBRUSH_PATH:-/path/to/edit-data/MagicBrush_HF}"
MAX_SAMPLES="${MAX_SAMPLES:-60}"
CFG_SCALE="${CFG_SCALE:-1.0}"
GPUS_CSV="${GPUS:-0,1,2,3}"
NO_CLIP="${NO_CLIP:-1}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/path/to/cache}"
export HF_HOME="${HF_HOME:-/path/to/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/path/to/cache/huggingface/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/path/to/cache/huggingface/transformers}"

SMALL_TARGETS="cap_proj.fc1,cap_proj.fc2"
LARGE_TARGETS="wqkv,wo,w1,w2,w3,cap_proj.fc1,cap_proj.fc2"

IFS=',' read -r -a GPU_LIST <<< "${GPUS_CSV}"
if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "No GPUs configured. Set GPUS=0,1,2,3." >&2
  exit 1
fi

RUN_ROOT="outputs/report_benchmark_magicbrush60"
LOG_DIR="outputs/logs/report_benchmark_magicbrush60"
mkdir -p "${RUN_ROOT}/samples" "${RUN_ROOT}/benchmark" "${LOG_DIR}"

run_one() {
  local exp_name="$1"
  local gpu="$2"
  local lora_ckpt="${3:-}"
  local lora_targets="${4:-${LARGE_TARGETS}}"

  local out_root="${RUN_ROOT}/samples/${exp_name}"
  local bench_root="${RUN_ROOT}/benchmark/${exp_name}"
  local log_path="${LOG_DIR}/${exp_name}.log"

  mkdir -p "${out_root}" "${bench_root}"
  echo "[$(date '+%F %T')] start ${exp_name} on CUDA_VISIBLE_DEVICES=${gpu}" | tee "${log_path}"

  local lora_args=()
  if [[ -n "${lora_ckpt}" ]]; then
    lora_args=(
      --lora-ckpt "${lora_ckpt}"
      --lora-rank 8
      --lora-alpha 16
      --lora-dropout 0.05
      --lora-target-modules "${lora_targets}"
    )
  fi

  local clip_args=()
  if [[ "${NO_CLIP}" == "1" ]]; then
    clip_args=(--no-clip)
  fi

  (
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" autoregressive/sample/sample_edit_folder.py \
      --gpt-ckpt "${BASE_CKPT}" \
      "${lora_args[@]}" \
      --vq-ckpt "${VQ_CKPT}" \
      --image-size 512 \
      --gpt-model GPT-XL \
      --gpt-mode joint_cls_emb \
      --testset magicbrush \
      --magicbrush-path "${MAGICBRUSH_PATH}" \
      --output-dir "${out_root}" \
      --max-samples "${MAX_SAMPLES}" \
      --cfg-scale "${CFG_SCALE}" \
      --top-k 1000 \
      --top-p 1.0 \
      --temperature 1.0 \
      --mixed-precision bf16

    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" tools/evaluate_magicbrush_outputs.py \
      --magicbrush-path "${MAGICBRUSH_PATH}" \
      --samples-dir "${out_root}/magicbrush/samples/txt_${CFG_SCALE}" \
      --output-dir "${bench_root}" \
      --cfg-scale "${CFG_SCALE}" \
      --max-samples "${MAX_SAMPLES}" \
      --image-size 512 \
      "${clip_args[@]}"

    echo "[$(date '+%F %T')] done ${exp_name}"
  ) >> "${log_path}" 2>&1
}

wait_for_slot() {
  local max_jobs="${#GPU_LIST[@]}"
  while [[ "$(jobs -rp | wc -l)" -ge "${max_jobs}" ]]; do
    sleep 10
  done
}

launch() {
  local exp_name="$1"
  local lora_ckpt="${2:-}"
  local lora_targets="${3:-${LARGE_TARGETS}}"
  local gpu="${GPU_LIST[$((LAUNCHED % ${#GPU_LIST[@]}))]}"
  wait_for_slot
  run_one "${exp_name}" "${gpu}" "${lora_ckpt}" "${lora_targets}" &
  echo "$!" > "${LOG_DIR}/${exp_name}.pid"
  LAUNCHED=$((LAUNCHED + 1))
}

LAUNCHED=0

launch "pretrained_magicbrush60_report" "" "${LARGE_TARGETS}"
launch "negative_lora_1500_magicbrush60_report" "checkpoints/editar_negative_lora/checkpoints/0001500.pt" "${LARGE_TARGETS}"
launch "mb_only_small_lora_1000_magicbrush60_report" "outputs/checkpoints/mb_only_small_lora_lr2e5/checkpoints/0001000.pt" "${SMALL_TARGETS}"
launch "mb_only_large_lora_5000_magicbrush60_report" "outputs/checkpoints/mb_only_large_lora_lr2e5/checkpoints/0005000.pt" "${LARGE_TARGETS}"
launch "mb_only_mask_lora_8000_magicbrush60_report" "outputs/checkpoints/mb_only_mask_lora_lr2e5/checkpoints/0008000.pt" "${LARGE_TARGETS}"

wait

"${PYTHON}" - <<'PY'
import json
from pathlib import Path

root = Path("outputs/report_benchmark_magicbrush60/benchmark")
experiments = [
    "pretrained_magicbrush60_report",
    "negative_lora_1500_magicbrush60_report",
    "mb_only_small_lora_1000_magicbrush60_report",
    "mb_only_large_lora_5000_magicbrush60_report",
    "mb_only_mask_lora_8000_magicbrush60_report",
]

rows = []
for exp in experiments:
    path = root / exp / "summary.json"
    if not path.exists():
        rows.append({"experiment": exp, "status": "missing"})
        continue
    data = json.loads(path.read_text())
    data = {"experiment": exp, "status": "done", **data}
    rows.append(data)

out = root.parent / "magicbrush60_report_summary.json"
out.write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
PY

echo "[$(date '+%F %T')] all MagicBrush-60 report benchmark jobs finished"
