#!/bin/bash
set -euo pipefail

cd "${REPO_DIR:-/path/to/EditAR}"

ACCELERATE="${ACCELERATE:-/path/to/EditAR/.venv/bin/accelerate}"
PYTHON="${PYTHON:-/path/to/EditAR/.venv/bin/python}"
BASE_CKPT="${BASE_CKPT:-checkpoints/editar/editar_release/editar_release.pt}"
VQ_CKPT="${VQ_CKPT:-pretrained_models/vq_ds16_t2i.pt}"
MAGICBRUSH_PATH="${MAGICBRUSH_PATH:-/path/to/edit-data/MagicBrush_HF}"

GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
GRAD_ACCUM=${GRAD_ACCUM:-1}
EPOCHS=${EPOCHS:-20}
LR=${LR:-2e-5}
MIN_LR=${MIN_LR:-1e-6}
WARMUP_STEPS=${WARMUP_STEPS:-200}
CKPT_EVERY=${CKPT_EVERY:-200}
EVAL_EVERY=${EVAL_EVERY:-1000}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-16}
CFG_SCALE=${CFG_SCALE:-1.0}

SMALL_LORA_TARGETS="cap_proj.fc1,cap_proj.fc2"
LARGE_LORA_TARGETS="wqkv,wo,w1,w2,w3,cap_proj.fc1,cap_proj.fc2"

mkdir -p outputs/logs/magicbrush_only_lora_sweep outputs/checkpoints outputs/inference outputs/benchmark

run_train() {
  local exp_name="$1"
  local gpus="$2"
  local port="$3"
  shift 3

  mkdir -p "outputs/logs/${exp_name}" "outputs/checkpoints/${exp_name}"
  echo "[$(date '+%F %T')] train ${exp_name} on CUDA_VISIBLE_DEVICES=${gpus}" | tee -a "outputs/logs/magicbrush_only_lora_sweep/${exp_name}.launcher.log"

  CUDA_VISIBLE_DEVICES="${gpus}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    NCCL_IB_DISABLE=1 \
    NCCL_P2P_DISABLE=1 \
    NCCL_ASYNC_ERROR_HANDLING=1 \
    TORCH_NCCL_BLOCKING_WAIT=1 \
    "${ACCELERATE}" launch \
      --num_processes 2 \
      --num_machines 1 \
      --main_process_port "${port}" \
      --mixed_precision bf16 \
      --dynamo_backend no \
      autoregressive/train/train_edit.py \
      --output-dir "outputs/checkpoints/${exp_name}" \
      --vq-ckpt "${VQ_CKPT}" \
      --image-size 512 \
      --gpt-model GPT-XL \
      --gpt-mode joint_cls_emb \
      --gpt-ckpt "${BASE_CKPT}" \
      --no-compile \
      --num-workers 4 \
      --global-batch-size "${GLOBAL_BATCH_SIZE}" \
      --gradient-accumulation-steps "${GRAD_ACCUM}" \
      --epochs "${EPOCHS}" \
      --lr "${LR}" \
      --lr-scheduler warmup_cosine \
      --warmup-steps "${WARMUP_STEPS}" \
      --min-lr "${MIN_LR}" \
      --ckpt-every "${CKPT_EVERY}" \
      --use-wandb \
      --use-lora \
      --lora-rank 8 \
      --lora-alpha 16 \
      --lora-dropout 0.05 \
      --dataset-list magicbrush \
      --magicbrush-path "${MAGICBRUSH_PATH}" \
      --magicbrush-prob 1.0 \
      "$@" \
      > "outputs/logs/${exp_name}/train.log" 2>&1
}

eval_checkpoint() {
  local exp_name="$1"
  local gpus="$2"
  local ckpt_path="$3"
  local step_name
  step_name="$(basename "${ckpt_path}" .pt)"

  local eval_name="${exp_name}_${step_name}_magicbrush${EVAL_MAX_SAMPLES}"
  local out_root="outputs/inference/${eval_name}"
  local bench_root="outputs/benchmark/${eval_name}"
  local log_path="outputs/logs/magicbrush_only_lora_sweep/${eval_name}.eval.log"

  mkdir -p "${out_root}" "${bench_root}"
  echo "[$(date '+%F %T')] eval ${eval_name} on CUDA_VISIBLE_DEVICES=${gpus}" | tee -a "${log_path}"

  CUDA_VISIBLE_DEVICES="${gpus}" "${PYTHON}" autoregressive/sample/sample_edit_folder.py \
    --gpt-ckpt "${BASE_CKPT}" \
    --lora-ckpt "${ckpt_path}" \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lora-dropout 0.05 \
    --vq-ckpt "${VQ_CKPT}" \
    --image-size 512 \
    --gpt-model GPT-XL \
    --gpt-mode joint_cls_emb \
    --testset magicbrush \
    --magicbrush-path "${MAGICBRUSH_PATH}" \
    --output-dir "${out_root}" \
    --max-samples "${EVAL_MAX_SAMPLES}" \
    --cfg-scale "${CFG_SCALE}" \
    --top-k 1000 \
    --top-p 1.0 \
    --temperature 1.0 \
    --mixed-precision bf16 \
    >> "${log_path}" 2>&1

  CUDA_VISIBLE_DEVICES="${gpus}" "${PYTHON}" tools/evaluate_magicbrush_outputs.py \
    --magicbrush-path "${MAGICBRUSH_PATH}" \
    --samples-dir "${out_root}/magicbrush/samples/txt_${CFG_SCALE}" \
    --output-dir "${bench_root}" \
    --cfg-scale "${CFG_SCALE}" \
    --max-samples "${EVAL_MAX_SAMPLES}" \
    >> "${log_path}" 2>&1
}

eval_experiment() {
  local exp_name="$1"
  local eval_gpu="$2"

  shopt -s nullglob
  for ckpt_path in "outputs/checkpoints/${exp_name}/checkpoints/"*.pt; do
    local step_name step_num
    step_name="$(basename "${ckpt_path}" .pt)"
    step_num=$((10#${step_name}))
    if (( step_num % EVAL_EVERY == 0 )); then
      eval_checkpoint "${exp_name}" "${eval_gpu}" "${ckpt_path}"
    fi
  done
}

summarize_results() {
  "${PYTHON}" - <<'PY'
import json
from pathlib import Path

root = Path("outputs/benchmark")
rows = []
for path in sorted(root.glob("mb_only_*_magicbrush16/summary.json")):
    data = json.loads(path.read_text())
    data["experiment"] = path.parent.name
    data["status"] = "done"
    rows.append(data)
out = root / "magicbrush_only_lora_sweep16_summary.json"
out.write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
PY
}

queue_a() {
  run_train "mb_only_small_lora_lr2e5" "0,1" "25201" \
    --lora-target-modules "${SMALL_LORA_TARGETS}"
  eval_experiment "mb_only_small_lora_lr2e5" "0"

  run_train "mb_only_large_lora_lr2e5" "0,1" "25201" \
    --lora-target-modules "${LARGE_LORA_TARGETS}"
  eval_experiment "mb_only_large_lora_lr2e5" "0"
}

queue_b() {
  run_train "mb_only_mask_lora_lr2e5" "2,3" "25202" \
    --lora-target-modules "${LARGE_LORA_TARGETS}" \
    --use-mask-weighted-loss \
    --lambda-edit 1.5 \
    --lambda-bg 1.0
  eval_experiment "mb_only_mask_lora_lr2e5" "2"

  run_train "mb_only_negative_lora_lr2e5" "2,3" "25202" \
    --lora-target-modules "${LARGE_LORA_TARGETS}" \
    --use-mask-weighted-loss \
    --lambda-edit 1.5 \
    --lambda-bg 1.0 \
    --use-negative-contrastive \
    --negative-contrastive-weight 0.03 \
    --negative-contrastive-margin 0.1
  eval_experiment "mb_only_negative_lora_lr2e5" "2"
}

queue_a > outputs/logs/magicbrush_only_lora_sweep/queue_a.log 2>&1 &
pid_a=$!
echo "${pid_a}" > outputs/logs/magicbrush_only_lora_sweep/queue_a.pid

queue_b > outputs/logs/magicbrush_only_lora_sweep/queue_b.log 2>&1 &
pid_b=$!
echo "${pid_b}" > outputs/logs/magicbrush_only_lora_sweep/queue_b.pid

wait "${pid_a}" "${pid_b}"
summarize_results

echo "[$(date '+%F %T')] magicbrush-only LoRA sweep finished"
