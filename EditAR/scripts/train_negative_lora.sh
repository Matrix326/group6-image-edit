#!/bin/bash
set -euo pipefail

cd "${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

PIPE_PATH="${PIPE_PATH:-/path/to/edit-data/PIPE_5k_HF}"
MAGICBRUSH_PATH="${MAGICBRUSH_PATH:-/path/to/edit-data/MagicBrush_HF}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc-per-node=2 --master-port=25002 \
        autoregressive/train/train_edit.py \
        --output-dir checkpoints/editar_negative_lora \
        --vq-ckpt pretrained_models/vq_ds16_t2i.pt \
        --image-size 512 \
        --gpt-model GPT-XL \
        --gpt-mode 'joint_cls_emb' \
        --gpt-ckpt checkpoints/editar/editar_release/editar_release.pt \
        --no-compile \
        --num-workers 4 \
        --global-batch-size 4 \
        --gradient-accumulation-steps 8 \
        --epochs 4 \
        --lr 1e-4 \
        --lr-scheduler warmup_cosine \
        --warmup-steps 100 \
        --min-lr 1e-6 \
        --ckpt-every 1000 \
        --use-wandb \
        --use-lora \
        --lora-rank 8 \
        --lora-alpha 16 \
        --lora-dropout 0.05 \
        --use-negative-contrastive \
        --negative-contrastive-weight 0.1 \
        --negative-contrastive-margin 0.2 \
        --use-mask-weighted-loss \
        --lambda-edit 3.0 \
        --lambda-bg 1.0 \
        --use-distill \
        --distill-mode dinov2 \
        --distill-loss-weight 0.5 \
        --dataset-list pipe magicbrush \
        --pipe-path "${PIPE_PATH}" \
        --pipe-prob 0.4 \
        --magicbrush-path "${MAGICBRUSH_PATH}" \
        --magicbrush-prob 1.0
