#!/bin/bash
set -euo pipefail

cd "${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

torchrun --nnodes=1 --nproc-per-node=8 --master-port=25001 \
        autoregressive/train/train_edit.py \
        --output-dir checkpoints/editar \
        --vq-ckpt pretrained_models/vq_ds16_t2i.pt \
        --image-size 512 \
        --gpt-model GPT-XL \
        --gpt-mode 'joint_cls_emb' \
        --gpt-ckpt pretrained_models/t2i_XL_stage2_512.pt \
        --num-workers 4 \
        --epochs 4 \
        --use-distill \
        --distill-mode dinov2 \
        --distill-loss-weight 0.5 \
        --dataset-list multigendepth multigencanny conditionsegmentation pipe seedxunsplash \
        --multigendepth-path data/MultiGen-20M_depth_HF/ \
        --multigendepth-prob 0.15 \
        --multigencanny-path data/MultiGen-20M_depth_HF/ \
        --multigencanny-prob 0.3 \
        --conditionsegmentation-path data/Condition_Segmentation/ \
        --conditionsegmentation-prob 0.45 \
        --pipe-path data/PIPE_HF \
        --pipe-prob 0.7 \
        --seedxunsplash-path data/Seedx_Unsplash_HF \
        --seedxunsplash-prob 1.0
