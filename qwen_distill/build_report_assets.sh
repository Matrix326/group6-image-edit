#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/share/home/group6/Project/group6-image-edit}
REPORT=${REPORT:-${ROOT}/reports/qwen_distill}

test -f "${ROOT}/checkpoints/base4_to_base40/step_distill_adapter_best.pt"
test -f "${ROOT}/logs/base4_to_base40/train_step_loss.csv"
test -f "${ROOT}/logs/base4_to_base40/step_loss_curve.png"
test -f "${REPORT}/qwen_distill_dev60_summary.md"
test -f "${REPORT}/qwen_distill_dev60_eval.json"
test -f "${REPORT}/figures/sample_representative_one_row.png"

echo "Report assets are ready under ${REPORT}"
