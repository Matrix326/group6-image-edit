# Qwen-Image-Edit Baseline Reproduction

This document records the Qwen-Image-Edit baseline used in our MagicBrush experiments. It is intentionally separate from the KeepEdit LoRA workflow: this baseline runs the original Qwen-Image-Edit model without training.

## Setup

Default local paths:

```text
project repo:      /share/home/group6/Project/group6-image-edit
Qwen baseline:     /share/home/group6/Project/qwen-image-edit-baseline
Qwen model:        /share/home/group6/Project/qwen-image-edit-baseline/models/Qwen-Image-Edit-2511
MagicBrush dev60:  /share/home/group6/our_project/artifacts/qwen_distill_dev60_final/magicbrush_dev60_keepedit.json
```

Install this repository package:

```bash
cd /share/home/group6/Project/group6-image-edit
pip install -e ".[train]"
```

The Qwen baseline runner reuses the existing Qwen-Image-Edit loader in:

```text
/share/home/group6/Project/qwen-image-edit-baseline/scripts/run_qwen_edit_benchmark.py
```

## Baseline Definition

Input:

```text
source image + editing instruction
```

Output:

```text
Qwen-Image-Edit(source, instruction)
```

We use two inference settings:

| Method | Steps | true_cfg_scale | Role |
|---|---:|---:|---|
| Qwen-4step | 4 | 4.0 | fast student |
| Qwen-40step | 40 | 4.0 | high-quality teacher |

The target image and mask from MagicBrush are used only for evaluation.

## Run Qwen-40step Baseline

```bash
GPU=0 \
MAX_SAMPLES=60 \
bash scripts/qwen_distill/run_qwen_baseline_dev60.sh
```

Outputs:

```text
outputs/qwen_image_edit_baseline_dev60/
  dev_000000/
    input.png
    target.png
    prompt.txt
    base40.png
    base40.json
  qwen_distill_lora_speed_results.json
```

## Cache Qwen-4step and Qwen-40step for Distillation

```bash
GPU=0 \
MAX_SAMPLES=60 \
bash scripts/qwen_distill/cache_qwen_4step_40step.sh
```

This generates paired student/teacher outputs:

```text
outputs/qwen_step_distill/cache_dev60/
  dev_xxxxxx/base4.png
  dev_xxxxxx/base40.png
  qwen_distill_lora_speed_results.json
```

## Existing Reproduction Artifacts

The submitted artifact bundle includes the final dev60 summary and evaluation JSON:

```text
reports/qwen_distill/qwen_distill_dev60_summary.md
reports/qwen_distill/qwen_distill_dev60_eval.json
```

Main dev60 results:

| Method | SSIM(Target) | PSNR(Target) | BG-SSIM | BG-PSNR |
|---|---:|---:|---:|---:|
| Qwen-4step | 0.7535 | 16.2793 | 0.6018 | 18.3761 |
| Qwen-40step | 0.7804 | 17.9871 | 0.6910 | 20.2761 |
| Qwen-4step + Adapter | 0.8213 | 19.2725 | 0.7453 | 21.7426 |

## Notes

- The original Qwen model weights are not stored in this repository.
- The repository stores only lightweight adapter checkpoints, logs, curves, and summary files.
- To rerun full Qwen inference, make sure CUDA is visible and the Qwen baseline environment can import `run_qwen_edit_benchmark.py`.
