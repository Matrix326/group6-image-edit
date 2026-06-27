# Qwen-Image-Edit 4-Step Self-Distillation Adapter

## Method Summary

The goal is to keep Qwen-Image-Edit fast while improving low-step quality. We freeze Qwen-Image-Edit and train a lightweight gated residual adapter:

```text
input image + Qwen-4step output -> adapter -> refined output
```

The 40-step Qwen output is used as the teacher:

```text
student = Qwen-4step(source, instruction)
teacher = Qwen-40step(source, instruction)
```

The adapter is small and operates in image space, so it does not require full Qwen fine-tuning.

## Adapter Architecture

The adapter input has 9 channels:

```text
source image        3 channels
student image       3 channels
student - source    3 channels
```

It predicts a bounded RGB correction and a spatial alpha gate:

```text
delta = tanh(pred_rgb) * 0.25
alpha = sigmoid(pred_alpha)
output = clamp(student + alpha * delta, 0, 1)
```

This design encourages local residual refinement instead of full-image regeneration.

## Loss Function

Training loss:

```text
L = L1(output, teacher)
  + 0.25 * L1(grad(output), grad(teacher))
  + 0.05 * (1 - SSIM_proxy(output, teacher))
  + 0.02 * mean(alpha)
```

Loss terms:

| Term | Purpose |
|---|---|
| L1 | match the 40-step teacher |
| edge loss | preserve edges and avoid blurry averages |
| SSIM proxy | preserve global structure |
| alpha sparsity | discourage unnecessary edits |

## Training Command

First export a metadata JSON from a cache containing both `base4` and `base40` variants:

```bash
CACHE_JSON=outputs/qwen_step_distill/cache_dev60/qwen_distill_lora_speed_results.json \
OUT=outputs/qwen_step_distill/step_distill_train.json \
bash scripts/qwen_distill/export_step_distill_metadata.sh
```

The training script expects a metadata JSON with:

```json
{
  "sample_id": "train_000001",
  "prompt": "edit instruction",
  "input_image": "/path/to/input.png",
  "student_image": "/path/to/base4.png",
  "teacher_image": "/path/to/base40.png",
  "target_image": "/path/to/target.png"
}
```

Run:

```bash
METADATA=/path/to/step_distill_train.json \
GPU=0 \
bash scripts/qwen_distill/train_step_distill_adapter.sh
```

Important defaults:

```text
image size: 512
hidden dim: 128
batch size: 8
epochs: 688
learning rate: 3e-5
```

## Evaluation Command

```bash
STUDENT_JSON=outputs/qwen_step_distill/cache_dev60/qwen_distill_lora_speed_results.json \
CKPT=checkpoints/base4_to_base40/step_distill_adapter_best.pt \
bash scripts/qwen_distill/eval_step_distill_adapter.sh
```

## Included Training Evidence

```text
checkpoints/base4_to_base40/step_distill_adapter_best.pt
logs/base4_to_base40/train_step_loss.csv
logs/base4_to_base40/step_loss_curve.png
reports/qwen_distill/figures/sample_representative_one_row.png
reports/qwen_distill/qwen_distill_dev60_summary.md
reports/qwen_distill/qwen_distill_dev60_eval.json
```

Final training record:

```text
epochs: 688
optimization steps: 54,343
batch size: 8
```

Adapter overhead:

```text
pure adapter forward: about 0.004s/image
adapter with preprocessing: about 0.08s/image
overhead vs 13.7s Qwen-4step: < 1%
```
