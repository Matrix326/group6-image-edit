# EditAR 代码整理与运行说明

## 核心改动位置速查

为了方便快速定位本部分相对原始 EditAR baseline 的主要改动，下面列出报告中最终保留的 LoRA 相关代码位置。路径均相对于小组总仓库 `group6-image-edit/`。

| 改动内容 | 代码相对路径 | 说明 |
| --- | --- | --- |
| LoRA 模块实现 | `EditAR/autoregressive/models/lora.py` | 定义 `LoRALinear`、LoRA 注入、仅训练 LoRA 参数、保存 LoRA adapter state dict 等工具函数。 |
| GPT loss 支持 token 加权 | `EditAR/autoregressive/models/gpt_edit.py` | 在 `Transformer.forward()` 中加入 `token_loss_weight` 和 `compute_loss`，用于 Region LoRA 的 mask-weighted CE。 |
| LoRA / Region / Contrast 训练入口 | `EditAR/autoregressive/train/train_edit.py` | 加入 `--use-lora`、`--lora-target-modules`、`--use-mask-weighted-loss`、`--use-negative-contrastive` 等训练参数。 |
| Region LoRA mask 权重 | `EditAR/autoregressive/train/mask_weight.py` | 将 MagicBrush mask 下采样到视觉 token 网格，并生成编辑区/背景区 token loss 权重。 |
| Contrast LoRA 对比损失 | `EditAR/autoregressive/train/contrastive.py` | 实现正确指令与错误指令之间的 margin contrastive loss。 |
| MagicBrush 训练集加载 | `EditAR/dataset/Edit_MagicBrush.py` | 读取 MagicBrush train split，并返回训练所需的输入图、目标图、文本和 mask。 |
| MagicBrush 推理集加载 | `EditAR/dataset/Edit_MagicBrush_eval.py` | 支持 MagicBrush dev/eval 推理与可视化保存。 |
| 数据集注册 | `EditAR/dataset/build.py` | 在原始 mixed dataset builder 中加入 MagicBrush，并为无 mask 数据补默认 mask。 |
| LoRA checkpoint 推理 | `EditAR/autoregressive/sample/sample_edit_example.py`、`EditAR/autoregressive/sample/sample_edit_folder.py` | 支持加载 adapter-only LoRA checkpoint，并加入 MagicBrush batch inference 入口。 |
| MagicBrush 指标评估 | `EditAR/tools/evaluate_magicbrush_outputs.py` | 计算报告表格中的 Target--Output、Input--Output、BG-SSIM、Edit Change 等指标。 |
| VQ / token 误差诊断 | `EditAR/tools/vq_reconstruct_benchmark.py`、`EditAR/tools/diagnose_magicbrush_tokens.py` | 用于报告中 VQ 重建误差和自回归 token 分布偏移分析。 |
| 复现实验脚本 | `EditAR/scripts/run_magicbrush_only_lora_sweep_a800.sh`、`EditAR/scripts/run_magicbrush60_report_benchmark_parallel_a800.sh` | 分别用于 LoRA 消融训练/评估和 MagicBrush-60 报告表格复现。 |

## 项目简介

EditAR 是一个基于自回归建模的图像编辑代码库。整体流程是：先用 VQ 图像 tokenizer 将输入图像和目标编辑图像离散化为视觉 token；再用 GPT 风格的 Transformer 接收 FLAN-T5 编码后的文本指令、输入图像 token 和任务模式 embedding；最后以 next-token prediction 的方式逐步预测编辑后图像的视觉 token，并通过 VQ decoder 解码回 RGB 图像。

本 release 版本保留两类代码：

- EditAR 原始 baseline 代码。
- 大作业报告中最终写入的 LoRA 改进与分析代码。

报告中涉及的 LoRA 变体包括：

- Backbone LoRA：在 Transformer attention、FFN 和 `cap_proj` 上注入 LoRA。
- Region LoRA：在 Backbone LoRA 基础上加入编辑区域 mask 加权 token loss。
- Contrast LoRA：在 Region LoRA 基础上加入负文本指令对比损失。
- Projection LoRA：只在 `cap_proj.fc1` 和 `cap_proj.fc2` 上注入 LoRA。

未进入最终报告的早期探索方法已经从 release 代码中移除。

## 环境配置

安装 Python 依赖：

```bash
bash scripts/install.sh
source .venv/bin/activate
```

代码主要按 Python 3.10、PyTorch 2.2.1、CUDA GPU 和混合精度训练/推理环境整理。

## 预训练权重

先创建默认目录：

```bash
mkdir -p pretrained_models/t5-ckpt pretrained_models checkpoints/editar/editar_release
```

下载 FLAN-T5-XL 文本编码器：

```bash
huggingface-cli download google/flan-t5-xl \
  --local-dir pretrained_models/t5-ckpt/flan-t5-xl
```

下载 LlamaGen 的 VQ tokenizer 权重：

```bash
wget -O pretrained_models/vq_ds16_t2i.pt \
  https://huggingface.co/peizesun/llamagen_t2i/resolve/main/vq_ds16_t2i.pt
```

运行原始 EditAR 推理时，将 release checkpoint 放到：

```text
checkpoints/editar/editar_release/editar_release.pt
```

如果需要从 LlamaGen T2I 初始化训练 baseline，将该权重放到：

```text
pretrained_models/t2i_XL_stage2_512.pt
```

## 数据准备

原始 baseline 训练脚本默认读取 `data/` 下已经处理好的 Hugging Face 数据集：

```text
data/
  MultiGen-20M_depth_HF/
  Condition_Segmentation/
  PIPE_HF/
  Seedx_Unsplash_HF/
  MagicBrush_HF/
```

本报告实验主要使用 MagicBrush。建议将处理后的 MagicBrush 数据放在：

```text
data/MagicBrush_HF
```

如果数据放在其他位置，可以通过环境变量传入：

```bash
MAGICBRUSH_PATH=/path/to/edit-data/MagicBrush_HF bash scripts/run_magicbrush_only_lora_sweep_a800.sh
```

## Baseline 推理

单图编辑示例：

```bash
python autoregressive/sample/sample_edit_example.py \
  --gpt-ckpt checkpoints/editar/editar_release/editar_release.pt \
  --vq-ckpt pretrained_models/vq_ds16_t2i.pt \
  --t5-path pretrained_models/t5-ckpt \
  --cfg-scale 3 \
  --seed 83
```

MagicBrush 批量推理：

```bash
python autoregressive/sample/sample_edit_folder.py \
  --testset magicbrush \
  --magicbrush-path data/MagicBrush_HF \
  --gpt-ckpt checkpoints/editar/editar_release/editar_release.pt \
  --vq-ckpt pretrained_models/vq_ds16_t2i.pt \
  --t5-path pretrained_models/t5-ckpt \
  --output-dir outputs/samples \
  --max-samples 60 \
  --cfg-scale 1.0
```

对已经生成的 MagicBrush 结果计算指标：

```bash
python tools/evaluate_magicbrush_outputs.py \
  --magicbrush-path data/MagicBrush_HF \
  --samples-dir outputs/samples/magicbrush/samples/txt_1.0 \
  --output-dir outputs/benchmark/pretrained_magicbrush60 \
  --cfg-scale 1.0 \
  --max-samples 60 \
  --image-size 512
```

## 训练脚本

原始 baseline 训练：

```bash
bash scripts/train.sh
```

运行报告中的 LoRA 消融训练，包括 Projection LoRA、Backbone LoRA、Region LoRA 和 Contrast LoRA：

```bash
REPO_DIR=/path/to/EditAR \
ACCELERATE=/path/to/EditAR/.venv/bin/accelerate \
PYTHON=/path/to/EditAR/.venv/bin/python \
MAGICBRUSH_PATH=/path/to/edit-data/MagicBrush_HF \
bash scripts/run_magicbrush_only_lora_sweep_a800.sh
```

该 sweep 中各变体对应关系为：

- Projection LoRA：`cap_proj.fc1,cap_proj.fc2`。
- Backbone LoRA：`wqkv,wo,w1,w2,w3,cap_proj.fc1,cap_proj.fc2`。
- Region LoRA：在 Backbone LoRA 基础上加入 `--use-mask-weighted-loss`。
- Contrast LoRA：在 Region LoRA 基础上加入 `--use-negative-contrastive`。

单独运行 Contrast LoRA 训练：

```bash
bash scripts/train_negative_lora.sh
```

## 报告指标复现

运行 MagicBrush-60 报告表格 benchmark：

```bash
REPO_DIR=/path/to/EditAR \
PYTHON=/path/to/EditAR/.venv/bin/python \
MAGICBRUSH_PATH=/path/to/edit-data/MagicBrush_HF \
bash scripts/run_magicbrush60_report_benchmark_parallel_a800.sh
```

脚本会评估原始 EditAR baseline 和报告中的 LoRA checkpoint，最终汇总到：

```text
outputs/report_benchmark_magicbrush60/magicbrush60_report_summary.json
```

## 诊断工具

VQ 重建误差诊断：

```bash
REPO_DIR=/path/to/EditAR \
PYTHON=/path/to/EditAR/.venv/bin/python \
HF_DATASET=/path/to/edit-data/MagicBrush_HF \
bash scripts/run_vq_reconstruction_benchmark_a800.sh
```

MagicBrush token 级自回归误差诊断：

```bash
REPO_DIR=/path/to/EditAR \
PYTHON=/path/to/EditAR/.venv/bin/python \
MAGICBRUSH_PATH=/path/to/edit-data/MagicBrush_HF \
bash scripts/run_magicbrush8_token_diagnostics_a800.sh
```
