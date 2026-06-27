# group6-image-edit

本仓库是小组图像编辑方向的整合仓库，包含下面内容：

1. **KeepEdit**：基于 Qwen-Image-Edit-2511，完成 Qwen2511 base、GT-LoRA、MTP LoRA、MoE Teacher LoRA 四条主线，并提供训练、推理、评估、可视化和 Hugging Face 发布资源。
2. **Qwen-Image-Edit baseline + step distillation**：重新整理 Qwen-Image-Edit baseline 复现流程，并提供 4-step 到 40-step 的轻量 gated residual adapter 蒸馏代码、脚本、checkpoint、训练日志、loss 曲线和 dev60 评测结果。
3. **InstructPix2Pix + MagicBrush baseline**：基于经典 InstructPix2Pix，完成 MagicBrush 上的 baseline 推理、多候选 oracle、局部 crop 编辑、背景保持 rerank 与 soft mask fusion。
4. **layer**:基于 Qwen-Image-Edit LoRA 微调、Qwen-Image-Layered 图像分层 和 CLIP 图层推荐 的局部图像编辑流程。完成 LoRA 微调，图片分解，CLIP 推荐，局部编辑，重新合成的任务。

1. 项目

本仓库只管理代码、脚本、配置、文档和小型结果摘要。数据和权重体积较大，可以按 README 下载，也可以用软链接或环境变量指向已有路径。

## 1. 总体目录结构

```text
group6-image-edit/
├── README.md                         # 小组总说明与运行入口
├── requirement.txt                   # 额外依赖记录，当前主要依赖见 pyproject.toml / 子模块说明
├── pyproject.toml                    # KeepEdit Python 包配置
├── Makefile                          # KeepEdit 常用命令快捷入口
│
├── instruct-pix2pix/                 # InstructPix2Pix + MagicBrush 扩展脚本
│   ├── README_InstructPix2Pix_MagicBrush.md
│   ├── run_magicbrush_dev.py
│   ├── run_magicbrush_p2p_oracle.py
│   ├── run_magicbrush_p2p_oracle_crop.py
│   ├── rerank_background_preserve.py
│   └── run_magicbrush_ultimate.py
│
├── layer/                            # 分图层修改代码
│   ├── train_qwen_edit_lora_pair.py
│   ├── try0_lora.py
│   ├── untitled.py
│   └── layer_README.md
│
├── configs/keepedit/                 # KeepEdit 配置
├── scripts/keepedit/                 # KeepEdit 数据、训练、评估、上传脚本
│   └── qwen_distill/                 # Qwen baseline / step-distill 训练与评估脚本
├── src/keepedit/                     # KeepEdit Python package
├── docs/keepedit/                    # KeepEdit 方法文档
├── logs/keepedit/                    # KeepEdit 指标、loss 曲线、可视化摘要
├── hf_release/keepedit/              # KeepEdit HF data/weights README
├── reports/                          # 实验结果、报告与可视化图表
│
├── datas/                            # 可选：运行时数据目录，不强制进入 git
├── checkpoints/                      # 可选：运行时权重目录，不强制进入 git
├── externals/keepedit/               # 可选：DiffSynth-Studio / EditAR 等外部依赖
├── models/                           # 可选：模型文件目录
└── outputs/                          # 可选：推理/评估输出目录
```

建议从仓库根目录运行 KeepEdit 命令；InstructPix2Pix 命令建议进入 `instruct-pix2pix/` 后运行，因为脚本依赖官方 `edit_cli.py`、`configs/generate.yaml` 和 checkpoint 的相对路径。

## 2. 环境配置

### 2.1 KeepEdit 环境

KeepEdit 使用 Python package 方式安装。推荐 Python 3.10+。

```bash
cd /home/shiliangzhi/work-space/wyt/ourproject/group6-image-edit

conda create -n keepedit python=3.10 -y
conda activate keepedit
pip install -e ".[all]"
```

如果使用已有环境，例如 `hw4diff`：

```bash
conda activate hw4diff
pip install -e ".[all]"
```

KeepEdit 依赖的主要库包括：PyTorch、DiffSynth-Studio、diffusers、transformers、peft、safetensors、datasets、Pillow、scikit-image、pandas、PyYAML、tqdm、open-clip 等。具体依赖见 `pyproject.toml`。

### 2.2 InstructPix2Pix 环境

`instruct-pix2pix/` 中主要放的是小组新增脚本，不是完整官方工程。运行前需要准备官方 InstructPix2Pix 代码环境，使目录中存在：

```text
instruct-pix2pix/edit_cli.py
instruct-pix2pix/configs/generate.yaml
instruct-pix2pix/checkpoints/instruct-pix2pix-00-22000.ckpt
```

推荐使用官方 InstructPix2Pix 环境：

```bash
cd instruct-pix2pix
conda env create -f environment.yaml
conda activate ip2p
```

如果当前目录没有 `environment.yaml`，请先按官方仓库补齐 InstructPix2Pix 代码和环境文件，或者把本目录的 MagicBrush 脚本复制进官方 InstructPix2Pix 工程中运行。

## 3. 数据与权重准备

### 3.1 MagicBrush 数据格式

两个模块都围绕 MagicBrush 做实验。推荐整理成如下结构：

```text
benchmarks/magicbrush/prepared_full/dev/
├── manifest.json
├── images/
│   ├── input/
│   ├── target/
│   └── mask/
```

`manifest.json` 中每个样本至少包含：

```json
{
  "id": "34_2",
  "instruction": "Open the zebra's mouth.",
  "input_image": "images/input/34_2.png",
  "target_image": "images/target/34_2.png",
  "mask_image": "images/mask/34_2.png"
}
```

KeepEdit 发布数据下载后会自动恢复为内部 JSONL 格式；InstructPix2Pix 脚本则主要使用上述 `manifest.json` 格式。

### 3.2 InstructPix2Pix 权重

需要官方 checkpoint：

```text
instruct-pix2pix/checkpoints/instruct-pix2pix-00-22000.ckpt
```

部分增强脚本也可使用你自己训练或对齐后的 checkpoint，例如：

```text
instruct-pix2pix/checkpoints/ip2p_finetuned_ours/ip2p_magicbrush_alignment.ckpt
```

### 3.3 KeepEdit 发布资源

KeepEdit 相关资源：

| 资源 | 链接 |
| --- | --- |
| KeepEdit LoRA 权重 | [Yitaallen/keepedit-release-weights](https://huggingface.co/Yitaallen/keepedit-release-weights) |
| KeepEdit 发布数据 | [Yitaallen/keepedit-release-data](https://huggingface.co/datasets/Yitaallen/keepedit-release-data) |
| MagicBrush | [osunlp/MagicBrush](https://huggingface.co/datasets/osunlp/MagicBrush) |
| Qwen-Image-Edit-2511 | [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) |
| Qwen-Image | [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) |
| Qwen3-VL-8B-Instruct | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| InstructPix2Pix | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix) |
| DiffSynth-Studio | [modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) |

下载 KeepEdit LoRA 权重：

```bash
huggingface-cli download Yitaallen/keepedit-release-weights \
  --repo-type model \
  --local-dir checkpoints \
  --local-dir-use-symlinks False
```

下载 KeepEdit 数据并解包：

```bash
huggingface-cli download Yitaallen/keepedit-release-data \
  --repo-type dataset \
  --local-dir . \
  --local-dir-use-symlinks False

bash scripts/keepedit/unpack_release_data_archives.sh
```

下载 Qwen、Pix2Pix、EditAR、Qwen3-VL 等基座资源：

```bash
bash scripts/keepedit/download_required_assets.sh
python scripts/keepedit/check_required_assets.py
```

如果已有数据和权重，不需要重新迁移。可以通过环境变量覆盖脚本默认路径，例如：

```bash
DEV_REQUESTS=/path/to/dev.jsonl \
LORA_PATH=/path/to/lora_dir \
bash scripts/keepedit/evaluate_qwen_edit_experiment.sh
```

## 4. KeepEdit 模块运行方法和结果

### 4.0 Qwen-Image-Edit baseline 与 4-step 蒸馏

本仓库包含一个 Qwen-Image-Edit baseline + 4-step self-distillation adapter 运行流程。
该流程主要包括 baseline 结果生成、student/teacher 缓存创建、adapter 训练与评估。

相关目录：

```text
docs/qwen_image_edit_baseline.md
docs/qwen_step_distill_adapter.md
scripts/qwen_distill/
src/keepedit/qwen_distill/
checkpoints/base4_to_base40/
logs/base4_to_base40/
reports/qwen_distill/
```

#### 4.0.1 Baseline 定义

Baseline 输入：

```text
source image + editing instruction
```

Baseline 输出：

```text
Qwen-Image-Edit(source, instruction)
```

MagicBrush 的 target/mask 仅用于评估，不作为模型输入。
默认配置：

```text
Qwen-4step: 4 steps, fast student
Qwen-40step: 40 steps, high-quality teacher
```

#### 4.0.2 运行 baseline

从仓库根目录运行：

```bash
cd /share/home/group6/Project/group6-image-edit
GPU=0 MAX_SAMPLES=60 bash scripts/qwen_distill/run_qwen_baseline_dev60.sh
```

如果需要同时生成 4-step 和 40-step 结果，可执行：

```bash
VARIANTS="base4:4:4.0:none,base40:40:4.0:none" GPU=0 MAX_SAMPLES=60 bash scripts/qwen_distill/cache_qwen_4step_40step.sh
```

#### 4.0.3 生成蒸馏缓存

缓存 student/teacher 结果用于 adapter 训练：

```bash
GPU=0 MAX_SAMPLES=60 bash scripts/qwen_distill/cache_qwen_4step_40step.sh
```

默认输出目录：

```text
outputs/qwen_step_distill/cache_dev60/
```

可用环境变量覆盖默认路径：

```bash
QWEN_ROOT=/path/to/qwen-image-edit-baseline MANIFEST=/path/to/magicbrush_dev60.json OUT=/path/to/cache_out GPU=0 bash scripts/qwen_distill/cache_qwen_4step_40step.sh
```

#### 4.0.4 训练 adapter

先导出训练 metadata：

```bash
CACHE_JSON=outputs/qwen_step_distill/cache_dev60/qwen_distill_lora_speed_results.json OUT=outputs/qwen_step_distill/step_distill_train.json bash scripts/qwen_distill/export_step_distill_metadata.sh
```

然后运行训练：

```bash
METADATA=outputs/qwen_step_distill/step_distill_train.json GPU=0 bash scripts/qwen_distill/train_step_distill_adapter.sh
```

训练脚本默认参数：

```text
IMAGE_SIZE=512
HIDDEN=128
BATCH_SIZE=8
EPOCHS=688
LR=3e-5
VAL_COUNT=10
```

#### 4.0.5 评估 adapter

使用 student cache 和训练 checkpoint 评估 adapter：

```bash
STUDENT_JSON=outputs/qwen_step_distill/cache_dev60/qwen_distill_lora_speed_results.json CKPT=checkpoints/base4_to_base40/step_distill_adapter_best.pt bash scripts/qwen_distill/eval_step_distill_adapter.sh
```

默认评估输出目录：

```text
outputs/qwen_step_distill/eval_base4_adapter_dev60/
```

#### 4.0.6 运行速览

主要执行：

```bash
bash scripts/qwen_distill/run_qwen_baseline_dev60.sh
bash scripts/qwen_distill/cache_qwen_4step_40step.sh
bash scripts/qwen_distill/export_step_distill_metadata.sh
bash scripts/qwen_distill/train_step_distill_adapter.sh
bash scripts/qwen_distill/eval_step_distill_adapter.sh
```

更多细节请参考：

```text
docs/qwen_image_edit_baseline.md
docs/qwen_step_distill_adapter.md
```

### 4.1 安装 KeepEdit 包

从仓库根目录执行：

```bash
cd /home/shiliangzhi/work-space/wyt/ourproject/group6-image-edit
pip install -e .
```

验证 package 能导入：

```bash
PYTHONPATH=src python -c "import keepedit; print(keepedit.__version__)"
```

### 4.2 评估 Qwen2511 Base

该实验不加载 LoRA，用 Qwen-Image-Edit-2511 原始模型直接编辑 MagicBrush dev。

```bash
EXPERIMENT_NAME=qwen2511_base \
LORA_PATH=none \
GPUS=0 \
bash scripts/keepedit/evaluate_qwen_edit_experiment.sh
```

常用参数：

```bash
QWEN_LIMIT=32                  # 只跑前 32 个样本，快速检查
QWEN_INFER_STEPS=28            # 推理步数
RUN_MLLM=1                     # 开启 Qwen3-VL/MLLM 评估
PARALLEL_GPUS=0,1,2,3          # 多卡并行分片评估
```

示例：

```bash
EXPERIMENT_NAME=qwen2511_base \
LORA_PATH=none \
PARALLEL_GPUS=0,1,2,3 \
QWEN_INFER_STEPS=28 \
RUN_MLLM=0 \
bash scripts/keepedit/evaluate_qwen_edit_experiment.sh
```

输出：

```text
data/outputs/magicbrush_dev_qwen2511_base/
logs/keepedit/summary/visual_gallery_magicbrush_dev_qwen2511_base/index.html
logs/keepedit/summary/magicbrush_dev_qwen2511_base_release_metrics.csv
logs/keepedit/summary/magicbrush_dev_qwen2511_base_release_metrics_summary.json
```

### 4.3 GT-LoRA 训练

GT-LoRA 用 MagicBrush 真实 target 作为监督目标，训练 Qwen2511 LoRA。训练数据形式是：

```text
edit_image = source image
prompt     = instruction
image      = ground-truth target image
```

运行：

```bash
GPUS=0,1,2,3 \
NUM_PROCESSES=4 \
QWEN_EPOCHS=2 \
QWEN_LR=1e-4 \
QWEN_RANK=32 \
bash scripts/keepedit/run_gt_lora_qwen_edit.sh
```

脚本内部会先生成 DiffSynth metadata：

```text
data/diffsynth/magicbrush_train_qwen2511_gt_onestage/metadata.json
```

然后调用 DiffSynth-Studio 的 Qwen-Image-Edit LoRA 训练脚本。输出权重：

```text
checkpoints/qwen_edit_2511_keepedit_gt_onestage/
```

训练完成后默认会评估 dev 集，输出指标和 gallery。

### 4.4 MTP LoRA 训练

MTP LoRA 是 KeepEdit 的紧凑改进版本，主要思想是用 mask-aware 训练目标强化编辑区域、约束背景区域，并加入少量 no-op/边界软化策略，让模型更明确地区分“该改”和“不该改”。

运行：

```bash
GPUS=0,1,2,3 \
NUM_PROCESSES=4 \
QWEN_EPOCHS=1 \
QWEN_LR=5e-5 \
QWEN_RANK=16 \
MASK_EDIT_WEIGHT=4.0 \
MASK_BG_WEIGHT=0.3 \
BOUNDARY_WEIGHT=0.15 \
bash scripts/keepedit/run_mtp_phasea.sh
```

输出权重：

```text
checkpoints/qwen_edit_2511_mtp_phasea/
```

输出评估：

```text
data/outputs/magicbrush_dev_qwen2511_mtp_phasea/
logs/keepedit/summary/visual_gallery_magicbrush_dev_qwen2511_mtp_phasea/index.html
logs/keepedit/summary/magicbrush_dev_qwen2511_mtp_phasea_release_metrics.csv
```

### 4.5 MoE-Fusion Teacher 构造

MoE-Fusion Teacher 使用三个专家候选：

```text
Pix2Pix
Qwen-Image-Edit
EditAR
```

训练期允许使用 target 和 mask，对不同专家候选进行区域级评分和融合：

```text
编辑区域：选择或融合 mask 内更接近 target、语义更符合指令的专家结果
背景区域：优先保留 source，避免非编辑区域漂移
边界区域：使用 feather / Laplacian blend / local refinement 降低接缝
```

构造专家候选和 MoE teacher：

```bash
GPUS=0,1,2,3 \
STAGE1_NUM_WORKERS=8 \
bash scripts/keepedit/run_keepedit_moe_fusion.sh
```

输出：

```text
data/candidates/magicbrush_train_pix2pix_qwen_editar/
data/candidates/magicbrush_dev_pix2pix_qwen_editar/
data/teachers/magicbrush_train_moe_fusion/
data/teachers/magicbrush_dev_moe_fusion/
logs/keepedit/summary/visual_gallery_magicbrush_dev_moe_fusion/index.html
```

### 4.6 MoE Teacher LoRA 训练

MoE Teacher LoRA 用 Stage 1 生成的 MoE teacher 图作为训练目标，让 Qwen2511 学习专家融合后的编辑方向。最终模型推理时不需要专家候选，只需要 source image + instruction。

运行：

```bash
GPUS=0,1,2,3 \
NUM_PROCESSES=4 \
QWEN_EPOCHS=1 \
QWEN_LR=5e-5 \
QWEN_RANK=32 \
bash scripts/keepedit/run_moe_teacher_lora.sh
```

输出权重：

```text
checkpoints/qwen_edit_2511_moe_teacher_onestage/
```

输出评估：

```text
data/outputs/magicbrush_dev_qwen2511_moe_teacher_onestage/
logs/keepedit/summary/visual_gallery_magicbrush_dev_qwen2511_moe_teacher_onestage/index.html
logs/keepedit/summary/magicbrush_dev_qwen2511_moe_teacher_onestage_release_metrics.csv
```

### 4.7 KeepEdit 评估指标与结果查看

#### 4.7.1 指标文件

统一评估脚本会生成：

```text
logs/keepedit/summary/magicbrush_dev_<experiment>_release_metrics.csv
logs/keepedit/summary/magicbrush_dev_<experiment>_release_metrics_summary.json
logs/keepedit/summary/magicbrush_dev_<experiment>_mllm_preference.jsonl   # RUN_MLLM=1 时生成
```

汇总表：

```text
logs/keepedit/summary/keepedit_release_full_metrics_comparison.csv
```

当前主要指标：

```text
Target--Output SSIM / PSNR     输出与目标图的整体相似度
BG-SSIM                        非编辑区域背景保持能力
Input--Output SSIM / PSNR      输出相对原图的改动幅度
Edit-Region Change             mask 内平均变化量，用于判断 under-edit / over-edit
MLLM preference                指令执行、目标相似、背景保持的视觉语言模型评估
```

#### 4.7.2 可视化结果

本地小型 gallery：

```text
logs/keepedit/summary/visual_gallery_magicbrush_dev_moe_fusion/index.html
logs/keepedit/summary/visual_gallery_magicbrush_dev_qwen2511_mtp_phasea/index.html
logs/keepedit/summary/visual_gallery_magicbrush_dev_qwen2511_moe_teacher_onestage/index.html
```

loss 曲线：

```text
logs/keepedit/loss_curves/three_lora_20260624/index.html
logs/keepedit/loss_curves/three_lora_20260624/three_lora_loss_curves_side_by_side.png
```

可以用本地 HTTP 服务查看：

```bash
python -m http.server 8899 --directory logs/keepedit
```

然后浏览器打开：

```text
http://localhost:8899/summary/visual_gallery_magicbrush_dev_qwen2511_moe_teacher_onestage/index.html
http://localhost:8899/loss_curves/three_lora_20260624/index.html
```

完整 gallery 在 Hugging Face：

- [MoE-Fusion Teacher gallery](https://huggingface.co/datasets/Yitaallen/keepedit-release-data/tree/main/reports/visual_gallery_magicbrush_dev_moe_fusion)
- [MTP LoRA gallery](https://huggingface.co/datasets/Yitaallen/keepedit-release-data/tree/main/reports/visual_gallery_magicbrush_dev_qwen2511_mtp_phasea)
- [MoE Teacher LoRA gallery](https://huggingface.co/datasets/Yitaallen/keepedit-release-data/tree/main/reports/visual_gallery_magicbrush_dev_qwen2511_moe_teacher_onestage)

#### 4.7.3 当前发布版指标

```text
Qwen2511 Base:
  Target--Output SSIM = 0.450
  BG-SSIM = 0.696

GT-LoRA:
  Target--Output SSIM = 0.696
  BG-SSIM = 0.821

MTP LoRA:
  Target--Output SSIM = 0.740
  BG-SSIM = 0.828

MoE Teacher LoRA:
  Target--Output SSIM = 0.763
  BG-SSIM = 0.852
```

结论：GT-LoRA 证明 Qwen2511 在 MagicBrush 上可被有效适配；MTP LoRA 在不依赖专家候选的情况下提升目标相似度和背景保持；MoE Teacher LoRA 通过多专家融合 teacher 蒸馏取得当前最好的客观指标。

## 5. InstructPix2Pix 模块运行方法

### 5.1 原始 baseline 批量推理

脚本：

```text
instruct-pix2pix/run_magicbrush_dev.py
```

作用：读取 MagicBrush `manifest.json`，逐样本运行官方 InstructPix2Pix baseline，并保存输出。

注意：当前 `run_magicbrush_dev.py` 中 `DATA_DIR` 是脚本顶部硬编码路径，运行前需要改成你的 MagicBrush dev 路径：

```python
DATA_DIR = "/path/to/benchmarks/magicbrush/prepared_full/dev"
OUTPUT_DIR = "./results/magicbrush_baseline_dev_outputs"
```

运行：

```bash
cd instruct-pix2pix
CUDA_VISIBLE_DEVICES=0 python run_magicbrush_dev.py
```

输出：

```text
instruct-pix2pix/results/magicbrush_baseline_dev_outputs/
```

### 5.2 Prompt + CFG/Seed Oracle 多候选搜索

脚本：

```text
instruct-pix2pix/run_magicbrush_p2p_oracle.py
```

作用：对每个样本生成多组候选，改变 prompt rewrite、CFG-text、CFG-image 和 seed；然后利用 target + mask 的 oracle 分数选择最优候选。该方法用于分析 InstructPix2Pix 的上限，而不是最终部署方案，因为它使用了 target。

运行示例：

```bash
cd instruct-pix2pix
CUDA_VISIBLE_DEVICES=0 python run_magicbrush_p2p_oracle.py \
  --data-dir /path/to/benchmarks/magicbrush/prepared_full/dev \
  --output-dir ./results/magicbrush_p2p_oracle_dev \
  --config ./configs/generate.yaml \
  --ckpt ./checkpoints/instruct-pix2pix-00-22000.ckpt \
  --steps 50 \
  --cfg-texts 7.5,9.0,10.5,12.0 \
  --cfg-images 1.0,1.2,1.5 \
  --seeds 0,1,2,3 \
  --use-rewrite \
  --limit 20
```

输出结构：

```text
results/magicbrush_p2p_oracle_dev/
├── candidates/          # 每个样本的多候选图
├── best/                # oracle 选出的最优图
└── scores.json          # 候选评分记录
```

### 5.3 Mask-Crop Oracle 局部编辑

脚本：

```text
instruct-pix2pix/run_magicbrush_p2p_oracle_crop.py
```

作用：使用 mask 定位编辑区域，对局部 crop 进行 InstructPix2Pix 编辑，再贴回原图；同时保留全图候选，与 crop 候选一起做 oracle 选择。适合小物体、局部属性修改等任务。

运行示例：

```bash
cd instruct-pix2pix
CUDA_VISIBLE_DEVICES=0 python run_magicbrush_p2p_oracle_crop.py \
  --data-dir /path/to/benchmarks/magicbrush/prepared_full/dev \
  --output-dir ./results/magicbrush_p2p_oracle_crop_dev \
  --config ./configs/generate.yaml \
  --ckpt ./checkpoints/instruct-pix2pix-00-22000.ckpt \
  --steps 50 \
  --cfg-texts 7.5,9.0,10.5,12.0 \
  --cfg-images 1.0,1.2,1.5 \
  --seeds 0,1,2,3 \
  --use-rewrite \
  --use-crop \
  --crop-expand 0.35 \
  --crop-min-size 128 \
  --limit 20
```

输出：

```text
results/magicbrush_p2p_oracle_crop_dev/
├── candidates/
├── best/
└── scores.json
```

### 5.4 Background-preserve Rerank + Fusion

脚本：

```text
instruct-pix2pix/rerank_background_preserve.py
```

作用：对 oracle/crop 产生的候选进行重排，评分同时考虑：

```text
mask 内接近 target        -> 编辑正确性
mask 外接近 input         -> 背景保持
全图接近 target           -> 整体一致性
```

然后使用 soft mask fusion 将候选编辑区域与原图背景融合，减少背景漂移。

运行示例：

```bash
cd instruct-pix2pix
python rerank_background_preserve.py \
  --data-dir /path/to/benchmarks/magicbrush/prepared_full/dev \
  --candidate-root ./results/magicbrush_p2p_oracle_crop_dev/candidates \
  --output-dir ./results/magicbrush_p2p_bg_preserve_dev \
  --w-edit 0.50 \
  --w-preserve 0.45 \
  --w-full 0.05 \
  --dilate 9 \
  --blur 7 \
  --limit 20
```

输出：

```text
results/magicbrush_p2p_bg_preserve_dev/
├── best/                # rerank 后的最佳候选
├── fused/               # soft mask fusion 后的图
└── scores.json          # 编辑/背景/全图评分
```

## 6. 常用命令速查

KeepEdit Makefile 快捷命令：

```bash
make assets      # 下载/检查 KeepEdit 所需基座模型与外部资源
make base-eval   # 评估 Qwen2511 base
make gt-lora     # 训练并评估 GT-LoRA
make mtp-lora    # 训练并评估 MTP LoRA
make moe-fusion  # 构造 MoE-Fusion Teacher
make moe-lora    # 训练并评估 MoE Teacher LoRA
make serve-reports
```

等价脚本命令：

```bash
bash scripts/keepedit/download_required_assets.sh
python scripts/keepedit/check_required_assets.py

EXPERIMENT_NAME=qwen2511_base LORA_PATH=none bash scripts/keepedit/evaluate_qwen_edit_experiment.sh
bash scripts/keepedit/run_gt_lora_qwen_edit.sh
bash scripts/keepedit/run_mtp_phasea.sh
bash scripts/keepedit/run_keepedit_moe_fusion.sh
bash scripts/keepedit/run_moe_teacher_lora.sh
```

## 9. 路径覆盖说明

默认路径适合从仓库根目录运行。常见覆盖变量：

```bash
ENV_NAME=hw4diff                         # conda 环境名
GPUS=0,1,2,3                             # 可见 GPU
NUM_PROCESSES=4                          # accelerate 进程数
TRAIN_REQUESTS=/path/to/train.jsonl      # 训练集 JSONL
DEV_REQUESTS=/path/to/dev.jsonl          # dev 集 JSONL
LORA_PATH=/path/to/lora_dir              # 评估时加载的 LoRA
DIFFSYNTH_ROOT=/path/to/DiffSynth-Studio # DiffSynth-Studio 路径
QWEN_MODEL_BASE=/path/to/diffsynth       # Qwen/DiffSynth 模型缓存根目录
RUN_MLLM=1                               # 开启 MLLM 评估
QWEN_LIMIT=32                            # 小规模调试样本数
```

示例：

```bash
ENV_NAME=hw4diff \
GPUS=0 \
DEV_REQUESTS=/data/my_magicbrush/dev.jsonl \
EXPERIMENT_NAME=qwen2511_mtp_phasea \
LORA_PATH=/data/my_weights/qwen_edit_2511_mtp_phasea \
bash scripts/keepedit/evaluate_qwen_edit_experiment.sh
```

## 10. 子文档

根 README 保留完整运行入口；更细节的算法说明见：

- [InstructPix2Pix + MagicBrush](instruct-pix2pix/README_InstructPix2Pix_MagicBrush.md)
- [Qwen2511 baseline](docs/keepedit/QWEN2511_BASELINE.md)
- [GT-LoRA workflow](docs/keepedit/GT_LORA_WORKFLOW.md)
- [MTP LoRA workflow](docs/keepedit/MTP_LORA_WORKFLOW.md)
- [MoE Teacher LoRA workflow](docs/keepedit/MOE_LORA_WORKFLOW.md)
- [KeepEdit Hugging Face release](docs/keepedit/HUGGINGFACE_RELEASE.md)
