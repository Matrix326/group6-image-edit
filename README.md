# group6-image-edit

本仓库是小组图像编辑方向的整合仓库。当前代码按子项目组织，各部分相互独立：

| 模块 | 位置 | 说明 |
| --- | --- | --- |
| KeepEdit | `KeepEdit/` | 完整 KeepEdit 子项目，包含 Qwen-Image-Edit-2511 base、GT-LoRA、MTP LoRA、MoE Teacher LoRA 的训练、评估、发布说明 |
| Qwen-Image-Edit baseline 与 4-step 蒸馏 | `qwen_distill/`、`docs/qwen_*.md` | 独立于 KeepEdit 的 baseline 复现与 4-step -> 40-step image-space adapter 蒸馏实验 |
| InstructPix2Pix + MagicBrush | `instruct-pix2pix/` | InstructPix2Pix baseline、多候选 oracle、局部 crop、背景保持 rerank 与 fusion |
| EditAR | `EditAR/`、`docs/editar/EditAR.md` | EditAR baseline 与 LoRA 消融相关代码和说明 |
| layer | `layer/` | 基于 Qwen-Image-Edit LoRA、Qwen-Image-Layered 和 CLIP 推荐的分层局部编辑流程 |

根目录只作为总入口；具体运行时请进入对应子目录或按该模块说明设置路径。

# 0. 总览

## 0.1 当前目录结构

```text
group6-image-edit/
├── README.md
├── KeepEdit/
│   ├── README_KeepEdit.md
│   ├── pyproject.toml
│   ├── environment.yml
│   ├── configs/
│   ├── docs/
│   ├── scripts/
│   ├── src/keepedit/
│   ├── hf_release/
│   └── reports/
├── qwen_distill/
│   ├── run_qwen_baseline_dev60.sh
│   ├── cache_qwen_4step_40step.sh
│   ├── export_step_distill_metadata.sh
│   ├── train_step_distill_adapter.sh
│   ├── eval_step_distill_adapter.sh
│   ├── build_report_assets.sh
│   └── reports/qwen_distill/
├── docs/
│   ├── qwen_image_edit_baseline.md
│   ├── qwen_step_distill_adapter.md
│   └── editar/EditAR.md
├── instruct-pix2pix/
│   ├── README_InstructPix2Pix_MagicBrush.md
│   ├── run_magicbrush_dev.py
│   ├── run_magicbrush_p2p_oracle.py
│   ├── run_magicbrush_p2p_oracle_crop.py
│   ├── rerank_background_preserve.py
│   └── run_magicbrush_ultimate.py
├── EditAR/
│   ├── README.md
│   ├── requirements.txt
│   ├── assets/
│   │   └── teaser.png
│   ├── autoregressive/
│   │   ├── models/
│   │   ├── sample/
│   │   └── train/
│   ├── dataset/
│   │   ├── Edit_MagicBrush.py
│   │   ├── Edit_MagicBrush_eval.py
│   │   ├── build.py
│   │   └── Condition_*.py
│   ├── scripts/
│   ├── tools/
│   │   ├── diagnose_magicbrush_tokens.py
│   │   ├── evaluate_magicbrush_outputs.py
│   │   └── vq_reconstruct_benchmark.py
│   ├── tokenizer/
│   ├── language/
│   ├── feature_encoders/
│   ├── utils/
│   └── report/
├── layer/
│   ├── layer_README.md
│   ├── train_qwen_edit_lora_pair.py
│   ├── try0_lora.py
│   └── untitled.py
└── logs/
    ├── base4_to_base40/
    │   ├── train_step_loss.csv
    │   └── step_loss_curve.png
    └── keepedit/
        ├── logs/
        ├── loss_curves/
        │   └── three_lora_20260624/
        └── summary/
            ├── keepedit_release_full_metrics_comparison.csv
            ├── magicbrush_dev_*_release_metrics.csv
            ├── magicbrush_dev_*_release_metrics_summary.json
            ├── magicbrush_dev_*_mllm_preference.jsonl
            └── visual_gallery_magicbrush_dev_*/
```

## 0.2 快速入口

KeepEdit 是独立 Python package：

```bash
cd KeepEdit
conda env create -f environment.yml
conda activate your_env
pip install -e ".[all]"
```

Qwen baseline 与蒸馏脚本在根目录的 `qwen_distill/` 下；这些脚本来自当时的实验环境，默认路径仍指向 `/share/home/group6/...`，迁移到新机器时需要显式覆盖 `ROOT`、`QWEN_ROOT`、`MANIFEST` 等变量。

```bash
ROOT=$(pwd) \
QWEN_ROOT=/path/to/qwen-image-edit-baseline \
MANIFEST=/path/to/magicbrush_dev60_keepedit.json \
GPU=0 \
MAX_SAMPLES=60 \
bash qwen_distill/run_qwen_baseline_dev60.sh
```

InstructPix2Pix、EditAR 和 layer 的入口见本文末尾“其它子项目”。

# 1. KeepEdit

KeepEdit 是一个面向指令图像编辑的研究项目。围绕 Qwen-Image-Edit-2511 展开，比较四条主线：

```text
1. Qwen2511 Base
2. GT-LoRA
3. MTP LoRA
4. MoE Teacher LoRA
```

所有可部署模型的推理形式都保持一致：

```text
source image + instruction -> edited image
```

## 1.1 仓库结构

```text
configs/        实验配置
docs/           中文流程文档
scripts/        数据、训练、评估和上传脚本
src/keepedit/   KeepEdit 核心代码
reports/        小型 release 指标文件
```

以下大目录不进入 git，需要手动完成配置：

```text
checkpoints/
data/
external/
```

它们通过 Hugging Face 或官方仓库下载。详见：

```text
docs/HUGGINGFACE_RELEASE.md
docs/LICENSE_NOTES.md
```

## 1.2 环境配置

推荐使用 conda：

```bash
conda env create -f environment.yml
conda activate your_env
pip install -e ".[all]"
```

项目依赖 DiffSynth-Studio 和 EditAR（候选图片的生成）：

```text
external/DiffSynth-Studio/
external/EditAR/
```

这些会通过下载脚本或手动 clone 准备。

## 1.3 下载权重和数据

### 1.3.1 公开资源地址

KeepEdit 发布资源：

| 资源 | 地址 | 说明 |
| --- | --- | --- |
| 发布权重 | [Yitaallen/keepedit-release-weights](https://huggingface.co/Yitaallen/keepedit-release-weights) | GT-LoRA、MTP LoRA、MoE Teacher LoRA 的 LoRA 权重 |
| 发布数据 | [Yitaallen/keepedit-release-data](https://huggingface.co/datasets/Yitaallen/keepedit-release-data) | 预处理数据、专家候选、MoE teacher、模型输出和完整可视化 gallery |

上游数据、模型与外部项目：

| 资源 | 地址 |
| --- | --- |
| MagicBrush 数据集 | [osunlp/MagicBrush](https://huggingface.co/datasets/osunlp/MagicBrush) |
| Qwen-Image-Edit-2511 | [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) |
| Qwen-Image | [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) |
| Qwen-Image-Edit | [Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) |
| InstructPix2Pix | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix) |
| Qwen3-VL-8B-Instruct | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| EditAR release checkpoint | [JitengMu/CVPR2025_EditAR_release](https://huggingface.co/datasets/JitengMu/CVPR2025_EditAR_release) |
| LlamaGen T2I / VQ tokenizer | [peizesun/llamagen_t2i](https://huggingface.co/peizesun/llamagen_t2i) |
| flan-t5-xl | [google/flan-t5-xl](https://huggingface.co/google/flan-t5-xl) |
| DiffSynth-Studio | [modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) |

### 1.3.2 下载 KeepEdit 发布权重

```bash
huggingface-cli download Yitaallen/keepedit-release-weights \
  --repo-type model \
  --local-dir checkpoints \
  --local-dir-use-symlinks False
```

下载后应得到：

```text
checkpoints/qwen_edit_2511_keepedit_gt_onestage/step-4404.safetensors
checkpoints/qwen_edit_2511_mtp_phasea/step-2269.safetensors
checkpoints/qwen_edit_2511_moe_teacher_onestage/step-2202.safetensors
```

### 1.3.3 下载 KeepEdit 发布数据

```bash
huggingface-cli download Yitaallen/keepedit-release-data \
  --repo-type dataset \
  --local-dir . \
  --local-dir-use-symlinks False

bash scripts/unpack_release_data_archives.sh
```

数据集仓库发布为少量归档分卷，下载后先得到：

```text
archives/data_processed.tar.000.part
archives/data_candidates.tar.000.part
archives/data_teachers.tar.000.part
...
archives/MANIFEST.sha256
```

`scripts/unpack_release_data_archives.sh` 会自动识别完整 `data_*.tar` 或分卷 `data_*.tar.*.part`。

解包后应包含：

```text
data/processed/
data/candidates/
data/teachers/
data/diffsynth/
data/outputs/
```

### 1.3.4 下载基座模型和外部依赖

```bash
bash scripts/download_required_assets.sh
python scripts/check_required_assets.py
```

该脚本会准备：

```text
Qwen/Qwen-Image-Edit-2511
Qwen/Qwen-Image
Qwen/Qwen-Image-Edit
timbrooks/instruct-pix2pix
Qwen3-VL-8B-Instruct
EditAR 相关权重
```

## 1.4 从 MagicBrush 原始数据重新预处理

如果不使用发布数据，可以从 MagicBrush parquet 重新生成 `data/processed`：

```bash
huggingface-cli download osunlp/MagicBrush \
  --repo-type dataset \
  --include "data/train-*.parquet" \
  --include "data/dev-*.parquet" \
  --local-dir data/raw/MagicBrush \
  --local-dir-use-symlinks False

python scripts/validate_magicbrush_parquet.py \
  --root data/raw/MagicBrush/data \
  --split train \
  --split dev

python scripts/prepare_magicbrush_parquet.py \
  --parquet_dir data/raw/MagicBrush \
  --split train \
  --out_dir data/processed/magicbrush_train

python scripts/prepare_magicbrush_parquet.py \
  --parquet_dir data/raw/MagicBrush \
  --split dev \
  --out_dir data/processed/magicbrush_dev
```

生成：

```text
data/processed/magicbrush_train/train.jsonl
data/processed/magicbrush_dev/dev.jsonl
```

## 1.5 Qwen2511 Base

不加载 LoRA，直接评估原始 Qwen-Image-Edit-2511：

```bash
EXPERIMENT_NAME=qwen2511_base \
LORA_PATH=none \
GPUS=0 \
bash scripts/evaluate_qwen_edit_experiment.sh
```

多卡并行：

```bash
EXPERIMENT_NAME=qwen2511_base \
LORA_PATH=none \
GPUS=0,1,2,3 \
PARALLEL_GPUS=0,1,2,3 \
bash scripts/evaluate_qwen_edit_experiment.sh
```

输出：

```text
data/outputs/magicbrush_dev_qwen2511_base/
reports/magicbrush_dev_qwen2511_base_release_metrics.csv
reports/magicbrush_dev_qwen2511_base_release_metrics_summary.json
```

详细说明见：

```text
docs/QWEN2511_BASELINE.md
```

## 1.6 GT-LoRA

GT-LoRA 直接使用 MagicBrush target 作为监督目标：

```text
source image + instruction -> target image
```

训练：

```bash
GPUS=0,1,2,3 \
NUM_PROCESSES=4 \
bash scripts/run_gt_lora_qwen_edit.sh
```

输出：

```text
checkpoints/qwen_edit_2511_keepedit_gt_onestage/step-4404.safetensors
data/outputs/magicbrush_dev_qwen2511_gt_onestage/
reports/magicbrush_dev_qwen2511_gt_onestage_release_metrics.csv
```

详细说明见：

```text
docs/GT_LORA_WORKFLOW.md
```

## 1.7 MTP LoRA

MTP 使用 mask-preserved clean target：

```text
G_bar = M_soft * target + (1 - M_soft) * source
```

它显式告诉模型：编辑区域学习 target，背景区域保持 source。

训练：

```bash
GPUS=0,1,2,3 \
NUM_PROCESSES=4 \
bash scripts/run_mtp_phasea.sh
```

输出：

```text
checkpoints/qwen_edit_2511_mtp_phasea/step-2269.safetensors
data/diffsynth/magicbrush_train_mtp_phasea/
data/outputs/magicbrush_dev_qwen2511_mtp_phasea/
reports/magicbrush_dev_qwen2511_mtp_phasea_release_metrics.csv
```

详细说明见：

```text
docs/MTP_LORA_WORKFLOW.md
docs/MTP_ALGORITHM.md
```

## 1.8 MoE Teacher LoRA

MoE 路线分两步：

```text
1. Pix2Pix / Qwen-Image-Edit / EditAR 生成专家候选
2. target + mask 监督下区域级融合，得到 MoE-Fusion Teacher
3. 用 teacher 训练 source-only Qwen2511 LoRA
```

构造专家候选和 teacher：

```bash
GPUS=0,1,2,3 bash scripts/run_keepedit_moe_fusion.sh
```

训练 MoE Teacher LoRA：

```bash
GPUS=0,1,2,3 \
NUM_PROCESSES=4 \
bash scripts/run_moe_teacher_lora.sh
```

输出：

```text
data/candidates/magicbrush_train_pix2pix_qwen_editar/
data/teachers/magicbrush_train_moe_fusion/
checkpoints/qwen_edit_2511_moe_teacher_onestage/step-2202.safetensors
data/outputs/magicbrush_dev_qwen2511_moe_teacher_onestage/
reports/magicbrush_dev_qwen2511_moe_teacher_onestage_release_metrics.csv
```

详细说明见：

```text
docs/MOE_LORA_WORKFLOW.md
```

## 1.9 统一评估

评估任意已有 LoRA：

```bash
EXPERIMENT_NAME=qwen2511_gt_onestage \
LORA_PATH=checkpoints/qwen_edit_2511_keepedit_gt_onestage \
bash scripts/evaluate_qwen_edit_experiment.sh

EXPERIMENT_NAME=qwen2511_mtp_phasea \
LORA_PATH=checkpoints/qwen_edit_2511_mtp_phasea \
bash scripts/evaluate_qwen_edit_experiment.sh

EXPERIMENT_NAME=qwen2511_moe_teacher_onestage \
LORA_PATH=checkpoints/qwen_edit_2511_moe_teacher_onestage \
bash scripts/evaluate_qwen_edit_experiment.sh
```

启用 MLLM：

```bash
RUN_MLLM=1 \
MLLM_BACKEND=qwen3_vl \
bash scripts/evaluate_qwen_edit_experiment.sh
```

最终总表：

```text
reports/keepedit_release_full_metrics_comparison.csv
```

## 1.10 查看可视化结果

完整可视化 gallery 已发布在 Hugging Face 数据仓库中；如果已经下载发布数据，也可以在本地直接打开对应 `index.html`。GitHub 上推荐点击 Hugging Face 入口查看完整文件夹。

| 内容 | 本地入口 | Hugging Face 入口 |
| --- | --- | --- |
| MoE-Fusion Teacher 可视化 | [reports/visual_gallery_magicbrush_dev_moe_fusion/index.html](reports/visual_gallery_magicbrush_dev_moe_fusion/index.html) | [HF: visual_gallery_magicbrush_dev_moe_fusion](https://huggingface.co/datasets/Yitaallen/keepedit-release-data/tree/main/reports/visual_gallery_magicbrush_dev_moe_fusion) |
| MTP LoRA 可视化 | [reports/visual_gallery_magicbrush_dev_qwen2511_mtp_phasea/index.html](reports/visual_gallery_magicbrush_dev_qwen2511_mtp_phasea/index.html) | [HF: visual_gallery_magicbrush_dev_qwen2511_mtp_phasea](https://huggingface.co/datasets/Yitaallen/keepedit-release-data/tree/main/reports/visual_gallery_magicbrush_dev_qwen2511_mtp_phasea) |
| MoE Teacher LoRA 可视化 | [reports/visual_gallery_magicbrush_dev_qwen2511_moe_teacher_onestage/index.html](reports/visual_gallery_magicbrush_dev_qwen2511_moe_teacher_onestage/index.html) | [HF: visual_gallery_magicbrush_dev_qwen2511_moe_teacher_onestage](https://huggingface.co/datasets/Yitaallen/keepedit-release-data/tree/main/reports/visual_gallery_magicbrush_dev_qwen2511_moe_teacher_onestage) |

可视化页面按样本展示输入图、目标图、prompt 和模型输出，适合快速检查模型是否真正执行了编辑、背景是否被破坏、以及不同方法之间的失败模式。

## 1.11 当前发布版指标

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

MoE Teacher LoRA 是当前客观指标最好的可部署模型；MTP LoRA 是更简洁、不依赖专家候选的强改进版本。

## 1.12 上传到 Hugging Face

```bash
bash scripts/pack_release_data_archives.sh
bash scripts/split_release_data_archives.sh

REPO_ID=Yitaallen/keepedit-release-data \
bash scripts/upload_split_release_archives.sh
```

说明见：

```text
docs/HUGGINGFACE_RELEASE.md
```

# 2. Qwen-Image-Edit baseline 与 4-step 蒸馏

这一部分是独立实验，不属于上面的 KeepEdit LoRA 工作流。它复现原始 Qwen-Image-Edit 在 MagicBrush dev60 上的 baseline，并训练一个轻量 image-space adapter，把 4-step 快速输出向 40-step teacher 输出对齐。

```text
source image + instruction -> Qwen-4step student
source image + instruction -> Qwen-40step teacher
source image + student image -> adapter -> refined image
```

MagicBrush 的 target image 和 mask 只用于评估，不作为 Qwen-Image-Edit 的输入。

## 2.1 相关文件

```text
qwen_distill/
├── run_qwen_baseline_dev60.sh        # 生成 Qwen-40step baseline
├── cache_qwen_4step_40step.sh        # 同时缓存 4-step student 和 40-step teacher
├── export_step_distill_metadata.sh   # 从缓存导出 adapter 训练 metadata
├── train_step_distill_adapter.sh     # 训练 gated residual adapter
├── eval_step_distill_adapter.sh      # 评估 adapter 输出
├── build_report_assets.sh            # 检查报告所需轻量产物
└── reports/qwen_distill/
    ├── qwen_distill_dev60_summary.md
    ├── qwen_distill_dev60_eval.json
    └── figures/sample_representative_one_row.png

docs/
├── qwen_image_edit_baseline.md
└── qwen_step_distill_adapter.md

logs/base4_to_base40/
├── train_step_loss.csv
└── step_loss_curve.png
```

大文件不在当前 git 目录中，复现时需要额外准备：

```text
Qwen-Image-Edit-2511 权重
Qwen-Image-Edit baseline 工程
MagicBrush dev60 manifest
outputs/qwen_step_distill/cache_dev60/
checkpoints/base4_to_base40/step_distill_adapter_best.pt
```

## 2.2 当前仓库状态

当前仓库保留了 Qwen 蒸馏实验的 shell 入口、文档、loss 曲线和 dev60 报告。`qwen_distill/*.sh` 脚本默认来自服务器实验环境：

```text
ROOT=/share/home/group6/Project/group6-image-edit
QWEN_ROOT=/share/home/group6/Project/qwen-image-edit-baseline
MANIFEST=/share/home/group6/our_project/artifacts/qwen_distill_dev60_final/magicbrush_dev60_keepedit.json
```

迁移到新机器时请显式覆盖这些路径。脚本内部会调用 `python -m keepedit.qwen_distill...`；如果当前环境无法导入该模块，需要使用原实验环境或恢复对应 Python 实现后再运行。

## 2.3 Baseline 定义

| 名称 | Steps | true_cfg_scale | 角色 |
| --- | ---: | ---: | --- |
| Qwen-4step | 4 | 4.0 | fast student |
| Qwen-40step | 40 | 4.0 | high-quality teacher |

Baseline 输入：

```text
source image + editing instruction
```

Baseline 输出：

```text
Qwen-Image-Edit(source, instruction)
```

## 2.4 运行 Qwen-40step baseline

```bash
ROOT=$(pwd) \
QWEN_ROOT=/path/to/qwen-image-edit-baseline \
MANIFEST=/path/to/magicbrush_dev60_keepedit.json \
OUT=$(pwd)/outputs/qwen_image_edit_baseline_dev60 \
GPU=0 \
MAX_SAMPLES=60 \
bash qwen_distill/run_qwen_baseline_dev60.sh
```

输出结构：

```text
outputs/qwen_image_edit_baseline_dev60/
├── dev_000000/
│   ├── input.png
│   ├── target.png
│   ├── prompt.txt
│   ├── base40.png
│   └── base40.json
└── qwen_distill_lora_speed_results.json
```

## 2.5 缓存 4-step student 与 40-step teacher

```bash
ROOT=$(pwd) \
QWEN_ROOT=/path/to/qwen-image-edit-baseline \
MANIFEST=/path/to/magicbrush_dev60_keepedit.json \
OUT=$(pwd)/outputs/qwen_step_distill/cache_dev60 \
GPU=0 \
MAX_SAMPLES=60 \
bash qwen_distill/cache_qwen_4step_40step.sh
```

等价地，也可以显式指定 variants：

```bash
VARIANTS="base4:4:4.0:none,base40:40:4.0:none" \
ROOT=$(pwd) \
QWEN_ROOT=/path/to/qwen-image-edit-baseline \
MANIFEST=/path/to/magicbrush_dev60_keepedit.json \
GPU=0 \
MAX_SAMPLES=60 \
bash qwen_distill/cache_qwen_4step_40step.sh
```

缓存结果：

```text
outputs/qwen_step_distill/cache_dev60/
├── dev_000000/
│   ├── base4.png
│   └── base40.png
└── qwen_distill_lora_speed_results.json
```

## 2.6 Adapter 方法

Adapter 冻结 Qwen-Image-Edit，只在图像空间做轻量残差修正：

```text
input image + Qwen-4step output -> adapter -> refined output
```

输入通道：

```text
source image        3 channels
student image       3 channels
student - source    3 channels
```

输出形式：

```text
delta = tanh(pred_rgb) * 0.25
alpha = sigmoid(pred_alpha)
output = clamp(student + alpha * delta, 0, 1)
```

训练目标是逼近 40-step teacher：

```text
L = L1(output, teacher)
  + 0.25 * L1(grad(output), grad(teacher))
  + 0.05 * (1 - SSIM_proxy(output, teacher))
  + 0.02 * mean(alpha)
```

## 2.7 导出训练 metadata

```bash
ROOT=$(pwd) \
CACHE_JSON=$(pwd)/outputs/qwen_step_distill/cache_dev60/qwen_distill_lora_speed_results.json \
OUT=$(pwd)/outputs/qwen_step_distill/step_distill_train.json \
bash qwen_distill/export_step_distill_metadata.sh
```

metadata 中每个样本包含：

```json
{
  "sample_id": "dev_000000",
  "prompt": "edit instruction",
  "input_image": "/path/to/input.png",
  "student_image": "/path/to/base4.png",
  "teacher_image": "/path/to/base40.png",
  "target_image": "/path/to/target.png"
}
```

## 2.8 训练 adapter

```bash
ROOT=$(pwd) \
METADATA=$(pwd)/outputs/qwen_step_distill/step_distill_train.json \
OUT=$(pwd)/outputs/qwen_step_distill/train_base4_to_base40 \
GPU=0 \
bash qwen_distill/train_step_distill_adapter.sh
```

脚本默认参数：

```text
IMAGE_SIZE=512
HIDDEN=128
BATCH_SIZE=8
EPOCHS=688
LR=3e-5
VAL_COUNT=10
```

已记录训练设置：

```text
epochs: 688
optimization steps: 54,343
batch size: 8
```

Adapter 开销：

```text
pure adapter forward: about 0.004s/image
adapter with preprocessing: about 0.08s/image
overhead vs 13.7s Qwen-4step: < 1%
```

## 2.9 评估 adapter

```bash
ROOT=$(pwd) \
STUDENT_JSON=$(pwd)/outputs/qwen_step_distill/cache_dev60/qwen_distill_lora_speed_results.json \
CKPT=$(pwd)/checkpoints/base4_to_base40/step_distill_adapter_best.pt \
OUT=$(pwd)/outputs/qwen_step_distill/eval_base4_adapter_dev60 \
bash qwen_distill/eval_step_distill_adapter.sh
```

默认输出：

```text
outputs/qwen_step_distill/eval_base4_adapter_dev60/
```

## 2.10 已提交 dev60 结果

当前仓库保留的报告：

```text
qwen_distill/reports/qwen_distill/qwen_distill_dev60_summary.md
qwen_distill/reports/qwen_distill/qwen_distill_dev60_eval.json
qwen_distill/reports/qwen_distill/figures/sample_representative_one_row.png
logs/base4_to_base40/train_step_loss.csv
logs/base4_to_base40/step_loss_curve.png
```

dev60 指标：

| Method | Samples | SSIM(Target) | PSNR(Target) | BG-SSIM | BG-PSNR | Edit Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen_4step | 60 | 0.7535 | 16.2793 | 0.6018 | 18.3761 | 0.0327 |
| qwen_40step | 60 | 0.7804 | 17.9871 | 0.6910 | 20.2761 | 0.0150 |
| qwen_4step_adapter | 60 | 0.8213 | 19.2725 | 0.7453 | 21.7426 | 0.0133 |
| open_lora4_4step | 60 | 0.8688 | 21.2100 | 0.8655 | 25.4724 | 0.0078 |
| open_lora4_adapter | 60 | 0.8687 | 21.1985 | 0.8647 | 25.4372 | 0.0079 |

结论：在未引入 LoRA 的 Qwen baseline 上，4-step adapter 明显提升 target 相似度和背景保持；在 open_lora4 结果上，adapter 变化很小，说明该 LoRA 输出本身已经接近目标，轻量残差修正空间有限。

## 2.11 报告资产检查

`build_report_assets.sh` 默认检查服务器路径下的 `reports/qwen_distill`。在当前仓库中，报告位于 `qwen_distill/reports/qwen_distill`，可以这样运行：

```bash
ROOT=$(pwd) \
REPORT=$(pwd)/qwen_distill/reports/qwen_distill \
bash qwen_distill/build_report_assets.sh
```

如果本地没有 `checkpoints/base4_to_base40/step_distill_adapter_best.pt`，该检查会失败；这表示缺少未纳入 git 的 adapter checkpoint。

## 2.12 详细文档

```text
docs/qwen_image_edit_baseline.md
docs/qwen_step_distill_adapter.md
```

# 3. InstructPix2Pix + MagicBrush

本部分基于官方 InstructPix2Pix 工程，在不重新训练模型的前提下，围绕 MagicBrush dev set 增加了批量推理、多候选 oracle、mask crop 局部编辑、背景保持 rerank 和 soft mask fusion 等 inference-time 方法。

对应说明文档：

```text
instruct-pix2pix/README_InstructPix2Pix_MagicBrush.md
```

## 3.1 环境与外部文件

`instruct-pix2pix/` 中主要保存小组新增脚本，不是完整官方工程。运行前需要准备官方 InstructPix2Pix 代码环境，使目录中存在：

```text
instruct-pix2pix/edit_cli.py
instruct-pix2pix/configs/generate.yaml
instruct-pix2pix/checkpoints/instruct-pix2pix-00-22000.ckpt
```

推荐使用官方环境：

```bash
cd instruct-pix2pix
conda env create -f environment.yaml
conda activate ip2p
```

官方 checkpoint 默认路径：

```text
instruct-pix2pix/checkpoints/instruct-pix2pix-00-22000.ckpt
```

部分增强脚本也可使用自己训练或对齐后的 checkpoint，例如：

```text
instruct-pix2pix/checkpoints/ip2p_finetuned_ours/ip2p_magicbrush_alignment.ckpt
```

## 3.2 MagicBrush 数据格式

脚本默认读取 MagicBrush `manifest.json`，推荐整理成：

```text
benchmarks/magicbrush/prepared_full/dev/
├── manifest.json
└── images/
    ├── input/
    ├── target/
    └── mask/
```

`manifest.json` 每个样本至少包含：

```json
{
  "id": "34_2",
  "instruction": "Open the zebra's mouth.",
  "input_image": "images/input/34_2.png",
  "target_image": "images/target/34_2.png",
  "mask_image": "images/mask/34_2.png"
}
```

## 3.3 原始 baseline 批量推理

脚本：

```text
instruct-pix2pix/run_magicbrush_dev.py
```

作用：读取 MagicBrush `manifest.json`，逐样本运行官方 InstructPix2Pix baseline，保存输出，并支持断点续跑。

注意：当前脚本顶部写有默认路径，运行前需要改成自己的 MagicBrush dev 路径和输出路径：

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

## 3.4 Prompt + CFG/Seed Oracle 多候选搜索

脚本：

```text
instruct-pix2pix/run_magicbrush_p2p_oracle.py
```

作用：对每个样本生成多组候选，改变 prompt rewrite、CFG-text、CFG-image 和 seed；然后利用 target + mask 的 oracle 分数选择最优候选。该方法用于分析 InstructPix2Pix 的上限，不是最终部署方案，因为它使用了 target。

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

## 3.5 Mask-Crop Oracle 局部编辑

脚本：

```text
instruct-pix2pix/run_magicbrush_p2p_oracle_crop.py
```

作用：使用 mask 定位编辑区域，对局部 crop 进行 InstructPix2Pix 编辑，再贴回原图；同时保留全图候选，与 crop 候选一起做 oracle 选择。该方法适合小物体、局部属性修改等任务。

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

## 3.6 Background-preserve Rerank + Fusion

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

## 3.7 Step-level latent preserve 实验

脚本：

```text
instruct-pix2pix/run_magicbrush_ultimate.py
```

作用：在 K-Diffusion 采样过程中用 MagicBrush mask 对 latent 做逐步背景保护，前期锁定背景区域，后期逐渐放开边缘以降低接缝。该脚本当前路径、样本数和 checkpoint 在文件顶部硬编码，默认只处理前 20 个样本。

运行前需要按本机环境修改：

```python
DATA_DIR = "/path/to/benchmarks/magicbrush/prepared_full/dev"
OUTPUT_DIR = "./results/magicbrush_p2p_perfect_outputs"
ckpt_path = "./checkpoints/ip2p_finetuned_ours/ip2p_magicbrush_alignment.ckpt"
```

运行：

```bash
cd instruct-pix2pix
CUDA_VISIBLE_DEVICES=0 python run_magicbrush_ultimate.py
```

## 3.8 推荐实验流程与指标

推荐流程：

```text
baseline
  -> oracle multi-candidate
  -> mask-crop oracle
  -> background-preserve rerank + fusion
```

常用输出根目录：

```text
instruct-pix2pix/results/
├── magicbrush_baseline_dev_outputs/
├── magicbrush_p2p_oracle_dev/
├── magicbrush_p2p_oracle_crop_dev/
├── magicbrush_p2p_bg_preserve_dev/
└── magicbrush_p2p_perfect_outputs/
```

常用评估指标：

```text
Full SSIM / PSNR
Edit-region SSIM
Background SSIM / PSNR
Edit Change
```

总结：Oracle 多候选用于估计上限，crop 强化小区域编辑，background-preserve rerank 和 fusion 改善背景稳定性；这些方法主要是 inference-time analysis，会在编辑正确性和背景保持之间形成取舍。

# 4. EditAR 改进分析

本部分基于原始 EditAR 自回归图像编辑框架，保留 baseline，并加入报告中的 LoRA 改进与分析代码。详细说明见：

```text
docs/editar/EditAR.md
```

## 4.1 项目简介

EditAR 的整体流程是：先用 VQ 图像 tokenizer 将输入图像和目标编辑图像离散化为视觉 token；再用 GPT 风格 Transformer 接收 FLAN-T5 编码后的文本指令、输入图像 token 和任务模式 embedding；最后以 next-token prediction 的方式逐步预测编辑后图像 token，并通过 VQ decoder 解码回 RGB 图像。

本仓库保留两类内容：

```text
EditAR 原始 baseline 代码
报告中最终写入的 LoRA 改进与分析代码
```

报告涉及的 LoRA 变体：

| 变体 | 说明 |
| --- | --- |
| Projection LoRA | 只在 `cap_proj.fc1` 和 `cap_proj.fc2` 注入 LoRA |
| Backbone LoRA | 在 Transformer attention、FFN 和 `cap_proj` 注入 LoRA |
| Region LoRA | Backbone LoRA + MagicBrush mask 加权 token loss |
| Contrast LoRA | Region LoRA + 正确/错误指令 margin contrastive loss |

## 4.2 核心改动位置

路径均相对于仓库根目录。

| 改动内容 | 代码路径 | 说明 |
| --- | --- | --- |
| LoRA 模块实现 | `EditAR/autoregressive/models/lora.py` | 定义 `LoRALinear`、LoRA 注入、仅训练 LoRA 参数、保存 adapter state dict 等工具 |
| GPT loss token 加权 | `EditAR/autoregressive/models/gpt_edit.py` | 在 `Transformer.forward()` 中加入 `token_loss_weight` 和 `compute_loss` |
| LoRA / Region / Contrast 训练入口 | `EditAR/autoregressive/train/train_edit.py` | 加入 `--use-lora`、`--lora-target-modules`、`--use-mask-weighted-loss`、`--use-negative-contrastive` 等参数 |
| Region LoRA mask 权重 | `EditAR/autoregressive/train/mask_weight.py` | 将 MagicBrush mask 下采样到视觉 token 网格，生成编辑区/背景区 token loss 权重 |
| Contrast LoRA 对比损失 | `EditAR/autoregressive/train/contrastive.py` | 实现正确指令与错误指令之间的 margin contrastive loss |
| MagicBrush 训练集 | `EditAR/dataset/Edit_MagicBrush.py` | 读取 MagicBrush train split，返回输入图、目标图、文本和 mask |
| MagicBrush 推理集 | `EditAR/dataset/Edit_MagicBrush_eval.py` | 支持 MagicBrush dev/eval 推理和可视化保存 |
| 数据集注册 | `EditAR/dataset/build.py` | 在 mixed dataset builder 中加入 MagicBrush，并为无 mask 数据补默认 mask |
| LoRA checkpoint 推理 | `EditAR/autoregressive/sample/sample_edit_example.py`、`EditAR/autoregressive/sample/sample_edit_folder.py` | 支持加载 adapter-only LoRA checkpoint 和 MagicBrush batch inference |
| MagicBrush 指标评估 | `EditAR/tools/evaluate_magicbrush_outputs.py` | 计算 Target--Output、Input--Output、BG-SSIM、Edit Change 等指标 |
| VQ / token 诊断 | `EditAR/tools/vq_reconstruct_benchmark.py`、`EditAR/tools/diagnose_magicbrush_tokens.py` | 用于 VQ 重建误差和自回归 token 分布偏移分析 |
| 复现实验脚本 | `EditAR/scripts/run_magicbrush_only_lora_sweep_a800.sh`、`EditAR/scripts/run_magicbrush60_report_benchmark_parallel_a800.sh` | LoRA 消融训练/评估和 MagicBrush-60 表格复现 |

## 4.3 环境配置

```bash
cd EditAR
bash scripts/install.sh
source .venv/bin/activate
```

代码主要按 Python 3.10、PyTorch 2.2.1、CUDA GPU 和混合精度训练/推理环境整理。

## 4.4 预训练权重

创建默认目录：

```bash
cd EditAR
mkdir -p pretrained_models/t5-ckpt pretrained_models checkpoints/editar/editar_release
```

下载 FLAN-T5-XL 文本编码器：

```bash
huggingface-cli download google/flan-t5-xl \
  --local-dir pretrained_models/t5-ckpt/flan-t5-xl
```

下载 LlamaGen VQ tokenizer：

```bash
wget -O pretrained_models/vq_ds16_t2i.pt \
  https://huggingface.co/peizesun/llamagen_t2i/resolve/main/vq_ds16_t2i.pt
```

运行原始 EditAR 推理时，将 release checkpoint 放到：

```text
EditAR/checkpoints/editar/editar_release/editar_release.pt
```

如果需要从 LlamaGen T2I 初始化训练 baseline，将权重放到：

```text
EditAR/pretrained_models/t2i_XL_stage2_512.pt
```

## 4.5 数据准备

原始 baseline 训练脚本默认读取 `EditAR/data/` 下已经处理好的 Hugging Face 数据集：

```text
EditAR/data/
├── MultiGen-20M_depth_HF/
├── Condition_Segmentation/
├── PIPE_HF/
├── Seedx_Unsplash_HF/
└── MagicBrush_HF/
```

本报告实验主要使用 MagicBrush，推荐放在：

```text
EditAR/data/MagicBrush_HF
```

如果数据放在其他位置，可以通过环境变量传入：

```bash
cd EditAR
MAGICBRUSH_PATH=/path/to/edit-data/MagicBrush_HF \
bash scripts/run_magicbrush_only_lora_sweep_a800.sh
```

## 4.6 Baseline 推理与评估

单图编辑示例：

```bash
cd EditAR
python autoregressive/sample/sample_edit_example.py \
  --gpt-ckpt checkpoints/editar/editar_release/editar_release.pt \
  --vq-ckpt pretrained_models/vq_ds16_t2i.pt \
  --t5-path pretrained_models/t5-ckpt \
  --cfg-scale 3 \
  --seed 83
```

MagicBrush 批量推理：

```bash
cd EditAR
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

计算 MagicBrush 指标：

```bash
cd EditAR
python tools/evaluate_magicbrush_outputs.py \
  --magicbrush-path data/MagicBrush_HF \
  --samples-dir outputs/samples/magicbrush/samples/txt_1.0 \
  --output-dir outputs/benchmark/pretrained_magicbrush60 \
  --cfg-scale 1.0 \
  --max-samples 60 \
  --image-size 512
```

## 4.7 训练与 LoRA 消融

原始 baseline 训练：

```bash
cd EditAR
bash scripts/train.sh
```

运行报告中的 LoRA 消融训练，包括 Projection LoRA、Backbone LoRA、Region LoRA 和 Contrast LoRA：

```bash
cd EditAR
REPO_DIR=$(pwd) \
ACCELERATE=$(pwd)/.venv/bin/accelerate \
PYTHON=$(pwd)/.venv/bin/python \
MAGICBRUSH_PATH=/path/to/edit-data/MagicBrush_HF \
bash scripts/run_magicbrush_only_lora_sweep_a800.sh
```

单独运行 Contrast LoRA 训练：

```bash
cd EditAR
bash scripts/train_negative_lora.sh
```

## 4.8 报告指标复现

运行 MagicBrush-60 报告表格 benchmark：

```bash
cd EditAR
REPO_DIR=$(pwd) \
PYTHON=$(pwd)/.venv/bin/python \
MAGICBRUSH_PATH=/path/to/edit-data/MagicBrush_HF \
bash scripts/run_magicbrush60_report_benchmark_parallel_a800.sh
```

输出汇总：

```text
EditAR/outputs/report_benchmark_magicbrush60/magicbrush60_report_summary.json
```

## 4.9 诊断工具

VQ 重建误差诊断：

```bash
cd EditAR
REPO_DIR=$(pwd) \
PYTHON=$(pwd)/.venv/bin/python \
HF_DATASET=/path/to/edit-data/MagicBrush_HF \
bash scripts/run_vq_reconstruction_benchmark_a800.sh
```

MagicBrush token 级自回归误差诊断：

```bash
cd EditAR
REPO_DIR=$(pwd) \
PYTHON=$(pwd)/.venv/bin/python \
MAGICBRUSH_PATH=/path/to/edit-data/MagicBrush_HF \
bash scripts/run_magicbrush8_token_diagnostics_a800.sh
```

# 5. layer

本部分实现了一个基于 Qwen-Image-Edit LoRA 微调、Qwen-Image-Layered 图像分层和 CLIP 图层推荐的局部图像编辑流程。完整说明见：

```text
layer/layer_README.md
```

## 5.1 功能概览

该流程包含：

```text
成对 old/new 数据训练 Qwen-Image-Edit LoRA
Qwen-Image-Layered 将输入图像分解为多个 RGBA layer
CLIP 计算目标文本与各图层的相似度并推荐目标图层
用户确认推荐图层，或通过 --layer 手动指定图层
Qwen-Image-Edit 加载 LoRA 只编辑目标图层
将编辑后的图层与其他图层重新合成最终图像
保存最终结果图、对比图和编辑后的单独图层
```

## 5.2 代码结构

```text
layer/
├── layer_README.md                  # 本模块说明
├── train_qwen_edit_lora_pair.py     # Qwen-Image-Edit 成对 old/new LoRA 微调脚本
├── try0_lora.py                     # 图层分解、CLIP 推荐、LoRA 编辑与重合成脚本
└── untitled.py                      # 独立 CLIP 图文相似度图层推荐工具
```

## 5.3 环境依赖

建议使用带 GPU 的 Linux 环境，例如 AutoDL。主要依赖：

```bash
pip install torch torchvision
pip install diffusers transformers accelerate peft safetensors
pip install pillow numpy
pip install bitsandbytes
pip install modelscope
```

如果服务器不能联网下载模型，建议提前下载到本地路径，并在脚本中指定本地模型目录。

## 5.4 模型准备

本模块需要：

| 模型 | 作用 |
| --- | --- |
| `Qwen/Qwen-Image-Layered` | 图像分层，将输入图片拆分为多个 RGBA layer |
| `Qwen/Qwen-Image-Edit` | 图像编辑基础模型，用于 LoRA 微调和推理编辑 |
| `openai/clip-vit-base-patch32` | CLIP 图文相似度模型，用于自动推荐目标图层 |

当前 `try0_lora.py` 顶部默认使用本地路径：

```python
LAYERED_MODEL_PATH = "/root/autodl-tmp/layer"
EDIT_MODEL_PATH = "/root/autodl-tmp/edit"
```

迁移到其他机器时，需要改成自己的本地模型路径。

## 5.5 LoRA 训练数据

LoRA 微调数据由三部分组成：

```text
/root/autodl-tmp/000/
├── old/
│   ├── 001.png
│   ├── 002.png
│   └── ...
├── new/
│   ├── 001.png
│   ├── 002.png
│   └── ...
└── untitled.txt
```

推荐保持 `old` 和 `new` 中文件名一致：

```text
old/001.png  <->  new/001.png
old/002.png  <->  new/002.png
```

`untitled.txt` 支持两种格式。只有一行时，所有训练图片共用同一个 prompt：

```text
switch to a robot wearing armor
```

多行时，每一行对应一组训练图片：

```text
switch to a robot wearing armor
change the clothes to blue
make the background snowy
```

## 5.6 Qwen-Image-Edit LoRA 微调

脚本：

```text
layer/train_qwen_edit_lora_pair.py
```

示例命令：

```bash
cd layer
accelerate launch train_qwen_edit_lora_pair.py \
  --pretrained_model_name_or_path /root/autodl-tmp/edit \
  --old_dir /root/autodl-tmp/000/old \
  --new_dir /root/autodl-tmp/000/new \
  --prompt_file /root/autodl-tmp/000/untitled.txt \
  --output_dir /root/autodl-tmp/qwen_edit_lora_out \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --rank 8 \
  --lora_alpha 16 \
  --learning_rate 1e-4 \
  --max_train_steps 800 \
  --mixed_precision bf16 \
  --use_8bit_adam
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--pretrained_model_name_or_path` | Qwen-Image-Edit 基座模型路径 |
| `--old_dir`、`--new_dir` | 训练输入图和目标图目录 |
| `--prompt_file` | 编辑指令文本文件 |
| `--output_dir` | LoRA 输出目录 |
| `--resolution` | 训练分辨率，默认 512 |
| `--rank`、`--lora_alpha`、`--lora_dropout` | LoRA 配置 |
| `--train_batch_size`、`--gradient_accumulation_steps` | batch 和梯度累计 |
| `--max_train_steps`、`--learning_rate` | 训练步数和学习率 |
| `--mixed_precision` | `no`、`fp16` 或 `bf16` |
| `--checkpointing_steps` | 中间 LoRA checkpoint 保存间隔 |
| `--validation_steps` | 验证推理间隔，0 表示关闭 |

训练完成后输出：

```text
/root/autodl-tmp/qwen_edit_lora_out/pytorch_lora_weights.safetensors
/root/autodl-tmp/qwen_edit_lora_out/training_config.json
```

## 5.7 图层分解、推荐与局部编辑

脚本：

```text
layer/try0_lora.py
```

概念上，图层识别文本和图像编辑 prompt 可以分开控制；但当前 `try0_lora.py` 实际暴露的参数是 `--prompt`、`--layer`、`--layers`、`--strength`、`--resolution` 和 LoRA 相关参数，没有单独的 `--target-text`。如需用独立文本推荐图层，可先用 `untitled.py` 对分解后的 layer 排序，再把选中的编号传给 `try0_lora.py --layer`。

自动推荐或交互选择图层：

```bash
cd layer
python try0_lora.py /root/autodl-tmp/test.png \
  --prompt "把红色衣服改成蓝色" \
  --lora /root/autodl-tmp/qwen_edit_lora_out \
  --lora-weight-name pytorch_lora_weights.safetensors \
  --lora-scale 0.8 \
  --layers 4 \
  --resolution 640 \
  --strength 0.8
```

手动指定图层，跳过推荐确认：

```bash
cd layer
python try0_lora.py /root/autodl-tmp/test.png \
  --prompt "把红色衣服改成蓝色" \
  --layer 2 \
  --lora /root/autodl-tmp/qwen_edit_lora_out \
  --lora-weight-name pytorch_lora_weights.safetensors \
  --lora-scale 0.8
```

运行流程：

```text
输入图片
-> Qwen-Image-Layered 分解图层
-> 保存 layer_0.png、layer_1.png、...
-> CLIP 根据目标文本或 prompt 推荐目标图层
-> 用户确认或手动选择图层
-> Qwen-Image-Edit 加载 LoRA 编辑目标图层
-> 替换目标图层
-> 重新合成最终图片
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `image` | 输入图片路径 |
| `--prompt` / `-p` | 图像编辑指令，必填 |
| `--layer` / `-l` | 手动指定目标图层编号 |
| `--layers` / `-n` | 分解图层数量，默认 4 |
| `--strength` / `-s` | 修改强度，默认 0.8 |
| `--resolution` / `-r` | 处理分辨率，默认 640 |
| `--lora` | LoRA 路径，可以是目录或 `.safetensors` 文件 |
| `--lora-weight-name` | 当 `--lora` 是目录时指定权重文件名 |
| `--lora-scale` | LoRA 强度，常用 0.5 到 1.2 |
| `--lora-adapter-name` | LoRA adapter 名称，默认 `edit_lora` |

## 5.8 独立 CLIP 图层推荐

脚本：

```text
layer/untitled.py
```

用于在一组候选 layer 图片中，根据文本描述计算 CLIP 相似度并排序。

示例：

```bash
cd layer
python untitled.py \
  --images debug_layers/layer_0.png debug_layers/layer_1.png debug_layers/layer_2.png debug_layers/layer_3.png \
  --text "red clothes" \
  --clip-model-name openai/clip-vit-base-patch32 \
  --topk 4
```

输出包括每张图层的相似度、匹配概率、排序结果和最匹配图层路径。

## 5.9 输出结果

`try0_lora.py` 会自动保存：

```text
layer/output/result_时间戳.jpg
layer/output/compare_时间戳.jpg
layer/output/edited_layer_图层编号_时间戳.png
```

其中：

| 文件 | 说明 |
| --- | --- |
| `result_时间戳.jpg` | 最终编辑结果 |
| `compare_时间戳.jpg` | 原图与编辑结果的左右对比图 |
| `edited_layer_图层编号_时间戳.png` | 被修改后的单独图层 |

图层分解阶段还会保存调试图层，例如：

```text
layer/debug_layers/layer_0.png
layer/debug_layers/layer_1.png
...
```

## 5.10 注意事项

- `try0_lora.py` 默认使用本地模型路径 `/root/autodl-tmp/layer` 和 `/root/autodl-tmp/edit`，换机器必须修改。
- 局部编辑质量依赖分层结果；如果 Qwen-Image-Layered 没有把目标区域分到清晰图层，后续编辑会受影响。
- CLIP 推荐只用于粗定位，复杂指令建议手动指定 `--layer`。
- LoRA 只加载到 Qwen-Image-Edit 编辑模型，不影响 Qwen-Image-Layered 分层模型。
