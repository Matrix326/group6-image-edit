# InstructPix2Pix + MagicBrush Custom Evaluation Pipeline (README)

## 1. 项目说明

本项目基于官方 InstructPix2Pix 开源代码（https://github.com/timothybrooks/instruct-pix2pix），
在**不重新训练模型的前提下**，我们额外开发了多个推理与评测脚本，用于在 MagicBrush benchmark 上分析和改进图像编辑效果。

---

## 2. 环境要求

### 2.1 基础环境（官方 InstructPix2Pix）

建议使用官方环境：

- Python 3.8.5
- PyTorch 1.11.0
- CUDA 11.3
- diffusers / k-diffusion / transformers 等依赖

可通过：

```bash
conda env create -f environment.yaml
conda activate ip2p
```

---

## 3. 模型权重

需要下载官方 checkpoint：

```
checkpoints/instruct-pix2pix-00-22000.ckpt
```

下载脚本：

```bash
bash scripts/download_checkpoints.sh
```

---

## 4. 数据集（MagicBrush）

数据路径结构：

```
benchmarks/magicbrush/prepared_full/dev/
├── manifest.json
├── images/
│   ├── input/
│   ├── target/
│   └── mask/
```

---

## 5. Baseline 运行方式

### 单张图片测试（官方脚本）

```bash
CUDA_VISIBLE_DEVICES=0 python edit_cli.py \
  --ckpt ./checkpoints/instruct-pix2pix-00-22000.ckpt \
  --input ./input.png \
  --output ./output.png \
  --edit "turn the person into a robot" \
  --steps 50 \
  --cfg-text 7.5 \
  --cfg-image 1.5
```

---

## 6. 我们新增的核心脚本

我们在官方 repo 基础上新增以下脚本：

---

## 6.1 MagicBrush baseline 批量测试

### 脚本

```
run_magicbrush_dev.py
```

### 功能

- 读取 MagicBrush manifest.json
- 批量运行 InstructPix2Pix baseline
- 保存输出结果
- 支持断点续跑

### 运行方式

```bash
CUDA_VISIBLE_DEVICES=0 python run_magicbrush_dev.py
```

---

## 6.2 Prompt + CFG/Seed Oracle（多候选增强）

### 脚本

```
run_magicbrush_p2p_oracle.py
```

### 功能

- Prompt rewrite（指令增强）
- 多 CFG-text / CFG-image / seed 组合生成候选图
- 使用 oracle（target + mask SSIM）选择最优结果

### 运行方式

```bash
CUDA_VISIBLE_DEVICES=0 python run_magicbrush_p2p_oracle.py \
  --data-dir /path/to/magicbrush/dev \
  --ckpt ./checkpoints/instruct-pix2pix-00-22000.ckpt \
  --steps 50 \
  --cfg-texts 7.5,9.0,10.5,12.0 \
  --cfg-images 1.0,1.2,1.5 \
  --seeds 0,1,2,3 \
  --limit 20
```

### 核心思想

> 多参数采样 + oracle rerank 提升候选上限

---

## 6.3 Mask-Crop Oracle（局部编辑）

### 脚本

```
run_magicbrush_p2p_oracle_crop.py
```

### 功能

- 使用 mask 定位编辑区域
- crop 局部区域进行 P2P 编辑
- 再贴回原图
- 与全图候选一起做 oracle 选择

### 运行方式

```bash
CUDA_VISIBLE_DEVICES=0 python run_magicbrush_p2p_oracle_crop.py \
  --use-crop \
  --crop-expand 0.35 \
  --crop-min-size 128 \
  --limit 20
```

### 核心思想

> 将全图编辑转为局部编辑，提高小目标任务表现

---

## 6.4 Background-preserve Rerank + Fusion

### 脚本

```
rerank_background_preserve.py
```

### 功能

- 引入背景保持评分（mask外区域）
- 同时考虑：
  - 编辑区域质量
  - 背景保持质量
- 使用 soft mask fusion 恢复背景

### 运行方式

```bash
python rerank_background_preserve.py \
  --w-edit 0.50 \
  --w-preserve 0.45 \
  --w-full 0.05
```

### 核心思想

> 在“编辑正确性”和“背景保持”之间做平衡

---

## 7. 推荐实验流程

```
baseline
  ↓
oracle multi-candidate
  ↓
mask-crop oracle
  ↓
background-preserve rerank
```

---

## 8. 输出结构

```
results/
├── baseline/
├── oracle/
├── oracle_crop/
├── bg_preserve/
```

---

## 9. 评价指标

- Full SSIM
- Edit-region SSIM
- Background SSIM
- PSNR

---

## 10. 实验总结

- Oracle：提升编辑上限
- Crop：增强局部编辑
- BG-preserve：增强背景稳定性
- 存在明显 trade-off

---

## 11. 注意事项

- 不重新训练模型
- 全部为 inference-time methods
- 基于 MagicBrush dev set
