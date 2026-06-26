# Qwen-Image-Edit LoRA 图层局部编辑系统

本项目实现了一个基于 **Qwen-Image-Edit LoRA 微调**、**Qwen-Image-Layered 图像分层** 和 **CLIP 图层推荐** 的局部图像编辑流程。系统可以先对 Qwen-Image-Edit 进行 LoRA 微调，然后在推理阶段将输入图片分解为多个图层，利用 CLIP 自动推荐需要修改的图层，并由用户确认后执行局部编辑，最后将修改后的图层与其他图层重新合成。

## 1. 项目功能

本项目主要包含以下功能：

* 使用成对数据对 Qwen-Image-Edit 进行 LoRA 微调
* 支持 `old 原图 + prompt + new 目标图` 的训练数据格式
* 使用 Qwen-Image-Layered 对输入图像进行图层分解
* 自动保存分解后的图层到 `debug_layers/`
* 使用 CLIP 计算目标文本与各图层之间的相似度
* 自动推荐最可能需要编辑的图层编号
* 支持用户确认推荐结果或手动选择图层
* 加载训练好的 LoRA 权重进行局部图像编辑
* 将编辑后的图层与其他图层重新合成最终图片
* 自动保存最终结果图、对比图和编辑后的单独图层

## 2. 项目结构

```text
project/
├── train_qwen_edit_lora_pair.py     # Qwen-Image-Edit LoRA 微调训练脚本
├── try0_lora.py                     # 图层分解、CLIP 推荐、LoRA 编辑与合成脚本
├── untitled.py                      # CLIP 图文相似度推荐图层脚本
└── layer_README.md                        # 项目说明文件
```

## 3. 环境依赖

建议使用带 GPU 的 Linux 环境运行，例如 AutoDL。

主要依赖包括：

```bash
pip install torch torchvision
pip install diffusers transformers accelerate peft safetensors
pip install pillow numpy
pip install bitsandbytes
```

如果使用 ModelScope 或其他 CLIP 模型，也需要根据实际情况安装对应依赖。

```bash
pip install modelscope
```

如果服务器不能联网下载模型，建议提前将模型下载到本地路径，然后在脚本中指定本地模型目录。

## 4. 数据准备

LoRA 微调数据由三部分组成：

* `old/`：编辑前的原图
* `new/`：编辑后的目标图
* `untitled.txt`：文本编辑指令 prompt

数据目录示例：

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

推荐保持 `old` 和 `new` 中的文件名一致，例如：

```text
old/001.png  <->  new/001.png
old/002.png  <->  new/002.png
```

`untitled.txt` 支持两种格式。

如果只有一行 prompt，则所有训练图片共用同一个 prompt：

```text
switch to a robot wearing armor
```

如果有多行 prompt，则每一行对应一组训练图片：

```text
switch to a robot wearing armor
change the clothes to blue
make the background snowy
```

## 5. LoRA 微调训练

使用 `train_qwen_edit_lora_pair.py` 对 Qwen-Image-Edit 进行 LoRA 微调。

示例命令：

```bash
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

训练完成后，会在输出目录中生成 LoRA 权重文件：

```text
/root/autodl-tmp/qwen_edit_lora_out/pytorch_lora_weights.safetensors
```

## 6. 推理与局部编辑

完成 LoRA 微调后，使用 `try0_lora.py` 进行图层局部编辑。

推荐使用 `--target-text` 和 `--prompt` 分别控制图层识别和图像编辑。

* `--target-text`：用于 CLIP 判断哪个图层需要修改
* `--prompt`：用于 Qwen-Image-Edit 执行具体编辑

示例命令：

```bash
python try0_lora.py /root/autodl-tmp/test.png \
  --prompt "把红色衣服改成蓝色" \
  --target-text "red clothes" \
  --lora /root/autodl-tmp/qwen_edit_lora_out \
  --lora-weight-name pytorch_lora_weights.safetensors \
  --lora-scale 0.8 \
  --layers 4 \
  --resolution 640 \
  --strength 0.8
```

运行流程如下：

```text
输入图片
↓
Qwen-Image-Layered 分解图层
↓
保存 layer_0.png、layer_1.png、...
↓
CLIP 根据 target-text 推荐目标图层
↓
用户确认或手动选择图层
↓
Qwen-Image-Edit 加载 LoRA 编辑目标图层
↓
替换目标图层
↓
重新合成最终图片
```

## 7. 手动指定图层

如果已经知道要编辑的图层编号，可以直接使用 `--layer` 参数跳过 CLIP 推荐：

```bash
python try0_lora.py /root/autodl-tmp/test.png \
  --prompt "把红色衣服改成蓝色" \
  --layer 2 \
  --lora /root/autodl-tmp/qwen_edit_lora_out \
  --lora-weight-name pytorch_lora_weights.safetensors \
  --lora-scale 0.8
```

此时程序会直接编辑指定图层，不再调用 CLIP 自动推荐。

## 8. 输出结果

推理完成后，系统会自动保存以下文件：

```text
output/result_时间戳.jpg
output/compare_时间戳.jpg
output/edited_layer_图层编号_时间戳.png
```

其中：

* `result_时间戳.jpg`：最终编辑结果
* `compare_时间戳.jpg`：原图与编辑结果的左右对比图
* `edited_layer_图层编号_时间戳.png`：被修改后的单独图层

## 9. 模型下载

本项目需要提前下载以下模型。

| 模型                             | 作用                        |
| ------------------------------ | -------------------------- |
| `Qwen/Qwen-Image-Layered`      | 图像分层，将输入图片拆分为多个 RGBA layer |
| `Qwen/Qwen-Image-Edit`         | 图像编辑基础模型，用于 LoRA 微调和推理编辑   |
| `openai/clip-vit-base-patch32` | CLIP 图文相似度模型，用于自动推荐目标图层    |

