#!/usr/bin/env python3
"""
Qwen-Image 智能图层编辑系统 - 双GPU显存优化版 + Qwen-Image-Edit LoRA 支持

用法示例：
python try0_lora.py input.png \
  --prompt "把衣服改成黑色" \
  --layer 2 \
  --lora /root/autodl-tmp/my_lora/pytorch_lora_weights.safetensors \
  --lora-scale 0.8
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from diffusers import QwenImageLayeredPipeline, QwenImageEditPipeline
from PIL import Image, ImageDraw
from pathlib import Path
import argparse
import gc
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ========== 配置区域 ==========
LAYERED_MODEL_PATH = "/root/autodl-tmp/layer"
EDIT_MODEL_PATH = "/root/autodl-tmp/edit"
DTYPE = torch.float16
# =============================


class LayerEditor:
    def __init__(self, lora_path=None, lora_weight_name=None, lora_scale=1.0, lora_adapter_name="edit_lora"):
        print("=" * 60)
        print("Qwen-Image 智能图层编辑器 - 双GPU版 + LoRA")
        print("=" * 60)

        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if self.num_gpus >= 2:
            print(f"✅ 检测到 {self.num_gpus} 个GPU")
            for i in range(self.num_gpus):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                torch.cuda.empty_cache()
        elif self.num_gpus == 1:
            print("⚠️ 仅检测到1个GPU")
        else:
            print("❌ 未检测到GPU")

        self.layered_pipe = None
        self.edit_pipe = None
        self.original_size = None

        # LoRA 配置：只给 Qwen-Image-Edit 模型加载，不影响分层模型
        self.lora_path = lora_path
        self.lora_weight_name = lora_weight_name
        self.lora_scale = float(lora_scale)
        self.lora_adapter_name = lora_adapter_name
        self._edit_lora_loaded = False

        if self.lora_path:
            print(f"🔧 已启用 Edit LoRA: {self.lora_path}")
            print(f"   LoRA scale: {self.lora_scale}")

        print("✅ 初始化完成\n")

    def _load_layered_model(self):
        if self.layered_pipe is None:
            print("📦 加载分层模型...")
            if self.num_gpus >= 2:
                self.layered_pipe = QwenImageLayeredPipeline.from_pretrained(
                    LAYERED_MODEL_PATH,
                    torch_dtype=DTYPE,
                    local_files_only=True,
                    device_map="balanced",
                )
            else:
                device = "cuda" if self.num_gpus == 1 else "cpu"
                self.layered_pipe = QwenImageLayeredPipeline.from_pretrained(
                    LAYERED_MODEL_PATH,
                    torch_dtype=DTYPE if device == "cuda" else torch.float32,
                    local_files_only=True,
                ).to(device)

            if hasattr(self.layered_pipe, "vae"):
                try:
                    self.layered_pipe.vae.enable_slicing()
                    self.layered_pipe.vae.enable_tiling()
                except Exception:
                    pass
            print("   ✓ 分层模型加载完成\n")

    def _load_edit_model(self):
        if self.edit_pipe is None:
            print("📦 加载编辑模型...")
            if self.num_gpus >= 2:
                self.edit_pipe = QwenImageEditPipeline.from_pretrained(
                    EDIT_MODEL_PATH,
                    torch_dtype=DTYPE,
                    local_files_only=True,
                    device_map="balanced",
                )
            else:
                device = "cuda" if self.num_gpus == 1 else "cpu"
                self.edit_pipe = QwenImageEditPipeline.from_pretrained(
                    EDIT_MODEL_PATH,
                    torch_dtype=DTYPE if device == "cuda" else torch.float32,
                    local_files_only=True,
                ).to(device)

            if hasattr(self.edit_pipe, "vae"):
                try:
                    self.edit_pipe.vae.enable_slicing()
                    self.edit_pipe.vae.enable_tiling()
                except Exception:
                    pass

            # 关键：编辑模型加载完成后，再加载 LoRA
            self._load_edit_lora()
            print("   ✓ 编辑模型加载完成\n")

    def _resolve_lora_source(self):
        """兼容两种 LoRA 写法：目录，或单个 .safetensors 文件。"""
        lora_path = Path(self.lora_path)

        if self.lora_weight_name:
            return str(lora_path), self.lora_weight_name

        if lora_path.is_file():
            return str(lora_path.parent), lora_path.name

        return str(lora_path), None

    def _load_edit_lora(self):
        """给 QwenImageEditPipeline 加载 LoRA。"""
        if not self.lora_path or self._edit_lora_loaded:
            return

        if not hasattr(self.edit_pipe, "load_lora_weights"):
            raise RuntimeError(
                "当前 diffusers / QwenImageEditPipeline 不支持 load_lora_weights，"
                "请升级 diffusers，或确认该 Pipeline 是否带有 QwenImageLoraLoaderMixin。"
            )

        source, weight_name = self._resolve_lora_source()
        print("🔧 正在加载 Edit LoRA...")
        print(f"   source: {source}")
        if weight_name:
            print(f"   weight_name: {weight_name}")

        kwargs = {"adapter_name": self.lora_adapter_name}
        if weight_name:
            kwargs["weight_name"] = weight_name

        # 有些旧版 diffusers 不支持 adapter_name 参数，所以做一次降级兼容
        loaded_with_named_adapter = True
        try:
            self.edit_pipe.load_lora_weights(source, **kwargs)
        except TypeError:
            loaded_with_named_adapter = False
            kwargs.pop("adapter_name", None)
            self.edit_pipe.load_lora_weights(source, **kwargs)

        # 调整 LoRA 强度。不同 diffusers 版本的 set_adapters 签名可能略有不同。
        if hasattr(self.edit_pipe, "set_adapters") and loaded_with_named_adapter:
            try:
                self.edit_pipe.set_adapters([self.lora_adapter_name], adapter_weights=[self.lora_scale])
            except TypeError:
                try:
                    self.edit_pipe.set_adapters(self.lora_adapter_name, self.lora_scale)
                except Exception as e:
                    print(f"   ⚠️ set_adapters 设置 scale 失败，将在推理时尝试 cross_attention_kwargs: {e}")

        self._edit_lora_loaded = True
        print("   ✓ Edit LoRA 加载完成")

    def _unload_layered_model(self):
        if self.layered_pipe is not None:
            print("   🧹 卸载分层模型...")
            del self.layered_pipe
            self.layered_pipe = None
            torch.cuda.empty_cache()
            gc.collect()
            print("   ✓ 分层模型已卸载")

    def _unload_edit_model(self):
        if self.edit_pipe is not None:
            print("   🧹 卸载编辑模型...")
            try:
                if hasattr(self.edit_pipe, "unload_lora_weights"):
                    self.edit_pipe.unload_lora_weights()
            except Exception:
                pass
            del self.edit_pipe
            self.edit_pipe = None
            self._edit_lora_loaded = False
            torch.cuda.empty_cache()
            gc.collect()
            print("   ✓ 编辑模型已卸载")

    def decompose(self, image_path, num_layers=4, resolution=640):
        print(f"🔪 正在分解图片: {image_path}")

        self._load_layered_model()

        original = Image.open(image_path).convert("RGBA")
        self.original_size = original.size
        print(f"   📐 图片尺寸: {self.original_size[0]}x{self.original_size[1]}")

        print(f"   ⚙️ 正在分层 (图层数: {num_layers}, 分辨率: {resolution})...")

        result = self.layered_pipe(
            image=original,
            layers=num_layers,
            num_inference_steps=40,
            resolution=resolution,
            true_cfg_scale=4.0,
            negative_prompt=" ",
        )

        raw = result.images
        print(f"   🔍 原始输出类型: {type(raw)}, 长度: {len(raw) if hasattr(raw, '__len__') else 'N/A'}")

        layers = raw
        while isinstance(layers, list) and len(layers) == 1 and isinstance(layers[0], list):
            layers = layers[0]
            print(f"   🔍 解包一层，新长度: {len(layers)}")

        if isinstance(layers, list):
            print(f"   ✓ 最终获取 {len(layers)} 个图层")
        else:
            raise ValueError(f"无法提取图层，最终类型: {type(layers)}")

        debug_dir = Path("./debug_layers")
        debug_dir.mkdir(exist_ok=True)
        valid_count = 0
        for i, layer in enumerate(layers):
            if hasattr(layer, "save"):
                layer.save(debug_dir / f"layer_{i}.png")
                print(f"   图层 {i}: {layer.size}, mode={layer.mode}")
                valid_count += 1
            else:
                print(f"   ⚠️ 图层 {i} 类型异常: {type(layer)}")

        print(f"   📁 已保存 {valid_count}/{len(layers)} 个图层到 {debug_dir}/")

        self._unload_layered_model()
        return layers, original

    def edit_layer(self, layer, prompt, strength=0.8):
        print(f"✏️ 正在修改图层: {prompt}")

        self._load_edit_model()

        call_kwargs = dict(
            image=layer,
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            strength=strength,
        )

        # 部分 diffusers pipeline 支持 cross_attention_kwargs 调 LoRA scale；
        # 如果当前 pipeline 不支持，会自动回退到普通调用。
        if self.lora_path:
            call_kwargs["cross_attention_kwargs"] = {"scale": self.lora_scale}

        try:
            result = self.edit_pipe(**call_kwargs)
        except TypeError as e:
            if "cross_attention_kwargs" in call_kwargs:
                print(f"   ⚠️ 当前 pipeline 不接受 cross_attention_kwargs，改用默认 LoRA scale: {e}")
                call_kwargs.pop("cross_attention_kwargs", None)
                result = self.edit_pipe(**call_kwargs)
            else:
                raise

        edited = result.images[0] if isinstance(result.images, list) else result.images

        if not hasattr(edited, "save"):
            raise TypeError(f"编辑结果类型错误: {type(edited)}")

        if edited.size != layer.size:
            edited = edited.resize(layer.size, Image.Resampling.LANCZOS)

        print("   ✓ 修改完成")
        self._unload_edit_model()
        return edited

    def composite(self, layers):
        print("🔨 正在合成图层...")

        final = Image.new("RGBA", self.original_size, (0, 0, 0, 0))

        for layer in layers:
            if hasattr(layer, "size"):
                if layer.size != self.original_size:
                    layer = layer.resize(self.original_size, Image.Resampling.LANCZOS)
                final = Image.alpha_composite(final, layer)

        white_bg = Image.new("RGB", self.original_size, (255, 255, 255))
        white_bg.paste(final, mask=final.split()[3])

        print("   ✓ 合成完成")
        return white_bg

    def _create_compare(self, original, edited):
        w, h = original.size
        compare = Image.new("RGB", (w * 2, h), (255, 255, 255))
        compare.paste(original, (0, 0))
        compare.paste(edited, (w, 0))
        draw = ImageDraw.Draw(compare)
        draw.line([(w, 0), (w, h)], fill="red", width=3)
        return compare

    def run(self, image_path, prompt, target_layer=None, num_layers=4, strength=0.8, resolution=640):
        print("\n" + "=" * 60)
        print("开始处理")
        print("=" * 60)

        layers, original = self.decompose(image_path, num_layers, resolution)

        if target_layer is None:
            print(f"\n📋 可用图层: 0-{len(layers) - 1}")
            for i in range(min(len(layers), 8)):
                print(f"   图层 {i}: debug_layers/layer_{i}.png")

            while True:
                try:
                    target_layer = int(input(f"\n👉 请输入要修改的图层编号 (0-{len(layers) - 1}): "))
                    if 0 <= target_layer < len(layers):
                        break
                except ValueError:
                    pass
        else:
            print(f"\n🎯 使用指定图层: {target_layer}")

        print()
        edited = self.edit_layer(layers[target_layer], prompt, strength)
        layers[target_layer] = edited

        print()
        result = self.composite(layers)

        output_dir = Path("./output")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_path = output_dir / f"result_{timestamp}.jpg"
        result.save(output_path, quality=95)

        compare = self._create_compare(original.convert("RGB"), result)
        compare_path = output_dir / f"compare_{timestamp}.jpg"
        compare.save(compare_path)

        edited_layer_path = output_dir / f"edited_layer_{target_layer}_{timestamp}.png"
        edited.save(edited_layer_path)

        print("\n" + "=" * 60)
        print("🎉 处理完成！")
        print("=" * 60)
        print(f"📁 编辑结果: {output_path}")
        print(f"📁 对比图片: {compare_path}")
        print(f"📁 修改后的图层: {edited_layer_path}")

        return str(output_path)

    def cleanup(self):
        print("\n🧹 正在清理资源...")
        self._unload_layered_model()
        self._unload_edit_model()
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ 清理完成")


def main():
    parser = argparse.ArgumentParser(description="智能图层图片编辑器 + Qwen-Image-Edit LoRA")
    parser.add_argument("image", help="输入图片路径")
    parser.add_argument("--prompt", "-p", required=True, help="修改指令")
    parser.add_argument("--layer", "-l", type=int, help="目标图层编号")
    parser.add_argument("--layers", "-n", type=int, default=4, help="分解图层数量")
    parser.add_argument("--strength", "-s", type=float, default=0.8, help="修改强度")
    parser.add_argument("--resolution", "-r", type=int, default=640, help="处理分辨率")

    # 新增：LoRA 参数
    parser.add_argument("--lora", type=str, default=None, help="Qwen-Image-Edit LoRA 路径：可以是目录，也可以是 .safetensors 文件")
    parser.add_argument("--lora-weight-name", type=str, default=None, help="LoRA 文件名；当 --lora 是目录时使用")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="LoRA 强度，常用 0.5-1.2，默认 1.0")
    parser.add_argument("--lora-adapter-name", type=str, default="edit_lora", help="LoRA adapter 名称，默认 edit_lora")

    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"❌ 文件不存在: {args.image}")
        return

    if args.lora:
        lora_path = Path(args.lora)
        if not lora_path.exists():
            print(f"❌ LoRA 路径不存在: {args.lora}")
            return

    editor = None
    try:
        editor = LayerEditor(
            lora_path=args.lora,
            lora_weight_name=args.lora_weight_name,
            lora_scale=args.lora_scale,
            lora_adapter_name=args.lora_adapter_name,
        )
        editor.run(
            image_path=args.image,
            prompt=args.prompt,
            target_layer=args.layer,
            num_layers=args.layers,
            strength=args.strength,
            resolution=args.resolution,
        )
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if editor:
            editor.cleanup()


if __name__ == "__main__":
    main()
