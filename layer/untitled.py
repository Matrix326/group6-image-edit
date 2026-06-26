#!/usr/bin/env python3
"""
可复用 CLIP 图文匹配模块

用途：
- 给定一组图片路径和一段文本，计算每张图片与文本的 CLIP 相似度；
- 返回最匹配图片以及完整排序；
- 可被 try0_lora.py 调用，用于自动推荐 Qwen-Image 分割后的 layer 编号。

命令行示例：
python untitled.py \
  --images ./debug_layers/layer_0.png ./debug_layers/layer_1.png ./debug_layers/layer_2.png \
  --text "red clothes"
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


ImagePath = Union[str, Path]


def _natural_key(path: ImagePath):
    """让 layer_2.png 排在 layer_10.png 前面。"""
    text = str(path)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _load_images(image_paths: Sequence[ImagePath]):
    images: List[Image.Image] = []
    valid_paths: List[str] = []

    for path in image_paths:
        path = str(path)
        if os.path.exists(path):
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_paths.append(path)
                print(f"✓ 已加载: {path}")
            except Exception as e:
                print(f"✗ 图片读取失败: {path}，原因: {e}")
        else:
            print(f"✗ 文件不存在: {path}")

    return images, valid_paths


def recommend_best_image(
    image_paths: Sequence[ImagePath],
    text_description: str,
    clip_model_name: str = "openai/clip-vit-base-patch32",
    device: Optional[str] = None,
    topk: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """
    使用 CLIP 从 image_paths 中推荐与 text_description 最匹配的图片。

    Args:
        image_paths: 图片路径列表。
        text_description: 用于匹配的文本，例如 "red clothes"。
        clip_model_name: transformers CLIP 模型名或本地路径。
        device: cuda / cpu；None 时自动判断。
        topk: 输出排序数量；None 表示全部输出。
        verbose: 是否打印过程和排序。

    Returns:
        dict:
        {
            "best_path": str,
            "best_index": int,          # 在有效图片列表 valid_paths 中的索引
            "best_score": float,
            "best_prob": float,
            "ranking": [
                {"rank": 1, "index": 0, "path": "...", "score": 12.3, "prob": 0.8},
                ...
            ],
            "valid_paths": ["...", ...]
        }
    """
    if not text_description or not text_description.strip():
        raise ValueError("text_description 不能为空")

    image_paths = sorted([str(p) for p in image_paths], key=_natural_key)
    if not image_paths:
        raise ValueError("image_paths 不能为空")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print("正在加载 CLIP 模型...")
        print(f"模型: {clip_model_name}")
        print(f"设备: {device}")

    model = CLIPModel.from_pretrained(clip_model_name).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    model.eval()

    if verbose:
        print("模型加载完成！\n")

    images, valid_paths = _load_images(image_paths)
    if not images:
        raise RuntimeError("没有找到有效的图片文件")

    inputs = processor(
        text=[text_description],
        images=images,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # logits_per_image 通常形状为 [num_images, 1]
        logits = outputs.logits_per_image.squeeze()
        if logits.ndim == 0:
            logits = logits.unsqueeze(0)
        probs = torch.softmax(logits, dim=0)

    ranking = []
    for idx, (path, score, prob) in enumerate(zip(valid_paths, logits, probs)):
        ranking.append(
            {
                "index": idx,
                "path": path,
                "score": float(score.item()),
                "prob": float(prob.item()),
            }
        )

    ranking.sort(key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(ranking, start=1):
        item["rank"] = rank

    if topk is not None:
        shown_ranking = ranking[: max(1, int(topk))]
    else:
        shown_ranking = ranking

    if verbose:
        print("\n" + "=" * 60)
        print(f"目标文本: '{text_description}'")
        print("=" * 60)
        for item in shown_ranking:
            print(f"\n排名 {item['rank']}: {os.path.basename(item['path'])}")
            print(f"  路径: {item['path']}")
            print(f"  相似度分数: {item['score']:.6f}")
            print(f"  匹配概率: {item['prob']:.2%}")

        best = ranking[0]
        print("\n" + "=" * 60)
        print(f"🎯 最匹配的图片: {os.path.basename(best['path'])}")
        print(f"   相似度分数: {best['score']:.6f}")
        print(f"   置信度: {best['prob']:.2%}")
        print("=" * 60)

    best = ranking[0]
    return {
        "best_path": best["path"],
        "best_index": best["index"],
        "best_score": best["score"],
        "best_prob": best["prob"],
        "ranking": ranking,
        "valid_paths": valid_paths,
    }


def main():
    parser = argparse.ArgumentParser(description="CLIP 图文相似度匹配工具")
    parser.add_argument("--images", nargs="+", required=True, help="候选图片路径列表")
    parser.add_argument("--text", required=True, help="用于匹配的文本描述")
    parser.add_argument("--clip-model-name", default="openai/clip-vit-base-patch32", help="CLIP 模型名称或本地路径")
    parser.add_argument("--device", default=None, help="cuda 或 cpu；默认自动判断")
    parser.add_argument("--topk", type=int, default=None, help="只显示前 K 个结果")
    args = parser.parse_args()

    result = recommend_best_image(
        image_paths=args.images,
        text_description=args.text,
        clip_model_name=args.clip_model_name,
        device=args.device,
        topk=args.topk,
        verbose=True,
    )
    return result["best_path"]


if __name__ == "__main__":
    main()
