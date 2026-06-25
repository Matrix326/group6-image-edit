import os
import json
import math
import argparse
import shutil

import torch
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from torch import autocast
import k_diffusion as K

from skimage.metrics import structural_similarity as ssim

# 从本地 edit_cli.py 引入
from edit_cli import load_model_from_config, CFGDenoiser


def parse_float_list(s):
    return [float(x) for x in s.split(",") if x.strip()]


def parse_int_list(s):
    return [int(x) for x in s.split(",") if x.strip()]


def rewrite_prompt(prompt):
    """
    针对 MagicBrush 短指令做增强，让 P2P 更敢改。
    """
    raw = prompt.strip()
    p = raw.lower()

    if any(k in p for k in ["put", "add", "place", "insert"]):
        return raw + ", clearly add the requested object, make it visible, realistic, and complete"

    if any(k in p for k in ["replace", "change", "turn", "make", "switch"]):
        return raw + ", make the transformation obvious, realistic, and complete"

    if any(k in p for k in ["background", "sky", "mountain", "river", "water", "sea"]):
        return raw + ", edit the background clearly and realistically"

    return raw + ", make the requested edit clearly visible, realistic, and complete"


def preprocess_image(input_image):
    """
    和 InstructPix2Pix edit_cli.py 类似：
    把图像 resize 到 Stable Diffusion 适合的 64 倍数尺寸。
    """
    width, height = input_image.size
    factor = 512 / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)

    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64

    input_image = ImageOps.fit(
        input_image,
        (width, height),
        method=Image.Resampling.LANCZOS,
    )

    return input_image


def run_ip2p_once(
    model,
    model_wrap,
    model_wrap_cfg,
    null_token,
    input_image,
    prompt,
    device,
    steps=50,
    cfg_text=7.5,
    cfg_image=1.5,
    seed=42,
):
    """
    单次 InstructPix2Pix 推理。
    """

    with torch.no_grad(), autocast("cuda"), model.ema_scope():
        cond = {}
        cond["c_crossattn"] = [model.get_learned_conditioning([prompt])]

        img_tensor = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        img_tensor = rearrange(img_tensor, "h w c -> 1 c h w").to(device)
        cond["c_concat"] = [model.encode_first_stage(img_tensor).mode()]

        uncond = {}
        uncond["c_crossattn"] = [null_token]
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

        sigmas = model_wrap.get_sigmas(steps)

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": cfg_text,
            "image_cfg_scale": cfg_image,
        }

        generator = torch.Generator(device=device).manual_seed(seed)
        z = torch.randn(
            cond["c_concat"][0].shape,
            device=device,
            generator=generator,
        ) * sigmas[0]

        z = K.sampling.sample_euler_ancestral(
            model_wrap_cfg,
            z,
            sigmas,
            extra_args=extra_args,
            disable=True,
        )

        x = model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")

        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())

    return edited_image


def load_rgb_np(path, size):
    img = Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    return np.array(img)


def load_mask_np(path, size):
    mask = Image.open(path).convert("L").resize(size, Image.Resampling.NEAREST)
    mask = np.array(mask)
    mask = (mask > 127).astype(np.uint8)
    return mask


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return x1, y1, x2 + 1, y2 + 1


def safe_ssim(img_a, img_b):
    """
    img_a, img_b: H W C, uint8
    """
    h, w = img_a.shape[:2]
    if h < 7 or w < 7:
        return 0.0
    return ssim(img_a, img_b, channel_axis=2, data_range=255)


def mask_region_ssim(candidate, target, mask):
    """
    只在 mask 的 bounding box 区域比较 candidate 和 target。
    这样比直接乘 mask 更稳定。
    """
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return 0.0

    x1, y1, x2, y2 = bbox

    cand_crop = candidate[y1:y2, x1:x2]
    targ_crop = target[y1:y2, x1:x2]

    return safe_ssim(cand_crop, targ_crop)


def full_image_ssim(candidate, target):
    return safe_ssim(candidate, target)


def select_best_by_target(candidate_dir, target_path, mask_path, image_size):
    """
    Oracle rerank:
    用 target + mask 从多个 P2P 候选里面选出最好的一张。

    注意：这个用了 target，是 dev 集分析/上限，不是真实测试集方法。
    """

    target = load_rgb_np(target_path, image_size)
    mask = load_mask_np(mask_path, image_size)

    best_score = -1e9
    best_path = None
    best_detail = None

    for name in os.listdir(candidate_dir):
        if not name.endswith(".png"):
            continue

        cand_path = os.path.join(candidate_dir, name)
        candidate = load_rgb_np(cand_path, image_size)

        full_score = full_image_ssim(candidate, target)
        mask_score = mask_region_ssim(candidate, target, mask)

        # 权重：更重视编辑区域是否接近 target
        score = 0.4 * full_score + 0.6 * mask_score

        if score > best_score:
            best_score = score
            best_path = cand_path
            best_detail = {
                "candidate": name,
                "score": float(score),
                "full_ssim": float(full_score),
                "mask_region_ssim": float(mask_score),
            }

    return best_path, best_score, best_detail


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=str,
        default="/share/home/group6/Project/qwen-image-edit-baseline/benchmarks/magicbrush/prepared_full/dev",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/magicbrush_p2p_oracle_dev",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/generate.yaml",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./checkpoints/instruct-pix2pix-00-22000.ckpt",
    )

    parser.add_argument("--steps", type=int, default=50)

    # 这里是重点：多组参数
    parser.add_argument(
        "--cfg-texts",
        type=str,
        default="7.5,9.0,10.5,12.0",
        help="逗号分隔，例如 7.5,9.0,10.5,12.0",
    )
    parser.add_argument(
        "--cfg-images",
        type=str,
        default="1.0,1.2,1.5",
        help="逗号分隔，例如 1.0,1.2,1.5",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3",
        help="逗号分隔，例如 0,1,2,3",
    )

    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--use-rewrite", action="store_true")

    args = parser.parse_args()

    cfg_text_list = parse_float_list(args.cfg_texts)
    cfg_image_list = parse_float_list(args.cfg_images)
    seed_list = parse_int_list(args.seeds)

    data_dir = args.data_dir
    output_dir = args.output_dir

    candidate_root = os.path.join(output_dir, "candidates")
    best_dir = os.path.join(output_dir, "p2p_oracle_best")
    log_path = os.path.join(output_dir, "oracle_scores.jsonl")

    os.makedirs(candidate_root, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    if args.limit > 0:
        data_list = data_list[: args.limit]

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        print("Using CUDA device:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU. This will be very slow.")

    print("正在加载 InstructPix2Pix 模型...")
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt)
    model.eval().to(device)

    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    print("开始 P2P 多候选生成 + Oracle 选择")
    print("cfg_text_list:", cfg_text_list)
    print("cfg_image_list:", cfg_image_list)
    print("seed_list:", seed_list)
    print("steps:", args.steps)
    print("use_rewrite:", args.use_rewrite)
    print("output_dir:", output_dir)

    for item in tqdm(data_list):
        img_id = item["id"]
        raw_prompt = item["instruction"]

        if args.use_rewrite:
            prompt = rewrite_prompt(raw_prompt)
        else:
            prompt = raw_prompt

        input_path = os.path.join(data_dir, item["input_image"])
        target_path = os.path.join(data_dir, item["target_image"])
        mask_path = os.path.join(data_dir, item["mask_image"])

        candidate_dir = os.path.join(candidate_root, img_id)
        os.makedirs(candidate_dir, exist_ok=True)

        best_output_path = os.path.join(best_dir, f"{img_id}.png")

        try:
            raw_input_image = Image.open(input_path).convert("RGB")
            input_image = preprocess_image(raw_input_image.copy())
            image_size = input_image.size

            # 1. 多参数、多 seed 生成候选
            for cfg_text in cfg_text_list:
                for cfg_image in cfg_image_list:
                    for seed in seed_list:
                        cand_name = f"{img_id}_t{cfg_text}_i{cfg_image}_s{seed}.png"
                        cand_path = os.path.join(candidate_dir, cand_name)

                        if os.path.exists(cand_path):
                            continue

                        edited_image = run_ip2p_once(
                            model=model,
                            model_wrap=model_wrap,
                            model_wrap_cfg=model_wrap_cfg,
                            null_token=null_token,
                            input_image=input_image,
                            prompt=prompt,
                            device=device,
                            steps=args.steps,
                            cfg_text=cfg_text,
                            cfg_image=cfg_image,
                            seed=seed,
                        )

                        edited_image.save(cand_path)

            # 2. 用 target + mask 选择最好候选
            best_path, best_score, best_detail = select_best_by_target(
                candidate_dir=candidate_dir,
                target_path=target_path,
                mask_path=mask_path,
                image_size=image_size,
            )

            if best_path is None:
                print(f"[警告] {img_id} 没有候选图")
                continue

            shutil.copy(best_path, best_output_path)

            log_item = {
                "id": img_id,
                "img_id": item.get("img_id"),
                "turn_index": item.get("turn_index"),
                "raw_instruction": raw_prompt,
                "used_prompt": prompt,
                "input_image": item["input_image"],
                "target_image": item["target_image"],
                "mask_image": item["mask_image"],
                "best_output": best_output_path,
                "best_candidate_path": best_path,
                "best_score": float(best_score),
                "best_detail": best_detail,
                "cfg_texts": cfg_text_list,
                "cfg_images": cfg_image_list,
                "seeds": seed_list,
                "steps": args.steps,
                "use_rewrite": args.use_rewrite,
            }

            with open(log_path, "a", encoding="utf-8") as fw:
                fw.write(json.dumps(log_item, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"\n[错误] 样本 {img_id} 失败: {e}")
            continue

    print("\n完成！")
    print("候选图目录:", candidate_root)
    print("Oracle best 目录:", best_dir)
    print("分数日志:", log_path)


if __name__ == "__main__":
    main()