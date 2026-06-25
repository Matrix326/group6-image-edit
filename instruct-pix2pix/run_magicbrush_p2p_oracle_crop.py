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


# =========================
# 工具函数
# =========================

def parse_float_list(s):
    return [float(x) for x in s.split(",") if x.strip()]


def parse_int_list(s):
    return [int(x) for x in s.split(",") if x.strip()]


def rewrite_prompt(prompt):
    """
    针对 MagicBrush 的短指令做 prompt 增强。
    目标：让 InstructPix2Pix 更敢改、更明显地执行指令。
    """
    raw = prompt.strip()
    p = raw.lower()

    if any(k in p for k in ["put", "add", "place", "insert"]):
        return (
            raw
            + ". Clearly add the requested object into the image. "
            + "The new object must be visible, realistic, and located as instructed."
        )

    if any(k in p for k in ["replace", "switch"]):
        return (
            raw
            + ". Replace the target object completely with the requested new object. "
            + "Make the replacement obvious and realistic."
        )

    if any(k in p for k in ["make", "turn", "change"]):
        return (
            raw
            + ". Make the requested visual change clearly visible and complete. "
            + "Do not leave the original attribute unchanged."
        )

    if any(k in p for k in ["background", "mountain", "mountains", "river", "sky", "water", "sea"]):
        return (
            raw
            + ". Modify the background clearly and realistically. "
            + "The requested background element should be obvious."
        )

    return (
        raw
        + ". Make the requested edit clearly visible, realistic, and complete."
    )


def preprocess_image(input_image):
    """
    复刻 InstructPix2Pix edit_cli.py 的尺寸规整逻辑：
    resize 到适合 Stable Diffusion 的 64 倍数尺寸。
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


def get_expanded_bbox_from_mask(mask_image, expand_ratio=0.35, min_size=128):
    """
    根据 MagicBrush mask 得到扩张后的编辑区域 bbox。

    mask_image: PIL L
    expand_ratio: 在原 mask bbox 基础上扩大比例
    min_size: crop 最小尺寸，避免区域太小导致 P2P 看不清

    返回:
        (x1, y1, x2, y2) 或 None
    """
    mask = np.array(mask_image.convert("L"))
    ys, xs = np.where(mask > 127)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    w = x2 - x1 + 1
    h = y2 - y1 + 1

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    new_w = max(int(w * (1.0 + expand_ratio)), min_size)
    new_h = max(int(h * (1.0 + expand_ratio)), min_size)

    H, W = mask.shape

    nx1 = max(0, int(cx - new_w / 2))
    ny1 = max(0, int(cy - new_h / 2))
    nx2 = min(W, int(cx + new_w / 2))
    ny2 = min(H, int(cy + new_h / 2))

    # 避免非法 bbox
    if nx2 <= nx1 or ny2 <= ny1:
        return None

    return nx1, ny1, nx2, ny2


def paste_crop_back(full_image, edited_crop, bbox):
    """
    把编辑后的 crop 贴回整图。

    full_image: PIL RGB，resize 后的整图
    edited_crop: PIL RGB，P2P 对 crop 的输出
    bbox: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox

    full_image = full_image.copy()
    edited_crop = edited_crop.resize((x2 - x1, y2 - y1), Image.Resampling.LANCZOS)

    full_image.paste(edited_crop, (x1, y1))
    return full_image


# =========================
# InstructPix2Pix 推理
# =========================

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


def run_crop_edit_once(
    model,
    model_wrap,
    model_wrap_cfg,
    null_token,
    full_input_image,
    mask_image,
    prompt,
    device,
    steps=50,
    cfg_text=11.0,
    cfg_image=0.9,
    seed=42,
    expand_ratio=0.35,
    min_size=128,
):
    """
    局部 crop 编辑：
    1. 根据 mask 找 bbox
    2. 裁剪 bbox 区域
    3. 对 crop 跑 P2P
    4. 把编辑后的 crop 贴回整图
    """

    bbox = get_expanded_bbox_from_mask(
        mask_image,
        expand_ratio=expand_ratio,
        min_size=min_size,
    )

    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox

    crop = full_input_image.crop((x1, y1, x2, y2))
    crop = preprocess_image(crop)

    edited_crop = run_ip2p_once(
        model=model,
        model_wrap=model_wrap,
        model_wrap_cfg=model_wrap_cfg,
        null_token=null_token,
        input_image=crop,
        prompt=prompt,
        device=device,
        steps=steps,
        cfg_text=cfg_text,
        cfg_image=cfg_image,
        seed=seed,
    )

    edited_full = paste_crop_back(full_input_image, edited_crop, bbox)
    return edited_full


# =========================
# Oracle 选择评分
# =========================

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
    使用 target + mask 从候选中选最好的一张。

    注意：
    - 这是 Oracle 上限实验；
    - 它用了 target，不是真实 test-time 方法；
    - 但可以证明 P2P 多候选搜索是否存在更优结果。
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

        # 更重视编辑区域
        score = 0.2 * full_score + 0.8 * mask_score

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


# =========================
# 主函数
# =========================

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
        default="./results/magicbrush_p2p_oracle_crop_dev",
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

    # 全图候选参数
    parser.add_argument(
        "--cfg-texts",
        type=str,
        default="9.0,11.0,13.0",
        help="全图候选 cfg-text，逗号分隔，例如 9.0,11.0,13.0",
    )
    parser.add_argument(
        "--cfg-images",
        type=str,
        default="0.8,1.0,1.2",
        help="全图候选 cfg-image，逗号分隔，例如 0.8,1.0,1.2",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3",
        help="随机种子，逗号分隔，例如 0,1,2,3",
    )

    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--use-rewrite", action="store_true")

    # crop 候选开关与参数
    parser.add_argument("--use-crop", action="store_true")
    parser.add_argument("--crop-expand", type=float, default=0.35)
    parser.add_argument("--crop-min-size", type=int, default=128)

    # crop 候选参数，默认更激进
    parser.add_argument(
        "--crop-cfg-texts",
        type=str,
        default="11.0,13.0,15.0",
        help="crop 候选 cfg-text",
    )
    parser.add_argument(
        "--crop-cfg-images",
        type=str,
        default="0.7,0.9,1.1",
        help="crop 候选 cfg-image",
    )

    args = parser.parse_args()

    cfg_text_list = parse_float_list(args.cfg_texts)
    cfg_image_list = parse_float_list(args.cfg_images)
    seed_list = parse_int_list(args.seeds)

    crop_cfg_text_list = parse_float_list(args.crop_cfg_texts)
    crop_cfg_image_list = parse_float_list(args.crop_cfg_images)

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

    print("\n========== P2P Oracle Crop Search ==========")
    print("data_dir:", data_dir)
    print("output_dir:", output_dir)
    print("steps:", args.steps)
    print("use_rewrite:", args.use_rewrite)
    print("use_crop:", args.use_crop)
    print("full cfg_text_list:", cfg_text_list)
    print("full cfg_image_list:", cfg_image_list)
    print("crop cfg_text_list:", crop_cfg_text_list)
    print("crop cfg_image_list:", crop_cfg_image_list)
    print("seed_list:", seed_list)
    print("crop_expand:", args.crop_expand)
    print("crop_min_size:", args.crop_min_size)
    print("num_samples:", len(data_list))
    print("===========================================\n")

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
            raw_mask_image = Image.open(mask_path).convert("L")

            input_image = preprocess_image(raw_input_image.copy())
            image_size = input_image.size

            # =========================
            # 1A. 全图候选
            # =========================
            for cfg_text in cfg_text_list:
                for cfg_image in cfg_image_list:
                    for seed in seed_list:
                        cand_name = f"{img_id}_full_t{cfg_text}_i{cfg_image}_s{seed}.png"
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

            # =========================
            # 1B. 局部 crop 候选
            # =========================
            if args.use_crop:
                resized_mask_image = raw_mask_image.resize(
                    input_image.size,
                    Image.Resampling.NEAREST,
                )

                for cfg_text in crop_cfg_text_list:
                    for cfg_image in crop_cfg_image_list:
                        for seed in seed_list:
                            cand_name = f"{img_id}_crop_t{cfg_text}_i{cfg_image}_s{seed}.png"
                            cand_path = os.path.join(candidate_dir, cand_name)

                            if os.path.exists(cand_path):
                                continue

                            edited_full = run_crop_edit_once(
                                model=model,
                                model_wrap=model_wrap,
                                model_wrap_cfg=model_wrap_cfg,
                                null_token=null_token,
                                full_input_image=input_image,
                                mask_image=resized_mask_image,
                                prompt=prompt,
                                device=device,
                                steps=args.steps,
                                cfg_text=cfg_text,
                                cfg_image=cfg_image,
                                seed=seed,
                                expand_ratio=args.crop_expand,
                                min_size=args.crop_min_size,
                            )

                            if edited_full is not None:
                                edited_full.save(cand_path)

            # =========================
            # 2. Oracle 选最优候选
            # =========================
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
                "full_cfg_texts": cfg_text_list,
                "full_cfg_images": cfg_image_list,
                "crop_cfg_texts": crop_cfg_text_list,
                "crop_cfg_images": crop_cfg_image_list,
                "seeds": seed_list,
                "steps": args.steps,
                "use_rewrite": args.use_rewrite,
                "use_crop": args.use_crop,
                "crop_expand": args.crop_expand,
                "crop_min_size": args.crop_min_size,
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