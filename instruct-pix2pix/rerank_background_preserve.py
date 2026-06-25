import os
import json
import argparse
import shutil

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def safe_ssim(img_a, img_b):
    h, w = img_a.shape[:2]
    if h < 7 or w < 7:
        return 0.0
    return ssim(img_a, img_b, channel_axis=2, data_range=255)


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    return x1, y1, x2 + 1, y2 + 1


def load_rgb(path, size):
    return np.array(
        Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    )


def load_mask(path, size):
    mask = np.array(
        Image.open(path).convert("L").resize(size, Image.Resampling.NEAREST)
    )
    return (mask > 127).astype(np.uint8)


def mask_bbox_ssim(candidate, target, mask):
    """
    只在 mask 的 bbox 区域比较 candidate 和 target。
    表示编辑区域是否接近目标。
    """
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return 0.0

    x1, y1, x2, y2 = bbox

    cand_crop = candidate[y1:y2, x1:x2]
    targ_crop = target[y1:y2, x1:x2]

    return safe_ssim(cand_crop, targ_crop)


def outside_mask_ssim(candidate, input_img, mask):
    """
    计算 mask 外区域 candidate 和 input 的相似度。
    表示背景/非编辑区域是否被保持。
    """
    outside = 1 - mask

    if outside.sum() < 10:
        return 0.0

    # 用 input 填充 mask 内，只让 SSIM 主要反映 mask 外区域差异
    outside3 = outside[..., None]

    cand_keep = candidate * outside3 + input_img * (1 - outside3)
    input_keep = input_img

    return safe_ssim(cand_keep.astype(np.uint8), input_keep.astype(np.uint8))


def outside_mask_psnr(candidate, input_img, mask):
    outside = 1 - mask

    if outside.sum() < 10:
        return 0.0

    outside3 = outside[..., None]

    cand_keep = candidate * outside3 + input_img * (1 - outside3)
    input_keep = input_img

    return psnr(input_keep.astype(np.uint8), cand_keep.astype(np.uint8), data_range=255)


def full_ssim(candidate, target):
    return safe_ssim(candidate, target)


def make_soft_mask(mask_path, size, dilate=9, blur=7):
    """
    生成 soft mask：
    - 先二值化
    - 适当膨胀，避免边缘漏改
    - 高斯模糊，避免拼接边界生硬
    """
    mask = Image.open(mask_path).convert("L").resize(size, Image.Resampling.NEAREST)

    mask_np = np.array(mask)
    mask_np = (mask_np > 127).astype(np.uint8) * 255
    mask = Image.fromarray(mask_np)

    if dilate > 0:
        kernel = dilate if dilate % 2 == 1 else dilate + 1
        mask = mask.filter(ImageFilter.MaxFilter(kernel))

    if blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(blur))

    return mask


def soft_mask_fusion(input_path, candidate_path, mask_path, output_path, size, dilate=9, blur=7):
    input_img = Image.open(input_path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    cand_img = Image.open(candidate_path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    soft_mask = make_soft_mask(mask_path, size, dilate=dilate, blur=blur)

    input_np = np.array(input_img).astype(np.float32)
    cand_np = np.array(cand_img).astype(np.float32)
    mask_np = np.array(soft_mask).astype(np.float32) / 255.0
    mask_np = mask_np[..., None]

    final_np = mask_np * cand_np + (1.0 - mask_np) * input_np
    final_np = np.clip(final_np, 0, 255).astype(np.uint8)

    Image.fromarray(final_np).save(output_path)


def select_best_background_preserve(
    candidate_dir,
    input_path,
    target_path,
    mask_path,
    image_size,
    w_edit=0.50,
    w_preserve=0.45,
    w_full=0.05,
):
    """
    选择标准：
    1. mask 内尽量接近 target
    2. mask 外尽量接近 input
    3. 全图稍微参考 target

    score = w_edit * mask_in_target_ssim
          + w_preserve * mask_out_input_ssim
          + w_full * full_target_ssim
    """

    input_img = load_rgb(input_path, image_size)
    target_img = load_rgb(target_path, image_size)
    mask = load_mask(mask_path, image_size)

    best_score = -1e9
    best_path = None
    best_detail = None

    for name in os.listdir(candidate_dir):
        if not name.endswith(".png"):
            continue

        cand_path = os.path.join(candidate_dir, name)
        candidate = load_rgb(cand_path, image_size)

        edit_score = mask_bbox_ssim(candidate, target_img, mask)
        preserve_score = outside_mask_ssim(candidate, input_img, mask)
        full_score = full_ssim(candidate, target_img)
        preserve_psnr = outside_mask_psnr(candidate, input_img, mask)

        score = (
            w_edit * edit_score
            + w_preserve * preserve_score
            + w_full * full_score
        )

        if score > best_score:
            best_score = score
            best_path = cand_path
            best_detail = {
                "candidate": name,
                "score": float(score),
                "edit_mask_ssim_vs_target": float(edit_score),
                "preserve_outside_ssim_vs_input": float(preserve_score),
                "preserve_outside_psnr_vs_input": float(preserve_psnr),
                "full_ssim_vs_target": float(full_score),
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
        "--candidate-root",
        type=str,
        default="./results/magicbrush_p2p_oracle_crop_dev/candidates",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/magicbrush_p2p_bg_preserve_dev",
    )
    parser.add_argument("--limit", type=int, default=20)

    # 评分权重
    parser.add_argument("--w-edit", type=float, default=0.50)
    parser.add_argument("--w-preserve", type=float, default=0.45)
    parser.add_argument("--w-full", type=float, default=0.05)

    # soft fusion 参数
    parser.add_argument("--dilate", type=int, default=9)
    parser.add_argument("--blur", type=int, default=7)

    args = parser.parse_args()

    data_dir = args.data_dir
    candidate_root = args.candidate_root
    output_dir = args.output_dir

    best_raw_dir = os.path.join(output_dir, "best_raw")
    best_fusion_dir = os.path.join(output_dir, "best_fusion")
    os.makedirs(best_raw_dir, exist_ok=True)
    os.makedirs(best_fusion_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "bg_preserve_scores.jsonl")

    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    if args.limit > 0:
        data_list = data_list[: args.limit]

    print("========== Background-Preserved Rerank ==========")
    print("data_dir:", data_dir)
    print("candidate_root:", candidate_root)
    print("output_dir:", output_dir)
    print("weights:", args.w_edit, args.w_preserve, args.w_full)
    print("fusion dilate:", args.dilate)
    print("fusion blur:", args.blur)
    print("num samples:", len(data_list))
    print("=================================================\n")

    for item in tqdm(data_list):
        img_id = item["id"]

        input_path = os.path.join(data_dir, item["input_image"])
        target_path = os.path.join(data_dir, item["target_image"])
        mask_path = os.path.join(data_dir, item["mask_image"])

        candidate_dir = os.path.join(candidate_root, img_id)

        if not os.path.exists(candidate_dir):
            print(f"[跳过] 没有候选目录: {candidate_dir}")
            continue

        raw_input_img = Image.open(input_path).convert("RGB")

        # 与 P2P 生成候选时的尺寸保持一致
        width, height = raw_input_img.size
        factor = 512 / max(width, height)
        import math
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        image_size = (width, height)

        best_path, best_score, best_detail = select_best_background_preserve(
            candidate_dir=candidate_dir,
            input_path=input_path,
            target_path=target_path,
            mask_path=mask_path,
            image_size=image_size,
            w_edit=args.w_edit,
            w_preserve=args.w_preserve,
            w_full=args.w_full,
        )

        if best_path is None:
            print(f"[跳过] {img_id} 没有可用候选")
            continue

        best_raw_path = os.path.join(best_raw_dir, f"{img_id}.png")
        best_fusion_path = os.path.join(best_fusion_dir, f"{img_id}.png")

        shutil.copy(best_path, best_raw_path)

        soft_mask_fusion(
            input_path=input_path,
            candidate_path=best_path,
            mask_path=mask_path,
            output_path=best_fusion_path,
            size=image_size,
            dilate=args.dilate,
            blur=args.blur,
        )

        log_item = {
            "id": img_id,
            "img_id": item.get("img_id"),
            "turn_index": item.get("turn_index"),
            "instruction": item["instruction"],
            "input_image": item["input_image"],
            "target_image": item["target_image"],
            "mask_image": item["mask_image"],
            "best_candidate": best_path,
            "best_raw_output": best_raw_path,
            "best_fusion_output": best_fusion_path,
            "best_score": float(best_score),
            "detail": best_detail,
            "weights": {
                "w_edit": args.w_edit,
                "w_preserve": args.w_preserve,
                "w_full": args.w_full,
            },
            "fusion": {
                "dilate": args.dilate,
                "blur": args.blur,
            },
        }

        with open(log_path, "a", encoding="utf-8") as fw:
            fw.write(json.dumps(log_item, ensure_ascii=False) + "\n")

    print("\n完成！")
    print("Raw best:", best_raw_dir)
    print("Fusion best:", best_fusion_dir)
    print("Log:", log_path)


if __name__ == "__main__":
    main()