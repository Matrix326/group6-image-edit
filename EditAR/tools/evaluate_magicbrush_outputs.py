import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from PIL import Image


def decode_image(image, mode="RGB"):
    if isinstance(image, Image.Image):
        return image.convert(mode)
    if isinstance(image, dict) and image.get("bytes") is not None:
        import io

        return Image.open(io.BytesIO(image["bytes"])).convert(mode)
    if isinstance(image, (bytes, bytearray)):
        import io

        return Image.open(io.BytesIO(image)).convert(mode)
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert(mode)
    if isinstance(image, str):
        return Image.open(image).convert(mode)
    raise TypeError(f"Unsupported image type: {type(image)}")


def center_crop_resize(img, target_size=512, resample=Image.BILINEAR):
    s = min(img.size)
    scale = target_size / s
    new_size = (round(scale * img.size[0]), round(scale * img.size[1]))
    img = img.resize(new_size, resample)
    x0 = (img.width - target_size) // 2
    y0 = (img.height - target_size) // 2
    return img.crop((x0, y0, x0 + target_size, y0 + target_size))


def load_mask(mask_img, target_size=512):
    mask = center_crop_resize(decode_image(mask_img, "L"), target_size, Image.NEAREST)
    mask = np.asarray(mask)
    return (mask > 127).astype(np.float32)


def mse(a, b, mask=None):
    diff = (a - b) ** 2
    if mask is None:
        return float(diff.mean())
    denom = float(mask.sum())
    if denom <= 0:
        return math.nan
    return float((diff * mask[:, :, None]).sum() / (denom * diff.shape[2]))


def psnr_from_mse(value):
    if value <= 0:
        return math.inf
    return float(-10.0 * math.log10(value))


def rgb_to_luma(img):
    return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


def ssim_from_luma(x, y):
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    c1 = 0.01**2
    c2 = 0.03**2
    mux = float(x.mean())
    muy = float(y.mean())
    vx = float(x.var())
    vy = float(y.var())
    cov = float(((x - mux) * (y - muy)).mean())
    return float(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux**2 + muy**2 + c1) * (vx + vy + c2)))


def ssim(a, b):
    return ssim_from_luma(rgb_to_luma(a), rgb_to_luma(b))


def bg_ssim(pred, source, mask):
    if mask is None:
        return math.nan
    bg = 1.0 - mask
    valid = bg > 0.5
    if not valid.any():
        return math.nan
    return ssim_from_luma(rgb_to_luma(pred)[valid], rgb_to_luma(source)[valid])


def edit_change(pred, source, mask):
    if mask is None:
        return math.nan
    return float((np.abs(rgb_to_luma(pred) - rgb_to_luma(source)) * mask).mean())


class ClipImageSimilarity:
    def __init__(self, model_name, device):
        import torch
        from transformers import AutoProcessor, CLIPModel

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _features(self, pil_images):
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features

    def __call__(self, pred_image, target_image):
        features = self._features([pred_image, target_image])
        return float((features[0] * features[1]).sum().detach().cpu().item())


def sample_path(samples_dir, index, cfg_scale):
    return samples_dir / f"magicbrush_{index:05d}_sample_txt_{cfg_scale}.png"


def mean(values):
    clean = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return float(np.mean(clean)) if clean else math.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--magicbrush-path", required=True)
    parser.add_argument("--samples-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cfg-scale", default="1.0")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-clip", action="store_true")
    args = parser.parse_args()

    dataset = load_from_disk(args.magicbrush_path)
    if hasattr(dataset, "keys"):
        dataset = dataset["dev"] if "dev" in dataset else next(iter(dataset.values()))

    samples_dir = Path(args.samples_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_similarity = None if args.no_clip else ClipImageSimilarity(args.clip_model, args.device)

    rows = []
    limit = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
    for index in range(limit):
        path = sample_path(samples_dir, index, args.cfg_scale)
        if not path.exists():
            path_alt = sample_path(samples_dir, index, str(float(args.cfg_scale)))
            path = path_alt if path_alt.exists() else path
        if not path.exists():
            continue

        data = dataset[index]
        pred_pil = center_crop_resize(Image.open(path).convert("RGB"), args.image_size, Image.BILINEAR)
        source_pil = center_crop_resize(decode_image(data["source_img"]), args.image_size, Image.BILINEAR)
        target_pil = center_crop_resize(decode_image(data["target_img"]), args.image_size, Image.BILINEAR)
        pred = np.asarray(pred_pil).astype(np.float32) / 255.0
        source = np.asarray(source_pil).astype(np.float32) / 255.0
        target = np.asarray(target_pil).astype(np.float32) / 255.0
        mask = load_mask(data["mask_img"], args.image_size) if data.get("mask_img") is not None else None

        row = {
            "index": index,
            "sample": str(path),
            "ssim": ssim(pred, target),
            "bg_ssim": bg_ssim(pred, source, mask),
            "edit_chg": edit_change(pred, source, mask),
            "ssim_in": ssim(pred, source),
            "psnr": psnr_from_mse(mse(pred, target)),
            "clip": clip_similarity(pred_pil, target_pil) if clip_similarity is not None else math.nan,
        }
        rows.append(row)

    fieldnames = [
        "index",
        "sample",
        "ssim",
        "bg_ssim",
        "edit_chg",
        "ssim_in",
        "psnr",
        "clip",
    ]
    with open(output_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {"num_samples": len(rows)}
    for key in fieldnames[2:]:
        summary[key] = mean([row[key] for row in rows])
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
