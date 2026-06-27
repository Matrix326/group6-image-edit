import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tokenizer.tokenizer_image.vq_model import VQ_models


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


def center_crop_resize(img, target_size=512, resample=Image.BICUBIC):
    if img.mode != "RGB":
        img = whiten_transparency(img)
    s = min(img.size)
    scale = target_size / s
    new_size = (round(scale * img.size[0]), round(scale * img.size[1]))
    img = img.resize(new_size, resample)
    x0 = (img.width - target_size) // 2
    y0 = (img.height - target_size) // 2
    return img.crop((x0, y0, x0 + target_size, y0 + target_size))


def whiten_transparency(img):
    if img.mode == "RGB":
        return img
    rgba = np.array(img.convert("RGBA"))
    if not (rgba[:, :, 3] < 255).any():
        return img.convert("RGB")
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = (1.0 - alpha) * 255.0 + alpha * rgba[:, :, :3].astype(np.float32)
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


def pil_to_tensor(img, image_size):
    img = center_crop_resize(img, image_size)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def tensor_to_uint8(x):
    x = torch.clamp(127.5 * x + 128.0, 0, 255)
    return x.permute(1, 2, 0).to("cpu", dtype=torch.uint8).numpy()


def as_float01(arr):
    return arr.astype(np.float32) / 255.0


def mse(a, b):
    return float(((a - b) ** 2).mean())


def mae(a, b):
    return float(np.abs(a - b).mean())


def psnr_from_mse(value):
    if value <= 0:
        return math.inf
    return float(-10.0 * math.log10(value))


def ssim_simple(a, b):
    ya = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    yb = 0.299 * b[:, :, 0] + 0.587 * b[:, :, 1] + 0.114 * b[:, :, 2]
    c1 = 0.01**2
    c2 = 0.03**2
    mux = float(ya.mean())
    muy = float(yb.mean())
    vx = float(ya.var())
    vy = float(yb.var())
    cov = float(((ya - mux) * (yb - muy)).mean())
    return float(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux**2 + muy**2 + c1) * (vx + vy + c2)))


def mean(values):
    clean = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return float(np.mean(clean)) if clean else math.nan


def load_vq_model(args, device):
    model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
    )
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    if "ema" in checkpoint:
        weights = checkpoint["ema"]
    elif "model" in checkpoint:
        weights = checkpoint["model"]
    elif "state_dict" in checkpoint:
        weights = checkpoint["state_dict"]
    else:
        raise ValueError("Unsupported checkpoint format: expected one of ema/model/state_dict")
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    return model


def iter_folder_images(input_dir, max_samples):
    paths = sorted(
        p for p in Path(input_dir).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if max_samples is not None:
        paths = paths[:max_samples]
    for index, path in enumerate(paths):
        yield {
            "index": index,
            "name": path.stem,
            "source": str(path),
            "image": Image.open(path).convert("RGB"),
        }


def iter_hf_images(dataset_path, split, image_field, max_samples):
    from datasets import load_from_disk

    dataset = load_from_disk(dataset_path)
    if hasattr(dataset, "keys"):
        dataset = dataset[split] if split in dataset else next(iter(dataset.values()))
    limit = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    for index in range(limit):
        row = dataset[index]
        yield {
            "index": index,
            "name": f"{image_field}_{index:05d}",
            "source": f"{dataset_path}:{split}:{image_field}:{index}",
            "image": decode_image(row[image_field]),
        }


def save_visualization(original, reconstruction, output_path):
    diff = np.abs(original.astype(np.int16) - reconstruction.astype(np.int16)).astype(np.uint8)
    diff = np.clip(diff * 4, 0, 255).astype(np.uint8)

    label_h = 24
    h, w = original.shape[:2]
    canvas = Image.new("RGB", (w * 3, h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (title, arr) in enumerate(
        [("input", original), ("reconstruction", reconstruction), ("abs diff x4", diff)]
    ):
        x0 = i * w
        draw.text((x0 + 8, 5), title, fill=(0, 0, 0))
        canvas.paste(Image.fromarray(arr), (x0, label_h))
    canvas.save(output_path)


def run(args):
    torch.set_grad_enabled(False)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = load_vq_model(args, device)

    output_dir = Path(args.output_dir)
    recon_dir = output_dir / "reconstructions"
    visual_dir = output_dir / "visualizations"
    token_dir = output_dir / "tokens" if args.save_tokens else None
    recon_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)
    if token_dir:
        token_dir.mkdir(parents=True, exist_ok=True)

    if args.input_dir:
        items = iter_folder_images(args.input_dir, args.max_samples)
    elif args.hf_dataset:
        items = iter_hf_images(args.hf_dataset, args.hf_split, args.hf_image_field, args.max_samples)
    else:
        raise ValueError("Pass either --input-dir or --hf-dataset")

    batch = []
    rows = []
    total = 0

    def flush():
        nonlocal batch, rows, total
        if not batch:
            return
        x = torch.stack([item["tensor"] for item in batch], dim=0).to(device)
        latent, _, [_, _, indices] = model.encode(x)
        samples = model.decode_code(indices, latent.shape)
        if samples.shape[-1] != args.image_size or samples.shape[-2] != args.image_size:
            samples = F.interpolate(samples, size=(args.image_size, args.image_size), mode="bicubic")

        indices_cpu = indices.detach().reshape(len(batch), -1).to("cpu")
        for local_i, item in enumerate(batch):
            original = item["original"]
            reconstruction = tensor_to_uint8(samples[local_i])
            restored = as_float01(reconstruction)
            target = as_float01(original)
            item_mse = mse(restored, target)
            flat_tokens = indices_cpu[local_i].numpy()
            safe_name = f"{item['index']:05d}_{item['name']}"
            recon_path = recon_dir / f"{safe_name}_recon.png"
            visual_path = visual_dir / f"{safe_name}_visual.png"
            Image.fromarray(reconstruction).save(recon_path)
            save_visualization(original, reconstruction, visual_path)
            if token_dir:
                np.save(token_dir / f"{safe_name}_tokens.npy", flat_tokens)

            rows.append(
                {
                    "index": item["index"],
                    "name": item["name"],
                    "source": item["source"],
                    "reconstruction": str(recon_path),
                    "visualization": str(visual_path),
                    "mse": item_mse,
                    "mae": mae(restored, target),
                    "psnr": psnr_from_mse(item_mse),
                    "ssim": ssim_simple(restored, target),
                    "token_count": int(flat_tokens.size),
                    "unique_tokens": int(np.unique(flat_tokens).size),
                    "token_min": int(flat_tokens.min()),
                    "token_max": int(flat_tokens.max()),
                }
            )
            total += 1
            if total % args.log_every == 0:
                print(f"processed {total} images")
        batch = []

    for item in items:
        img = center_crop_resize(item["image"], args.image_size)
        arr = np.asarray(img).astype(np.uint8)
        item["original"] = arr
        item["tensor"] = pil_to_tensor(img, args.image_size)
        batch.append(item)
        if len(batch) >= args.batch_size:
            flush()
    flush()

    fieldnames = [
        "index",
        "name",
        "source",
        "reconstruction",
        "visualization",
        "mse",
        "mae",
        "psnr",
        "ssim",
        "token_count",
        "unique_tokens",
        "token_min",
        "token_max",
    ]
    with open(output_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "num_images": len(rows),
        "image_size": args.image_size,
        "vq_model": args.vq_model,
        "vq_ckpt": args.vq_ckpt,
        "mean_mse": mean([row["mse"] for row in rows]),
        "mean_mae": mean([row["mae"] for row in rows]),
        "mean_psnr": mean([row["psnr"] for row in rows]),
        "mean_ssim": mean([row["ssim"] for row in rows]),
        "mean_unique_tokens": mean([row["unique_tokens"] for row in rows]),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--hf-dataset", default=None)
    parser.add_argument("--hf-split", default="dev")
    parser.add_argument("--hf-image-field", default="source_img")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vq-model", choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", required=True)
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-tokens", action="store_true")
    parser.add_argument("--log-every", type=int, default=16)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
