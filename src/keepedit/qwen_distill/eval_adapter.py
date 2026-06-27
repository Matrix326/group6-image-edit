from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from keepedit.image_ops import save_image as save_rgb
from keepedit.io import dump_json as save_json
from keepedit.qwen_distill.dataset import image_tensor
from keepedit.qwen_distill.metrics import compute_metrics, load_json, save_text
from keepedit.qwen_distill.model import StepDistillAdapter


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


def font(size: int):
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def fit(image: Image.Image, w: int, h: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(w / image.width, h / image.height)
    resized = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w, h), (245, 245, 245))
    canvas.paste(resized, ((w - resized.width) // 2, (h - resized.height) // 2))
    return canvas


def avg(rows: list[dict], key: str) -> float:
    vals = [float(row["metrics"].get(key, 0.0)) for row in rows if key in row["metrics"]]
    return sum(vals) / max(1, len(vals))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_results_json", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--variant_name", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--grid_rows", type=int, default=20)
    args = parser.parse_args()

    rows = load_json(args.student_results_json)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StepDistillAdapter(hidden=args.hidden).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for row in rows:
        sample_dir = args.output_dir / row["sample_id"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        input_path = Path(row["input_image"])
        target_path = Path(row["target_image"])
        mask_path = Path(row["mask_image"]) if row.get("mask_image") else None
        student_path = Path(row["output_image"])
        input_tensor = image_tensor(input_path, args.image_size).unsqueeze(0).to(device)
        student_tensor = image_tensor(student_path, args.image_size).unsqueeze(0).to(device)
        with torch.inference_mode():
            output, _ = model(input_tensor, student_tensor)
        adapted = tensor_to_image(output[0])
        adapted_path = sample_dir / f"{args.variant_name}.png"
        save_rgb(adapted_path, adapted)
        results.append(
            {
                "sample_id": row["sample_id"],
                "prompt": row["prompt"],
                "variant": args.variant_name,
                "input_image": str(input_path),
                "target_image": str(target_path),
                "mask_image": str(mask_path) if mask_path else None,
                "student_image": str(student_path),
                "output_image": str(adapted_path),
                "metrics": compute_metrics(input_path, adapted_path, target_path, mask_path),
            }
        )

    save_json(args.output_dir / f"{args.variant_name}_eval.json", results)
    lines = [
        f"# {args.variant_name} Summary",
        "",
        "| Method | Samples | SSIM(Target) | PSNR(Target) | BG-SSIM | BG-PSNR | Edit Change |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| {name} | {n} | {ssim:.4f} | {psnr:.4f} | {bg:.4f} | {bgpsnr:.4f} | {chg:.4f} |".format(
            name=args.variant_name,
            n=len(results),
            ssim=avg(results, "ssim_target_output"),
            psnr=avg(results, "psnr_target_output"),
            bg=avg(results, "bg_ssim"),
            bgpsnr=avg(results, "bg_psnr"),
            chg=avg(results, "edit_region_change"),
        ),
    ]
    save_text(args.output_dir / f"{args.variant_name}_summary.md", "\n".join(lines) + "\n")

    cols = [("input_image", "Input"), ("target_image", "Target"), ("student_image", "Student"), ("output_image", "Adapter")]
    show_rows = results[: args.grid_rows]
    cell_w, cell_h = 220, 220
    margin, gap, header_h, prompt_h = 24, 16, 32, 58
    width = margin * 2 + len(cols) * cell_w + (len(cols) - 1) * gap
    row_h = header_h + cell_h + prompt_h
    height = margin * 2 + len(show_rows) * row_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(show_rows):
        y0 = margin + i * row_h
        for j, (key, label) in enumerate(cols):
            x0 = margin + j * (cell_w + gap)
            draw.text((x0 + cell_w / 2, y0), label, fill=(20, 20, 20), font=font(17), anchor="ma")
            canvas.paste(fit(Image.open(row[key]), cell_w, cell_h), (x0, y0 + header_h))
        draw.text((width / 2, y0 + header_h + cell_h + 16), row["prompt"][:120], fill=(60, 60, 60), font=font(15), anchor="ma")
    canvas.save(args.output_dir / f"{args.variant_name}_grid.png")


if __name__ == "__main__":
    main()
