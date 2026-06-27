from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from keepedit.image_ops import load_rgb, save_image as save_rgb
from keepedit.qwen_distill.manifest import load_manifest
from keepedit.qwen_distill.metrics import compute_metrics, load_json, save_json, save_text


def add_qwen_scripts_to_path(project_root: Path) -> None:
    scripts_dir = project_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def parse_variants(raw: str) -> list[tuple[str, int, float, str]]:
    variants = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        # name:steps:true_cfg:lora_or_none
        parts = item.split(":")
        if len(parts) != 4:
            raise ValueError(f"Invalid variant {item!r}; expected name:steps:true_cfg:lora_or_none")
        variants.append((parts[0], int(parts[1]), float(parts[2]), parts[3]))
    return variants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--project_root", type=Path, default=Path("/share/home/group6/our_project/baselines/qwen-image-edit-baseline"))
    parser.add_argument("--model_path", default="models/Qwen-Image-Edit-2511")
    parser.add_argument("--lora_path", type=Path, default=None)
    parser.add_argument("--lora_weight_name", default=None)
    parser.add_argument("--max_samples", type=int, default=4)
    parser.add_argument(
        "--variants",
        default="base40:40:4.0:none,base15:15:4.0:none,distill15:15:1.0:lora,distill8:8:1.0:lora",
        help="Comma-separated variants: name:steps:true_cfg:lora_or_none",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Skip variants with existing PNG/JSON outputs and reuse their JSON rows.")
    args = parser.parse_args()

    add_qwen_scripts_to_path(args.project_root)
    from run_qwen_edit_benchmark import load_qwen_edit_pipeline

    import torch

    pipe = load_qwen_edit_pipeline(model_path=str((args.project_root / args.model_path).resolve()), strict=True)
    requests = load_manifest(args.manifest)[: args.max_samples]
    variants = parse_variants(args.variants)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lora_loaded = False
    if any(kind == "lora" for _, _, _, kind in variants):
        if args.lora_path is None or not args.lora_path.exists():
            raise RuntimeError(f"LoRA variant requested but --lora_path is missing or does not exist: {args.lora_path}")
        lora_path = args.lora_path
        weight_name = args.lora_weight_name
        if lora_path.is_file():
            weight_name = lora_path.name
            lora_path = lora_path.parent
        elif weight_name is None and (lora_path / "model.safetensors").exists():
            weight_name = "model.safetensors"
        if weight_name:
            pipe.load_lora_weights(str(lora_path), weight_name=weight_name, adapter_name="distill_lora")
        else:
            pipe.load_lora_weights(str(lora_path), adapter_name="distill_lora")
        lora_loaded = True

    rows = []
    for idx, request in enumerate(requests):
        sample_dir = args.output_dir / request.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        input_image = load_rgb(request.input_image)
        save_rgb(sample_dir / "input.png", input_image)
        if request.target_image:
            save_rgb(sample_dir / "target.png", load_rgb(request.target_image))
        save_text(sample_dir / "prompt.txt", request.instruction + "\n")
        for name, steps, true_cfg, kind in variants:
            out_path = sample_dir / f"{name}.png"
            json_path = sample_dir / f"{name}.json"
            if args.resume and out_path.exists() and json_path.exists():
                rows.append(load_json(json_path))
                print(f"[{idx + 1}/{len(requests)}] {request.sample_id} {name} resume=skip", flush=True)
                continue
            if kind == "lora":
                if not lora_loaded:
                    raise RuntimeError("LoRA was not loaded")
                pipe.set_adapters(["distill_lora"], adapter_weights=[1.0])
            else:
                if lora_loaded:
                    pipe.disable_lora()
            start = time.perf_counter()
            with torch.inference_mode():
                output = pipe(
                    image=[input_image],
                    prompt=request.instruction,
                    negative_prompt=" ",
                    num_inference_steps=steps,
                    true_cfg_scale=true_cfg,
                    guidance_scale=1.0,
                    generator=torch.manual_seed(args.seed + idx),
                ).images[0]
            elapsed = time.perf_counter() - start
            save_rgb(out_path, output)
            metrics = compute_metrics(request.input_image, out_path, request.target_image, request.mask_image)
            row = {
                "sample_id": request.sample_id,
                "prompt": request.instruction,
                "variant": name,
                "steps": steps,
                "true_cfg_scale": true_cfg,
                "uses_lora": kind == "lora",
                "elapsed_sec": elapsed,
                "sec_per_step": elapsed / max(steps, 1),
                "output_image": str(out_path),
                "input_image": str(sample_dir / "input.png"),
                "target_image": str(sample_dir / "target.png") if request.target_image else None,
                "mask_image": str(request.mask_image) if request.mask_image else None,
                "metrics": metrics,
            }
            save_json(sample_dir / f"{name}.json", row)
            rows.append(row)
            print(f"[{idx + 1}/{len(requests)}] {request.sample_id} {name} steps={steps} lora={kind == 'lora'} time={elapsed:.2f}s", flush=True)
    if lora_loaded:
        pipe.disable_lora()
    save_json(args.output_dir / "qwen_distill_lora_speed_results.json", rows)
    print(f"Qwen distill LoRA speed benchmark finished: rows={len(rows)} output={args.output_dir}")


if __name__ == "__main__":
    main()
