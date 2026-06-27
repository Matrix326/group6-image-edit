from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _rgb_array(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def _mask_array(path: str | Path | None, size: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.ones((size[1], size[0]), dtype=bool)
    mask = Image.open(path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask) > 127


def _masked_mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return 0.0
    diff = (a - b) ** 2
    return float(diff[mask].mean())


def _masked_psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    mse = _masked_mse(a, b, mask)
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _masked_l1(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return 0.0
    return float(np.abs(a - b)[mask].mean())


def compute_metrics(
    input_image: str | Path,
    output_image: str | Path,
    target_image: str | Path | None,
    mask_image: str | Path | None = None,
) -> dict[str, float]:
    output = _rgb_array(output_image)
    size = (output.shape[1], output.shape[0])
    input_arr = _rgb_array(input_image, size=size)
    target_arr = _rgb_array(target_image, size=size) if target_image else input_arr
    mask = _mask_array(mask_image, size)
    bg_mask = ~mask

    metrics = {
        "ssim_target_output": float(structural_similarity(target_arr, output, channel_axis=2, data_range=1.0)),
        "psnr_target_output": float(peak_signal_noise_ratio(target_arr, output, data_range=1.0)),
        "ssim_input_output": float(structural_similarity(input_arr, output, channel_axis=2, data_range=1.0)),
        "psnr_input_output": float(peak_signal_noise_ratio(input_arr, output, data_range=1.0)),
        "edit_region_change": _masked_l1(input_arr, output, mask),
        "inside_target_l1": _masked_l1(target_arr, output, mask),
        "outside_delta_mean": _masked_l1(input_arr, output, bg_mask),
        "bg_psnr": _masked_psnr(input_arr, output, bg_mask),
    }
    if bg_mask.sum() > 8:
        bg_input = input_arr.copy()
        bg_output = output.copy()
        bg_input[mask] = 0.0
        bg_output[mask] = 0.0
        metrics["bg_ssim"] = float(structural_similarity(bg_input, bg_output, channel_axis=2, data_range=1.0))
    else:
        metrics["bg_ssim"] = metrics["ssim_input_output"]
    return metrics
