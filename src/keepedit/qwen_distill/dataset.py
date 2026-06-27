from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def image_tensor(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    arr = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


@dataclass
class StepDistillItem:
    sample_id: str
    prompt: str
    input_image: Path
    teacher_image: Path
    student_image: Path
    target_image: Path | None


class QwenStepDistillDataset(Dataset):
    def __init__(self, metadata_json: Path, size: int = 512) -> None:
        rows = json.loads(metadata_json.read_text(encoding="utf-8"))
        self.items = [
            StepDistillItem(
                sample_id=str(row.get("sample_id", index)),
                prompt=str(row.get("prompt", "")),
                input_image=Path(row["input_image"]),
                teacher_image=Path(row["teacher_image"]),
                student_image=Path(row["student_image"]),
                target_image=Path(row["target_image"]) if row.get("target_image") else None,
            )
            for index, row in enumerate(rows)
        ]
        self.size = size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.items[index]
        return {
            "sample_id": item.sample_id,
            "prompt": item.prompt,
            "input": image_tensor(item.input_image, self.size),
            "teacher": image_tensor(item.teacher_image, self.size),
            "student": image_tensor(item.student_image, self.size),
            "target": image_tensor(item.target_image or item.teacher_image, self.size),
        }
