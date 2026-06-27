from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EditRequest:
    sample_id: str
    instruction: str
    input_image: Path
    target_image: Path | None = None
    mask_image: Path | None = None


def _resolve(path: str | None, root: Path) -> Path | None:
    if not path:
        return None
    value = Path(path)
    return value if value.is_absolute() else root / value


def _load_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["samples", "data", "items"]:
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unsupported manifest format: {path}")


def load_manifest(path: str | Path) -> list[EditRequest]:
    manifest = Path(path)
    root = manifest.parent
    requests: list[EditRequest] = []
    for index, row in enumerate(_load_rows(manifest)):
        sample_id = str(row.get("sample_id") or row.get("id") or row.get("name") or f"sample_{index:06d}")
        instruction = str(row.get("instruction") or row.get("prompt") or row.get("edit") or "")
        input_image = _resolve(row.get("input_image") or row.get("source_image") or row.get("image"), root)
        if input_image is None:
            raise ValueError(f"Missing input image for {sample_id}")
        requests.append(
            EditRequest(
                sample_id=sample_id,
                instruction=instruction,
                input_image=input_image,
                target_image=_resolve(row.get("target_image") or row.get("output_image") or row.get("target"), root),
                mask_image=_resolve(row.get("mask_image") or row.get("mask"), root),
            )
        )
    return requests
