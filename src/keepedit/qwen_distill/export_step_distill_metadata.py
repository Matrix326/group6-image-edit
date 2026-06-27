from __future__ import annotations

import argparse
from pathlib import Path

from keepedit.qwen_distill.metrics import load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_results_json", required=True, type=Path)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--student_variant", default="base4")
    parser.add_argument("--teacher_variant", default="base40")
    args = parser.parse_args()

    rows = load_json(args.cache_results_json)
    by_sample: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_sample.setdefault(str(row["sample_id"]), {})[str(row["variant"])] = row

    items = []
    for sample_id, variants in sorted(by_sample.items()):
        student = variants.get(args.student_variant)
        teacher = variants.get(args.teacher_variant)
        if student is None or teacher is None:
            continue
        items.append(
            {
                "sample_id": sample_id,
                "prompt": student.get("prompt", ""),
                "input_image": student["input_image"],
                "student_image": student["output_image"],
                "teacher_image": teacher["output_image"],
                "target_image": student.get("target_image"),
            }
        )
    save_json(args.output_json, items)
    print(f"wrote {len(items)} distillation metadata rows to {args.output_json}")


if __name__ == "__main__":
    main()
