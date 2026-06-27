from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Subset

from keepedit.io import dump_json as save_json
from keepedit.qwen_distill.dataset import QwenStepDistillDataset
from keepedit.qwen_distill.model import StepDistillAdapter

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def save_model_state(model: nn.Module, output: Path) -> None:
    torch.save(unwrap_model(model).state_dict(), output)


def load_model_state(model: nn.Module, checkpoint: Path, device: str) -> None:
    state = torch.load(checkpoint, map_location=device)
    unwrap_model(model).load_state_dict(state)


def reserve_cuda_memory(target_gib: float, device: str) -> list[torch.Tensor]:
    if target_gib <= 0 or device != "cuda":
        return []
    reserved = []
    bytes_per_gib = 1024**3
    target_bytes = int(target_gib * bytes_per_gib)
    chunk_bytes = 1024**3
    allocated = 0
    while allocated < target_bytes:
        this_chunk = min(chunk_bytes, target_bytes - allocated)
        numel = max(1, this_chunk // 2)
        reserved.append(torch.empty(numel, dtype=torch.float16, device=device))
        allocated += numel * 2
    return reserved


def gradient_map(image: torch.Tensor) -> torch.Tensor:
    dx = image[..., :, 1:] - image[..., :, :-1]
    dy = image[..., 1:, :] - image[..., :-1, :]
    dx = torch.nn.functional.pad(dx, (0, 1, 0, 0))
    dy = torch.nn.functional.pad(dy, (0, 0, 0, 1))
    return torch.abs(dx) + torch.abs(dy)


def draw_loss_curve(history: list[dict[str, float]], output: Path) -> None:
    width, height, margin = 900, 520, 60
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    draw.rectangle([margin, margin, margin + plot_w, margin + plot_h], outline=(210, 210, 210))
    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row.get("val_loss") for row in history if row.get("val_loss") is not None]
    max_epoch = max(epochs) if epochs else 1
    max_loss = max(train_losses + val_losses) if train_losses or val_losses else 1.0

    def make_points(key: str) -> list[tuple[int, int]]:
        pts = []
        for row in history:
            loss = row.get(key)
            if loss is None:
                continue
            x = margin + int((row["epoch"] - 1) / max(max_epoch - 1, 1) * plot_w)
            y = margin + plot_h - int(loss / max(max_loss, 1e-8) * plot_h)
            pts.append((x, y))
        return pts

    pts = make_points("train_loss")
    val_pts = make_points("val_loss")
    if len(pts) > 1:
        draw.line(pts, fill=(220, 50, 47), width=3)
    if len(val_pts) > 1:
        draw.line(val_pts, fill=(30, 120, 210), width=3)
    draw.text((margin, 20), "Qwen Step-Distill Adapter Loss", fill=(20, 20, 20))
    draw.text((margin + 500, 20), "red=train blue=val", fill=(20, 20, 20))
    final_train = train_losses[-1] if train_losses else 0.0
    final_val = val_losses[-1] if val_losses else None
    footer = f"epochs={len(history)} final_train={final_train:.6f}"
    if final_val is not None:
        footer += f" final_val={final_val:.6f}"
    draw.text((margin, height - 35), footer, fill=(20, 20, 20))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def draw_step_loss_curve(history: list[dict[str, float]], output: Path) -> None:
    width, height, margin = 1100, 520, 60
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    draw.rectangle([margin, margin, margin + plot_w, margin + plot_h], outline=(210, 210, 210))
    if not history:
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
        return
    steps = [int(row["global_step"]) for row in history]
    losses = [float(row["loss"]) for row in history]
    max_step = max(steps) if steps else 1
    min_loss = min(losses) if losses else 0.0
    max_loss = max(losses) if losses else 1.0
    span = max(max_loss - min_loss, 1e-8)
    pts = []
    for step, loss in zip(steps, losses):
        x = margin + int((step - 1) / max(max_step - 1, 1) * plot_w)
        y = margin + plot_h - int((loss - min_loss) / span * plot_h)
        pts.append((x, y))
    if len(pts) > 1:
        draw.line(pts, fill=(220, 50, 47), width=2)
    draw.text((margin, 20), "Qwen Step-Distill Adapter Step Loss", fill=(20, 20, 20))
    footer = f"steps={len(history)} final={losses[-1]:.6f} min={min_loss:.6f} max={max_loss:.6f}"
    draw.text((margin, height - 35), footer, fill=(20, 20, 20))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def draw_val_curve(history: list[dict[str, float]], output: Path) -> None:
    rows = [row for row in history if row.get("val_ssim_teacher") is not None]
    width, height, margin = 900, 520, 60
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    draw.rectangle([margin, margin, margin + plot_w, margin + plot_h], outline=(210, 210, 210))
    max_epoch = max([row["epoch"] for row in rows], default=1)
    ssim_vals = [row["val_ssim_teacher"] for row in rows]
    psnr_vals = [row["val_psnr_teacher"] for row in rows]
    max_psnr = max(psnr_vals, default=1.0)

    ssim_pts = []
    psnr_pts = []
    for row in rows:
        x = margin + int((row["epoch"] - 1) / max(max_epoch - 1, 1) * plot_w)
        ssim_y = margin + plot_h - int(row["val_ssim_teacher"] * plot_h)
        psnr_y = margin + plot_h - int((row["val_psnr_teacher"] / max(max_psnr, 1e-8)) * plot_h)
        ssim_pts.append((x, ssim_y))
        psnr_pts.append((x, psnr_y))
    if len(ssim_pts) > 1:
        draw.line(ssim_pts, fill=(30, 150, 70), width=3)
    if len(psnr_pts) > 1:
        draw.line(psnr_pts, fill=(230, 130, 20), width=3)
    draw.text((margin, 20), "Validation Metrics to 40-step Teacher", fill=(20, 20, 20))
    draw.text((margin + 500, 20), "green=SSIM orange=PSNR(scaled)", fill=(20, 20, 20))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def ssim_proxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c1 = 0.01**2
    c2 = 0.03**2
    dims = (-3, -2, -1)
    mu_a = a.mean(dim=dims)
    mu_b = b.mean(dim=dims)
    var_a = ((a - mu_a[:, None, None, None]) ** 2).mean(dim=dims)
    var_b = ((b - mu_b[:, None, None, None]) ** 2).mean(dim=dims)
    cov = ((a - mu_a[:, None, None, None]) * (b - mu_b[:, None, None, None])).mean(dim=dims)
    score = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
    return score.clamp(-1.0, 1.0).mean()


def psnr_tensor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((a - b) ** 2, dim=(-3, -2, -1)).clamp_min(1e-8)
    return (20.0 * torch.log10(torch.tensor(1.0, device=a.device) / torch.sqrt(mse))).mean()


def compute_loss(
    model: StepDistillAdapter,
    batch: dict,
    device: str,
    l1: nn.Module,
    use_amp: bool,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor]:
    input_image = batch["input"].to(device)
    student = batch["student"].to(device)
    teacher = batch["teacher"].to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp and device == "cuda"):
        output, alpha = model(input_image, student)
        recon_loss = l1(output, teacher)
        edge_loss = l1(gradient_map(output), gradient_map(teacher))
        ssim_loss = 1.0 - ssim_proxy(output.float(), teacher.float())
        alpha_loss = alpha.mean()
        loss = recon_loss + 0.25 * edge_loss + 0.05 * ssim_loss + 0.02 * alpha_loss
    parts = {
        "recon_l1": float(recon_loss.detach().float().item()),
        "edge_l1": float(edge_loss.detach().float().item()),
        "ssim_loss": float(ssim_loss.detach().float().item()),
        "alpha_mean": float(alpha_loss.detach().float().item()),
    }
    return loss, parts, output, alpha


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    import numpy as np

    arr = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).astype("uint8"))


def fit(image: Image.Image, w: int, h: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(w / image.width, h / image.height)
    resized = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w, h), (245, 245, 245))
    canvas.paste(resized, ((w - resized.width) // 2, (h - resized.height) // 2))
    return canvas


def make_sample_grid_image(model: StepDistillAdapter, dataset: QwenStepDistillDataset, indices: list[int], device: str) -> Image.Image:
    cols = [("input", "Input"), ("target", "Target"), ("student", "Qwen-4step"), ("adapted", "Adapter"), ("teacher", "Qwen-40step")]
    cell_w, cell_h = 220, 220
    margin, gap, header_h, prompt_h = 24, 14, 30, 56
    width = margin * 2 + len(cols) * cell_w + (len(cols) - 1) * gap
    row_h = header_h + cell_h + prompt_h
    height = margin * 2 + len(indices) * row_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    model.eval()
    for row_idx, index in enumerate(indices):
        batch = dataset[index]
        input_tensor = batch["input"].unsqueeze(0).to(device)
        student_tensor = batch["student"].unsqueeze(0).to(device)
        with torch.inference_mode():
            adapted, _ = model(input_tensor, student_tensor)
        images = {
            "input": tensor_to_image(batch["input"]),
            "target": tensor_to_image(batch["target"]),
            "student": tensor_to_image(batch["student"]),
            "adapted": tensor_to_image(adapted[0]),
            "teacher": tensor_to_image(batch["teacher"]),
        }
        y0 = margin + row_idx * row_h
        for col_idx, (key, label) in enumerate(cols):
            x0 = margin + col_idx * (cell_w + gap)
            draw.text((x0 + cell_w / 2, y0), label, fill=(20, 20, 20), anchor="ma")
            canvas.paste(fit(images[key], cell_w, cell_h), (x0, y0 + header_h))
        draw.text((width / 2, y0 + header_h + cell_h + 16), str(batch.get("prompt", ""))[:130], fill=(60, 60, 60), anchor="ma")
    return canvas


def save_sample_grid(model: StepDistillAdapter, dataset: QwenStepDistillDataset, indices: list[int], device: str, output: Path) -> Image.Image:
    canvas = make_sample_grid_image(model, dataset, indices, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return canvas


def split_indices(n: int, val_ratio: float, val_count: int | None, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if val_count is None:
        val_n = int(round(n * val_ratio))
    else:
        val_n = val_count
    val_n = max(0, min(n - 1 if n > 1 else 0, val_n))
    val_indices = sorted(indices[:val_n])
    train_indices = sorted(indices[val_n:])
    return train_indices, val_indices


def evaluate(model: StepDistillAdapter, loader: DataLoader, device: str, l1: nn.Module, use_amp: bool) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.inference_mode():
        for batch in loader:
            loss, parts, output, _ = compute_loss(model, batch, device, l1, use_amp)
            teacher = batch["teacher"].to(device)
            student = batch["student"].to(device)
            batch_n = len(teacher)
            metrics = {
                "val_loss": float(loss.detach().float().item()),
                "val_ssim_teacher": float(ssim_proxy(output.float(), teacher.float()).item()),
                "val_psnr_teacher": float(psnr_tensor(output.float(), teacher.float()).item()),
                "val_student_ssim_teacher": float(ssim_proxy(student.float(), teacher.float()).item()),
                "val_student_psnr_teacher": float(psnr_tensor(student.float(), teacher.float()).item()),
                **parts,
            }
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * batch_n
            count += batch_n
    return {key: value / max(count, 1) for key, value in totals.items()}


def write_csv(history: list[dict[str, float]], output: Path) -> None:
    keys = [
        "epoch",
        "lr",
        "train_loss",
        "train_recon_l1",
        "train_edge_l1",
        "train_ssim_loss",
        "train_alpha_mean",
        "val_loss",
        "val_ssim_teacher",
        "val_psnr_teacher",
        "val_student_ssim_teacher",
        "val_student_psnr_teacher",
        "best_score",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_step_csv(history: list[dict[str, float]], output: Path) -> None:
    keys = [
        "global_step",
        "epoch",
        "step_in_epoch",
        "lr",
        "loss",
        "recon_l1",
        "edge_l1",
        "ssim_loss",
        "alpha_mean",
        "batch_size",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key, "") for key in keys})


def pil_to_tensorboard_image(image: Image.Image) -> torch.Tensor:
    import numpy as np

    arr = np.asarray(image.convert("RGB"), dtype="float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_json", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--init_checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--val_count", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--sample_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--max_sample_rows", type=int, default=4)
    parser.add_argument("--log_step_every", type=int, default=10)
    parser.add_argument("--reserve_memory_gib", type=float, default=0.0)
    parser.add_argument("--tensorboard_dir", type=Path, default=None)
    parser.add_argument("--no_tensorboard", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--data_parallel", action="store_true", help="Use torch.nn.DataParallel when multiple CUDA devices are visible.")
    args = parser.parse_args()

    dataset = QwenStepDistillDataset(args.metadata_json, size=args.image_size)
    if len(dataset) == 0:
        raise RuntimeError("empty step distill dataset")
    train_indices, val_indices = split_indices(len(dataset), args.val_ratio, args.val_count, args.seed)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices) if val_indices else None
    loader = DataLoader(train_dataset, batch_size=min(args.batch_size, len(train_dataset)), shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=min(args.batch_size, len(val_dataset)), shuffle=False)
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reserved_memory = reserve_cuda_memory(args.reserve_memory_gib, device)
    if reserved_memory:
        reserved_gib = sum(t.numel() * t.element_size() for t in reserved_memory) / (1024**3)
        print(f"reserved_cuda_memory_gib={reserved_gib:.2f}", flush=True)
    model = StepDistillAdapter(hidden=args.hidden).to(device)
    if args.data_parallel and device == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"using DataParallel with {torch.cuda.device_count()} CUDA devices")
    if args.init_checkpoint is not None:
        load_model_state(model, args.init_checkpoint, device)
        print(f"loaded init checkpoint: {args.init_checkpoint}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp and device == "cuda")
    l1 = nn.L1Loss()
    history = []
    step_history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_lines = []
    tb_writer = None
    if not args.no_tensorboard:
        if SummaryWriter is None:
            print("tensorboard unavailable: install tensorboard to enable event logging", flush=True)
        else:
            tb_dir = args.tensorboard_dir or (args.output_dir / "tb")
            tb_writer = SummaryWriter(log_dir=str(tb_dir))
            tb_writer.add_text("config/metadata_json", str(args.metadata_json), 0)
            tb_writer.add_text("config/output_dir", str(args.output_dir), 0)
            tb_writer.add_scalar("config/image_size", args.image_size, 0)
            tb_writer.add_scalar("config/hidden", args.hidden, 0)
            tb_writer.add_scalar("config/batch_size", args.batch_size, 0)
            tb_writer.add_scalar("config/train_samples", len(train_indices), 0)
            tb_writer.add_scalar("config/val_samples", len(val_indices), 0)
    step_csv_path = args.output_dir / "train_step_loss.csv"
    step_csv_keys = [
        "global_step",
        "epoch",
        "step_in_epoch",
        "lr",
        "loss",
        "recon_l1",
        "edge_l1",
        "ssim_loss",
        "alpha_mean",
        "batch_size",
    ]
    with step_csv_path.open("w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=step_csv_keys).writeheader()
    use_amp = not args.no_amp
    best_score = -math.inf
    best_epoch = 0
    sample_pool = list(range(len(dataset)))
    random.Random(args.seed + 1337).shuffle(sample_pool)
    sample_indices = sorted(sample_pool[: args.max_sample_rows])
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        part_totals: dict[str, float] = {}
        count = 0
        for step_in_epoch, batch in enumerate(loader, start=1):
            loss, parts, _, _ = compute_loss(model, batch, device, l1, use_amp)
            batch_n = len(batch["input"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            total += float(loss.item()) * batch_n
            for key, value in parts.items():
                part_totals[key] = part_totals.get(key, 0.0) + value * batch_n
            count += batch_n
            global_step += 1
            step_row = {
                "global_step": global_step,
                "epoch": epoch + 1,
                "step_in_epoch": step_in_epoch,
                "lr": opt.param_groups[0]["lr"],
                "loss": float(loss.detach().float().item()),
                "recon_l1": parts.get("recon_l1"),
                "edge_l1": parts.get("edge_l1"),
                "ssim_loss": parts.get("ssim_loss"),
                "alpha_mean": parts.get("alpha_mean"),
                "batch_size": batch_n,
            }
            step_history.append(step_row)
            with step_csv_path.open("a", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, fieldnames=step_csv_keys).writerow(step_row)
            if tb_writer is not None:
                tb_writer.add_scalar("step/loss", step_row["loss"], global_step)
                tb_writer.add_scalar("step/recon_l1", step_row["recon_l1"], global_step)
                tb_writer.add_scalar("step/edge_l1", step_row["edge_l1"], global_step)
                tb_writer.add_scalar("step/ssim_loss", step_row["ssim_loss"], global_step)
                tb_writer.add_scalar("step/alpha_mean", step_row["alpha_mean"], global_step)
                tb_writer.add_scalar("step/lr", step_row["lr"], global_step)
            if args.log_step_every > 0 and global_step % args.log_step_every == 0:
                step_line = f"step={global_step} epoch={epoch + 1} step_in_epoch={step_in_epoch} loss={step_row['loss']:.6f} lr={step_row['lr']:.8f}"
                print(step_line, flush=True)
                log_lines.append(step_line)
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "lr": scheduler.get_last_lr()[0],
            "train_loss": total / max(count, 1),
            "train_recon_l1": part_totals.get("recon_l1", 0.0) / max(count, 1),
            "train_edge_l1": part_totals.get("edge_l1", 0.0) / max(count, 1),
            "train_ssim_loss": part_totals.get("ssim_loss", 0.0) / max(count, 1),
            "train_alpha_mean": part_totals.get("alpha_mean", 0.0) / max(count, 1),
        }
        should_eval = val_loader is not None and ((epoch + 1) % args.eval_every == 0 or epoch == 0 or epoch + 1 == args.epochs)
        if should_eval:
            val_metrics = evaluate(model, val_loader, device, l1, use_amp)
            row.update(val_metrics)
            score = row["val_ssim_teacher"] + row["val_psnr_teacher"] / 100.0
            if score > best_score:
                best_score = score
                best_epoch = epoch + 1
                save_model_state(model, args.output_dir / "step_distill_adapter_best.pt")
        row["best_score"] = best_score if best_score > -math.inf else None
        history.append(row)
        if tb_writer is not None:
            tb_writer.add_scalar("epoch/train_loss", row["train_loss"], epoch + 1)
            tb_writer.add_scalar("epoch/train_recon_l1", row["train_recon_l1"], epoch + 1)
            tb_writer.add_scalar("epoch/train_edge_l1", row["train_edge_l1"], epoch + 1)
            tb_writer.add_scalar("epoch/train_ssim_loss", row["train_ssim_loss"], epoch + 1)
            tb_writer.add_scalar("epoch/train_alpha_mean", row["train_alpha_mean"], epoch + 1)
            tb_writer.add_scalar("epoch/lr", row["lr"], epoch + 1)
            if row.get("val_loss") is not None:
                tb_writer.add_scalar("val/loss", row["val_loss"], epoch + 1)
                tb_writer.add_scalar("val/ssim_teacher", row["val_ssim_teacher"], epoch + 1)
                tb_writer.add_scalar("val/psnr_teacher", row["val_psnr_teacher"], epoch + 1)
                tb_writer.add_scalar("val/student_ssim_teacher", row["val_student_ssim_teacher"], epoch + 1)
                tb_writer.add_scalar("val/student_psnr_teacher", row["val_student_psnr_teacher"], epoch + 1)
                tb_writer.add_scalar("val/best_score", row["best_score"] or 0.0, epoch + 1)
        if (epoch + 1) % args.save_every == 0:
            rolling_ckpt = args.output_dir / "step_distill_adapter_latest_interval.pt"
            save_model_state(model, rolling_ckpt)
        if sample_indices and ((epoch + 1) % args.sample_every == 0 or epoch == 0 or epoch + 1 == args.epochs):
            sample_grid = save_sample_grid(model, dataset, sample_indices, device, args.output_dir / f"samples_epoch_{epoch + 1:04d}.png")
            if tb_writer is not None:
                tb_writer.add_image("samples/grid", pil_to_tensorboard_image(sample_grid), epoch + 1)
        line = f"epoch={row['epoch']} train_loss={row['train_loss']:.6f}"
        if row.get("val_loss") is not None:
            line += f" val_loss={row['val_loss']:.6f} val_ssim_teacher={row['val_ssim_teacher']:.4f} val_psnr_teacher={row['val_psnr_teacher']:.4f}"
        line += f" lr={row['lr']:.8f}"
        print(line)
        log_lines.append(line)
    save_model_state(model, args.output_dir / "step_distill_adapter_last.pt")
    if not (args.output_dir / "step_distill_adapter_best.pt").exists():
        save_model_state(model, args.output_dir / "step_distill_adapter_best.pt")
        best_epoch = args.epochs
    save_model_state(model, args.output_dir / "step_distill_adapter.pt")
    save_json(
        args.output_dir / "training_meta.json",
        {
            "history": history,
            "image_size": args.image_size,
            "hidden": args.hidden,
            "num_samples": len(dataset),
            "train_samples": len(train_indices),
            "val_samples": len(val_indices),
            "best_epoch": best_epoch,
            "best_score": best_score if best_score > -math.inf else None,
            "optimizer_steps": len(step_history),
            "loss": "L1 + 0.25 edge + 0.05 SSIM-proxy + 0.02 alpha",
        },
    )
    write_csv(history, args.output_dir / "train_loss.csv")
    write_step_csv(step_history, args.output_dir / "train_step_loss.csv")
    save_json(args.output_dir / "val_metrics.json", [row for row in history if row.get("val_loss") is not None])
    (args.output_dir / "train.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    draw_loss_curve(history, args.output_dir / "loss_curve.png")
    draw_step_loss_curve(step_history, args.output_dir / "step_loss_curve.png")
    draw_val_curve(history, args.output_dir / "val_curve.png")
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()


if __name__ == "__main__":
    main()
