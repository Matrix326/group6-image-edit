#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen-Image-Edit LoRA paired-image trainer

Dataset layout expected by default:
  /root/autodl-tmp/000/old/        # input / condition images
  /root/autodl-tmp/000/new/        # target / edited images
  /root/autodl-tmp/000/untitled.txt # one prompt for all images, or one prompt per image

This script trains LoRA layers on Qwen-Image-Edit's transformer only.
It pre-caches prompt embeddings and VAE latents, then unloads VAE/text_encoder
so the training loop mostly keeps only the transformer + LoRA on GPU.
"""

import argparse
import copy
import gc
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import QwenImageEditPipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

try:
    from diffusers.training_utils import compute_loss_weighting_for_sd3
except Exception:  # diffusers old version fallback
    compute_loss_weighting_for_sd3 = None

try:
    from diffusers.training_utils import _collate_lora_metadata
except Exception:
    _collate_lora_metadata = None

try:
    from diffusers.training_utils import cast_training_params
except Exception:
    cast_training_params = None

logger = get_logger(__name__)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser("Train a LoRA for Qwen-Image-Edit with paired old/new images.")

    # Your paths
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="/root/autodl-tmp/edit")
    parser.add_argument("--old_dir", type=str, default="/root/autodl-tmp/000/old")
    parser.add_argument("--new_dir", type=str, default="/root/autodl-tmp/000/new")
    parser.add_argument("--prompt_file", type=str, default="/root/autodl-tmp/000/untitled.txt")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/qwen_edit_lora_out")

    # Data / model
    parser.add_argument("--resolution", type=int, default=512, help="Training square resolution. Use 512 first to test.")
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--no_local_files_only", action="store_false", dest="local_files_only")
    parser.add_argument("--repeats", type=int, default=1, help="Repeat dataset N times per epoch.")

    # LoRA
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_layers",
        type=str,
        default="to_k,to_q,to_v,to_out.0",
        help='Comma-separated target module names. Example: "to_k,to_q,to_v,to_out.0"',
    )

    # Training
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=800)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--use_8bit_adam", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--logging_dir", type=str, default="logs")

    # Flow-matching timestep sampling
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--weighting_scheme", type=str, default="none", choices=["none", "sigma_sqrt", "logit_normal", "mode", "cosmap"])
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Only used if transformer.config.guidance_embeds=True")

    # Saving / validation
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--validation_steps", type=int, default=0, help="0 disables validation inference during training.")
    parser.add_argument("--validation_index", type=int, default=0)
    parser.add_argument("--validation_num_inference_steps", type=int, default=30)
    parser.add_argument("--validation_lora_scale", type=float, default=1.0)

    return parser.parse_args()


def image_paths(folder: str) -> List[Path]:
    paths = [p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()]
    return sorted(paths, key=lambda p: p.name)


def pair_old_new(old_dir: str, new_dir: str) -> List[Tuple[Path, Path]]:
    old_paths = image_paths(old_dir)
    new_paths = image_paths(new_dir)
    if not old_paths:
        raise ValueError(f"No images found in old_dir: {old_dir}")
    if not new_paths:
        raise ValueError(f"No images found in new_dir: {new_dir}")

    old_by_stem = {p.stem: p for p in old_paths}
    new_by_stem = {p.stem: p for p in new_paths}
    common = sorted(set(old_by_stem) & set(new_by_stem))

    if common:
        pairs = [(old_by_stem[s], new_by_stem[s]) for s in common]
        missing_old = sorted(set(new_by_stem) - set(old_by_stem))
        missing_new = sorted(set(old_by_stem) - set(new_by_stem))
        if missing_old:
            print(f"⚠️ new 中有 {len(missing_old)} 个文件没有同名 old，将忽略。")
        if missing_new:
            print(f"⚠️ old 中有 {len(missing_new)} 个文件没有同名 new，将忽略。")
        return pairs

    if len(old_paths) != len(new_paths):
        raise ValueError(
            "old/new 没有同名文件，而且数量不同，无法安全配对。"
            f" old={len(old_paths)}, new={len(new_paths)}"
        )
    print("⚠️ old/new 没有同名 stem，改用按文件名排序后一一配对。")
    return list(zip(old_paths, new_paths))


def read_prompts(prompt_file: str, num_pairs: int) -> List[str]:
    text = Path(prompt_file).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {prompt_file}")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) == 1:
        return lines * num_pairs
    if len(lines) == num_pairs:
        return lines
    raise ValueError(
        f"prompt 行数不匹配：{prompt_file} 中有 {len(lines)} 行非空 prompt，"
        f"但 old/new 配对数量是 {num_pairs}。请改成 1 行，或正好 {num_pairs} 行。"
    )


class PairedEditDataset(Dataset):
    def __init__(self, old_dir: str, new_dir: str, prompt_file: str, resolution: int, repeats: int = 1):
        self.pairs = pair_old_new(old_dir, new_dir)
        self.prompts = read_prompts(prompt_file, len(self.pairs))
        self.resolution = resolution
        self.repeats = max(1, int(repeats))
        self.transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.pairs) * self.repeats

    def _load_pil(self, path: Path) -> Image.Image:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((self.resolution, self.resolution), Image.Resampling.BICUBIC)
        return img

    def __getitem__(self, idx: int):
        real_idx = idx % len(self.pairs)
        old_path, new_path = self.pairs[real_idx]
        old_pil = self._load_pil(old_path)
        new_pil = self._load_pil(new_path)
        return {
            "old_pil": old_pil,
            "old_tensor": self.transform(old_pil),
            "new_tensor": self.transform(new_pil),
            "prompt": self.prompts[real_idx],
            "old_path": str(old_path),
            "new_path": str(new_path),
        }


def pil_collate(examples: Sequence[Dict]):
    return {
        "old_pil": [e["old_pil"] for e in examples],
        "old_tensor": torch.stack([e["old_tensor"] for e in examples]),
        "new_tensor": torch.stack([e["new_tensor"] for e in examples]),
        "prompt": [e["prompt"] for e in examples],
        "old_path": [e["old_path"] for e in examples],
        "new_path": [e["new_path"] for e in examples],
    }


@dataclass
class CachedSample:
    target_latents: torch.Tensor       # [C, 1, H, W], CPU
    condition_latents: torch.Tensor    # [C, 1, H, W], CPU
    prompt_embeds: torch.Tensor        # [S, D], CPU
    prompt_mask: Optional[torch.Tensor]  # [S], CPU or None
    prompt: str
    old_path: str
    new_path: str


class CachedTensorDataset(Dataset):
    def __init__(self, samples: List[CachedSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def cached_collate(samples: Sequence[CachedSample]):
    target_latents = torch.stack([s.target_latents for s in samples], dim=0)
    condition_latents = torch.stack([s.condition_latents for s in samples], dim=0)

    max_len = max(s.prompt_embeds.shape[0] for s in samples)
    hidden = samples[0].prompt_embeds.shape[-1]
    prompt_embeds = target_latents.new_zeros((len(samples), max_len, hidden), dtype=samples[0].prompt_embeds.dtype)
    prompt_mask = torch.zeros((len(samples), max_len), dtype=torch.long)

    for i, s in enumerate(samples):
        seq_len = s.prompt_embeds.shape[0]
        prompt_embeds[i, :seq_len] = s.prompt_embeds
        if s.prompt_mask is None:
            prompt_mask[i, :seq_len] = 1
        else:
            prompt_mask[i, :seq_len] = s.prompt_mask.to(torch.long)

    if bool(prompt_mask.all()):
        prompt_mask = None

    return {
        "target_latents": target_latents,
        "condition_latents": condition_latents,
        "prompt_embeds": prompt_embeds,
        "prompt_mask": prompt_mask,
        "prompts": [s.prompt for s in samples],
        "old_paths": [s.old_path for s in samples],
        "new_paths": [s.new_path for s in samples],
    }


def get_weight_dtype(mixed_precision: str):
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def encode_vae_image(pipe: QwenImageEditPipeline, pixel_values: torch.Tensor, dtype: torch.dtype, device: torch.device):
    # Qwen VAE expects [B, C, F, H, W]. Our tensors are [B, C, H, W].
    pixel_values = pixel_values.to(device=device, dtype=dtype).unsqueeze(2)
    latents = pipe._encode_vae_image(pixel_values, generator=None)
    return latents


def cache_dataset(
    args,
    accelerator: Accelerator,
    pipe: QwenImageEditPipeline,
    dataset: PairedEditDataset,
    weight_dtype: torch.dtype,
) -> List[CachedSample]:
    loader = DataLoader(
        dataset,
        batch_size=1,  # safer for huge text encoder / VAE; training batch size is separate
        shuffle=False,
        collate_fn=pil_collate,
        num_workers=0,
    )
    samples: List[CachedSample] = []
    device = accelerator.device

    pipe.vae.eval()
    pipe.text_encoder.eval()

    for batch in tqdm(loader, desc="Caching prompt embeddings and latents", disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            prompt_embeds, prompt_mask = pipe.encode_prompt(
                image=batch["old_pil"],
                prompt=batch["prompt"],
                device=device,
                max_sequence_length=args.max_sequence_length,
            )
            target_latents = encode_vae_image(pipe, batch["new_tensor"], weight_dtype, device)
            condition_latents = encode_vae_image(pipe, batch["old_tensor"], weight_dtype, device)

        samples.append(
            CachedSample(
                target_latents=target_latents[0].detach().cpu(),
                condition_latents=condition_latents[0].detach().cpu(),
                prompt_embeds=prompt_embeds[0].detach().cpu(),
                prompt_mask=None if prompt_mask is None else prompt_mask[0].detach().cpu(),
                prompt=batch["prompt"][0],
                old_path=batch["old_path"][0],
                new_path=batch["new_path"][0],
            )
        )

    return samples


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float,
    logit_std: float,
    mode_scale: float,
    device: torch.device,
):
    if weighting_scheme == "logit_normal":
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device=device)
        return torch.sigmoid(u)
    if weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device=device)
        return 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    # "sigma_sqrt", "cosmap" and "none" can use uniform sampling; loss weighting handles post-weighting.
    return torch.rand(size=(batch_size,), device=device)


def loss_weighting(weighting_scheme: str, sigmas: torch.Tensor):
    if compute_loss_weighting_for_sd3 is not None:
        return compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)
    return torch.ones_like(sigmas)


def save_lora(output_dir: str, transformer, weight_name: str = "pytorch_lora_weights.safetensors"):
    os.makedirs(output_dir, exist_ok=True)
    transformer_lora_layers = get_peft_model_state_dict(transformer)
    kwargs = {}
    if _collate_lora_metadata is not None:
        try:
            kwargs.update(_collate_lora_metadata({"transformer": transformer}))
        except Exception:
            pass
    QwenImageEditPipeline.save_lora_weights(
        save_directory=output_dir,
        transformer_lora_layers=transformer_lora_layers,
        weight_name=weight_name,
        safe_serialization=True,
        **kwargs,
    )


def run_validation(args, accelerator, transformer, sample: CachedSample, step: int, weight_dtype: torch.dtype):
    # Minimal validation: reload pipeline and generated image for one sample.
    # This costs memory, so keep validation_steps=0 unless you need it.
    if args.validation_steps <= 0:
        return
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    tmp_lora_dir = Path(args.output_dir) / f"_tmp_lora_step_{step}"
    save_lora(str(tmp_lora_dir), accelerator.unwrap_model(transformer))

    pipe = QwenImageEditPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
        local_files_only=args.local_files_only,
    ).to(accelerator.device)
    pipe.load_lora_weights(str(tmp_lora_dir))
    try:
        pipe.set_adapters(["default"], adapter_weights=[args.validation_lora_scale])
    except Exception:
        pass

    image = Image.open(sample.old_path).convert("RGB").resize((args.resolution, args.resolution), Image.Resampling.BICUBIC)
    with torch.no_grad():
        out = pipe(
            image=image,
            prompt=sample.prompt,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.validation_num_inference_steps,
            guidance_scale=args.guidance_scale if getattr(pipe.transformer.config, "guidance_embeds", False) else None,
        ).images[0]
    val_dir = Path(args.output_dir) / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    out.save(val_dir / f"step_{step}.png")

    pipe.to("cpu")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def main(args):
    logging_dir = Path(args.output_dir) / args.logging_dir
    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision if args.mixed_precision != "no" else None,
        log_with=args.report_to,
        project_config=project_config,
    )

    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    weight_dtype = get_weight_dtype(args.mixed_precision)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading Qwen-Image-Edit pipeline...")
    pipe = QwenImageEditPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
        local_files_only=args.local_files_only,
    ).to(accelerator.device)

    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.transformer.requires_grad_(False)

    if args.gradient_checkpointing and hasattr(pipe.transformer, "enable_gradient_checkpointing"):
        pipe.transformer.enable_gradient_checkpointing()

    target_modules = [x.strip() for x in args.lora_layers.split(",") if x.strip()]
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    pipe.transformer.add_adapter(lora_config)

    if args.mixed_precision == "fp16" and cast_training_params is not None:
        cast_training_params([pipe.transformer], dtype=torch.float32)

    trainable = [p for p in pipe.transformer.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable)
    logger.info(f"Trainable LoRA params: {num_trainable:,}")

    raw_dataset = PairedEditDataset(args.old_dir, args.new_dir, args.prompt_file, args.resolution, args.repeats)
    if accelerator.is_main_process:
        print(f"✅ 配对样本数: {len(raw_dataset.pairs)}，repeats={args.repeats}，训练样本数={len(raw_dataset)}")
        print(f"✅ LoRA target modules: {target_modules}")

    # Cache prompt embeddings + VAE latents before training to reduce memory.
    cached_samples = cache_dataset(args, accelerator, pipe, raw_dataset, weight_dtype)

    # Keep only transformer after cache.
    scheduler = copy.deepcopy(pipe.scheduler)
    scheduler.set_timesteps(args.num_train_timesteps, device=accelerator.device)
    vae_scale_factor = pipe.vae_scale_factor
    transformer = pipe.transformer

    pipe.to("cpu")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    cached_dataset = CachedTensorDataset(cached_samples)
    train_dataloader = DataLoader(
        cached_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=cached_collate,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError as e:
            raise ImportError("--use_8bit_adam requires bitsandbytes: pip install bitsandbytes") from e
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        [{"params": [p for p in transformer.parameters() if p.requires_grad], "lr": args.learning_rate}],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None or args.max_train_steps <= 0:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("qwen-image-edit-paired-lora", config=vars(args))

    def get_sigmas(timesteps: torch.Tensor, n_dim: int, dtype: torch.dtype):
        sigmas = scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num cached samples = {len(cached_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Per-device batch size = {args.train_batch_size}")
    logger.info(f"  Total effective batch size = {total_batch_size}")
    logger.info(f"  Max train steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, desc="Steps")
    global_step = 0

    for epoch in range(args.num_train_epochs):
        transformer.train()
        for batch in train_dataloader:
            with accelerator.accumulate(transformer):
                target_latents = batch["target_latents"].to(device=accelerator.device, dtype=weight_dtype)
                condition_latents = batch["condition_latents"].to(device=accelerator.device, dtype=weight_dtype)
                prompt_embeds = batch["prompt_embeds"].to(device=accelerator.device, dtype=weight_dtype)
                prompt_mask = batch["prompt_mask"]
                if prompt_mask is not None:
                    prompt_mask = prompt_mask.to(device=accelerator.device)

                noise = torch.randn_like(target_latents)
                bsz = target_latents.shape[0]

                u = compute_density_for_timestep_sampling(
                    args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                    device=accelerator.device,
                )
                indices = (u * scheduler.config.num_train_timesteps).long().clamp(0, len(scheduler.timesteps) - 1)
                timesteps = scheduler.timesteps[indices].to(device=accelerator.device)
                sigmas = get_sigmas(timesteps, n_dim=target_latents.ndim, dtype=target_latents.dtype)

                noisy_target = (1.0 - sigmas) * target_latents + sigmas * noise

                # Match QwenImageEditPipeline.prepare_latents layout:
                #   target/generated latents are packed from [B, 1, C, H, W]
                #   condition image latents are packed from [B, C, 1, H, W]
                target_for_pack = noisy_target.permute(0, 2, 1, 3, 4).contiguous()
                packed_target = QwenImageEditPipeline._pack_latents(
                    target_for_pack,
                    batch_size=bsz,
                    num_channels_latents=target_latents.shape[1],
                    height=target_latents.shape[3],
                    width=target_latents.shape[4],
                )
                packed_condition = QwenImageEditPipeline._pack_latents(
                    condition_latents.contiguous(),
                    batch_size=bsz,
                    num_channels_latents=condition_latents.shape[1],
                    height=condition_latents.shape[3],
                    width=condition_latents.shape[4],
                )
                hidden_states = torch.cat([packed_target, packed_condition], dim=1)

                latent_h = args.resolution // vae_scale_factor // 2
                latent_w = args.resolution // vae_scale_factor // 2
                img_shapes = [[(1, latent_h, latent_w), (1, latent_h, latent_w)] for _ in range(bsz)]

                guidance = None
                unwrapped = accelerator.unwrap_model(transformer)
                if getattr(unwrapped.config, "guidance_embeds", False):
                    guidance = torch.full((bsz,), args.guidance_scale, device=accelerator.device, dtype=torch.float32)

                model_pred = transformer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_mask,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    img_shapes=img_shapes,
                    return_dict=False,
                )[0]

                # The transformer predicts only the target part; discard condition positions if returned.
                model_pred = model_pred[:, : packed_target.shape[1]]
                model_pred = QwenImageEditPipeline._unpack_latents(
                    model_pred,
                    height=args.resolution,
                    width=args.resolution,
                    vae_scale_factor=vae_scale_factor,
                )

                target = noise - target_latents
                weighting = loss_weighting(args.weighting_scheme, sigmas)
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    dim=1,
                ).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if accelerator.is_main_process and args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0:
                    ckpt_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                    save_lora(str(ckpt_dir), accelerator.unwrap_model(transformer))
                    logger.info(f"Saved LoRA checkpoint to {ckpt_dir}")

                if args.validation_steps > 0 and global_step % args.validation_steps == 0:
                    val_idx = min(max(args.validation_index, 0), len(cached_samples) - 1)
                    run_validation(args, accelerator, transformer, cached_samples[val_idx], global_step, weight_dtype)

                if global_step >= args.max_train_steps:
                    break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_transformer = accelerator.unwrap_model(transformer)
        save_lora(args.output_dir, final_transformer)
        config = {
            "base_model": args.pretrained_model_name_or_path,
            "old_dir": args.old_dir,
            "new_dir": args.new_dir,
            "prompt_file": args.prompt_file,
            "resolution": args.resolution,
            "rank": args.rank,
            "lora_alpha": args.lora_alpha,
            "lora_layers": args.lora_layers,
            "max_train_steps": args.max_train_steps,
            "learning_rate": args.learning_rate,
        }
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "training_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n🎉 训练完成，LoRA 已保存到: {args.output_dir}")
        print("   主要权重文件通常是 pytorch_lora_weights.safetensors")

    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
