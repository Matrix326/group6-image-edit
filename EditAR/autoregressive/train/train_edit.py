# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT
#   nanoGPT: https://github.com/karpathy/nanoGPT
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from glob import glob
import time
import argparse
import os
import math
import inspect
import contextlib
from natsort import natsorted
from tqdm import tqdm
import wandb
# os.environ["WANDB_API_KEY"] = 'WANDB_CREDENTIAL'

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.distributed import init_distributed_mode
from utils.logger import create_logger
from dataset.build import build_dataset
from autoregressive.models.gpt_edit import GPT_models
from autoregressive.models.lora import count_trainable_parameters, inject_lora, lora_state_dict, mark_only_lora_as_trainable
from autoregressive.train.contrastive import margin_contrastive_loss
from autoregressive.train.mask_weight import mask_to_token_loss_weight
from tokenizer.tokenizer_image.vq_model import VQ_models

from language.t5 import T5Embedder
from feature_encoders.build import Semantic_Encoder

def compute_mask_loss_stats(logits, targets, token_loss_weight, lambda_edit, lambda_bg):
    if token_loss_weight is None:
        return None

    with torch.no_grad():
        loss_all = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="none",
        ).view(targets.shape)
        weight = token_loss_weight.to(device=loss_all.device, dtype=loss_all.dtype)
        edit_mask = weight > ((float(lambda_edit) + float(lambda_bg)) * 0.5)
        bg_mask = ~edit_mask

        edit_count = edit_mask.float().sum()
        bg_count = bg_mask.float().sum()
        return {
            "weight_mean": weight.mean(),
            "weight_max": weight.max(),
            "edit_token_frac": edit_mask.float().mean(),
            "edit_ce": loss_all[edit_mask].mean() if edit_count > 0 else torch.zeros((), device=loss_all.device),
            "bg_ce": loss_all[bg_mask].mean() if bg_count > 0 else torch.zeros((), device=loss_all.device),
        }

def creat_optimizer(model, weight_decay, learning_rate, betas, logger):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer

def get_learning_rate(args, step, total_steps):
    if args.lr_scheduler == "constant":
        return args.lr

    if args.lr_scheduler == "warmup_cosine":
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return args.lr * float(step + 1) / float(args.warmup_steps)

        decay_steps = max(1, total_steps - args.warmup_steps)
        decay_step = min(max(0, step - args.warmup_steps), decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_step / decay_steps))
        return args.min_lr + cosine * (args.lr - args.min_lr)

    raise ValueError(f"Unknown lr scheduler: {args.lr_scheduler}")

def set_optimizer_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    assert args.warmup_steps >= 0, "warmup_steps must be non-negative."
    assert args.min_lr <= args.lr, "min_lr must be less than or equal to lr."

    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    if rank==0 and args.use_wandb:
        wandb.init(
            project="editAR",
            name=os.path.basename(args.output_dir),
        )
        wandb.define_metric("train/*", step_metric="train/global_step")

    # Setup an experiment folder:
    checkpoint_dir = f"{args.output_dir}/checkpoints"
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(args.output_dir)
        logger.info(f"Experiment directory created at {args.output_dir}")

    else:
        logger = create_logger(None)

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # setup tokenizer
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    assert os.path.exists(args.t5_path)
    t5_model = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )

    assert os.path.exists(args.vq_ckpt)
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint

    # Setup model
    latent_size = args.image_size // args.downsample_size
    model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        model_mode=args.gpt_mode,
        resid_dropout_p=args.dropout_p,
        ffn_dropout_p=args.dropout_p,
        token_dropout_p=args.token_dropout_p,
        distill_mode=args.distill_mode,
    ).to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer_state = None
    extra_trainable = []

    if args.use_lora:
        target_modules = [target.strip() for target in args.lora_target_modules.split(",") if target.strip()]
        replaced_modules = inject_lora(
            model,
            target_modules=target_modules,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        extra_trainable = [keyword.strip() for keyword in args.lora_extra_trainable.split(",") if keyword.strip()]
        mark_only_lora_as_trainable(model, extra_trainable_keywords=extra_trainable)
        logger.info(f"LoRA injected into {replaced_modules} Linear modules: {target_modules}")
        logger.info(f"LoRA extra trainable parameter keywords: {extra_trainable}")
        logger.info(f"Trainable GPT Parameters: {count_trainable_parameters(model):,}")

    if args.use_distill:
        semantic_encoder = Semantic_Encoder(args.distill_mode, precision, device)

    dataset = build_dataset(args, llm_tokenizer=t5_model.tokenizer)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images")

    # Prepare models for training:
    folder_names = natsorted([folder_name for folder_name in os.listdir(checkpoint_dir)])

    if len(folder_names) > 0:
        model_path = os.path.join(checkpoint_dir, folder_names[-1])
        checkpoint = torch.load(model_path, map_location="cpu")
        if args.use_lora and checkpoint.get("checkpoint_type") == "lora_adapter":
            if args.gpt_ckpt:
                base_checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
                model.load_state_dict(base_checkpoint["model"], strict=False)
                del base_checkpoint
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint["model"], strict=not args.use_lora)
        if "optimizer" in checkpoint:
            optimizer_state = checkpoint["optimizer"]
        if "steps" in checkpoint:
            train_steps = checkpoint["steps"]
        else:
            train_steps = 0
        start_epoch = int(train_steps / int(len(dataset) / (args.global_batch_size * max(1, args.gradient_accumulation_steps))))
        del checkpoint
        logger.info(f"Resume training from checkpoint: {model_path}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    elif args.gpt_ckpt:
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        # if "optimizer" in checkpoint:
        #     optimizer.load_state_dict(checkpoint["optimizer"])
        train_steps = 0
        del checkpoint
        start_epoch = 0
        logger.info(f"Load pretrained checkpoint: {args.gpt_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0

    # Setup optimizer after optional LoRA injection and checkpoint loading so all
    # newly added trainable adapter parameters are included.
    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
        except ValueError as exc:
            logger.info(f"Skip optimizer resume because parameter groups changed: {exc}")

    if not args.no_compile:
        logger.info("compiling the model... (may take several minutes)")
        model = torch.compile(model) # requires PyTorch 2.0        
    
    model = DDP(model.to(device), device_ids=[args.gpu])
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    accumulation_steps = max(1, args.gradient_accumulation_steps)
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    running_llm_loss = 0
    running_distill_loss = 0
    running_contrastive_loss = 0
    running_weight_mean = 0
    running_weight_max = 0
    running_edit_token_frac = 0
    running_edit_ce = 0
    running_bg_ce = 0
    running_mask_stats_steps = 0
    start_time = time.time()

    optimizer.zero_grad(set_to_none=True)
    logger.info(
        f"Micro global batch size: {args.global_batch_size}, "
        f"gradient accumulation steps: {accumulation_steps}, "
        f"effective global batch size: {args.global_batch_size * accumulation_steps}"
    )
    steps_per_epoch = math.ceil(len(loader) / accumulation_steps)
    total_train_steps = max(1, args.epochs * steps_per_epoch)
    logger.info(
        f"LR scheduler: {args.lr_scheduler}, base_lr={args.lr}, min_lr={args.min_lr}, "
        f"warmup_steps={args.warmup_steps}, total_train_steps={total_train_steps}"
    )
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        progress_bar = tqdm(
            total=steps_per_epoch,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            dynamic_ncols=True,
            disable=(rank != 0),
        )
        progress_step_time = time.time()
        accum_loss = 0
        accum_llm_loss = 0
        accum_distill_loss = 0
        accum_contrastive_loss = 0
        accum_weight_mean = 0
        accum_weight_max = 0
        accum_edit_token_frac = 0
        accum_edit_ce = 0
        accum_bg_ce = 0
        accum_mask_stats_steps = 0
        accum_micro_steps = 0
        batch_iter = iter(loader)
        batch_idx = 0
        while batch_idx < len(loader):
            micro_batches = []
            for _ in range(accumulation_steps):
                if batch_idx >= len(loader):
                    break
                micro_batches.append(next(batch_iter))
                batch_idx += 1

            group_size = len(micro_batches)
            if group_size == 0:
                break

            group_txt_embs = []
            for batch in micro_batches:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                input_ids_attn_mask = batch['input_ids_attn_mask'].to(device, non_blocking=True)
                with torch.no_grad():
                    input_txt_embs = t5_model.model(
                        input_ids=input_ids,
                        attention_mask=input_ids_attn_mask,
                    )['last_hidden_state'].detach()
                group_txt_embs.append(input_txt_embs)

            negative_txt_pool = None
            local_pool_size = 0
            if args.use_negative_contrastive:
                local_txt_pool = torch.cat(group_txt_embs, dim=0)
                local_pool_size = local_txt_pool.shape[0]
                if dist.get_world_size() > 1:
                    gathered_txt_pools = [torch.empty_like(local_txt_pool) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_txt_pools, local_txt_pool)
                    negative_txt_pool = torch.cat(gathered_txt_pools, dim=0)
                else:
                    negative_txt_pool = local_txt_pool

            local_txt_offset = 0
            loss_divisor = group_size
            for micro_idx, (batch, input_txt_embs) in enumerate(zip(micro_batches, group_txt_embs)):
                update_grad = micro_idx == group_size - 1
                sync_context = contextlib.nullcontext() if update_grad else model.no_sync()

                # input_img, edited_img, target_ids, input_ids, prompts
                input_img = batch['input_img'].to(device, non_blocking=True)
                edited_img = batch['edited_img'].to(device, non_blocking=True)
                input_mode = batch['mode'].to(device, non_blocking=True)

                # process image ids to embeddings
                with torch.no_grad():
                    _, _, [_, _, input_img_indices] = vq_model.encode(input_img)
                    _, _, [_, _, edited_img_indices] = vq_model.encode(edited_img)
                    input_img_indices = input_img_indices.reshape(input_img.shape[0], -1)
                    edited_img_indices = edited_img_indices.reshape(edited_img.shape[0], -1)
                token_loss_weight = None
                if args.use_mask_weighted_loss:
                    if '_mask' in batch:
                        token_loss_weight = mask_to_token_loss_weight(
                            batch['_mask'].to(device, non_blocking=True),
                            latent_size=latent_size,
                            lambda_edit=args.lambda_edit,
                            lambda_bg=args.lambda_bg,
                        )
                    else:
                        token_loss_weight = torch.ones(
                            edited_img_indices.shape,
                            device=device,
                            dtype=torch.float32,
                        )
                with sync_context:
                    with torch.cuda.amp.autocast(dtype=ptdtype):
                        logits, llm_loss, llm_features = model(
                            input_txt_embs=input_txt_embs,
                            input_img_indices=input_img_indices,
                            edited_img_indices=edited_img_indices,
                            input_mode=input_mode,
                            token_loss_weight=token_loss_weight,
                        )
                        contrastive_loss = torch.zeros((), device=device, dtype=llm_loss.dtype)
                        if args.use_negative_contrastive and negative_txt_pool is not None and negative_txt_pool.shape[0] > 1:
                            wrong_indices = torch.randint(
                                0,
                                negative_txt_pool.shape[0],
                                (input_txt_embs.shape[0],),
                                device=device,
                            )
                            self_indices = rank * local_pool_size + local_txt_offset + torch.arange(
                                input_txt_embs.shape[0],
                                device=device,
                            )
                            wrong_indices = torch.where(
                                wrong_indices == self_indices,
                                (wrong_indices + 1) % negative_txt_pool.shape[0],
                                wrong_indices,
                            )
                            wrong_txt_embs = negative_txt_pool[wrong_indices]
                            _, wrong_llm_loss, _ = model(
                                input_txt_embs=wrong_txt_embs,
                                input_img_indices=input_img_indices,
                                edited_img_indices=edited_img_indices,
                                input_mode=input_mode,
                                token_loss_weight=token_loss_weight,
                            )
                            contrastive_loss = args.negative_contrastive_weight * margin_contrastive_loss(
                                llm_loss,
                                wrong_llm_loss,
                                args.negative_contrastive_margin,
                            )
                        if args.use_distill:
                            distill_loss = args.distill_loss_weight * semantic_encoder.compute_distill_loss(edited_img, llm_features)
                            raw_loss = llm_loss + distill_loss + contrastive_loss
                        else:
                            distill_loss = torch.zeros((), device=device, dtype=llm_loss.dtype)
                            raw_loss = llm_loss + contrastive_loss

                    loss = raw_loss / loss_divisor
                    scaler.scale(loss).backward()

                local_txt_offset += input_txt_embs.shape[0]
                accum_loss += raw_loss.item()
                accum_llm_loss += llm_loss.item()
                accum_distill_loss += distill_loss.item()
                accum_contrastive_loss += contrastive_loss.item()
                mask_stats = compute_mask_loss_stats(
                    logits.detach(),
                    edited_img_indices,
                    token_loss_weight,
                    args.lambda_edit,
                    args.lambda_bg,
                )
                if mask_stats is not None:
                    accum_weight_mean += mask_stats["weight_mean"].item()
                    accum_weight_max += mask_stats["weight_max"].item()
                    accum_edit_token_frac += mask_stats["edit_token_frac"].item()
                    accum_edit_ce += mask_stats["edit_ce"].item()
                    accum_bg_ce += mask_stats["bg_ce"].item()
                    accum_mask_stats_steps += 1
                accum_micro_steps += 1

            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            current_lr = get_learning_rate(args, train_steps, total_train_steps)
            set_optimizer_lr(optimizer, current_lr)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            current_loss = accum_loss / accum_micro_steps
            running_loss += current_loss
            running_llm_loss += accum_llm_loss / accum_micro_steps
            if args.use_distill:
                running_distill_loss += accum_distill_loss / accum_micro_steps
            if args.use_negative_contrastive:
                running_contrastive_loss += accum_contrastive_loss / accum_micro_steps
            if accum_mask_stats_steps > 0:
                running_weight_mean += accum_weight_mean / accum_mask_stats_steps
                running_weight_max += accum_weight_max / accum_mask_stats_steps
                running_edit_token_frac += accum_edit_token_frac / accum_mask_stats_steps
                running_edit_ce += accum_edit_ce / accum_mask_stats_steps
                running_bg_ce += accum_bg_ce / accum_mask_stats_steps
                running_mask_stats_steps += 1
            accum_loss = 0
            accum_llm_loss = 0
            accum_distill_loss = 0
            accum_contrastive_loss = 0
            accum_weight_mean = 0
            accum_weight_max = 0
            accum_edit_token_frac = 0
            accum_edit_ce = 0
            accum_bg_ce = 0
            accum_mask_stats_steps = 0
            accum_micro_steps = 0

            log_steps += 1
            train_steps += 1
            if rank == 0:
                now = time.time()
                step_seconds = now - progress_step_time
                progress_step_time = now
                if step_seconds > 1:
                    speed = f"{step_seconds:.2f}s/step"
                else:
                    speed = f"{1 / max(step_seconds, 1e-12):.2f}step/s"
                progress_bar.set_postfix(loss=f"{current_loss:.4f}", speed=speed, step=train_steps)
                progress_bar.update(1)
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_llm_loss = None
                avg_distill_loss = None
                avg_contrastive_loss = None
                avg_llm_loss = torch.tensor(running_llm_loss / log_steps, device=device)
                dist.all_reduce(avg_llm_loss, op=dist.ReduceOp.SUM)
                avg_llm_loss = avg_llm_loss.item() / dist.get_world_size()
                if args.use_distill:
                    avg_distill_loss = torch.tensor(running_distill_loss / log_steps, device=device)
                    dist.all_reduce(avg_distill_loss, op=dist.ReduceOp.SUM)
                    avg_distill_loss = avg_distill_loss.item() / dist.get_world_size()
                if args.use_negative_contrastive:
                    avg_contrastive_loss = torch.tensor(running_contrastive_loss / log_steps, device=device)
                    dist.all_reduce(avg_contrastive_loss, op=dist.ReduceOp.SUM)
                    avg_contrastive_loss = avg_contrastive_loss.item() / dist.get_world_size()
                avg_mask_stats = None
                if running_mask_stats_steps > 0:
                    mask_stats_denom = max(1, running_mask_stats_steps)
                    avg_mask_stats = {
                        "weight_mean": torch.tensor(running_weight_mean / mask_stats_denom, device=device),
                        "weight_max": torch.tensor(running_weight_max / mask_stats_denom, device=device),
                        "edit_token_frac": torch.tensor(running_edit_token_frac / mask_stats_denom, device=device),
                        "edit_ce": torch.tensor(running_edit_ce / mask_stats_denom, device=device),
                        "bg_ce": torch.tensor(running_bg_ce / mask_stats_denom, device=device),
                    }
                    for key, value in list(avg_mask_stats.items()):
                        dist.all_reduce(value, op=dist.ReduceOp.SUM)
                        avg_mask_stats[key] = value.item() / dist.get_world_size()
                if args.use_distill:
                    log_msg = f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, LLM Loss: {avg_llm_loss:.4f}, Distill Loss: {avg_distill_loss:.4f}"
                    if avg_contrastive_loss is not None:
                        log_msg += f", Contrastive Loss: {avg_contrastive_loss:.4f}"
                    logger.info(f"{log_msg}, Train Steps/Sec: {steps_per_sec:.2f}")
                else:
                    log_msg = f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, LLM Loss: {avg_llm_loss:.4f}"
                    if avg_contrastive_loss is not None:
                        log_msg += f", Contrastive Loss: {avg_contrastive_loss:.4f}"
                    if avg_mask_stats is not None:
                        log_msg += (
                            f", Mask edit_frac: {avg_mask_stats['edit_token_frac']:.4f}, "
                            f"edit_ce: {avg_mask_stats['edit_ce']:.4f}, "
                            f"bg_ce: {avg_mask_stats['bg_ce']:.4f}, "
                            f"weight_mean: {avg_mask_stats['weight_mean']:.4f}, "
                            f"weight_max: {avg_mask_stats['weight_max']:.4f}"
                        )
                    logger.info(f"{log_msg}, Train Steps/Sec: {steps_per_sec:.2f}")
                if rank==0 and args.use_wandb:
                    wandb_dic = {}
                    wandb_dic['train/global_step'] = train_steps
                    wandb_dic['train/loss'] = avg_loss
                    if avg_llm_loss is not None:
                        wandb_dic['train/llm_loss'] = avg_llm_loss
                    if avg_mask_stats is not None:
                        wandb_dic['train/mask_weight_mean'] = avg_mask_stats['weight_mean']
                        wandb_dic['train/mask_weight_max'] = avg_mask_stats['weight_max']
                        wandb_dic['train/mask_edit_token_frac'] = avg_mask_stats['edit_token_frac']
                        wandb_dic['train/edit_ce'] = avg_mask_stats['edit_ce']
                        wandb_dic['train/bg_ce'] = avg_mask_stats['bg_ce']
                    if args.use_distill:
                        wandb_dic['train/distill_loss'] = avg_distill_loss
                    if avg_contrastive_loss is not None:
                        wandb_dic['train/contrastive_loss'] = avg_contrastive_loss
                    wandb_dic['train/lr'] = current_lr
                    wandb.log(wandb_dic)
                # Reset monitoring variables:
                running_loss = 0
                running_llm_loss = 0
                running_distill_loss = 0
                running_contrastive_loss = 0
                running_weight_mean = 0
                running_weight_max = 0
                running_edit_token_frac = 0
                running_edit_ce = 0
                running_bg_ce = 0
                running_mask_stats_steps = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    if not args.no_compile:
                        base_model = model.module._orig_mod
                    else:
                        base_model = model.module
                    if args.use_lora:
                        checkpoint = {
                            "checkpoint_type": "lora_adapter",
                            "model": lora_state_dict(base_model, extra_trainable_keywords=extra_trainable),
                            "steps": train_steps,
                            "args": args,
                            "base_model_ckpt": args.gpt_ckpt,
                            "lora_target_modules": args.lora_target_modules,
                            "lora_extra_trainable": args.lora_extra_trainable,
                        }
                    else:
                        checkpoint = {
                            "checkpoint_type": "full_model",
                            "model": base_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "steps": train_steps,
                            "args": args
                        }
                    if not args.no_local_save:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                    # cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
                    # torch.save(checkpoint, cloud_checkpoint_path)
                    # logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")
                dist.barrier()

            if args.max_train_steps > 0 and train_steps >= args.max_train_steps:
                logger.info(f"Reached max_train_steps={args.max_train_steps}; stopping early.")
                break


        progress_bar.close()
        if args.max_train_steps > 0 and train_steps >= args.max_train_steps:
            break

    if rank==0:
        # output COMPLETE
        os.makedirs(args.output_dir, exist_ok=True)
        file_path = os.path.join(args.output_dir, 'COMPLETE')
        # Create the empty file
        with open(file_path, 'w') as f:
            pass  # Just create an empty file and close it

    logger.info("Done!")
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-distill", action='store_true')
    parser.add_argument("--distill-mode", type=str, choices=['dinov2', 'clip'], default=None)
    parser.add_argument("--distill-loss-weight", type=float, default=0.5, help="distill loss weight")
    parser.add_argument("--use-negative-contrastive", action='store_true', help="add wrong-instruction margin loss")
    parser.add_argument("--negative-contrastive-weight", type=float, default=0.1, help="weight for wrong-instruction margin loss")
    parser.add_argument("--negative-contrastive-margin", type=float, default=0.2, help="margin for correct-vs-wrong instruction CE")
    parser.add_argument("--use-mask-weighted-loss", action='store_true', help="upweight target tokens inside edit masks when masks are available")
    parser.add_argument("--lambda-edit", type=float, default=3.0, help="token CE weight for edit-region visual tokens")
    parser.add_argument("--lambda-bg", type=float, default=1.0, help="token CE weight for background visual tokens")
    parser.add_argument("--use-lora", action='store_true', help="train LoRA adapters instead of all GPT parameters")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", type=str, default="wqkv,wo,w1,w2,w3,cap_proj.fc1,cap_proj.fc2")
    parser.add_argument("--lora-extra-trainable", type=str, default="alignment", help="comma-separated parameter name fragments kept trainable with LoRA")
    parser.add_argument("--use-wandb", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default='./pretrained_models/vq_ds16_t2i.pt', help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i', 'edit'], default="edit")
    parser.add_argument("--gpt-mode", type=str, choices=['img_cls_emb', 'joint_cls_emb'], default=None)
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path", type=float, default=0.0, help="drop_path_rate of attention and ffn")
    parser.add_argument("--no-compile", action='store_true')
    parser.add_argument("--output-dir", type=str, default="checkpoints/test")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=512)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", type=str, default="constant", choices=["constant", "warmup_cosine"])
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10, help="optimizer-step interval for aggregating and uploading train metrics to wandb")
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--max-train-steps", type=int, default=0, help="optional optimizer-step cap for smoke tests")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--t5-path", type=str, default='pretrained_models/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--dataset-list", nargs='+', default=['seedxunsplash', 'pipe'])
    parser.add_argument("--multigendepth-path", type=str, default='data/MultiGen-20M_depth_HF')
    parser.add_argument("--multigendepth-prob", type=float, default=0.15)
    parser.add_argument("--multigencanny-path", type=str, default='data/MultiGen-20M_depth_HF')
    parser.add_argument("--multigencanny-prob", type=float, default=0.30)
    parser.add_argument("--conditionsegmentation-path", type=str, default='data/Condition_Segmentation')
    parser.add_argument("--conditionsegmentation-prob", type=float, default=0.45)
    parser.add_argument("--pipe-path", type=str, default='data/PIPE_HF')
    parser.add_argument("--pipe-prob", type=float, default=0.7)
    parser.add_argument("--seedxunsplash-path", type=str, default='data/Seedx_Unsplash_HF')
    parser.add_argument("--seedxunsplash-prob", type=float, default=1.0)
    parser.add_argument("--magicbrush-path", type=str, default='data/MagicBrush_HF')
    parser.add_argument("--magicbrush-prob", type=float, default=1.0)
    args = parser.parse_args()
    main(args)
