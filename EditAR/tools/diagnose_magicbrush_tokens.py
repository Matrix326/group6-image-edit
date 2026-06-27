import argparse
import csv
import json
import math
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

from autoregressive.models.generate_edit import top_k_top_p_filtering
from autoregressive.models.gpt_edit import GPT_models
from dataset.Edit_MagicBrush_eval import MagicBrush_Eval_Dataset
from language.t5 import T5Embedder
from tokenizer.tokenizer_image.vq_model import VQ_models


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def tensor_image_to_uint8(tensor):
    tensor = torch.clamp(tensor.detach().cpu(), -1, 1)
    array = (tensor.permute(1, 2, 0).numpy() + 1) * 127.5
    return np.clip(array, 0, 255).astype(np.uint8)


def encode_instruction(t5_model, text, device):
    tokens_and_mask = t5_model.tokenizer(
        text,
        max_length=120,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = tokens_and_mask["input_ids"].to(device)
    attention_mask = tokens_and_mask["attention_mask"].to(device)
    with torch.no_grad():
        return t5_model.model(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"].detach()


def save_triptych(path, source_tensor, target_tensor, pred_tensor, title):
    images = [
        tensor_image_to_uint8(source_tensor),
        tensor_image_to_uint8(target_tensor),
        tensor_image_to_uint8(pred_tensor),
    ]
    labels = ["source", "target", "prediction"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, image, label in zip(axes, images, labels):
        ax.imshow(image)
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_heatmap(path, values, title, cmap="viridis", vmin=None, vmax=None):
    grid = np.asarray(values).reshape(32, 32)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_lineplot(path, series, title, ylabel):
    fig, ax = plt.subplots(figsize=(8, 3))
    for name, values in series:
        ax.plot(np.arange(len(values)), values, label=name, linewidth=1)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("target token position")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if len(series) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_hist(path, series, title, xlabel):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for name, values in series:
        ax.hist(values, bins=50, alpha=0.55, label=name)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    if len(series) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def codebook_vectors(vq_model, indices):
    quant = vq_model.quantize.get_codebook_entry(indices.reshape(-1), None)
    return quant.float()


def embedding_stats(vq_model, source_indices, target_indices, pred_indices):
    src = codebook_vectors(vq_model, source_indices)
    tgt = codebook_vectors(vq_model, target_indices)
    pred = codebook_vectors(vq_model, pred_indices)
    return {
        "pred_target_l2": torch.linalg.vector_norm(pred - tgt, dim=-1).cpu().numpy(),
        "source_target_l2": torch.linalg.vector_norm(src - tgt, dim=-1).cpu().numpy(),
        "source_pred_l2": torch.linalg.vector_norm(src - pred, dim=-1).cpu().numpy(),
        "pred_target_cosine": F.cosine_similarity(pred, tgt, dim=-1).cpu().numpy(),
        "source_target_cosine": F.cosine_similarity(src, tgt, dim=-1).cpu().numpy(),
        "source_pred_cosine": F.cosine_similarity(src, pred, dim=-1).cpu().numpy(),
    }


def filtered_probs_from_logits(logits, temperature, top_k, top_p):
    scaled = logits / max(temperature, 1e-5)
    filtered = scaled.clone()
    if top_k > 0 or top_p < 1.0:
        filtered = top_k_top_p_filtering(filtered, top_k=top_k, top_p=top_p)
    probs = F.softmax(filtered, dim=-1)
    return scaled, filtered, probs


def collect_step_stats(logits, target_token, source_token, temperature, top_k, top_p, sample_logits, topk_save):
    raw_logits = logits.float()
    raw_probs = F.softmax(raw_logits, dim=-1)
    scaled_logits, filtered_logits, sample_probs = filtered_probs_from_logits(raw_logits, temperature, top_k, top_p)

    if sample_logits:
        pred_token = torch.multinomial(sample_probs, num_samples=1)
    else:
        pred_token = torch.argmax(sample_probs, dim=-1, keepdim=True)

    top_probs, top_ids = torch.topk(sample_probs, k=min(topk_save, sample_probs.shape[-1]), dim=-1)
    top_raw_logits = torch.gather(raw_logits, 1, top_ids)
    target_token_2d = target_token.view(-1, 1)
    source_token_2d = source_token.view(-1, 1)
    pred_token_2d = pred_token.view(-1, 1)

    target_raw_prob = torch.gather(raw_probs, 1, target_token_2d)
    target_sample_prob = torch.gather(sample_probs, 1, target_token_2d)
    pred_raw_prob = torch.gather(raw_probs, 1, pred_token_2d)
    pred_sample_prob = torch.gather(sample_probs, 1, pred_token_2d)
    source_raw_prob = torch.gather(raw_probs, 1, source_token_2d)
    source_sample_prob = torch.gather(sample_probs, 1, source_token_2d)
    target_raw_logit = torch.gather(raw_logits, 1, target_token_2d)
    pred_raw_logit = torch.gather(raw_logits, 1, pred_token_2d)

    target_rank = (raw_logits > target_raw_logit).sum(dim=-1) + 1
    entropy = -(raw_probs * raw_probs.clamp_min(1e-12).log()).sum(dim=-1)
    top1_prob, top1_id = torch.max(raw_probs, dim=-1)

    return pred_token, {
        "top_ids": top_ids.detach().cpu().numpy(),
        "top_sample_probs": top_probs.detach().cpu().numpy(),
        "top_raw_logits": top_raw_logits.detach().cpu().numpy(),
        "pred_token": pred_token.squeeze(1).detach().cpu().numpy(),
        "top1_token": top1_id.detach().cpu().numpy(),
        "target_raw_prob": target_raw_prob.squeeze(1).detach().cpu().numpy(),
        "target_sample_prob": target_sample_prob.squeeze(1).detach().cpu().numpy(),
        "pred_raw_prob": pred_raw_prob.squeeze(1).detach().cpu().numpy(),
        "pred_sample_prob": pred_sample_prob.squeeze(1).detach().cpu().numpy(),
        "source_raw_prob": source_raw_prob.squeeze(1).detach().cpu().numpy(),
        "source_sample_prob": source_sample_prob.squeeze(1).detach().cpu().numpy(),
        "target_raw_logit": target_raw_logit.squeeze(1).detach().cpu().numpy(),
        "pred_raw_logit": pred_raw_logit.squeeze(1).detach().cpu().numpy(),
        "target_rank": target_rank.detach().cpu().numpy(),
        "entropy": entropy.detach().cpu().numpy(),
        "top1_raw_prob": top1_prob.detach().cpu().numpy(),
    }


def stats_from_logits(logits, target_indices, source_indices, topk_save):
    raw_logits = logits.float()
    raw_probs = F.softmax(raw_logits, dim=-1)
    top_probs, top_ids = torch.topk(raw_probs, k=min(topk_save, raw_probs.shape[-1]), dim=-1)
    top_raw_logits = torch.gather(raw_logits, 2, top_ids)
    pred_raw_prob, pred_token = torch.max(raw_probs, dim=-1)

    target_token_3d = target_indices.unsqueeze(-1)
    source_token_3d = source_indices.unsqueeze(-1)
    pred_token_3d = pred_token.unsqueeze(-1)

    target_raw_prob = torch.gather(raw_probs, 2, target_token_3d).squeeze(-1)
    source_raw_prob = torch.gather(raw_probs, 2, source_token_3d).squeeze(-1)
    target_raw_logit = torch.gather(raw_logits, 2, target_token_3d).squeeze(-1)
    pred_raw_logit = torch.gather(raw_logits, 2, pred_token_3d).squeeze(-1)
    target_rank = (raw_logits > target_raw_logit.unsqueeze(-1)).sum(dim=-1) + 1
    entropy = -(raw_probs * raw_probs.clamp_min(1e-12).log()).sum(dim=-1)

    return pred_token, {
        "top_ids": top_ids.detach().cpu().numpy(),
        "top_sample_probs": top_probs.detach().cpu().numpy(),
        "top_raw_logits": top_raw_logits.detach().cpu().numpy(),
        "pred_token": pred_token.detach().cpu().numpy(),
        "top1_token": pred_token.detach().cpu().numpy(),
        "target_raw_prob": target_raw_prob.detach().cpu().numpy(),
        "target_sample_prob": target_raw_prob.detach().cpu().numpy(),
        "pred_raw_prob": pred_raw_prob.detach().cpu().numpy(),
        "pred_sample_prob": pred_raw_prob.detach().cpu().numpy(),
        "source_raw_prob": source_raw_prob.detach().cpu().numpy(),
        "source_sample_prob": source_raw_prob.detach().cpu().numpy(),
        "target_raw_logit": target_raw_logit.detach().cpu().numpy(),
        "pred_raw_logit": pred_raw_logit.detach().cpu().numpy(),
        "target_rank": target_rank.detach().cpu().numpy(),
        "entropy": entropy.detach().cpu().numpy(),
        "top1_raw_prob": pred_raw_prob.detach().cpu().numpy(),
    }


def clear_kv_caches(model):
    for layer in model.layers:
        layer.attention.kv_cache = None


@torch.no_grad()
def teacher_forcing_with_diagnostics(model, input_txt_embs, input_img_indices, input_mode, target_indices, topk_save):
    clear_kv_caches(model)
    seq_len = input_img_indices.shape[1] + input_txt_embs.shape[1] + target_indices.shape[1] - 1
    input_pos = torch.arange(0, seq_len, device=input_img_indices.device)
    logits, _, _ = model(
        input_txt_embs=input_txt_embs,
        input_img_indices=input_img_indices,
        edited_img_indices=target_indices,
        input_mode=input_mode,
        input_pos=input_pos,
        compute_loss=False,
    )
    logits = logits[:, model.block_size + model.cls_token_num - 1 :].contiguous()
    return stats_from_logits(logits, target_indices, input_img_indices, topk_save)


@torch.no_grad()
def generate_with_diagnostics(
    model,
    input_txt_embs,
    input_img_indices,
    input_mode,
    target_indices,
    max_new_tokens,
    cfg_scale,
    cfg_interval,
    temperature,
    top_k,
    top_p,
    sample_logits,
    topk_save,
):
    if model.model_type != "edit":
        raise ValueError("diagnostic generation currently supports edit model only")

    use_guidance = cfg_scale > 1.0
    if use_guidance:
        input_txt_null = torch.zeros_like(input_txt_embs) + model.cls_embedding.uncond_embedding
        input_txt_combined = torch.cat([input_txt_embs, input_txt_null])
        input_img_combined = torch.cat([input_img_indices, input_img_indices])
        input_mode_combined = torch.cat([input_mode, input_mode])
    else:
        input_txt_combined = input_txt_embs
        input_img_combined = input_img_indices
        input_mode_combined = input_mode

    batch_size = input_img_indices.shape[0]
    prefix_len = input_txt_combined.shape[1] + input_img_combined.shape[1]
    max_seq_length = prefix_len + max_new_tokens
    cache_batch_size = batch_size * 2 if use_guidance else batch_size
    with torch.device(input_img_indices.device):
        model.setup_caches(
            max_batch_size=cache_batch_size,
            max_seq_length=max_seq_length,
            dtype=model.tok_embeddings.weight.dtype,
        )

    seq = torch.empty((batch_size, max_new_tokens), dtype=torch.int64, device=input_img_indices.device)
    records = []

    input_pos = torch.arange(0, prefix_len, device=input_img_indices.device)
    logits, _, _ = model(
        input_txt_embs=input_txt_combined,
        input_img_indices=input_img_combined,
        edited_img_indices=None,
        input_mode=input_mode_combined,
        input_pos=input_pos,
    )
    if use_guidance:
        cond_logits, null_logits = torch.split(logits, len(logits) // 2, dim=0)
        logits = null_logits + (cond_logits - null_logits) * cfg_scale
    target_token = target_indices[:, 0]
    source_token = input_img_indices[:, 0]
    next_token, step_stats = collect_step_stats(
        logits[:, -1, :],
        target_token,
        source_token,
        temperature,
        top_k,
        top_p,
        sample_logits,
        topk_save,
    )
    seq[:, 0:1] = next_token
    records.append(step_stats)

    input_pos = torch.tensor([prefix_len], device=input_img_indices.device, dtype=torch.int)
    cur_token = next_token
    for i in range(1, max_new_tokens):
        logits, _, _ = model(
            input_txt_embs=None,
            input_img_indices=None,
            edited_img_indices=torch.cat([cur_token, cur_token]) if use_guidance else cur_token,
            input_pos=input_pos,
        )
        guidance_enabled = cfg_scale > 1.0
        if cfg_interval > -1 and i > cfg_interval:
            guidance_enabled = False
        if use_guidance:
            cond_logits, null_logits = torch.split(logits, len(logits) // 2, dim=0)
            logits = null_logits + (cond_logits - null_logits) * cfg_scale if guidance_enabled else cond_logits

        target_token = target_indices[:, i]
        source_token = input_img_indices[:, i]
        next_token, step_stats = collect_step_stats(
            logits[:, -1, :],
            target_token,
            source_token,
            temperature,
            top_k,
            top_p,
            sample_logits,
            topk_save,
        )
        seq[:, i : i + 1] = next_token
        records.append(step_stats)
        cur_token = next_token
        input_pos += 1

    stacked = {}
    for key in records[0].keys():
        stacked[key] = np.stack([record[key] for record in records], axis=1)
    return seq, stacked


def load_models(args, device):
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
    ).to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint

    precision = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.mixed_precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size**2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        model_mode=args.gpt_mode,
        resid_dropout_p=args.dropout_p,
        ffn_dropout_p=args.dropout_p,
        token_dropout_p=args.token_dropout_p,
        distill_mode=args.distill_mode,
    ).to(device=device, dtype=precision)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    if "model" in checkpoint:
        model_weight = checkpoint["model"]
    elif "module" in checkpoint:
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise ValueError("cannot find model weights in checkpoint")
    gpt_model.load_state_dict(model_weight, strict=False)
    del checkpoint
    gpt_model.eval()
    return vq_model, gpt_model, precision, latent_size


def summarize_sample(sample_dir, sample_id, source_indices, target_indices, pred_indices, stats, emb, filename="summary.json"):
    pred = pred_indices.cpu().numpy().reshape(-1)
    source = source_indices.cpu().numpy().reshape(-1)
    target = target_indices.cpu().numpy().reshape(-1)
    target_prob = stats["target_raw_prob"][0]
    target_rank = stats["target_rank"][0]
    entropy = stats["entropy"][0]
    top1 = stats["top1_token"][0]
    pred_prob = stats["pred_raw_prob"][0]

    summary = {
        "sample_id": int(sample_id),
        "token_count": int(target.shape[0]),
        "pred_target_token_match_rate": float(np.mean(pred == target)),
        "top1_target_match_rate": float(np.mean(top1 == target)),
        "pred_source_token_match_rate": float(np.mean(pred == source)),
        "source_target_token_match_rate": float(np.mean(source == target)),
        "target_raw_prob_mean": float(np.mean(target_prob)),
        "target_raw_prob_median": float(np.median(target_prob)),
        "target_rank_mean": float(np.mean(target_rank)),
        "target_rank_median": float(np.median(target_rank)),
        "target_rank_le_10_rate": float(np.mean(target_rank <= 10)),
        "target_rank_le_100_rate": float(np.mean(target_rank <= 100)),
        "entropy_mean": float(np.mean(entropy)),
        "pred_raw_prob_mean": float(np.mean(pred_prob)),
    }
    for name, values in emb.items():
        summary[f"{name}_mean"] = float(np.mean(values))
        summary[f"{name}_median"] = float(np.median(values))
    save_json(os.path.join(sample_dir, filename), summary)
    return summary


def write_token_csv(path, source_indices, target_indices, pred_indices, stats, emb):
    source = source_indices.cpu().numpy().reshape(-1)
    target = target_indices.cpu().numpy().reshape(-1)
    pred = pred_indices.cpu().numpy().reshape(-1)
    rows = []
    for i in range(target.shape[0]):
        rows.append(
            {
                "pos": i,
                "row": i // 32,
                "col": i % 32,
                "source_token": int(source[i]),
                "target_token": int(target[i]),
                "pred_token": int(pred[i]),
                "top1_token": int(stats["top1_token"][0, i]),
                "pred_eq_target": int(pred[i] == target[i]),
                "top1_eq_target": int(stats["top1_token"][0, i] == target[i]),
                "pred_eq_source": int(pred[i] == source[i]),
                "target_raw_prob": float(stats["target_raw_prob"][0, i]),
                "target_sample_prob": float(stats["target_sample_prob"][0, i]),
                "pred_raw_prob": float(stats["pred_raw_prob"][0, i]),
                "pred_sample_prob": float(stats["pred_sample_prob"][0, i]),
                "source_raw_prob": float(stats["source_raw_prob"][0, i]),
                "target_raw_logit": float(stats["target_raw_logit"][0, i]),
                "pred_raw_logit": float(stats["pred_raw_logit"][0, i]),
                "target_rank": int(stats["target_rank"][0, i]),
                "entropy": float(stats["entropy"][0, i]),
                "pred_target_l2": float(emb["pred_target_l2"][i]),
                "source_target_l2": float(emb["source_target_l2"][i]),
                "source_pred_l2": float(emb["source_pred_l2"][i]),
                "pred_target_cosine": float(emb["pred_target_cosine"][i]),
            }
        )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_sample_artifacts(
    sample_dir,
    sample_id,
    source_img,
    target_img,
    pred_img,
    source_indices,
    target_indices,
    pred_indices,
    stats,
    emb,
    topk_save,
    prefix="",
    save_images=True,
):
    if save_images:
        save_triptych(
            os.path.join(sample_dir, f"{prefix}image_triptych.png"),
            source_img,
            target_img,
            pred_img,
            f"MagicBrush sample {sample_id}",
        )
        Image.fromarray(tensor_image_to_uint8(pred_img)).save(os.path.join(sample_dir, f"{prefix}prediction.png"))
        Image.fromarray(tensor_image_to_uint8(source_img)).save(os.path.join(sample_dir, "source.png"))
        Image.fromarray(tensor_image_to_uint8(target_img)).save(os.path.join(sample_dir, "target.png"))

    source = source_indices.cpu().numpy().reshape(-1)
    target = target_indices.cpu().numpy().reshape(-1)
    pred = pred_indices.cpu().numpy().reshape(-1)
    save_heatmap(os.path.join(sample_dir, f"{prefix}token_pred_eq_target.png"), pred == target, f"{prefix}pred token == target token", cmap="gray", vmin=0, vmax=1)
    save_heatmap(os.path.join(sample_dir, f"{prefix}token_source_eq_target.png"), source == target, "source token == target token", cmap="gray", vmin=0, vmax=1)
    save_heatmap(os.path.join(sample_dir, f"{prefix}target_rank_log10.png"), np.log10(stats["target_rank"][0]), f"{prefix}log10 target rank")
    save_heatmap(os.path.join(sample_dir, f"{prefix}target_raw_prob.png"), stats["target_raw_prob"][0], f"{prefix}raw softmax prob of target token")
    save_heatmap(os.path.join(sample_dir, f"{prefix}entropy.png"), stats["entropy"][0], f"{prefix}raw softmax entropy")
    save_heatmap(os.path.join(sample_dir, f"{prefix}embed_pred_target_l2.png"), emb["pred_target_l2"], f"{prefix}codebook embedding L2: pred vs target")
    save_heatmap(os.path.join(sample_dir, f"{prefix}embed_source_target_l2.png"), emb["source_target_l2"], "codebook embedding L2: source vs target")
    save_heatmap(os.path.join(sample_dir, f"{prefix}embed_source_pred_l2.png"), emb["source_pred_l2"], f"{prefix}codebook embedding L2: source vs pred")
    save_heatmap(os.path.join(sample_dir, f"{prefix}embed_pred_target_cosine.png"), emb["pred_target_cosine"], f"{prefix}codebook cosine: pred vs target", vmin=-1, vmax=1)
    if prefix == "teacher_forcing_":
        save_heatmap(os.path.join(sample_dir, "teacher_forcing_top1_pred_target_l2.png"), emb["pred_target_l2"], "teacher forcing top1 embedding L2: pred vs target")

    save_lineplot(
        os.path.join(sample_dir, f"{prefix}prob_curves.png"),
        [
            ("target raw prob", stats["target_raw_prob"][0]),
            ("pred raw prob", stats["pred_raw_prob"][0]),
            ("source raw prob", stats["source_raw_prob"][0]),
        ],
        "selected token probabilities from raw softmax",
        "probability",
    )
    save_lineplot(
        os.path.join(sample_dir, f"{prefix}rank_entropy_curves.png"),
        [
            ("log10 target rank", np.log10(stats["target_rank"][0])),
            ("entropy", stats["entropy"][0]),
        ],
        "target rank and distribution entropy",
        "value",
    )
    save_hist(
        os.path.join(sample_dir, f"{prefix}embedding_l2_hist.png"),
        [
            ("pred-target", emb["pred_target_l2"]),
            ("source-target", emb["source_target_l2"]),
            ("source-pred", emb["source_pred_l2"]),
        ],
        "codebook embedding L2 distributions",
        "L2 distance",
    )

    npz_name = "teacher_forcing_diagnostics.npz" if prefix == "teacher_forcing_" else f"{prefix}token_diagnostics.npz"
    np.savez_compressed(
        os.path.join(sample_dir, npz_name),
        source_tokens=source,
        target_tokens=target,
        pred_tokens=pred,
        top_ids=stats["top_ids"][0, :, :topk_save],
        top_sample_probs=stats["top_sample_probs"][0, :, :topk_save],
        top_raw_logits=stats["top_raw_logits"][0, :, :topk_save],
        target_raw_prob=stats["target_raw_prob"][0],
        target_sample_prob=stats["target_sample_prob"][0],
        pred_raw_prob=stats["pred_raw_prob"][0],
        pred_sample_prob=stats["pred_sample_prob"][0],
        source_raw_prob=stats["source_raw_prob"][0],
        target_rank=stats["target_rank"][0],
        entropy=stats["entropy"][0],
        **emb,
    )
    write_token_csv(os.path.join(sample_dir, f"{prefix}token_metrics.csv"), source_indices, target_indices, pred_indices, stats, emb)


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ensure_dir(args.output_dir)
    vq_model, gpt_model, precision, latent_size = load_models(args, device)
    print("models loaded")

    t5_model = T5Embedder(
        device=device,
        local_cache=True,
        cache_dir=args.t5_path,
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )
    dataset_args = SimpleNamespace(output_dir=args.output_dir, gpt_ckpt=args.gpt_ckpt)
    dataset = MagicBrush_Eval_Dataset(
        args=dataset_args,
        dataset_path=args.magicbrush_path,
        llm_tokenizer=t5_model.tokenizer,
        mode="dev",
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    summaries = []
    for idx, batch in enumerate(loader):
        if idx >= args.max_samples:
            break
        sample_id = int(batch["index"][0].item())
        sample_dir = os.path.join(args.output_dir, f"sample_{sample_id:05d}")
        ensure_dir(sample_dir)
        instruction = batch["_edit_txt"][0]
        print(f"[{idx + 1}/{args.max_samples}] sample={sample_id} instruction={instruction}")

        input_img = batch["input_img"].to(device, non_blocking=True)
        target_img = batch["edited_img"].to(device, non_blocking=True)
        input_mode = batch["mode"].to(device, non_blocking=True)
        input_txt_embs = encode_instruction(t5_model, instruction, device)

        with torch.no_grad():
            _, _, [_, _, source_indices] = vq_model.encode(input_img)
            _, _, [_, _, target_indices] = vq_model.encode(target_img)
            source_indices = source_indices.reshape(input_img.shape[0], -1)
            target_indices = target_indices.reshape(target_img.shape[0], -1)

        with open(os.path.join(sample_dir, "instruction.txt"), "w", encoding="utf-8") as f:
            f.write(instruction + "\n")
        sample_summary = {"sample_id": sample_id, "instruction": instruction}

        if args.diagnostic_mode in ("ar", "both"):
            with torch.no_grad():
                pred_indices, stats = generate_with_diagnostics(
                    gpt_model,
                    input_txt_embs,
                    source_indices,
                    input_mode,
                    target_indices,
                    max_new_tokens=latent_size**2,
                    cfg_scale=args.cfg_scale,
                    cfg_interval=args.cfg_interval,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    sample_logits=not args.greedy,
                    topk_save=args.topk_save,
                )
                qzshape = [input_img.shape[0], args.codebook_embed_dim, latent_size, latent_size]
                pred_img = vq_model.decode_code(pred_indices, qzshape)

            emb = embedding_stats(vq_model, source_indices[0], target_indices[0], pred_indices[0])
            save_sample_artifacts(
                sample_dir,
                sample_id,
                input_img[0],
                target_img[0],
                pred_img[0],
                source_indices[0],
                target_indices[0],
                pred_indices[0],
                stats,
                emb,
                args.topk_save,
            )
            ar_summary = summarize_sample(sample_dir, sample_id, source_indices[0], target_indices[0], pred_indices[0], stats, emb, filename="ar_summary.json")
            sample_summary["ar"] = ar_summary

        if args.diagnostic_mode in ("teacher_forcing", "both"):
            with torch.no_grad():
                tf_pred_indices, tf_stats = teacher_forcing_with_diagnostics(
                    gpt_model,
                    input_txt_embs,
                    source_indices,
                    input_mode,
                    target_indices,
                    args.topk_save,
                )
            tf_emb = embedding_stats(vq_model, source_indices[0], target_indices[0], tf_pred_indices[0])
            save_sample_artifacts(
                sample_dir,
                sample_id,
                input_img[0],
                target_img[0],
                input_img[0],
                source_indices[0],
                target_indices[0],
                tf_pred_indices[0],
                tf_stats,
                tf_emb,
                args.topk_save,
                prefix="teacher_forcing_",
                save_images=False,
            )
            tf_summary = summarize_sample(
                sample_dir,
                sample_id,
                source_indices[0],
                target_indices[0],
                tf_pred_indices[0],
                tf_stats,
                tf_emb,
                filename="teacher_forcing_summary.json",
            )
            sample_summary["teacher_forcing"] = tf_summary

        if "ar" in sample_summary and "teacher_forcing" in sample_summary:
            ar_summary = sample_summary["ar"]
            tf_summary = sample_summary["teacher_forcing"]
            sample_summary["gap"] = {
                "target_rank_median_ar_minus_tf": ar_summary["target_rank_median"] - tf_summary["target_rank_median"],
                "target_prob_mean_tf_minus_ar": tf_summary["target_raw_prob_mean"] - ar_summary["target_raw_prob_mean"],
                "top1_target_match_tf_minus_ar": tf_summary["top1_target_match_rate"] - ar_summary["top1_target_match_rate"],
                "pred_target_l2_mean_ar_minus_tf": ar_summary["pred_target_l2_mean"] - tf_summary["pred_target_l2_mean"],
            }

        save_json(os.path.join(sample_dir, "summary.json"), sample_summary)
        summaries.append(sample_summary)
        print_parts = []
        if "ar" in sample_summary:
            s = sample_summary["ar"]
            print_parts.append(
                "AR match={:.4f} top1_match={:.4f} rank_med={:.1f} pred_tgt_l2={:.4f}".format(
                    s["pred_target_token_match_rate"],
                    s["top1_target_match_rate"],
                    s["target_rank_median"],
                    s["pred_target_l2_mean"],
                )
            )
        if "teacher_forcing" in sample_summary:
            s = sample_summary["teacher_forcing"]
            print_parts.append(
                "TF top1_match={:.4f} rank_med={:.1f} pred_tgt_l2={:.4f}".format(
                    s["top1_target_match_rate"],
                    s["target_rank_median"],
                    s["pred_target_l2_mean"],
                )
            )
        print("  " + " | ".join(print_parts))

    aggregate = {"samples": summaries}
    if summaries:
        for section in ("ar", "teacher_forcing", "gap"):
            section_rows = [sample[section] for sample in summaries if section in sample]
            if not section_rows:
                continue
            numeric_keys = [key for key, value in section_rows[0].items() if isinstance(value, (int, float))]
            aggregate.setdefault("mean", {})[section] = {
                key: float(np.mean([sample[key] for sample in section_rows]))
                for key in numeric_keys
                if key != "sample_id"
            }
    save_json(os.path.join(args.output_dir, "summary.json"), aggregate)

    csv_path = os.path.join(args.output_dir, "summary.csv")
    if summaries:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            rows = []
            for sample in summaries:
                row = {"sample_id": sample["sample_id"], "instruction": sample["instruction"]}
                for section in ("ar", "teacher_forcing", "gap"):
                    if section not in sample:
                        continue
                    for key, value in sample[section].items():
                        if isinstance(value, (int, float)):
                            row[f"{section}.{key}"] = value
                rows.append(row)
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"done: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="outputs/diagnostics/magicbrush8_baseline")
    parser.add_argument("--magicbrush-path", type=str, default="./data/MagicBrush_HF")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--diagnostic-mode", type=str, choices=["ar", "teacher_forcing", "both"], default="both")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distill-mode", type=str, choices=["dinov2", "clip", "clipseg"], default=None)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default="./pretrained_models/vq_ds16_t2i.pt")
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, required=True)
    parser.add_argument("--gpt-type", type=str, choices=["c2i", "t2i", "edit"], default="edit")
    parser.add_argument("--gpt-mode", type=str, choices=["img_cls_emb", "joint_cls_emb"], default="joint_cls_emb")
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--cls-token-num", type=int, default=120)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--token-dropout-p", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=512)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--cfg-interval", type=int, default=-1)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true", help="use argmax instead of multinomial sampling")
    parser.add_argument("--topk-save", type=int, default=64)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--t5-path", type=str, default="pretrained_models/t5-ckpt")
    parser.add_argument("--t5-model-type", type=str, default="flan-t5-xl")
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    args = parser.parse_args()
    main(args)
