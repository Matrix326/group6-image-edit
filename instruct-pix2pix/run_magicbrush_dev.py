import os
import json
import math
import torch
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from torch import autocast
import k_diffusion as K

# 从你的本地 edit_cli.py 中直接引入模型加载类与 CFG 控制器
from edit_cli import load_model_from_config, CFGDenoiser

def main():
    # 1. 路径与参数配置
    DATA_DIR = "/share/home/group6/Project/qwen-image-edit-baseline/benchmarks/magicbrush/prepared_full/dev"
    OUTPUT_DIR = "./results/magicbrush_baseline_dev_outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 2. 读取 manifest.json
    json_path = os.path.join(DATA_DIR, "manifest.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)
        
    # 确保脚本里写的是这三行（强行绑定到它能看到的“第0张卡”）
    device_id = 0
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    
    config_path = "./configs/generate.yaml"
    ckpt_path = "./checkpoints/instruct-pix2pix-00-22000.ckpt"
    
    print("正在加载【纯净 Baseline 模型】...")
    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt_path)
    model.eval().to(device)
    
    # 4. 完美复刻 k-diffusion 独有的双层包装结构
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])
    
    print(f"点火成功！开始在 GPU 4 上进行 {len(data_list)} 组 Baseline 批量推理...")
    
    # 5. 评测主循环
    for item in tqdm(data_list):
        img_id = item["id"]
        prompt = item["instruction"]
        rel_input_path = item["input_image"]
        
        input_path = os.path.join(DATA_DIR, rel_input_path)
        output_path = os.path.join(OUTPUT_DIR, f"{img_id}_output.png")
        
        if os.path.exists(output_path):
            continue  # 断点续传
            
        try:
            # --- 完美的 k_diffusion 图像尺寸规整化预处理 ---
            input_image = Image.open(input_path).convert("RGB")
            width, height = input_image.size
            factor = 512 / max(width, height)
            factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
            width = int((width * factor) // 64) * 64
            height = int((height * factor) // 64) * 64
            input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)
            
            # --- 运行模型推理 ---
            with torch.no_grad(), autocast("cuda"), model.ema_scope():
                cond = {}
                cond["c_crossattn"] = [model.get_learned_conditioning([prompt])]
                
                img_tensor = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
                img_tensor = rearrange(img_tensor, "h w c -> 1 c h w").to(device)
                cond["c_concat"] = [model.encode_first_stage(img_tensor).mode()]
                
                uncond = {}
                uncond["c_crossattn"] = [null_token]
                uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]
                
                # 获取 K-diffusion 的 euler 时间步长表
                sigmas = model_wrap.get_sigmas(100) # steps=100
                
                extra_args = {
                    "cond": cond,
                    "uncond": uncond,
                    "text_cfg_scale": 7.5,
                    "image_cfg_scale": 1.5,
                }
                
                # 固定 seed=42 保证可复现性，大作业标准操作
                torch.manual_seed(42)
                z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
                
                # 真正调用 K-diffusion 核心去噪采样器
                z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args, disable=True)
                
                # VAE 解码与还原保存
                x = model.decode_first_stage(z)
                x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
                x = 255.0 * rearrange(x, "1 c h w -> h w c")
                
                edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
                edited_image.save(output_path)
                
        except Exception as e:
            print(f"\n[❌ 错误] 样本 {img_id} 运行失败: {e}")
            continue

    print(f"\n🔥 革命成功！原版 Baseline 推理全部完成，结果在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()