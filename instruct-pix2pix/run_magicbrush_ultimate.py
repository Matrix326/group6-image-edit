import os
import json
import math
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from torch import autocast
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import k_diffusion as K
from edit_cli import load_model_from_config, CFGDenoiser

# =====================================================================
# ⚡️ 核心改进：K-Diffusion 步级隐空间拦截拦截器（修复 P2P 崩溃的终极武器）
# =====================================================================
class P2PLatentBlender:
    def __init__(self, mask_tensor, z_orig, total_steps=100):
        # 对掩膜做轻微的高斯模糊，防止边缘过于生硬
        self.mask = TF.gaussian_blur(mask_tensor, kernel_size=[3, 3], sigma=[0.5, 0.5])
        self.z_orig = z_orig
        self.total_steps = total_steps

    def __call__(self, info):
        z = info["x"]              # 当前时间步模型生成的隐空间状态
        sigma = info["sigma"]      # 当前时间步的噪声标准差
        step_index = info["i"]     # 当前是第几步
        
        # MagicBrush Mask 中：1.0 代表要修改的区域，0.0 代表要保留的区域
        # 我们定义 preserve_mask：1.0 代表绝对要锁死的区域，0.0 代表放开修改的区域
        preserve_mask = 1.0 - self.mask
        
        # ⚡️ 动态时序解锁（边缘缝合核心）：
        # 前 85% 的时间步，死死锁住背景；最后 15 步（接近出图时），逐渐松开锁死权重
        # 这样模型能用最后的微调能力，把修改区（如篮子）和保护区（如桌面）的边缘进行自然缝合！
        if step_index > int(self.total_steps * 0.85):
            decay = 1.0 - (step_index - int(self.total_steps * 0.85)) / (self.total_steps * 0.15)
            preserve_mask = preserve_mask * max(decay, 0.0)
            
        # 给原始图像的 Latent 加上当前步同等强度的噪声，使其处于同一分布
        noise = torch.randn_like(self.z_orig)
        z_orig_noisy = self.z_orig + noise * sigma
        
        # ⚡️ 物理融合：保护区用带有标准噪声的原图强行覆盖，修改区沿用模型生成的特征
        mutated_z = z * (1.0 - preserve_mask) + z_orig_noisy * preserve_mask
        
        # 内存指针写回，直接干预 K-Diffusion 的下一步迭代
        z.copy_(mutated_z)


def main():
    DATA_DIR = "/share/home/group6/Project/qwen-image-edit-baseline/benchmarks/magicbrush/prepared_full/dev"
    # 新建一个纯净的输出目录
    OUTPUT_DIR = "./results/magicbrush_p2p_perfect_outputs" 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(os.path.join(DATA_DIR, "manifest.json"), "r", encoding="utf-8") as f:
        data_list = json.load(f)
        
    device_id = 0
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    
    # 3. 载入你先前训练好的重训权重（或者官方原始权重皆可，推荐使用你重训出来的ckpt）
    config_path = "./configs/generate.yaml"
    ckpt_path = "./checkpoints/ip2p_finetuned_ours/ip2p_magicbrush_alignment.ckpt" 
    
    print("正在加载优化版 P2P 核心网络...")
    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt_path)
    model.eval().to(device)
    
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])
    
    print(f"🚀 P2P 步级拦截系统就位！正在批量校正前 20 组样本...")
    
    for item in tqdm(data_list[:20]):
        img_id = item["id"]
        prompt = item["instruction"]
        input_path = os.path.join(DATA_DIR, item["input_image"])
        mask_path = os.path.join(DATA_DIR, item["mask_image"])
        output_path = os.path.join(OUTPUT_DIR, f"{img_id}_perfect.png")
        
        if os.path.exists(output_path): continue  
            
        try:
            # 尺寸预处理
            input_image = Image.open(input_path).convert("RGB")
            width, height = input_image.size
            factor = 512 / max(width, height)
            factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
            width = int((width * factor) // 64) * 64
            height = int((height * factor) // 64) * 64
            input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)
            
            # 读取当前图的真实二值掩膜，并缩放到 Latent 隐空间尺寸 (64x64 左右)
            mask_image = Image.open(mask_path).convert("L").resize((width // 8, height // 8), Image.Resampling.NEAREST)
            mask_np = np.array(mask_image).astype(np.float32) / 255.0
            mask_tensor = torch.from_numpy(mask_np[None, None, ...]).to(device)
            
            with torch.no_grad(), autocast("cuda"), model.ema_scope():
                cond = {"c_crossattn": [model.get_learned_conditioning([prompt])]}
                img_tensor = (2 * torch.tensor(np.array(input_image)).float() / 255 - 1).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # 编码原图的隐空间状态，留给 Blender 做每一步的基准对齐
                z_orig = model.encode_first_stage(img_tensor).mode()
                cond["c_concat"] = [z_orig]
                
                uncond = {"c_crossattn": [null_token], "c_concat": [torch.zeros_like(z_orig)]}
                
                # 总步数设为 100 步，确保质量
                total_steps = 100
                sigmas = model_wrap.get_sigmas(total_steps)
                
                # ⚡️ 核心：恢复官方最稳定、绝不崩溃的黄金平衡参数
                extra_args = {
                    "cond": cond,
                    "uncond": uncond,
                    "text_cfg_scale": 7.5,
                    "image_cfg_scale": 1.5,
                }
                
                torch.manual_seed(42)
                z = torch.randn_like(z_orig) * sigmas[0]
                
                # ⚡️ 实例化拦截器，并在 sample 采样中通过 callback 强行挂载
                blender = P2PLatentBlender(mask_tensor, z_orig, total_steps=total_steps)
                z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args, callback=blender, disable=True)
                
                # 解码输出
                x = torch.clamp((model.decode_first_stage(z) + 1.0) / 2.0, min=0.0, max=1.0)
                Image.fromarray((255.0 * rearrange(x, "1 c h w -> h w c")).type(torch.uint8).cpu().numpy()).save(output_path)
                
        except Exception as e:
            print(f"\n[❌ 错误] 样本 {img_id}: {e}")
            continue

    print(f"\n🏆 校正完成！完美解耦的 P2P 纯净结果已存放在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
    