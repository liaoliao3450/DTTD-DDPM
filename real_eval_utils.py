"""
实验评估工具函数
用于Table 1, Table 2等实验脚本
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from scipy import signal


def ensure_output_dir(output_dir: str = "paper_results/real_experiments") -> str:
    """确保输出目录存在"""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_checkpoint_mapping() -> Dict[str, str]:
    """获取模型名称到检查点路径的映射"""
    return {
        "DTTD-DDPM": "checkpoints/bci2a_enhanced/best_model.pth",
        "CVAE": "checkpoints/bci2a/baseline_cvae/best_model.pth",
        "cGAN": "checkpoints/bci2a_enhanced/baseline_cgan/best_model.pth",
        "Simple-DDPM": "checkpoints/bci2a/baseline_simple_ddpm/best_model.pth",
    }


def load_reconstruction_model(
    model_name: str,
    config: dict,
    device: torch.device
) -> Optional[nn.Module]:
    """加载重建模型"""
    checkpoint_mapping = get_checkpoint_mapping()
    
    if model_name not in checkpoint_mapping:
        print(f"[WARN] Unknown model: {model_name}")
        return None
    
    checkpoint_path = checkpoint_mapping[model_name]
    if not os.path.exists(checkpoint_path):
        print(f"[WARN] Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        if model_name == "DTTD-DDPM":
            from models import DTTDEnhanced
            model = DTTDEnhanced(config["model"]).to(device)
        elif model_name == "CVAE":
            from models.baselines import CVAE
            model = CVAE(
                input_channels=config["model"].get("input_channels", 9),
                output_channels=config["model"].get("output_channels", 22),
                time_steps=config["model"].get("time_steps", 1000),
                num_classes=config["model"].get("num_classes", 4),
                latent_dim=128
            ).to(device)
        elif model_name == "cGAN":
            from models.baselines import ConditionalGAN
            model = ConditionalGAN(
                input_channels=config["model"].get("input_channels", 9),
                output_channels=config["model"].get("output_channels", 22),
                time_steps=config["model"].get("time_steps", 1000),
                num_classes=config["model"].get("num_classes", 4)
            ).to(device)
        elif model_name == "Simple-DDPM":
            from models.baselines import SimpleDDPM
            model = SimpleDDPM(
                input_channels=config["model"].get("input_channels", 9),
                output_channels=config["model"].get("output_channels", 22),
                time_steps=config["model"].get("time_steps", 1000),
                num_classes=config["model"].get("num_classes", 4)
            ).to(device)
        else:
            return None
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        
        # 处理cGAN的特殊情况
        if model_name == "cGAN":
            generator_state_dict = {}
            for key, value in state_dict.items():
                if not key.startswith("generator."):
                    generator_state_dict[f"generator.{key}"] = value
                else:
                    generator_state_dict[key] = value
            model.load_state_dict(generator_state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
        
        model.eval()
        return model
        
    except Exception as e:
        print(f"[ERROR] Failed to load {model_name}: {e}")
        return None



def generate_samples_real(
    model: nn.Module,
    model_name: str,
    dataloader,
    device: torch.device,
    num_samples: int = 200
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """生成重建样本"""
    generated_list = []
    target_list = []
    labels_list = []
    
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc=f"Generating {model_name}"):
            # reconstruction_mode返回4个值: target_data, channel_indices, labels, subject_ids
            if len(batch_data) == 4:
                target_data, channel_indices, labels, subject_ids = batch_data
            else:
                target_data, channel_indices, labels = batch_data
            
            # 提取输入通道
            ch_idx = channel_indices[0].tolist() if channel_indices.dim() > 1 else channel_indices.tolist()
            input_data = target_data[:, ch_idx, :].to(device)
            labels = labels.to(device)
            target_data = target_data.to(device)
            
            # 根据模型类型生成
            if model_name == "DTTD-DDPM":
                noise_level = 0.01
                noisy_input = input_data + torch.randn_like(input_data) * noise_level
                t = torch.zeros(input_data.size(0), device=device, dtype=torch.long)
                generated = model(noisy_input, t, labels)
            elif model_name == "CVAE":
                generated, _, _ = model(input_data, labels)
            elif model_name == "cGAN":
                z = torch.randn(input_data.size(0), 128, device=device)
                generated = model(z, labels, input_data)
            elif model_name == "Simple-DDPM":
                generated = model.reconstruct(input_data, labels, num_inference_steps=1, noise_level=0.02)
            else:
                continue
            
            generated_list.append(generated.cpu())
            target_list.append(target_data.cpu())
            labels_list.append(labels.cpu())
            
            if len(generated_list) * dataloader.batch_size >= num_samples:
                break
    
    generated = torch.cat(generated_list, dim=0)[:num_samples]
    target = torch.cat(target_list, dim=0)[:num_samples]
    labels = torch.cat(labels_list, dim=0)[:num_samples]
    
    return generated, target, labels


def compute_reconstruction_metrics(
    generated: torch.Tensor,
    target: torch.Tensor,
    fs: int = 250
) -> Dict[str, float]:
    """计算重建质量指标"""
    gen_np = generated.numpy()
    tgt_np = target.numpy()
    
    # MSE
    mse = np.mean((gen_np - tgt_np) ** 2)
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # 拓扑相似度 (通道相关矩阵的相关性)
    topo_sims = []
    for i in range(len(gen_np)):
        gen_corr = np.corrcoef(gen_np[i])
        tgt_corr = np.corrcoef(tgt_np[i])
        # 取上三角
        gen_upper = gen_corr[np.triu_indices(gen_corr.shape[0], k=1)]
        tgt_upper = tgt_corr[np.triu_indices(tgt_corr.shape[0], k=1)]
        if len(gen_upper) > 0 and len(tgt_upper) > 0:
            corr = np.corrcoef(gen_upper, tgt_upper)[0, 1]
            if not np.isnan(corr):
                topo_sims.append(corr)
    topo_sim = np.mean(topo_sims) if topo_sims else 0.0
    
    # PSD相关性
    psd_corrs = []
    for i in range(len(gen_np)):
        for ch in range(gen_np.shape[1]):
            f_gen, psd_gen = signal.welch(gen_np[i, ch], fs=fs, nperseg=min(256, gen_np.shape[2]))
            f_tgt, psd_tgt = signal.welch(tgt_np[i, ch], fs=fs, nperseg=min(256, tgt_np.shape[2]))
            if len(psd_gen) > 0 and len(psd_tgt) > 0:
                corr = np.corrcoef(psd_gen, psd_tgt)[0, 1]
                if not np.isnan(corr):
                    psd_corrs.append(corr)
    psd_corr = np.mean(psd_corrs) if psd_corrs else 0.0
    
    # 频率相似度 (各频段能量相关性)
    freq_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    band_sims = []
    for i in range(min(50, len(gen_np))):  # 采样计算
        for ch in range(gen_np.shape[1]):
            f, psd_gen = signal.welch(gen_np[i, ch], fs=fs, nperseg=min(256, gen_np.shape[2]))
            _, psd_tgt = signal.welch(tgt_np[i, ch], fs=fs, nperseg=min(256, tgt_np.shape[2]))
            
            gen_bands = []
            tgt_bands = []
            for band_name, (low, high) in freq_bands.items():
                mask = (f >= low) & (f < high)
                if np.any(mask):
                    gen_bands.append(np.sum(psd_gen[mask]))
                    tgt_bands.append(np.sum(psd_tgt[mask]))
            
            if len(gen_bands) > 0:
                corr = np.corrcoef(gen_bands, tgt_bands)[0, 1]
                if not np.isnan(corr):
                    band_sims.append(corr)
    
    freq_sim = np.mean(band_sims) if band_sims else 0.0
    
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "topo_sim": float(topo_sim),
        "psd_corr": float(psd_corr),
        "freq_sim": float(freq_sim)
    }
