"""
统一的重建质量评估 - 确保所有方法使用相同的评估流程
"""
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from scipy import signal as scipy_signal
from scipy.stats import pearsonr
from scipy.interpolate import CubicSpline, Rbf
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models import DTTDEnhanced
from models.baselines import CVAE, Generator as cGANGenerator
from data import get_bci2a_dataloaders
from utils import load_config, get_device, set_seed


def compute_metrics(generated, target):
    """
    统一的指标计算函数
    
    Args:
        generated: numpy array [batch, 22, time]
        target: numpy array [batch, 22, time]
    
    Returns:
        dict with metrics
    """
    assert generated.shape == target.shape, f"Shape mismatch: {generated.shape} vs {target.shape}"
    
    n_samples, n_channels, n_times = generated.shape
    
    # 1. MSE (归一化)
    mse_raw = np.mean((generated - target) ** 2)
    target_var = np.var(target)
    nmse = mse_raw / (target_var + 1e-10)
    
    # 2. 拓扑相似度 - 通道间相关矩阵的Pearson相关
    topo_sims = []
    for i in range(min(n_samples, 100)):
        # 每个样本计算22x22的通道相关矩阵
        corr_gen = np.corrcoef(generated[i])  # [22, 22]
        corr_target = np.corrcoef(target[i])  # [22, 22]
        
        # 检查NaN
        if np.isnan(corr_gen).any() or np.isnan(corr_target).any():
            continue
        
        # 提取上三角（不含对角线）
        mask = np.triu(np.ones((n_channels, n_channels), dtype=bool), k=1)
        gen_vec = corr_gen[mask]
        target_vec = corr_target[mask]
        
        # Pearson相关
        corr, _ = pearsonr(gen_vec, target_vec)
        if not np.isnan(corr):
            topo_sims.append(corr)
    
    topo_sim = np.mean(topo_sims) if topo_sims else 0.0
    
    # 3. PSD相关 - 每个通道的功率谱相关
    psd_corrs = []
    for i in range(min(n_samples, 50)):
        for ch in range(n_channels):
            try:
                _, psd_gen = scipy_signal.welch(generated[i, ch], fs=250, nperseg=min(256, n_times))
                _, psd_target = scipy_signal.welch(target[i, ch], fs=250, nperseg=min(256, n_times))
                
                corr, _ = pearsonr(psd_gen, psd_target)
                if not np.isnan(corr):
                    psd_corrs.append(corr)
            except Exception as e:
                pass
    
    psd_corr = np.mean(psd_corrs) if psd_corrs else 0.0
    
    # 4. 频率相似度 - 5个频段的平均相关
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    band_sims = []
    for band_name, (low, high) in bands.items():
        try:
            sos = scipy_signal.butter(4, [low, high], btype='band', fs=250, output='sos')
            band_corrs = []
            
            for i in range(min(n_samples, 30)):
                for ch in range(n_channels):
                    gen_band = scipy_signal.sosfilt(sos, generated[i, ch])
                    target_band = scipy_signal.sosfilt(sos, target[i, ch])
                    
                    corr, _ = pearsonr(gen_band, target_band)
                    if not np.isnan(corr):
                        band_corrs.append(corr)
            
            if band_corrs:
                band_sims.append(np.mean(band_corrs))
        except Exception as e:
            pass
    
    freq_sim = np.mean(band_sims) if band_sims else 0.0
    
    return {
        'nmse': float(nmse),
        'topology_similarity': float(topo_sim),
        'psd_correlation': float(psd_corr),
        'frequency_similarity': float(freq_sim)
    }


def spline_interpolation(input_9ch, ch_indices):
    """样条插值重建22通道"""
    n_samples, _, n_times = input_9ch.shape
    output = np.zeros((n_samples, 22, n_times))
    
    sorted_idx = sorted(ch_indices)
    
    for i in range(n_samples):
        # 获取已知通道的值（按索引排序）
        known_vals = input_9ch[i, [ch_indices.index(idx) for idx in sorted_idx], :]
        
        for t in range(n_times):
            cs = CubicSpline(sorted_idx, known_vals[:, t], extrapolate=True)
            output[i, :, t] = cs(np.arange(22))
    
    return output


def kriging_interpolation(input_9ch, ch_indices):
    """克里金插值重建22通道"""
    n_samples, _, n_times = input_9ch.shape
    output = np.zeros((n_samples, 22, n_times))
    
    sorted_idx = sorted(ch_indices)
    
    for i in range(n_samples):
        known_vals = input_9ch[i, [ch_indices.index(idx) for idx in sorted_idx], :]
        
        for t in range(n_times):
            try:
                rbf = Rbf(sorted_idx, known_vals[:, t], function='multiquadric')
                output[i, :, t] = rbf(np.arange(22))
            except:
                output[i, :, t] = np.interp(np.arange(22), sorted_idx, known_vals[:, t])
    
    return output


def main():
    set_seed(42)
    device = get_device()
    config = load_config('configs/bci2a_enhanced_config.yaml')
    
    # 数据缩放因子
    DATA_SCALE = 1e5
    
    print("="*70)
    print("统一重建质量评估")
    print("="*70)
    print(f"数据缩放因子: {DATA_SCALE}")
    
    # 加载数据 - 只用Subject 1
    print("\n加载数据 (Subject 1)...")
    _, _, test_loader = get_bci2a_dataloaders(
        data_path=config['data']['data_path'],
        batch_size=32,
        subject_ids=[1],  # 只用Subject 1
        num_workers=0,
        reconstruction_mode=True
    )
    
    # 收集测试数据
    test_data_list = []
    test_labels_list = []
    ch_indices = None
    
    for batch in tqdm(test_loader, desc="Loading"):
        target_data, channel_indices, labels, _ = batch
        test_data_list.append(target_data)
        test_labels_list.append(labels)
        if ch_indices is None:
            ch_indices = channel_indices[0].tolist()
    
    # 使用全部测试数据
    test_data = torch.cat(test_data_list, dim=0)
    test_labels = torch.cat(test_labels_list, dim=0)
    
    print(f"\n测试数据: {test_data.shape}")
    print(f"输入通道索引: {ch_indices}")
    print(f"数据范围: [{test_data.min():.6f}, {test_data.max():.6f}]")
    
    # 提取9通道输入
    input_9ch = test_data[:, ch_indices, :].numpy()
    target_22ch = test_data.numpy()
    
    results = {}
    
    # ========== 1. DTTD-DDPM ==========
    print("\n" + "-"*50)
    print("评估 DTTD-DDPM...")
    
    model = DTTDEnhanced(config['model']).to(device)
    ckpt = torch.load('checkpoints/bci2a_enhanced/best_model.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    
    generated_list = []
    with torch.no_grad():
        for i in tqdm(range(0, len(test_data), 32), desc="DTTD-DDPM"):
            batch_data = test_data[i:i+32].to(device)
            batch_labels = test_labels[i:i+32].to(device)
            input_data = batch_data[:, ch_indices, :]
            
            # 缩放 -> 生成 -> 还原
            scaled_input = input_data * DATA_SCALE
            noisy_input = scaled_input + torch.randn_like(scaled_input) * 0.02
            t = torch.zeros(input_data.size(0), device=device, dtype=torch.long)
            generated = model(noisy_input, t, batch_labels)
            generated = generated / DATA_SCALE
            
            generated_list.append(generated.cpu().numpy())
    
    dttd_generated = np.concatenate(generated_list, axis=0)
    print(f"  生成数据范围: [{dttd_generated.min():.6f}, {dttd_generated.max():.6f}]")
    
    results['DTTD-DDPM'] = compute_metrics(dttd_generated, target_22ch)
    print(f"  结果: {results['DTTD-DDPM']}")
    
    # ========== 2. CVAE ==========
    print("\n" + "-"*50)
    print("评估 CVAE...")
    
    cvae_ckpt_path = 'checkpoints/bci2a/baseline_cvae/best_model.pth'
    if os.path.exists(cvae_ckpt_path):
        cvae_model = CVAE(input_channels=9, output_channels=22, time_steps=1000, 
                         num_classes=4, latent_dim=128).to(device)
        cvae_ckpt = torch.load(cvae_ckpt_path, map_location=device)
        cvae_model.load_state_dict(cvae_ckpt['model_state_dict'], strict=False)
        cvae_model.eval()
        
        generated_list = []
        with torch.no_grad():
            for i in tqdm(range(0, len(test_data), 32), desc="CVAE"):
                batch_data = test_data[i:i+32].to(device)
                batch_labels = test_labels[i:i+32].to(device)
                input_data = batch_data[:, ch_indices, :]
                
                scaled_input = input_data * DATA_SCALE
                generated, _, _ = cvae_model(scaled_input, batch_labels)
                generated = generated / DATA_SCALE
                
                generated_list.append(generated.cpu().numpy())
        
        cvae_generated = np.concatenate(generated_list, axis=0)
        print(f"  生成数据范围: [{cvae_generated.min():.6f}, {cvae_generated.max():.6f}]")
        
        results['CVAE'] = compute_metrics(cvae_generated, target_22ch)
        print(f"  结果: {results['CVAE']}")
    else:
        print(f"  [SKIP] checkpoint not found")
    
    # ========== 3. cGAN ==========
    print("\n" + "-"*50)
    print("评估 cGAN...")
    
    cgan_ckpt_path = 'checkpoints/bci2a_enhanced/baseline_cgan/best_model.pth'
    if os.path.exists(cgan_ckpt_path):
        cgan_model = cGANGenerator(input_channels=9, output_channels=22, time_steps=1000,
                                   num_classes=4, latent_dim=128).to(device)
        cgan_ckpt = torch.load(cgan_ckpt_path, map_location=device)
        cgan_model.load_state_dict(cgan_ckpt['model_state_dict'], strict=False)
        cgan_model.eval()
        
        generated_list = []
        with torch.no_grad():
            for i in tqdm(range(0, len(test_data), 32), desc="cGAN"):
                batch_data = test_data[i:i+32].to(device)
                batch_labels = test_labels[i:i+32].to(device)
                input_data = batch_data[:, ch_indices, :]
                
                batch_size = input_data.size(0)
                z = torch.randn(batch_size, 128).to(device)
                scaled_input = input_data * DATA_SCALE
                generated = cgan_model(z, batch_labels, scaled_input)
                generated = generated / DATA_SCALE
                
                generated_list.append(generated.cpu().numpy())
        
        cgan_generated = np.concatenate(generated_list, axis=0)
        print(f"  生成数据范围: [{cgan_generated.min():.6f}, {cgan_generated.max():.6f}]")
        
        results['cGAN'] = compute_metrics(cgan_generated, target_22ch)
        print(f"  结果: {results['cGAN']}")
    else:
        print(f"  [SKIP] checkpoint not found")
    
    # ========== 4. 样条插值 ==========
    print("\n" + "-"*50)
    print("评估 样条插值...")
    
    spline_generated = spline_interpolation(input_9ch, ch_indices)
    print(f"  生成数据范围: [{spline_generated.min():.6f}, {spline_generated.max():.6f}]")
    
    results['Spline'] = compute_metrics(spline_generated, target_22ch)
    print(f"  结果: {results['Spline']}")
    
    # ========== 5. 克里金插值 ==========
    print("\n" + "-"*50)
    print("评估 克里金插值...")
    
    kriging_generated = kriging_interpolation(input_9ch, ch_indices)
    print(f"  生成数据范围: [{kriging_generated.min():.6f}, {kriging_generated.max():.6f}]")
    
    results['Kriging'] = compute_metrics(kriging_generated, target_22ch)
    print(f"  结果: {results['Kriging']}")
    
    # ========== 保存结果 ==========
    output_path = 'paper_results/real_experiments/unified_reconstruction_metrics.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # ========== 打印汇总 ==========
    print("\n" + "="*80)
    print("重建质量指标汇总（统一评估）")
    print("="*80)
    print(f"{'方法':<15} {'拓扑相似度':>12} {'NMSE':>12} {'PSD相关':>12} {'频率相似度':>12}")
    print("-"*80)
    
    # 按拓扑相似度排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['topology_similarity'], reverse=True)
    for method, metrics in sorted_results:
        print(f"{method:<15} {metrics['topology_similarity']:>12.4f} {metrics['nmse']:>12.4f} "
              f"{metrics['psd_correlation']:>12.4f} {metrics['frequency_similarity']:>12.4f}")
    
    print(f"\n结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
