"""
快速计算重建质量指标（修正版拓扑相似度）
"""
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from scipy import signal as scipy_signal
from scipy.stats import pearsonr
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models import DTTDEnhanced
from models.baselines import CVAE, SimpleDDPM, Generator as cGANGenerator
from data import get_bci2a_dataloaders
from utils import load_config, get_device, set_seed


def compute_reconstruction_metrics(generated, target, exclude_known_channels=None):
    """计算重建质量指标（修正版）
    
    Args:
        generated: 生成的数据 [batch, channels, time]
        target: 目标数据 [batch, channels, time]
        exclude_known_channels: 排除已知通道（只在未知通道上计算MSE）
    """
    if torch.is_tensor(generated):
        generated = generated.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    
    # 确定要计算MSE的通道
    if exclude_known_channels is not None:
        all_channels = set(range(generated.shape[1]))
        unknown_channels = list(all_channels - set(exclude_known_channels))
        mse_raw = np.mean((generated[:, unknown_channels, :] - target[:, unknown_channels, :]) ** 2)
        target_var = np.var(target[:, unknown_channels, :])
    else:
        mse_raw = np.mean((generated - target) ** 2)
        target_var = np.var(target)
    
    # 归一化MSE (NMSE) - 相对于目标信号的方差，这样更有意义
    nmse = mse_raw / (target_var + 1e-10)
    mse = nmse  # 使用NMSE作为主要指标
    
    # Topology Similarity - 正确计算通道间相关矩阵
    topo_sims = []
    for i in range(min(generated.shape[0], 100)):
        # 计算通道间相关矩阵 [22, 22]
        corr_gen = np.corrcoef(generated[i])
        corr_target = np.corrcoef(target[i])
        
        # 提取上三角元素
        mask = np.triu(np.ones_like(corr_gen, dtype=bool), k=1)
        gen_vec = corr_gen[mask]
        target_vec = corr_target[mask]
        
        if len(gen_vec) > 1 and not np.isnan(gen_vec).any() and not np.isnan(target_vec).any():
            corr, _ = pearsonr(gen_vec, target_vec)
            if not np.isnan(corr):
                topo_sims.append(corr)
    
    topo_sim = np.mean(topo_sims) if topo_sims else 0.0
    
    # PSD Correlation - 在所有通道上计算
    psd_corrs = []
    for i in range(min(generated.shape[0], 50)):
        for ch in range(generated.shape[1]):
            try:
                f_gen, psd_gen = scipy_signal.welch(generated[i, ch], fs=250, nperseg=min(256, generated.shape[2]))
                f_target, psd_target = scipy_signal.welch(target[i, ch], fs=250, nperseg=min(256, target.shape[2]))
                corr, _ = pearsonr(psd_gen, psd_target)
                if not np.isnan(corr):
                    psd_corrs.append(corr)
            except:
                pass
    psd_corr = np.mean(psd_corrs) if psd_corrs else 0.0
    
    # Frequency Similarity - 在所有通道上计算
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
    band_sims = []
    for band_name, (low, high) in bands.items():
        try:
            sos = scipy_signal.butter(4, [low, high], btype='band', fs=250, output='sos')
            band_corrs = []
            for i in range(min(generated.shape[0], 20)):
                for ch in range(generated.shape[1]):
                    gen_band = scipy_signal.sosfilt(sos, generated[i, ch])
                    target_band = scipy_signal.sosfilt(sos, target[i, ch])
                    corr, _ = pearsonr(gen_band, target_band)
                    if not np.isnan(corr):
                        band_corrs.append(corr)
            if band_corrs:
                band_sims.append(np.mean(band_corrs))
        except:
            pass
    freq_sim = np.mean(band_sims) if band_sims else 0.0
    
    return {
        'mse': float(mse),
        'topology_similarity': float(topo_sim),
        'psd_correlation': float(psd_corr),
        'frequency_similarity': float(freq_sim)
    }


def main():
    set_seed(42)
    device = get_device()
    config = load_config('configs/bci2a_enhanced_config.yaml')
    
    print("加载数据...")
    _, _, test_loader = get_bci2a_dataloaders(
        data_path=config['data']['data_path'],
        batch_size=32,
        subject_ids=list(range(1, 10)),
        num_workers=0,
        reconstruction_mode=True
    )
    
    # 收集测试数据
    test_data_list = []
    test_labels_list = []
    ch_indices = None
    
    for batch in tqdm(test_loader, desc="Loading test data"):
        target_data, channel_indices, labels, _ = batch
        test_data_list.append(target_data)
        test_labels_list.append(labels)
        if ch_indices is None:
            ch_indices = channel_indices[0].tolist()
    
    test_data = torch.cat(test_data_list, dim=0)[:200]  # 只取200个样本
    test_labels = torch.cat(test_labels_list, dim=0)[:200]
    
    print(f"测试数据: {test_data.shape}")
    print(f"输入通道索引: {ch_indices}")
    
    results = {}
    
    # 1. DTTD-DDPM
    print("\n评估 DTTD-DDPM...")
    model = DTTDEnhanced(config['model']).to(device)
    ckpt = torch.load('checkpoints/bci2a_enhanced/best_model.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    
    # 获取数据缩放因子（模型训练时使用的）
    data_scale_factor = ckpt.get('data_scale_factor', 1e5)
    print(f"  使用 data_scale_factor: {data_scale_factor}")
    
    generated_list = []
    with torch.no_grad():
        for i in tqdm(range(0, len(test_data), 32), desc="Generating"):
            batch_data = test_data[i:i+32].to(device)
            batch_labels = test_labels[i:i+32].to(device)
            input_data = batch_data[:, ch_indices, :]
            
            # 缩放输入数据
            scaled_input = input_data * data_scale_factor
            noisy_input = scaled_input + torch.randn_like(scaled_input) * 0.02
            t = torch.zeros(input_data.size(0), device=device, dtype=torch.long)
            generated = model(noisy_input, t, batch_labels)
            
            # 还原缩放
            generated = generated / data_scale_factor
            generated_list.append(generated.cpu())
    
    generated = torch.cat(generated_list, dim=0)
    results['DTTD-DDPM'] = compute_reconstruction_metrics(generated, test_data, exclude_known_channels=ch_indices)
    print(f"  Topo Sim: {results['DTTD-DDPM']['topology_similarity']:.4f}")
    print(f"  MSE: {results['DTTD-DDPM']['mse']:.4f}")
    print(f"  PSD Corr: {results['DTTD-DDPM']['psd_correlation']:.4f}")
    print(f"  Freq Sim: {results['DTTD-DDPM']['frequency_similarity']:.4f}")
    
    # 2. 样条插值
    print("\n评估 样条插值...")
    from scipy.interpolate import CubicSpline
    
    spline_generated = []
    sorted_ch_indices = sorted(ch_indices)  # 排序后的索引
    
    for i in range(len(test_data)):
        sample = test_data[i].numpy()
        input_9ch = sample[sorted_ch_indices, :]  # 使用排序后的索引
        
        # 简单的通道插值
        output = np.zeros((22, sample.shape[1]))
        output[sorted_ch_indices, :] = input_9ch
        
        # 对缺失通道进行插值
        known_idx = np.array(sorted_ch_indices)
        all_idx = np.arange(22)
        
        for t in range(sample.shape[1]):
            known_vals = input_9ch[:, t]
            cs = CubicSpline(known_idx, known_vals, extrapolate=True)
            output[:, t] = cs(all_idx)
        
        spline_generated.append(output)
    
    spline_generated = np.array(spline_generated)
    results['Spline'] = compute_reconstruction_metrics(spline_generated, test_data.numpy(), exclude_known_channels=sorted_ch_indices)
    print(f"  Topo Sim: {results['Spline']['topology_similarity']:.4f}")
    print(f"  MSE: {results['Spline']['mse']:.4f}")
    print(f"  PSD Corr: {results['Spline']['psd_correlation']:.4f}")
    print(f"  Freq Sim: {results['Spline']['frequency_similarity']:.4f}")
    
    # 3. 克里金插值（简化版）
    print("\n评估 克里金插值...")
    from scipy.interpolate import Rbf
    
    kriging_generated = []
    for i in range(len(test_data)):
        sample = test_data[i].numpy()
        input_9ch = sample[sorted_ch_indices, :]  # 使用排序后的索引
        
        output = np.zeros((22, sample.shape[1]))
        known_idx = np.array(sorted_ch_indices)
        all_idx = np.arange(22)
        
        for t in range(sample.shape[1]):
            known_vals = input_9ch[:, t]
            try:
                rbf = Rbf(known_idx, known_vals, function='multiquadric')
                output[:, t] = rbf(all_idx)
            except:
                output[:, t] = np.interp(all_idx, known_idx, known_vals)
        
        kriging_generated.append(output)
    
    kriging_generated = np.array(kriging_generated)
    results['Kriging'] = compute_reconstruction_metrics(kriging_generated, test_data.numpy(), exclude_known_channels=sorted_ch_indices)
    print(f"  Topo Sim: {results['Kriging']['topology_similarity']:.4f}")
    print(f"  MSE: {results['Kriging']['mse']:.4f}")
    print(f"  PSD Corr: {results['Kriging']['psd_correlation']:.4f}")
    print(f"  Freq Sim: {results['Kriging']['frequency_similarity']:.4f}")
    
    # 4. CVAE
    print("\n评估 CVAE...")
    cvae_ckpt_path = 'checkpoints/bci2a/baseline_cvae/best_model.pth'
    if os.path.exists(cvae_ckpt_path):
        cvae_model = CVAE(
            input_channels=9,
            output_channels=22,
            time_steps=1000,
            num_classes=4,
            latent_dim=128
        ).to(device)
        cvae_ckpt = torch.load(cvae_ckpt_path, map_location=device)
        cvae_model.load_state_dict(cvae_ckpt.get('model_state_dict', cvae_ckpt), strict=False)
        cvae_model.eval()
        
        cvae_generated_list = []
        with torch.no_grad():
            for i in tqdm(range(0, len(test_data), 32), desc="Generating CVAE"):
                batch_data = test_data[i:i+32].to(device)
                batch_labels = test_labels[i:i+32].to(device)
                input_data = batch_data[:, ch_indices, :]
                
                # CVAE生成 - 使用相同的缩放因子
                scaled_input = input_data * data_scale_factor
                generated, _, _ = cvae_model(scaled_input, batch_labels)
                generated = generated / data_scale_factor
                cvae_generated_list.append(generated.cpu())
        
        cvae_generated = torch.cat(cvae_generated_list, dim=0)
        results['CVAE'] = compute_reconstruction_metrics(cvae_generated, test_data, exclude_known_channels=ch_indices)
        print(f"  Topo Sim: {results['CVAE']['topology_similarity']:.4f}")
        print(f"  MSE: {results['CVAE']['mse']:.4f}")
        print(f"  PSD Corr: {results['CVAE']['psd_correlation']:.4f}")
        print(f"  Freq Sim: {results['CVAE']['frequency_similarity']:.4f}")
    else:
        print(f"  [SKIP] CVAE checkpoint not found: {cvae_ckpt_path}")
    
    # 5. cGAN
    print("\n评估 cGAN...")
    cgan_ckpt_path = 'checkpoints/bci2a_enhanced/baseline_cgan/best_model.pth'
    if os.path.exists(cgan_ckpt_path):
        cgan_model = cGANGenerator(
            input_channels=9,
            output_channels=22,
            time_steps=1000,
            num_classes=4,
            latent_dim=128
        ).to(device)
        cgan_ckpt = torch.load(cgan_ckpt_path, map_location=device)
        cgan_model.load_state_dict(cgan_ckpt.get('model_state_dict', cgan_ckpt), strict=False)
        cgan_model.eval()
        
        cgan_generated_list = []
        with torch.no_grad():
            for i in tqdm(range(0, len(test_data), 32), desc="Generating cGAN"):
                batch_data = test_data[i:i+32].to(device)
                batch_labels = test_labels[i:i+32].to(device)
                input_data = batch_data[:, ch_indices, :]
                
                # cGAN生成 - 使用相同的缩放因子
                batch_size = input_data.size(0)
                z = torch.randn(batch_size, 128).to(device)
                scaled_input = input_data * data_scale_factor
                generated = cgan_model(z, batch_labels, scaled_input)
                generated = generated / data_scale_factor
                cgan_generated_list.append(generated.cpu())
        
        cgan_generated = torch.cat(cgan_generated_list, dim=0)
        results['cGAN'] = compute_reconstruction_metrics(cgan_generated, test_data, exclude_known_channels=ch_indices)
        print(f"  Topo Sim: {results['cGAN']['topology_similarity']:.4f}")
        print(f"  MSE: {results['cGAN']['mse']:.4f}")
        print(f"  PSD Corr: {results['cGAN']['psd_correlation']:.4f}")
        print(f"  Freq Sim: {results['cGAN']['frequency_similarity']:.4f}")
    else:
        print(f"  [SKIP] cGAN checkpoint not found: {cgan_ckpt_path}")
    
    # 保存结果
    output_path = 'paper_results/real_experiments/corrected_reconstruction_metrics.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到: {output_path}")
    
    # 打印汇总表格
    print("\n" + "="*70)
    print("重建质量指标汇总（修正版）")
    print("="*70)
    print(f"{'方法':<20} {'拓扑相似度':>12} {'MSE':>10} {'PSD相关':>10} {'频率相似度':>12}")
    print("-"*70)
    for method, metrics in results.items():
        print(f"{method:<20} {metrics['topology_similarity']:>12.4f} {metrics['mse']:>10.4f} {metrics['psd_correlation']:>10.4f} {metrics['frequency_similarity']:>12.4f}")


if __name__ == '__main__':
    main()
