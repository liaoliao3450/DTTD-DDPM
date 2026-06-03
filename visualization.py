"""
可视化脚本 (整合版)

生成论文所需的各种可视化图表:
1. 波形对比图 (waveform)
2. t-SNE分布图 (tsne)
3. 各被试条形图 (barplot)
4. 频谱对比图 (spectrum)

使用方法:
    python experiments/visualization.py --type all
    python experiments/visualization.py --type waveform
    python experiments/visualization.py --type barplot
"""
import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import signal
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

matplotlib.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.dttd_enhanced_v2 import DTTDEnhanced
from data import get_bci2a_dataloaders
from utils import load_config, get_device

# 设置绘图样式
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


# ==================== 波形对比图 ====================

def plot_waveform_comparison(config, checkpoint_path, output_dir):
    """生成波形对比图"""
    print("\n生成波形对比图...")
    
    device = get_device()
    
    # 加载测试数据
    _, _, test_loader = get_bci2a_dataloaders(
        data_path=config['data']['data_path'],
        batch_size=32,
        subject_ids=config['data'].get('subjects', list(range(1, 10))),
        num_workers=0,
        reconstruction_mode=True
    )
    
    # 获取一个batch的数据
    batch_data = next(iter(test_loader))
    if len(batch_data) == 4:
        target_data, channel_indices, labels, _ = batch_data
    else:
        target_data, channel_indices, labels = batch_data
    
    ch_idx = channel_indices[0].tolist() if channel_indices.dim() > 1 else channel_indices.tolist()
    input_data = target_data[:, ch_idx, :].to(device)
    labels_tensor = labels.to(device)
    target_data = target_data.to(device)
    
    # 选择一个样本
    sample_idx = 0
    input_sample = input_data[sample_idx:sample_idx+1]
    target_sample = target_data[sample_idx:sample_idx+1]
    label_sample = labels_tensor[sample_idx:sample_idx+1]
    
    # 加载DTTD模型
    model = DTTDEnhanced(config['model']).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    data_scale_factor = checkpoint.get('data_scale_factor', 1e5)
    
    # 生成重建
    with torch.no_grad():
        input_scaled = input_sample * data_scale_factor
        noisy_input = input_scaled + 0.02 * torch.randn_like(input_scaled)
        t = torch.zeros(1, device=device, dtype=torch.long)
        generated = model(noisy_input, t, label_sample)
        generated = generated / data_scale_factor
    
    # 转换为numpy
    target_np = target_sample.cpu().numpy()[0] * data_scale_factor
    generated_np = generated.cpu().numpy()[0] * data_scale_factor
    
    # 绘图
    channels_to_plot = [0, 5, 10, 15, 20]
    channel_names = ['Ch1', 'Ch6', 'Ch11', 'Ch16', 'Ch21']
    time_points = target_np.shape[1]
    time_axis = np.arange(time_points) / 250.0
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(len(channels_to_plot), 1, figure=fig, hspace=0.3)
    
    for idx, (ch_idx, ch_name) in enumerate(zip(channels_to_plot, channel_names)):
        ax = fig.add_subplot(gs[idx, 0])
        
        ax.plot(time_axis, target_np[ch_idx], color='black', linewidth=2, label='Original', alpha=0.8)
        ax.plot(time_axis, generated_np[ch_idx], color='#E74C3C', linewidth=1.5, label='DTTD-DDPM', alpha=0.7)
        
        ax.set_ylabel(ch_name, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([0, time_points/250.0])
        
        if idx == 0:
            ax.legend(loc='upper right', ncol=2, fontsize=9)
            ax.set_title('EEG Waveform Reconstruction', fontsize=14, fontweight='bold')
        
        if idx == len(channels_to_plot) - 1:
            ax.set_xlabel('Time (s)', fontsize=12)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'waveform_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 保存到: {output_path}")


# ==================== 各被试条形图 ====================

def plot_per_subject_barplot(results_file, output_dir):
    """生成各被试分类准确率条形图"""
    print("\n生成各被试条形图...")
    
    # 加载结果
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    def plot_scenario(scenario_data, scenario_name, filename):
        """绘制单个场景"""
        # 支持新格式 (classification_results.json)
        if isinstance(scenario_data, dict):
            # 新格式: {"S1": {...}, "S2": {...}, "average": {...}}
            subjects = [k for k in scenario_data.keys() if k != 'average']
            baseline_9ch = [scenario_data[s]['baseline_9ch']['accuracy'] * 100 for s in subjects]
            baseline_22ch = [scenario_data[s]['baseline_22ch']['accuracy'] * 100 for s in subjects]
            dttd_acc = [scenario_data[s]['dttd_augmented']['accuracy'] * 100 for s in subjects]
        else:
            # 旧格式: [{"subject_id": 1, ...}, ...]
            subjects = [f"S{d['subject_id']}" for d in scenario_data]
            baseline_9ch = [d['9ch']['accuracy'] * 100 for d in scenario_data]
            baseline_22ch = [d['22ch']['accuracy'] * 100 for d in scenario_data]
            dttd_acc = [d['DTTD-DDPM_recon']['accuracy'] * 100 for d in scenario_data]
        
        x = np.arange(len(subjects))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 6))
        bars1 = ax.bar(x - width, baseline_9ch, width, label='9通道基线', color='#2ECC71', alpha=0.8)
        bars2 = ax.bar(x, baseline_22ch, width, label='22通道基线', color='#3498DB', alpha=0.8)
        bars3 = ax.bar(x + width, dttd_acc, width, label='DTTD', color='#E74C3C', alpha=0.8)
        
        ax.set_xlabel('受试者', fontsize=12)
        ax.set_ylabel('准确率 (%)', fontsize=12)
        ax.set_title(f'{scenario_name}分类准确率对比', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(subjects)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 添加数值标签
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)
        
        # 添加平均值线
        ax.axhline(y=np.mean(baseline_9ch), color='#2ECC71', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(y=np.mean(baseline_22ch), color='#3498DB', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(y=np.mean(dttd_acc), color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.7)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{filename}_per_subject.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 保存: {output_path}")
        
        return {
            '9ch_mean': np.mean(baseline_9ch),
            '22ch_mean': np.mean(baseline_22ch),
            'dttd_mean': np.mean(dttd_acc)
        }
    
    stats = {}
    
    # 绘制各场景
    if 'within_session' in data:
        stats['被试内'] = plot_scenario(data['within_session'], '被试内', 'within_session')
    if 'cross_session' in data:
        stats['跨会话'] = plot_scenario(data['cross_session'], '跨会话', 'cross_session')
    if 'cross_subject' in data:
        stats['跨被试'] = plot_scenario(data['cross_subject'], '跨被试', 'cross_subject')
    
    # 绘制汇总图
    if len(stats) > 0:
        scenarios = list(stats.keys())
        baseline_9ch = [stats[s]['9ch_mean'] for s in scenarios]
        baseline_22ch = [stats[s]['22ch_mean'] for s in scenarios]
        dttd_means = [stats[s]['dttd_mean'] for s in scenarios]
        
        x = np.arange(len(scenarios))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width, baseline_9ch, width, label='9通道基线', color='#2ECC71', alpha=0.8)
        ax.bar(x, baseline_22ch, width, label='22通道基线', color='#3498DB', alpha=0.8)
        ax.bar(x + width, dttd_means, width, label='DTTD', color='#E74C3C', alpha=0.8)
        
        ax.set_xlabel('评估场景', fontsize=12)
        ax.set_ylabel('准确率 (%)', fontsize=12)
        ax.set_title('三种评估场景分类准确率对比', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'three_scenarios_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 保存: {output_path}")


# ==================== t-SNE分布图 ====================

def plot_tsne_comparison(config, checkpoint_path, output_dir, num_samples=500):
    """生成t-SNE分布对比图"""
    print("\n生成t-SNE分布图...")
    
    device = get_device()
    
    # 加载数据
    _, _, test_loader = get_bci2a_dataloaders(
        data_path=config['data']['data_path'],
        batch_size=32,
        subject_ids=config['data'].get('subjects', list(range(1, 10))),
        num_workers=0,
        reconstruction_mode=True
    )
    
    # 收集真实数据
    real_data_list, real_labels_list = [], []
    for batch_data in test_loader:
        if len(batch_data) == 4:
            target_data, channel_indices, labels, _ = batch_data
        else:
            target_data, channel_indices, labels = batch_data
        real_data_list.append(target_data)
        real_labels_list.append(labels)
        if len(real_data_list) * 32 >= num_samples:
            break
    
    real_data = torch.cat(real_data_list, dim=0)[:num_samples]
    real_labels = torch.cat(real_labels_list, dim=0)[:num_samples]
    
    ch_idx = channel_indices[0].tolist() if channel_indices.dim() > 1 else channel_indices.tolist()
    
    # 加载模型并生成数据
    model = DTTDEnhanced(config['model']).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    data_scale_factor = checkpoint.get('data_scale_factor', 1e5)
    
    # 生成数据
    generated_list = []
    with torch.no_grad():
        for i in range(0, len(real_data), 32):
            batch = real_data[i:i+32]
            batch_labels = real_labels[i:i+32]
            
            input_9ch = batch[:, ch_idx, :].to(device) * data_scale_factor
            batch_labels = batch_labels.to(device)
            
            t = torch.zeros(batch.size(0), dtype=torch.long, device=device)
            generated = model(input_9ch, t, batch_labels)
            generated = generated / data_scale_factor
            generated_list.append(generated.cpu())
    
    generated_data = torch.cat(generated_list, dim=0)
    
    # 提取特征 (通道均值)
    real_features = real_data.numpy().mean(axis=2)
    generated_features = generated_data.numpy().mean(axis=2)
    
    # 合并数据
    all_features = np.vstack([real_features, generated_features])
    all_labels = np.concatenate([real_labels.numpy(), real_labels.numpy()])
    all_sources = np.array(['Real'] * len(real_features) + ['Generated'] * len(generated_features))
    
    # 标准化
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)
    
    # t-SNE
    print("  运行t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    features_2d = tsne.fit_transform(all_features_scaled)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#E57373', '#64B5F6', '#81C784', '#FFD54F']
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    
    for source, marker, alpha in [('Real', 'o', 0.3), ('Generated', 's', 0.7)]:
        source_mask = all_sources == source
        for class_id in range(4):
            class_mask = all_labels == class_id
            mask = source_mask & class_mask
            
            if np.any(mask):
                ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                          c=colors[class_id], marker=marker, alpha=alpha, s=50,
                          label=f'{source} - {class_names[class_id]}')
    
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title('t-SNE: Real vs Generated EEG', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'tsne_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 保存到: {output_path}")


# ==================== 频谱对比图 ====================

def plot_spectrum_comparison(config, checkpoint_path, output_dir):
    """生成频谱对比图"""
    print("\n生成频谱对比图...")
    
    device = get_device()
    
    # 加载数据
    _, _, test_loader = get_bci2a_dataloaders(
        data_path=config['data']['data_path'],
        batch_size=32,
        subject_ids=config['data'].get('subjects', list(range(1, 10))),
        num_workers=0,
        reconstruction_mode=True
    )
    
    batch_data = next(iter(test_loader))
    if len(batch_data) == 4:
        target_data, channel_indices, labels, _ = batch_data
    else:
        target_data, channel_indices, labels = batch_data
    
    ch_idx = channel_indices[0].tolist() if channel_indices.dim() > 1 else channel_indices.tolist()
    
    # 加载模型
    model = DTTDEnhanced(config['model']).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    data_scale_factor = checkpoint.get('data_scale_factor', 1e5)
    
    # 生成数据
    input_data = target_data[:, ch_idx, :].to(device) * data_scale_factor
    labels_tensor = labels.to(device)
    
    with torch.no_grad():
        t = torch.zeros(target_data.size(0), dtype=torch.long, device=device)
        generated = model(input_data, t, labels_tensor)
        generated = generated / data_scale_factor
    
    # 计算PSD
    fs = config['data'].get('fs', 250)
    
    real_np = target_data.numpy()
    gen_np = generated.cpu().numpy()
    
    # 平均PSD
    def compute_avg_psd(data, fs):
        psds = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                f, psd = signal.welch(data[i, j], fs=fs, nperseg=min(256, data.shape[2]))
                psds.append(psd)
        return f, np.mean(psds, axis=0)
    
    f_real, psd_real = compute_avg_psd(real_np, fs)
    f_gen, psd_gen = compute_avg_psd(gen_np, fs)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogy(f_real, psd_real, 'b-', linewidth=2, label='Real', alpha=0.8)
    ax.semilogy(f_gen, psd_gen, 'r--', linewidth=2, label='Generated', alpha=0.8)
    
    # 标注频段
    freq_bands = {'δ': (0.5, 4), 'θ': (4, 8), 'α': (8, 13), 'β': (13, 30), 'γ': (30, 50)}
    colors = ['#FFE0B2', '#FFCC80', '#FFB74D', '#FFA726', '#FF9800']
    
    for (name, (low, high)), color in zip(freq_bands.items(), colors):
        ax.axvspan(low, high, alpha=0.2, color=color, label=f'{name} ({low}-{high}Hz)')
    
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Power Spectral Density', fontsize=12)
    ax.set_title('PSD Comparison: Real vs Generated', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 60])
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'spectrum_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 保存到: {output_path}")


# ==================== 主函数 ====================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='生成可视化图表')
    parser.add_argument('--config', default='configs/bci2a_enhanced_config.yaml')
    parser.add_argument('--checkpoint', default='checkpoints/dttd/best_model.pth')
    parser.add_argument('--results-file', default='paper_results/classification/classification_results.json')
    parser.add_argument('--output-dir', default='paper_results/figures')
    parser.add_argument('--type', default='all', choices=['all', 'waveform', 'barplot', 'tsne', 'spectrum'])
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    print("="*60)
    print("生成可视化图表")
    print("="*60)
    
    if args.type in ['all', 'waveform']:
        plot_waveform_comparison(config, args.checkpoint, args.output_dir)
    
    if args.type in ['all', 'barplot']:
        if os.path.exists(args.results_file):
            plot_per_subject_barplot(args.results_file, args.output_dir)
        else:
            print(f"[WARN] 结果文件不存在: {args.results_file}")
    
    if args.type in ['all', 'tsne']:
        plot_tsne_comparison(config, args.checkpoint, args.output_dir)
    
    if args.type in ['all', 'spectrum']:
        plot_spectrum_comparison(config, args.checkpoint, args.output_dir)
    
    print("\n" + "="*60)
    print("✓ 可视化完成!")
    print("="*60)


if __name__ == '__main__':
    main()
