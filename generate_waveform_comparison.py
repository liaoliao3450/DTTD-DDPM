"""
生成波形对比图
用于论文展示重建质量
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.dttd_enhanced_v2 import DTTDEnhanced
from models.baselines import CVAE, ConditionalGAN, SimpleDDPM
from models.traditional_baselines import SplineInterpolation, KrigingInterpolation
from data import get_bci2a_dataloaders
from utils import load_config, get_device

# 设置绘图样式
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def plot_waveform_comparison():
    """生成波形对比图"""
    
    # 加载配置和数据
    config = load_config('configs/bci2a_enhanced_config.yaml')
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
    # reconstruction_mode返回4个值
    if len(batch_data) == 4:
        target_data, channel_indices, labels, subject_ids = batch_data
    else:
        target_data, channel_indices, labels = batch_data
    
    # 提取输入数据
    ch_idx = channel_indices[0].tolist() if channel_indices.dim() > 1 else channel_indices.tolist()
    input_data = target_data[:, ch_idx, :].to(device)
    labels_tensor = labels.to(device)
    target_data = target_data.to(device)
    
    # 选择一个样本
    sample_idx = 0
    input_sample = input_data[sample_idx:sample_idx+1]
    target_sample = target_data[sample_idx:sample_idx+1]
    label_sample = labels_tensor[sample_idx:sample_idx+1]
    
    # 加载所有模型并生成重建
    results = {}
    
    print("Loading models and generating reconstructions...")
    
    # 1. DTTD-DDPM
    print("  [1/5] DTTD-DDPM...")
    dttd = DTTDEnhanced(config['model']).to(device)
    checkpoint = torch.load('checkpoints/bci2a_enhanced/best_model.pth', map_location=device)
    dttd.load_state_dict(checkpoint['model_state_dict'], strict=False)
    dttd.eval()
    
    # ⭐ 数据缩放因子（与训练时一致）
    data_scale_factor = 1e5
    
    with torch.no_grad():
        # 缩放输入数据
        input_scaled = input_sample * data_scale_factor
        noisy_input = input_scaled + 0.02 * torch.randn_like(input_scaled)
        t = torch.zeros(1, device=device, dtype=torch.long)
        generated = dttd(noisy_input, t, label_sample)
        # 缩放回原始尺度
        results['DTTD-DDPM'] = generated / data_scale_factor
    
    # 2. CVAE
    print("  [2/5] CVAE...")
    cvae = CVAE(input_channels=9, output_channels=22, time_steps=1000, num_classes=4, latent_dim=128).to(device)
    checkpoint = torch.load('checkpoints/bci2a/baseline_cvae/best_model.pth', map_location=device)
    cvae.load_state_dict(checkpoint['model_state_dict'])
    cvae.eval()
    
    with torch.no_grad():
        results['CVAE'], _, _ = cvae(input_sample, label_sample)
    
    # 3. cGAN
    print("  [3/5] cGAN...")
    cgan = ConditionalGAN(input_channels=9, output_channels=22, time_steps=1000, num_classes=4).to(device)
    checkpoint = torch.load('checkpoints/bci2a_enhanced/baseline_cgan/best_model.pth', map_location=device)
    
    # 只加载Generator部分（checkpoint中只有Generator的参数）
    generator_state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        generator_state_dict[f'generator.{key}'] = value
    cgan.load_state_dict(generator_state_dict, strict=False)
    cgan.eval()
    
    with torch.no_grad():
        z = torch.randn(1, 128).to(device)
        results['cGAN'] = cgan(z, label_sample, input_sample)
    
    # 4. Simple-DDPM
    print("  [4/5] Simple-DDPM...")
    simple = SimpleDDPM(input_channels=9, output_channels=22, time_steps=1000, num_classes=4).to(device)
    checkpoint = torch.load('checkpoints/bci2a/baseline_simple_ddpm/best_model.pth', map_location=device)
    simple.load_state_dict(checkpoint['model_state_dict'])
    simple.eval()
    
    with torch.no_grad():
        results['Simple-DDPM'] = simple.reconstruct(input_sample, label_sample, num_inference_steps=1, noise_level=0.02)
    
    # 5. Simple-CondDDIM
    print("  [5/7] Simple-CondDDIM...")
    with torch.no_grad():
        results['Simple-CondDDIM'] = simple.reconstruct_conditioned_ddim(input_sample, label_sample, num_inference_steps=50, eta=0.0, guidance_scale=2.0)

    # 6. Spline-Interpolation (traditional)
    print("  [6/7] Spline-Interpolation...")
    spline = SplineInterpolation(input_channels=9, output_channels=22)
    with torch.no_grad():
        results['Spline'] = spline.reconstruct(input_sample).to(device)

    # 7. Kriging (traditional)
    print("  [7/7] Kriging...")
    kriging = KrigingInterpolation(input_channels=9, output_channels=22)
    with torch.no_grad():
        results['Kriging'] = kriging.reconstruct(input_sample).to(device)
    
    # 转换为numpy并统一缩放以便可视化
    # ⭐ 原始EEG数据通常是微伏级别，数值很小，需要缩放才能在图上清晰显示
    # 所有数据统一乘以 data_scale_factor 使波形可见
    target_np = target_sample.cpu().numpy()[0] * data_scale_factor
    
    print(f"\n数据范围检查:")
    print(f"  原始信号 (缩放后): min={target_np.min():.4f}, max={target_np.max():.4f}, std={target_np.std():.4f}")
    
    # 计算原始信号的统计量，用于归一化异常输出
    target_mean = target_np.mean()
    target_std = target_np.std()
    
    for key in results:
        result_data = results[key].cpu().numpy()[0]
        # 所有方法的输出都统一缩放
        results[key] = result_data * data_scale_factor
        
        # 检查输出尺度是否异常（与原始信号差异超过10倍）
        result_std = results[key].std()
        if result_std > target_std * 10 or result_std < target_std / 10:
            print(f"  {key}: 尺度异常 (std={result_std:.4f})，进行归一化...")
            # 将输出归一化到与原始信号相同的尺度
            results[key] = (results[key] - results[key].mean()) / (result_std + 1e-8) * target_std + target_mean
        
        print(f"  {key}: min={results[key].min():.4f}, max={results[key].max():.4f}, std={results[key].std():.4f}")
    
    # 绘图
    print("\nGenerating waveform comparison plot...")
    
    # 选择几个代表性通道进行展示
    channels_to_plot = [0, 5, 10, 15, 20]  # C3, Cz, C4等
    channel_names = ['Ch1', 'Ch6', 'Ch11', 'Ch16', 'Ch21']
    time_points = 1000
    time_axis = np.arange(time_points) / 250.0  # 转换为秒
    
    # 创建图形
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(len(channels_to_plot), 1, figure=fig, hspace=0.3)
    
    methods = ['Original', 'DTTD-DDPM', 'Simple-CondDDIM', 'CVAE', 'cGAN', 'Simple-DDPM', 'Spline', 'Kriging']
    colors = ['black', '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#7F8C8D', '#34495E']
    linestyles = ['-', '-', '-', '--', '--', ':', '-.', '-.']
    
    for idx, (ch_idx, ch_name) in enumerate(zip(channels_to_plot, channel_names)):
        ax = fig.add_subplot(gs[idx, 0])
        
        # 绘制原始信号
        ax.plot(time_axis, target_np[ch_idx], color=colors[0], 
                linestyle=linestyles[0], linewidth=2, label='Original', alpha=0.8)
        
        # 绘制各方法的重建信号
        for method_idx, method in enumerate(methods[1:], 1):
            ax.plot(time_axis, results[method][ch_idx], 
                    color=colors[method_idx], linestyle=linestyles[method_idx], 
                    linewidth=1.5, label=method, alpha=0.7)
        
        ax.set_ylabel(ch_name, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([0, time_points/250.0])
        
        if idx == 0:
            ax.legend(loc='upper right', ncol=min(4, len(methods)), fontsize=9, framealpha=0.9)
            ax.set_title('EEG Waveform Reconstruction Comparison', fontsize=14, fontweight='bold')
        
        if idx == len(channels_to_plot) - 1:
            ax.set_xlabel('Time (s)', fontsize=12)
        else:
            ax.set_xticklabels([])
    
    plt.tight_layout()
    
    # 保存
    output_dir = 'paper_results/figures'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'waveform_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to: {output_path}")
    
    plt.close()
    
    # 生成单个通道的详细对比图
    print("\nGenerating detailed single-channel comparison...")
    
    # 动态网格以容纳所有方法（此处8个方法 → 2x4）
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    
    ch_idx = 10  # C4通道
    
    for ax_idx, method in enumerate(methods):
        ax = axes[ax_idx]
        
        if method == 'Original':
            signal = target_np[ch_idx]
            color = colors[0]
        else:
            signal = results[method][ch_idx]
            color = colors[ax_idx]
        
        ax.plot(time_axis, signal, color=color, linewidth=2)
        ax.set_title(method, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Amplitude (μV)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, time_points/250.0])
        
        # 计算MSE
        if method != 'Original':
            mse = np.mean((signal - target_np[ch_idx]) ** 2)
            ax.text(0.02, 0.98, f'MSE: {mse:.4f}', transform=ax.transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Detailed Reconstruction Comparison - Channel C4', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'waveform_detailed.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to: {output_path}")
    
    plt.close()
    
    print("\n✓ All waveform comparison plots generated successfully!")


if __name__ == '__main__':
    plot_waveform_comparison()

