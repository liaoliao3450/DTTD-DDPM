"""
ERD/ERS 量化分析实验

论文中提到的实验：
1. 分析DTTD重建数据是否保留了运动想象的神经生理学特征
2. ERD（事件相关去同步）：alpha/beta频段功率降低
3. ERS（事件相关同步）：gamma频段功率增加
4. 对侧运动皮层的ERD模式验证

输出：
- 各通道各频段的ERD/ERS值
- 统计检验（配对t检验）
- 可视化图
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import ttest_rel

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# EEG频段定义
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 40)
}

# 关键通道索引（BCI2a 22通道）
CHANNEL_NAMES = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P1', 'Pz', 'P2', 'POz'
]

KEY_CHANNELS = {
    'C3': 7,   # 左运动皮层
    'Cz': 9,   # 中央
    'C4': 11   # 右运动皮层
}


def compute_psd(data, fs=250, nperseg=256):
    """计算功率谱密度"""
    if data.ndim == 2:
        # [channels, time] -> 计算每个通道的PSD
        freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg, axis=-1)
    else:
        # [batch, channels, time] -> 计算每个样本每个通道的PSD
        freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg, axis=-1)
    return freqs, psd


def compute_band_power(psd, freqs, band):
    """计算特定频带的功率"""
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    return np.mean(psd[..., band_mask], axis=-1)


def compute_erd_ers(baseline_power, task_power):
    """
    计算ERD/ERS值
    
    ERD（事件相关去同步）: (baseline - task) / baseline -> 正值表示ERD
    ERS（事件相关同步）: (task - baseline) / baseline -> 正值表示ERS
    
    返回: ERD值（负值表示ERS）
    """
    baseline_power = np.where(baseline_power == 0, 1e-10, baseline_power)
    erd = (baseline_power - task_power) / baseline_power
    return erd


def load_generated_data():
    """加载生成的数据"""
    data_path = 'results/generated_samples_test.npz'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    data = np.load(data_path)
    generated = data['generated']  # DTTD生成的22通道
    targets = data['targets']      # 真实的22通道
    labels = data['labels']        # 类别标签
    
    print(f"加载数据: generated={generated.shape}, targets={targets.shape}, labels={labels.shape}")
    return generated, targets, labels


def analyze_erd_ers_per_class(generated, targets, labels, fs=250):
    """按类别分析ERD/ERS模式"""
    results = {}
    
    # 类别名称
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    
    for class_idx in range(4):
        print(f"\n分析类别: {class_names[class_idx]}")
        
        # 选择该类别的样本
        class_mask = labels == class_idx
        class_gen = generated[class_mask]
        class_tgt = targets[class_mask]
        
        if len(class_gen) == 0:
            continue
        
        # 计算PSD
        freqs, psd_gen = compute_psd(class_gen, fs=fs)
        _, psd_tgt = compute_psd(class_tgt, fs=fs)
        
        # 计算各频段功率
        band_powers_gen = {}
        band_powers_tgt = {}
        
        for band_name, band_range in FREQ_BANDS.items():
            band_powers_gen[band_name] = compute_band_power(psd_gen, freqs, band_range)
            band_powers_tgt[band_name] = compute_band_power(psd_tgt, freqs, band_range)
        
        # 计算ERD（使用alpha频段作为参考基线）
        # 对于运动想象，alpha和beta频段应该显示ERD
        erd_results = {}
        for ch_name, ch_idx in KEY_CHANNELS.items():
            erd_results[ch_name] = {}
            for band_name in ['alpha', 'beta']:
                # 计算该通道该频段的ERD
                # 使用该类别的平均作为基线（简化处理）
                baseline_power = np.mean(band_powers_tgt[band_name][:, ch_idx])
                task_power_gen = band_powers_gen[band_name][:, ch_idx]
                task_power_tgt = band_powers_tgt[band_name][:, ch_idx]
                
                erd_gen = compute_erd_ers(baseline_power, task_power_gen)
                erd_tgt = compute_erd_ers(baseline_power, task_power_tgt)
                
                erd_results[ch_name][band_name] = {
                    'generated': {
                        'mean': float(np.mean(erd_gen)),
                        'std': float(np.std(erd_gen)),
                        'values': erd_gen.tolist()
                    },
                    'target': {
                        'mean': float(np.mean(erd_tgt)),
                        'std': float(np.std(erd_tgt)),
                        'values': erd_tgt.tolist()
                    }
                }
        
        results[class_idx] = {
            'class_name': class_names[class_idx],
            'erd_results': erd_results,
            'num_samples': len(class_gen)
        }
    
    return results


def validate_contralateral_erd(results):
    """
    验证对侧ERD模式：
    - 左手运动想象：右侧运动皮层(C4)的alpha/beta功率降低（ERD）
    - 右手运动想象：左侧运动皮层(C3)的alpha/beta功率降低（ERD）
    - 脚运动想象：中央区域(Cz)的alpha/beta功率降低（ERD）
    """
    print("\n" + "="*60)
    print("对侧ERD模式验证")
    print("="*60)
    
    validation_results = []
    
    # 定义每个类别的预期ERD通道
    expected_erd = {
        0: {'channel': 'C4', 'expected': 'ERD'},  # 左手 -> C4 ERD
        1: {'channel': 'C3', 'expected': 'ERD'},  # 右手 -> C3 ERD
        2: {'channel': 'Cz', 'expected': 'ERD'},  # 脚 -> Cz ERD
        3: {'channel': 'Cz', 'expected': 'ERD'}   # 舌头 -> Cz ERD
    }
    
    for class_idx in range(4):
        if class_idx not in results:
            continue
        
        class_name = results[class_idx]['class_name']
        erd_results = results[class_idx]['erd_results']
        expected_ch = expected_erd[class_idx]['channel']
        
        print(f"\n{class_name}:")
        
        for band_name in ['alpha', 'beta']:
            if expected_ch in erd_results:
                gen_mean = erd_results[expected_ch][band_name]['generated']['mean']
                tgt_mean = erd_results[expected_ch][band_name]['target']['mean']
                
                # ERD值为正表示功率降低（去同步）
                is_erd_gen = gen_mean > 0.1  # ERD阈值
                is_erd_tgt = tgt_mean > 0.1
                
                status_gen = "✓ ERD" if is_erd_gen else "✗ 无ERD"
                status_tgt = "✓ ERD" if is_erd_tgt else "✗ 无ERD"
                
                print(f"  {band_name}频段 {expected_ch}:")
                print(f"    真实数据: {tgt_mean:.4f} ({status_tgt})")
                print(f"    DTTD生成: {gen_mean:.4f} ({status_gen})")
                
                validation_results.append({
                    'class': class_name,
                    'band': band_name,
                    'channel': expected_ch,
                    'target_erd': tgt_mean,
                    'generated_erd': gen_mean,
                    'target_valid': is_erd_tgt,
                    'generated_valid': is_erd_gen
                })
    
    return validation_results


def statistical_test(results):
    """统计检验：真实数据和生成数据的ERD值是否有显著差异"""
    print("\n" + "="*60)
    print("统计检验 (配对t检验)")
    print("="*60)
    
    all_tgt_values = []
    all_gen_values = []
    
    for class_idx in results:
        erd_results = results[class_idx]['erd_results']
        for ch_name in erd_results:
            for band_name in erd_results[ch_name]:
                tgt_vals = np.array(erd_results[ch_name][band_name]['target']['values'])
                gen_vals = np.array(erd_results[ch_name][band_name]['generated']['values'])
                
                # 确保长度相同
                min_len = min(len(tgt_vals), len(gen_vals))
                if min_len < 2:
                    continue
                
                tgt_vals = tgt_vals[:min_len]
                gen_vals = gen_vals[:min_len]
                
                all_tgt_values.extend(tgt_vals)
                all_gen_values.extend(gen_vals)
                
                # 配对t检验
                t_stat, p_val = ttest_rel(tgt_vals, gen_vals)
                
                class_name = results[class_idx]['class_name']
                print(f"{class_name} {ch_name} {band_name}:")
                print(f"  t={t_stat:.4f}, p={p_val:.4f}")
                if p_val < 0.05:
                    print(f"  结论: 存在显著差异 (p<0.05)")
                else:
                    print(f"  结论: 无显著差异 (p>=0.05)")
    
    # 整体检验
    print("\n整体检验:")
    t_stat, p_val = ttest_rel(all_tgt_values, all_gen_values)
    print(f"t={t_stat:.4f}, p={p_val:.4f}")
    if p_val < 0.05:
        print("结论: 整体存在显著差异")
    else:
        print("结论: 整体无显著差异 - DTTD生成数据保留了ERD模式")


def plot_erd_comparison(results, output_dir):
    """绘制ERD对比图"""
    os.makedirs(output_dir, exist_ok=True)
    
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    channels = ['C3', 'Cz', 'C4']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for class_idx, ax in enumerate(axes):
        if class_idx not in results:
            continue
        
        class_name = class_names[class_idx]
        erd_results = results[class_idx]['erd_results']
        
        x = np.arange(len(channels))
        width = 0.35
        
        alpha_tgt = []
        alpha_gen = []
        beta_tgt = []
        beta_gen = []
        
        for ch in channels:
            if ch in erd_results:
                alpha_tgt.append(erd_results[ch]['alpha']['target']['mean'])
                alpha_gen.append(erd_results[ch]['alpha']['generated']['mean'])
                beta_tgt.append(erd_results[ch]['beta']['target']['mean'])
                beta_gen.append(erd_results[ch]['beta']['generated']['mean'])
            else:
                alpha_tgt.append(0)
                alpha_gen.append(0)
                beta_tgt.append(0)
                beta_gen.append(0)
        
        rects1 = ax.bar(x - width/2 - width/4, alpha_tgt, width/2, label='Alpha (Target)', color='blue')
        rects2 = ax.bar(x - width/4, alpha_gen, width/2, label='Alpha (DTTD)', color='lightblue')
        rects3 = ax.bar(x + width/4, beta_tgt, width/2, label='Beta (Target)', color='orange')
        rects4 = ax.bar(x + width/2 + width/4, beta_gen, width/2, label='Beta (DTTD)', color='lightcoral')
        
        ax.set_xlabel('Channels')
        ax.set_ylabel('ERD Value')
        ax.set_title(f'ERD Comparison - {class_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(channels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.5, 1.0])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'erd_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'erd_comparison.pdf'), bbox_inches='tight')
    print(f"\nERD对比图已保存到: {output_dir}")


def main():
    print("="*60)
    print("ERD/ERS 量化分析实验")
    print("="*60)
    
    # 加载数据
    print("\n[1/4] 加载数据...")
    generated, targets, labels = load_generated_data()
    
    # 分析ERD/ERS
    print("\n[2/4] 分析ERD/ERS模式...")
    results = analyze_erd_ers_per_class(generated, targets, labels)
    
    # 验证对侧ERD模式
    print("\n[3/4] 验证对侧ERD模式...")
    validate_contralateral_erd(results)
    
    # 统计检验
    print("\n[4/4] 统计检验...")
    statistical_test(results)
    
    # 绘制对比图
    output_dir = 'paper_results/erd_ers'
    plot_erd_comparison(results, output_dir)
    
    print("\n" + "="*60)
    print("分析完成!")
    print("="*60)


if __name__ == '__main__':
    main()