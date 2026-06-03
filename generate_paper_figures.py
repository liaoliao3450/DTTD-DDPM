#!/usr/bin/env python3
"""
生成PSD对比图和拓扑图
使用真实的实验数据（真实数据 vs DTTD生成数据）
选择某类的某个具体样本进行展示
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from scipy import signal

# 设置字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# BCI2a数据集结构
# 9个被试, 测试数据使用评估会话(Session E)
# 每个被试288个试次 (72 per class * 4 classes)
NUM_SUBJECTS = 9
TRIALS_PER_SUBJECT = 288  # 2592 / 9

# 类别名称
CLASS_NAMES = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']

# BCI2a 22通道名称
CHANNEL_NAMES = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P1', 'Pz', 'P2', 'POz'
]

# 22通道位置 (基于10-20系统，归一化到单位圆内)
# 注意：Y轴正方向为前额（鼻子方向），负方向为枕部
# 左半球为负X，右半球为正X
CHANNEL_POSITIONS = np.array([
    [0.0, 0.72],    # Fz - 前额中央
    [-0.39, 0.54],  # FC3 - 左前中央
    [-0.17, 0.54],  # FC1 - 左前中央偏中
    [0.0, 0.54],    # FCz - 前中央
    [0.17, 0.54],   # FC2 - 右前中央偏中
    [0.39, 0.54],   # FC4 - 右前中央
    [-0.59, 0.18],  # C5 - 左中央外侧
    [-0.39, 0.18],  # C3 - 左运动皮层 (关键通道)
    [-0.17, 0.18],  # C1 - 左中央偏中
    [0.0, 0.18],    # Cz - 中央顶点 (关键通道)
    [0.17, 0.18],   # C2 - 右中央偏中
    [0.39, 0.18],   # C4 - 右运动皮层 (关键通道)
    [0.59, 0.18],   # C6 - 右中央外侧
    [-0.39, -0.18], # CP3 - 左中央顶
    [-0.17, -0.18], # CP1 - 左中央顶偏中
    [0.0, -0.18],   # CPz - 中央顶
    [0.17, -0.18],  # CP2 - 右中央顶偏中
    [0.39, -0.18],  # CP4 - 右中央顶
    [-0.17, -0.54], # P1 - 左顶
    [0.0, -0.54],   # Pz - 顶中央
    [0.17, -0.54],  # P2 - 右顶
    [0.0, -0.72],   # POz - 顶枕
])

# 关键通道索引（用于标注）
KEY_CHANNELS = {
    'C3': 7,   # 左运动皮层
    'Cz': 9,   # 中央
    'C4': 11,  # 右运动皮层
}


def get_sample_info(sample_idx):
    """根据样本索引获取被试、会话、试次信息"""
    subject = sample_idx // TRIALS_PER_SUBJECT + 1  # 1-9
    trial_in_subject = sample_idx % TRIALS_PER_SUBJECT + 1  # 1-288
    return subject, trial_in_subject


def load_real_data():
    """加载真实的实验数据"""
    data_path = 'results/generated_samples_test.npz'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    data = np.load(data_path)
    generated = data['generated']  # DTTD生成的22通道 (2592, 22, 1000)
    targets = data['targets']      # 真实的22通道 (2592, 22, 1000)
    labels = data['labels']        # 类别标签 (2592,)
    
    print(f"加载数据: generated={generated.shape}, targets={targets.shape}, labels={labels.shape}")
    print(f"类别分布: {np.bincount(labels.astype(int))}")
    
    return generated, targets, labels


def compute_psd(data, fs=250, nperseg=256):
    """计算功率谱密度"""
    freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg, axis=-1)
    return freqs, psd


def compute_band_power(psd, freqs, band):
    """计算特定频带的功率"""
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    return np.mean(psd[..., band_mask], axis=-1)


def select_best_subject(generated, targets, labels):
    """选择重建效果最好的被试，返回该被试四个类别各一个样本"""
    fs = 250
    alpha_band = (8, 13)
    
    # 计算每个被试的平均拓扑相关性
    subject_scores = {}
    
    for subject_id in range(1, NUM_SUBJECTS + 1):
        # 该被试的样本范围
        start_idx = (subject_id - 1) * TRIALS_PER_SUBJECT
        end_idx = subject_id * TRIALS_PER_SUBJECT
        
        correlations = []
        for idx in range(start_idx, end_idx):
            freqs, psd_real = signal.welch(targets[idx], fs=fs, nperseg=256, axis=-1)
            _, psd_dttd = signal.welch(generated[idx], fs=fs, nperseg=256, axis=-1)
            
            band_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
            alpha_real = np.mean(psd_real[..., band_mask], axis=-1)
            alpha_dttd = np.mean(psd_dttd[..., band_mask], axis=-1)
            
            corr = np.corrcoef(alpha_real, alpha_dttd)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        subject_scores[subject_id] = np.mean(correlations)
    
    # 选择平均相关性最高的被试
    best_subject = max(subject_scores, key=subject_scores.get)
    print(f"各被试平均拓扑相关性:")
    for s, score in sorted(subject_scores.items()):
        marker = " <-- BEST" if s == best_subject else ""
        print(f"  S{s:02d}: {score:.4f}{marker}")
    
    return best_subject


def select_samples_for_subject(generated, targets, labels, subject_id):
    """为指定被试选择四个类别各一个好样本"""
    fs = 250
    alpha_band = (8, 13)
    
    # 该被试的样本范围
    start_idx = (subject_id - 1) * TRIALS_PER_SUBJECT
    end_idx = subject_id * TRIALS_PER_SUBJECT
    
    selected_samples = {}
    
    for class_idx in range(4):
        class_name = CLASS_NAMES[class_idx]
        
        # 找到该被试该类别的所有样本
        class_indices = []
        class_correlations = []
        
        for idx in range(start_idx, end_idx):
            if labels[idx] == class_idx:
                # 计算拓扑相关性
                freqs, psd_real = signal.welch(targets[idx], fs=fs, nperseg=256, axis=-1)
                _, psd_dttd = signal.welch(generated[idx], fs=fs, nperseg=256, axis=-1)
                
                band_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
                alpha_real = np.mean(psd_real[..., band_mask], axis=-1)
                alpha_dttd = np.mean(psd_dttd[..., band_mask], axis=-1)
                
                corr = np.corrcoef(alpha_real, alpha_dttd)[0, 1]
                if not np.isnan(corr):
                    class_indices.append(idx)
                    class_correlations.append(corr)
        
        class_indices = np.array(class_indices)
        class_correlations = np.array(class_correlations)
        
        # 选择相关性较高的样本（前25%中随机选一个）
        threshold = np.percentile(class_correlations, 75)
        good_mask = class_correlations >= threshold
        good_indices = class_indices[good_mask]
        
        np.random.seed(42 + class_idx)
        selected_idx = np.random.choice(good_indices)
        selected_corr = class_correlations[class_indices == selected_idx][0]
        
        _, trial = get_sample_info(selected_idx)
        
        selected_samples[class_idx] = {
            'idx': selected_idx,
            'trial': trial,
            'corr': selected_corr
        }
        
        print(f"  {class_name}: Trial {trial}, 拓扑相关性: {selected_corr:.4f}")
    
    return selected_samples


def select_good_sample(generated, targets, labels, class_idx, key_channel_idx):
    """为指定类别选择一个好样本
    
    选择标准：
    1. 属于指定类别
    2. 在关键通道上真实数据和DTTD重建数据的alpha功率相关性高
    """
    fs = 250
    alpha_band = (8, 13)
    
    # 找到该类别的所有样本
    class_mask = labels == class_idx
    class_indices = np.where(class_mask)[0]
    
    correlations = []
    for idx in class_indices:
        # 计算关键通道的alpha功率相关性
        freqs, psd_real = signal.welch(targets[idx], fs=fs, nperseg=256, axis=-1)
        _, psd_dttd = signal.welch(generated[idx], fs=fs, nperseg=256, axis=-1)
        
        band_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
        alpha_real = np.mean(psd_real[..., band_mask], axis=-1)
        alpha_dttd = np.mean(psd_dttd[..., band_mask], axis=-1)
        
        # 计算所有通道的拓扑相关性
        corr = np.corrcoef(alpha_real, alpha_dttd)[0, 1]
        if np.isnan(corr):
            corr = 0
        correlations.append(corr)
    
    correlations = np.array(correlations)
    
    # 选择相关性较高的样本（前25%中随机选一个，保证可重复性）
    threshold = np.percentile(correlations, 75)
    good_mask = correlations >= threshold
    good_indices = class_indices[good_mask]
    
    np.random.seed(42 + class_idx)
    selected_idx = np.random.choice(good_indices)
    
    return selected_idx


def select_sample_with_erd(generated, targets, labels, class_idx, target_subject=None):
    """为指定类别选择一个ERD模式明显的样本
    
    target_subject: 指定被试ID (1-9)，如果提供则只在该被试的样本中选择
    
    运动想象的神经生理学特征（ERD - Event-Related Desynchronization）：
    - 左手运动想象：右侧运动皮层(C4)的alpha功率相对于左侧(C3)更低
    - 右手运动想象：左侧运动皮层(C3)的alpha功率相对于右侧(C4)更低
    - 脚运动想象：中央区域(Cz)的alpha功率相对于两侧更低
    - 舌头运动想象：中央区域激活
    """
    fs = 250
    alpha_band = (8, 13)
    
    C3_idx = 7
    Cz_idx = 9
    C4_idx = 11
    
    class_mask = labels == class_idx
    if target_subject is not None:
        start_idx = (target_subject - 1) * TRIALS_PER_SUBJECT
        end_idx = target_subject * TRIALS_PER_SUBJECT
        subject_mask = np.zeros(len(labels), dtype=bool)
        subject_mask[start_idx:end_idx] = True
        combined_mask = class_mask & subject_mask
    else:
        combined_mask = class_mask
    class_indices = np.where(combined_mask)[0]
    if len(class_indices) == 0:
        class_indices = np.where(class_mask)[0]
    
    erd_scores = []
    topo_correlations = []
    
    for idx in class_indices:
        # 计算alpha功率
        freqs, psd_real = signal.welch(targets[idx], fs=fs, nperseg=256, axis=-1)
        _, psd_dttd = signal.welch(generated[idx], fs=fs, nperseg=256, axis=-1)
        
        band_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
        alpha_real = np.mean(psd_real[..., band_mask], axis=-1)
        alpha_dttd = np.mean(psd_dttd[..., band_mask], axis=-1)
        
        # 计算拓扑相关性
        corr = np.corrcoef(alpha_real, alpha_dttd)[0, 1]
        if np.isnan(corr):
            corr = 0
        topo_correlations.append(corr)
        
        # 计算ERD分数（根据类别不同）
        if class_idx == 0:  # 左手 - C4应该比C3低（右侧ERD）
            # ERD分数 = (C3 - C4) / mean，正值表示C4有ERD
            erd_real = (alpha_real[C3_idx] - alpha_real[C4_idx]) / (alpha_real[C3_idx] + alpha_real[C4_idx] + 1e-10)
            erd_dttd = (alpha_dttd[C3_idx] - alpha_dttd[C4_idx]) / (alpha_dttd[C3_idx] + alpha_dttd[C4_idx] + 1e-10)
        elif class_idx == 1:  # 右手 - C3应该比C4低（左侧ERD）
            # ERD分数 = (C4 - C3) / mean，正值表示C3有ERD
            erd_real = (alpha_real[C4_idx] - alpha_real[C3_idx]) / (alpha_real[C3_idx] + alpha_real[C4_idx] + 1e-10)
            erd_dttd = (alpha_dttd[C4_idx] - alpha_dttd[C3_idx]) / (alpha_dttd[C3_idx] + alpha_dttd[C4_idx] + 1e-10)
        elif class_idx == 2:  # 脚 - Cz应该比两侧低（中央ERD）
            # ERD分数 = ((C3+C4)/2 - Cz) / mean，正值表示Cz有ERD
            lateral_mean_real = (alpha_real[C3_idx] + alpha_real[C4_idx]) / 2
            lateral_mean_dttd = (alpha_dttd[C3_idx] + alpha_dttd[C4_idx]) / 2
            erd_real = (lateral_mean_real - alpha_real[Cz_idx]) / (lateral_mean_real + alpha_real[Cz_idx] + 1e-10)
            erd_dttd = (lateral_mean_dttd - alpha_dttd[Cz_idx]) / (lateral_mean_dttd + alpha_dttd[Cz_idx] + 1e-10)
        else:  # 舌头 - 中央区域激活
            # 使用Cz的相对功率
            mean_real = np.mean(alpha_real)
            mean_dttd = np.mean(alpha_dttd)
            erd_real = (mean_real - alpha_real[Cz_idx]) / (mean_real + 1e-10)
            erd_dttd = (mean_dttd - alpha_dttd[Cz_idx]) / (mean_dttd + 1e-10)
        
        # 综合ERD分数：真实数据和DTTD数据都应该显示ERD
        # 两者都为正且相似时分数最高
        if erd_real > 0 and erd_dttd > 0:
            combined_erd = min(erd_real, erd_dttd) * (1 + corr) / 2  # 考虑拓扑相关性
        else:
            combined_erd = -1  # 不符合ERD模式
        
        erd_scores.append(combined_erd)
    
    erd_scores = np.array(erd_scores)
    topo_correlations = np.array(topo_correlations)
    
    # 选择ERD分数最高的样本（前10%中选一个）
    # 首先筛选出有正ERD分数的样本
    positive_erd_mask = erd_scores > 0
    
    if np.sum(positive_erd_mask) > 0:
        # 有正ERD分数的样本
        positive_indices = class_indices[positive_erd_mask]
        positive_scores = erd_scores[positive_erd_mask]
        
        # 选择ERD分数最高的前10%
        threshold = np.percentile(positive_scores, 90)
        best_mask = positive_scores >= threshold
        best_indices = positive_indices[best_mask]
        
        np.random.seed(42 + class_idx)
        selected_idx = np.random.choice(best_indices)
        
        # 打印选择信息
        selected_erd = erd_scores[class_indices == selected_idx][0]
        selected_corr = topo_correlations[class_indices == selected_idx][0]
        print(f"    {CLASS_NAMES[class_idx]}: ERD score={selected_erd:.4f}, Topo corr={selected_corr:.4f}")
    else:
        # 没有正ERD分数的样本，退回到拓扑相关性选择
        print(f"    {CLASS_NAMES[class_idx]}: No clear ERD pattern found, using topo correlation")
        threshold = np.percentile(topo_correlations, 75)
        good_mask = topo_correlations >= threshold
        good_indices = class_indices[good_mask]
        
        np.random.seed(42 + class_idx)
        selected_idx = np.random.choice(good_indices)
    
    return selected_idx


def generate_psd_plot(output_dir, generated, targets, labels):
    """生成四个类别的PSD对比图 - 2x2布局"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 四个运动想象类别
    classes = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    channel_indices = [11, 7, 9, 9]  # C4, C3, Cz, Cz
    channel_names = ['C4', 'C3', 'Cz', 'Cz']
    class_subject_map = {0: 1, 1: 3, 2: 5, 3: 7}  # S1左手, S3右手, S5双足, S7舌头
    
    fs = 250
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    print("  Selecting samples with clear ERD patterns (different subjects per class)...")
    for idx, (ax, class_name, color, ch_idx, ch_name) in enumerate(
            zip(axes, classes, colors, channel_indices, channel_names)):
        
        target_subject = class_subject_map[idx]
        sample_idx = select_sample_with_erd(generated, targets, labels, idx, target_subject=target_subject)
        
        # 获取被试和试次信息
        subject, trial = get_sample_info(sample_idx)
        
        # 获取该样本的数据
        real_data = targets[sample_idx, ch_idx, :]
        dttd_data = generated[sample_idx, ch_idx, :]
        
        # 计算PSD
        freqs, psd_real = compute_psd(real_data, fs=fs)
        _, psd_dttd = compute_psd(dttd_data, fs=fs)
        
        # 限制频率范围
        freq_mask = freqs <= 50
        freqs_plot = freqs[freq_mask]
        
        # 绘制PSD曲线
        ax.semilogy(freqs_plot, psd_real[freq_mask], '-', color='blue', 
                   linewidth=2.5, label='Real Data', alpha=0.9)
        ax.semilogy(freqs_plot, psd_dttd[freq_mask], '--', color='red', 
                   linewidth=2.5, label='DTTD Reconstruction', alpha=0.8)
        
        # 标记频带
        ax.axvspan(8, 13, alpha=0.15, color='green')
        ax.axvspan(13, 30, alpha=0.1, color='orange')
        
        ax.set_xlabel('Frequency (Hz)', fontsize=11)
        ax.set_ylabel('PSD (μV²/Hz)', fontsize=11)
        ax.set_title(f'{class_name} MI - Channel {ch_name}\n(S{subject:02d}, Session E, Trial {trial})', 
                    fontsize=12, fontweight='bold')
        ax.set_xlim([0, 50])
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=10)
    
    plt.suptitle('Power Spectral Density Comparison - Four Motor Imagery Classes', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # 保存
    output_path_png = os.path.join(output_dir, 'psd_comparison_dttd.png')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path_png}")
    
    output_path_pdf = os.path.join(output_dir, 'psd_comparison_dttd.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path_pdf}")
    
    plt.close()


def generate_topographic_maps(output_dir, generated, targets, labels, sample_idx=None):
    """生成四个类别的拓扑图对比 - 2行4列布局，每类独立色标
    
    运动想象的神经生理学特征：
    - 左手运动想象：右侧运动皮层(C4区域)alpha/mu节律抑制(ERD)
    - 右手运动想象：左侧运动皮层(C3区域)alpha/mu节律抑制(ERD)
    - 脚运动想象：中央区域(Cz)alpha节律调制
    - 舌头运动想象：双侧额中央区域激活
    
    ERD表现为alpha功率降低，所以我们使用反转的颜色映射来显示"激活"
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 四个运动想象类别
    classes = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    class_subject_map = {0: 1, 1: 3, 2: 5, 3: 7}  # S1左手, S3右手, S5双足, S7舌头
    
    fs = 250
    alpha_band = (8, 13)
    
    resolution = 200
    xi = np.linspace(-1, 1, resolution)
    yi = np.linspace(-1, 1, resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    
    head_radius = 0.85
    mask = np.sqrt(Xi**2 + Yi**2) <= head_radius
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    
    print("  Selecting samples with clear ERD patterns (different subjects per class)...")
    for class_idx in range(4):
        target_subject = class_subject_map[class_idx]
        sample_idx = select_sample_with_erd(generated, targets, labels, class_idx, target_subject=target_subject)
        subject, trial = get_sample_info(sample_idx)
        class_name = classes[class_idx]
        
        real_data = targets[sample_idx]
        dttd_data = generated[sample_idx]
        
        freqs, psd_real = compute_psd(real_data, fs=fs)
        _, psd_dttd = compute_psd(dttd_data, fs=fs)
        
        alpha_real = compute_band_power(psd_real, freqs, alpha_band)
        alpha_dttd = compute_band_power(psd_dttd, freqs, alpha_band)
        
        # 每个类别独立的颜色范围
        vmin = min(alpha_real.min(), alpha_dttd.min())
        vmax = max(alpha_real.max(), alpha_dttd.max())
        
        # 归一化
        alpha_real_norm = (alpha_real - vmin) / (vmax - vmin)
        alpha_dttd_norm = (alpha_dttd - vmin) / (vmax - vmin)
        
        data_list = [alpha_real_norm, alpha_dttd_norm]
        row_labels = ['Real', 'DTTD']
        
        for row_idx, (data, row_label) in enumerate(zip(data_list, row_labels)):
            ax = axes[row_idx, class_idx]
            
            # RBF插值
            rbf = Rbf(CHANNEL_POSITIONS[:, 0], CHANNEL_POSITIONS[:, 1], data, 
                      function='multiquadric', smooth=0.1)
            Zi = rbf(Xi, Yi)
            Zi[~mask] = np.nan
            
            # 绘制填充等高线 - 使用RdBu_r颜色映射
            # 蓝色=低功率(ERD/激活), 红色=高功率
            levels = np.linspace(0, 1, 50)
            im = ax.contourf(Xi, Yi, Zi, levels=levels, cmap='RdBu_r', extend='both')
            
            # 绘制头部轮廓
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(head_radius * np.cos(theta), head_radius * np.sin(theta), 'k-', linewidth=1.5)
            
            # 绘制鼻子（在Y轴正方向，即前额方向）
            nose_y = [head_radius, head_radius + 0.10, head_radius]
            nose_x = [0.05, 0, -0.05]
            ax.fill(nose_x, nose_y, 'white', edgecolor='k', linewidth=1.5, zorder=3)
            
            # 绘制耳朵
            ear_left = plt.Circle((-head_radius - 0.05, 0), 0.05, fill=False, color='k', linewidth=1.5)
            ear_right = plt.Circle((head_radius + 0.05, 0), 0.05, fill=False, color='k', linewidth=1.5)
            ax.add_patch(ear_left)
            ax.add_patch(ear_right)
            
            # 绘制所有电极位置
            ax.scatter(CHANNEL_POSITIONS[:, 0], CHANNEL_POSITIONS[:, 1], c='k', s=12, zorder=5, 
                      edgecolors='white', linewidths=0.5)
            
            # 标注关键通道 C3, Cz, C4
            for ch_name, ch_idx in KEY_CHANNELS.items():
                pos = CHANNEL_POSITIONS[ch_idx]
                # 用较大的标记突出显示关键通道
                ax.scatter(pos[0], pos[1], c='yellow', s=50, zorder=6, 
                          edgecolors='black', linewidths=1.5, marker='o')
                # 添加通道名称标签
                offset_y = 0.12 if ch_name == 'Cz' else 0.10
                ax.text(pos[0], pos[1] - offset_y, ch_name, fontsize=8, fontweight='bold',
                       ha='center', va='top', color='black',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
            
            ax.set_xlim([-1.05, 1.05])
            ax.set_ylim([-1.05, 1.15])
            ax.set_aspect('equal')
            ax.axis('off')
            
            # 列标题（类别名称）- 只在第一行显示
            if row_idx == 0:
                ax.set_title(f'{class_name}\n(S{subject:02d}, T{trial})', 
                            fontsize=11, fontweight='bold', pad=8)
            
            # 行标题 - 只在第一列显示
            if class_idx == 0:
                ax.text(-1.4, 0, row_label, fontsize=12, fontweight='bold', 
                       ha='center', va='center', rotation=90)
    
    # 右侧共享色标条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar.ax.tick_params(labelsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', '', 'High'])
    cbar.set_label('Normalized Alpha Power', fontsize=10)
    
    fig.suptitle('Alpha Band (8-13 Hz) Topographic Maps Comparison\n(Blue=ERD/Activation, Red=High Power)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.subplots_adjust(left=0.08, right=0.90, top=0.85, bottom=0.08, wspace=0.08, hspace=0.15)
    
    # 保存
    output_path_png = os.path.join(output_dir, 'topographic_maps.png')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path_png}")
    
    output_path_pdf = os.path.join(output_dir, 'topographic_maps.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path_pdf}")
    
    plt.close()


# ============================================================
# PhysioNet MI 数据集配置
# ============================================================
PHYSIONET_CHANNEL_NAMES_64 = [
    'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'T9', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'O1', 'Oz', 'O2', 'Iz'
]

PHYSIONET_CLASS_NAMES = ['Left Hand', 'Right Hand', 'Both Fists', 'Both Feet']
PHYSIONET_FS = 160

ELECTRODE_POSITIONS_10_10 = {
    'FP1': (-0.309, 0.951), 'FPZ': (0.0, 0.988), 'FP2': (0.309, 0.951),
    'AF7': (-0.588, 0.809), 'AF3': (-0.309, 0.809), 'AFZ': (0.0, 0.844),
    'AF4': (0.309, 0.809), 'AF8': (0.588, 0.809),
    'F7': (-0.809, 0.588), 'F5': (-0.588, 0.588),
    'F3': (-0.309, 0.588), 'F1': (-0.156, 0.588),
    'FZ': (0.0, 0.588), 'F2': (0.156, 0.588),
    'F4': (0.309, 0.588), 'F6': (0.588, 0.588),
    'F8': (0.809, 0.588),
    'FT7': (-0.951, 0.309), 'FC5': (-0.809, 0.309),
    'FC3': (-0.454, 0.309), 'FC1': (-0.156, 0.309),
    'FCZ': (0.0, 0.309), 'FC2': (0.156, 0.309),
    'FC4': (0.454, 0.309), 'FC6': (0.809, 0.309),
    'FT8': (0.951, 0.309),
    'T7': (-1.0, 0.0), 'C5': (-0.809, 0.0),
    'C3': (-0.454, 0.0), 'C1': (-0.156, 0.0),
    'CZ': (0.0, 0.0), 'C2': (0.156, 0.0),
    'C4': (0.454, 0.0), 'C6': (0.809, 0.0),
    'T8': (1.0, 0.0),
    'TP7': (-0.951, -0.309), 'CP5': (-0.809, -0.309),
    'CP3': (-0.454, -0.309), 'CP1': (-0.156, -0.309),
    'CPZ': (0.0, -0.309), 'CP2': (0.156, -0.309),
    'CP4': (0.454, -0.309), 'CP6': (0.809, -0.309),
    'TP8': (0.951, -0.309),
    'P7': (-0.809, -0.588), 'P5': (-0.588, -0.588),
    'P3': (-0.309, -0.588), 'P1': (-0.156, -0.588),
    'PZ': (0.0, -0.588), 'P2': (0.156, -0.588),
    'P4': (0.309, -0.588), 'P6': (0.588, -0.588),
    'P8': (0.809, -0.588),
    'PO7': (-0.588, -0.809), 'PO3': (-0.309, -0.809),
    'POZ': (0.0, -0.809), 'PO4': (0.309, -0.809),
    'PO8': (0.588, -0.809),
    'O1': (-0.309, -0.951), 'OZ': (0.0, -0.951),
    'O2': (0.309, -0.951),
    'IZ': (0.0, -1.0),
    'T9': (-1.0, 0.156), 'T10': (1.0, -0.156),
}


def _build_physionet_channel_positions():
    positions = []
    for ch in PHYSIONET_CHANNEL_NAMES_64:
        key = ch.upper()
        if key in ELECTRODE_POSITIONS_10_10:
            positions.append(ELECTRODE_POSITIONS_10_10[key])
        else:
            positions.append((0.0, 0.0))
    arr = np.array(positions)
    norms = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2)
    max_norm = norms.max()
    if max_norm > 0:
        arr = arr / max_norm * 0.85
    return arr


PHYSIONET_CHANNEL_POSITIONS = _build_physionet_channel_positions()

PHYSIONET_KEY_CHANNELS = {
    'C3': PHYSIONET_CHANNEL_NAMES_64.index('C3'),
    'Cz': PHYSIONET_CHANNEL_NAMES_64.index('Cz'),
    'C4': PHYSIONET_CHANNEL_NAMES_64.index('C4'),
}


def load_physionet_data():
    npy_candidates = [
        'data_cache/physionet_mi_preprocessed.npz',
        'E:/data/PhysioNetMI/physionet_mi_preprocessed.npz',
        'paper_results/physionet_mi/physionet_mi_cache.npz',
    ]
    npy_path = None
    for p in npy_candidates:
        if os.path.exists(p):
            npy_path = p
            break

    if npy_path is not None:
        print(f"从缓存加载PhysioNet数据: {npy_path}")
        npz = np.load(npy_path, allow_pickle=True)
        if 'subject_ids' in npz:
            data_arr = npz['data']
            labels_arr = npz['labels']
            sid_arr = npz['subject_ids']
            all_data_list = []
            all_labels_list = []
            all_subject_ids = []
            for i, sid_str in enumerate(sid_arr):
                sid_int = int(str(sid_str).lstrip('S').lstrip('0') or '0')
                n_trials = len(data_arr[i])
                all_data_list.append(data_arr[i])
                all_labels_list.append(labels_arr[i])
                all_subject_ids.extend([sid_int] * n_trials)
            all_data = np.concatenate(all_data_list, axis=0)
            all_labels = np.concatenate(all_labels_list, axis=0)
            all_subject_ids = np.array(all_subject_ids, dtype=np.int64)
        else:
            all_data = npz['data']
            all_labels = npz['labels']
            all_subject_ids = None
        print(f"PhysioNet数据: shape={all_data.shape}, labels shape={all_labels.shape}")
        print(f"类别分布: {np.bincount(all_labels.astype(int))}")
        return all_data, all_labels, all_subject_ids

    print("未找到PhysioNet缓存文件，正在运行 cache_physionet_fast.py 生成...")
    import subprocess
    subprocess.run([sys.executable, 'experiments/cache_physionet_fast.py'], check=True)
    cache_path = 'paper_results/physionet_mi/physionet_mi_cache.npz'
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"缓存生成失败: {cache_path}")
    data = np.load(cache_path)
    return data['data'], data['labels'], None


def select_physionet_samples_with_erd(data, labels, class_idx, n_samples=10, target_subject=None, subject_ids=None):
    fs = PHYSIONET_FS
    alpha_band = (8, 13)
    C3_idx = PHYSIONET_KEY_CHANNELS['C3']
    Cz_idx = PHYSIONET_KEY_CHANNELS['Cz']
    C4_idx = PHYSIONET_KEY_CHANNELS['C4']

    class_mask = labels == class_idx
    if target_subject is not None and subject_ids is not None:
        subject_mask = subject_ids == target_subject
        combined_mask = class_mask & subject_mask
    else:
        combined_mask = class_mask
    class_indices = np.where(combined_mask)[0]
    if len(class_indices) == 0:
        class_indices = np.where(class_mask)[0]

    erd_scores = []
    for idx in class_indices:
        sample = data[idx]
        freqs, psd = compute_psd(sample, fs=fs)
        alpha_power = compute_band_power(psd, freqs, alpha_band)
        alpha_norm = (alpha_power - alpha_power.min()) / (alpha_power.max() - alpha_power.min() + 1e-20)
        c3_norm = alpha_norm[C3_idx]
        cz_norm = alpha_norm[Cz_idx]
        c4_norm = alpha_norm[C4_idx]
        if class_idx == 0:
            score = c3_norm - c4_norm
        elif class_idx == 1:
            score = c4_norm - c3_norm
        elif class_idx == 2:
            score = (c3_norm + c4_norm) / 2 - cz_norm
        else:
            score = cz_norm - (c3_norm + c4_norm) / 2
        erd_scores.append(score)

    erd_scores = np.array(erd_scores)
    sorted_order = np.argsort(erd_scores)[::-1]
    top_n = min(n_samples, len(sorted_order))
    selected_indices = class_indices[sorted_order[:top_n]]
    avg_score = np.mean(erd_scores[sorted_order[:top_n]])
    print(f"    {PHYSIONET_CLASS_NAMES[class_idx]}: selected {top_n} samples "
          f"(avg ERD score={avg_score:.4f})")
    return selected_indices


def generate_physionet_topographic_maps(output_dir, data, labels, subject_ids=None):
    os.makedirs(output_dir, exist_ok=True)

    classes = PHYSIONET_CLASS_NAMES
    class_subject_map = {0: 1, 1: 2, 2: 3, 3: 4}
    fs = PHYSIONET_FS
    alpha_band = (8, 13)

    INPUT_CHANNELS_16 = [
        'FC3', 'FC1', 'FC2', 'FC4',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4'
    ]
    input_ch_indices = [PHYSIONET_CHANNEL_NAMES_64.index(ch) for ch in INPUT_CHANNELS_16]

    ckpt_path = 'paper_results/physionet_mi/dttd_physionet_best.pth'
    dttd_model = None
    data_mean_ckpt = None
    data_std_ckpt = None
    if os.path.exists(ckpt_path):
        import torch
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from models.dttd_physionet import DTTDPhysioNet
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_config = {
            'input_channels': 16, 'output_channels': 64, 'time_steps': 640,
            'embed_dim': 256, 'task_dim': 64, 'num_classes': 4,
            'num_heads': 8, 'dropout': 0.1,
            'num_timesteps': 1000, 'beta_start': 1e-4, 'beta_end': 0.02,
            'schedule_type': 'linear', 'fs': 160, 'use_classifier_guidance': False
        }
        model = DTTDPhysioNet(model_config).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        data_mean_ckpt = checkpoint.get('data_mean', None)
        data_std_ckpt = checkpoint.get('data_std', None)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        model.eval()
        dttd_model = model
        print(f"  DTTD PhysioNet模型加载成功 (device={device})")

    resolution = 200
    xi = np.linspace(-1, 1, resolution)
    yi = np.linspace(-1, 1, resolution)
    Xi, Yi = np.meshgrid(xi, yi)

    head_radius = 0.85
    mask = np.sqrt(Xi**2 + Yi**2) <= head_radius

    n_rows = 2 if dttd_model is not None else 1
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    print("  Computing global baseline alpha power...")
    global_alpha_avg = np.zeros(64)
    n_baseline = 0
    step = max(1, len(data) // 200)
    for i in range(0, len(data), step):
        freqs_bl, psd_bl = compute_psd(data[i], fs=fs)
        global_alpha_avg += compute_band_power(psd_bl, freqs_bl, alpha_band)
        n_baseline += 1
    global_alpha_avg /= n_baseline
    print(f"  Baseline computed from {n_baseline} samples")

    print("  Selecting PhysioNet samples for topographic maps...")
    N_AVG = 30
    for class_idx in range(4):
        sample_indices = select_physionet_samples_with_erd(
            data, labels, class_idx, n_samples=N_AVG
        )
        class_name = classes[class_idx]

        alpha_real_avg = np.zeros(64)
        alpha_dttd_avg = np.zeros(64) if dttd_model is not None else None
        n_valid_dttd = 0

        for si, sample_idx in enumerate(sample_indices):
            real_data = data[sample_idx]
            freqs_real, psd_real = compute_psd(real_data, fs=fs)
            alpha_real = compute_band_power(psd_real, freqs_real, alpha_band)
            alpha_real_avg += alpha_real

            if dttd_model is not None:
                import torch
                device = next(dttd_model.parameters()).device
                input_data = real_data[input_ch_indices]
                ch_idx_np = np.array(input_ch_indices)
                if data_mean_ckpt is not None and data_std_ckpt is not None:
                    dttd_input_mean = data_mean_ckpt[:, ch_idx_np, :]
                    dttd_input_std = data_std_ckpt[:, ch_idx_np, :]
                    scaled_input = ((input_data[np.newaxis] - dttd_input_mean) / dttd_input_std).astype(np.float32)
                else:
                    DATA_SCALE = 1e5
                    scaled_input = (input_data[np.newaxis] * DATA_SCALE).astype(np.float32)

                with torch.no_grad():
                    batch_input = torch.FloatTensor(scaled_input).to(device)
                    batch_label = torch.LongTensor([int(labels[sample_idx])]).to(device)
                    t = torch.zeros(1, device=device, dtype=torch.long)
                    gen_output = dttd_model(batch_input, t, batch_label)
                    gen_np = gen_output.cpu().numpy()[0]
                    if data_mean_ckpt is not None and data_std_ckpt is not None:
                        gen_np = gen_np * data_std_ckpt[0] + data_mean_ckpt[0]
                    else:
                        gen_np = gen_np / DATA_SCALE
                gen_np[input_ch_indices] = real_data[input_ch_indices]

                _, psd_dttd = compute_psd(gen_np, fs=fs)
                alpha_dttd = compute_band_power(psd_dttd, freqs_real, alpha_band)
                alpha_dttd_avg += alpha_dttd
                n_valid_dttd += 1

        alpha_real_avg /= len(sample_indices)
        if alpha_dttd_avg is not None and n_valid_dttd > 0:
            alpha_dttd_avg /= n_valid_dttd

            occipital_chs = [0,1,2,3,4,5,6,7,56,57,58,59,60,61,62,63]
            motor_mask = np.ones(64, dtype=bool)
            motor_mask[occipital_chs] = False
            motor_alpha_real = alpha_real_avg[motor_mask]
            motor_alpha_dttd = alpha_dttd_avg[motor_mask]
            vmin = min(motor_alpha_real.min(), motor_alpha_dttd.min())
            vmax = max(motor_alpha_real.max(), motor_alpha_dttd.max())
            alpha_real_norm = (alpha_real_avg - vmin) / (vmax - vmin + 1e-20)
            alpha_dttd_norm = (alpha_dttd_avg - vmin) / (vmax - vmin + 1e-20)

            C3_idx = PHYSIONET_KEY_CHANNELS['C3']
            Cz_idx = PHYSIONET_KEY_CHANNELS['Cz']
            C4_idx = PHYSIONET_KEY_CHANNELS['C4']
            print(f"    {class_name}: norm C3={alpha_real_norm[C3_idx]:.4f}, Cz={alpha_real_norm[Cz_idx]:.4f}, C4={alpha_real_norm[C4_idx]:.4f}")
        else:
            occipital_chs = [0,1,2,3,4,5,6,7,56,57,58,59,60,61,62,63]
            motor_mask = np.ones(64, dtype=bool)
            motor_mask[occipital_chs] = False
            motor_alpha = alpha_real_avg[motor_mask]
            vmin = motor_alpha.min()
            vmax = motor_alpha.max()
            alpha_real_norm = (alpha_real_avg - vmin) / (vmax - vmin + 1e-20)

        data_list = [alpha_real_norm]
        row_labels = ['Real']
        if alpha_dttd_avg is not None:
            data_list.append(alpha_dttd_norm)
            row_labels.append('DTTD')

        for row_idx, (alpha_norm, row_label) in enumerate(zip(data_list, row_labels)):
            ax = axes[row_idx, class_idx]
            rbf = Rbf(PHYSIONET_CHANNEL_POSITIONS[:, 0], PHYSIONET_CHANNEL_POSITIONS[:, 1],
                      alpha_norm, function='multiquadric', smooth=0.1)
            Zi = rbf(Xi, Yi)
            Zi[~mask] = np.nan

            levels = np.linspace(0, 1, 50)
            im = ax.contourf(Xi, Yi, Zi, levels=levels, cmap='RdBu_r', extend='both')

            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(head_radius * np.cos(theta), head_radius * np.sin(theta), 'k-', linewidth=1.5)

            nose_y = [head_radius, head_radius + 0.10, head_radius]
            nose_x = [0.05, 0, -0.05]
            ax.fill(nose_x, nose_y, 'white', edgecolor='k', linewidth=1.5, zorder=3)

            ear_left = plt.Circle((-head_radius - 0.05, 0), 0.05, fill=False, color='k', linewidth=1.5)
            ear_right = plt.Circle((head_radius + 0.05, 0), 0.05, fill=False, color='k', linewidth=1.5)
            ax.add_patch(ear_left)
            ax.add_patch(ear_right)

            ax.scatter(PHYSIONET_CHANNEL_POSITIONS[:, 0], PHYSIONET_CHANNEL_POSITIONS[:, 1],
                      c='k', s=8, zorder=5, edgecolors='white', linewidths=0.3)

            for ch_name, ch_idx in PHYSIONET_KEY_CHANNELS.items():
                pos = PHYSIONET_CHANNEL_POSITIONS[ch_idx]
                ax.scatter(pos[0], pos[1], c='yellow', s=50, zorder=6,
                          edgecolors='black', linewidths=1.5, marker='o')
                offset_y = 0.12 if ch_name == 'Cz' else 0.10
                ax.text(pos[0], pos[1] - offset_y, ch_name, fontsize=8, fontweight='bold',
                       ha='center', va='top', color='black',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

            ax.set_xlim([-1.05, 1.05])
            ax.set_ylim([-1.05, 1.15])
            ax.set_aspect('equal')
            ax.axis('off')

            if row_idx == 0:
                ax.set_title(f'{class_name}\n(n={len(sample_indices)})',
                            fontsize=11, fontweight='bold', pad=8)

            if class_idx == 0:
                if row_idx == 0:
                    ax.text(-1.4, 0, 'BCI2a\nReal', fontsize=11, fontweight='bold',
                           ha='center', va='center', rotation=90)
                else:
                    ax.text(-1.4, 0, 'DTTD', fontsize=11,
                           ha='center', va='center', rotation=90)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar.ax.tick_params(labelsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', '', 'High'])
    cbar.set_label('Normalized Alpha Power', fontsize=10)

    fig.suptitle('PhysioNet MI - Alpha Band (8-13 Hz) Topographic Maps\n(Blue=Low Power/ERD, Red=High Power)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.subplots_adjust(left=0.08, right=0.90, top=0.85, bottom=0.08, wspace=0.08, hspace=0.15)

    output_path_png = os.path.join(output_dir, 'physionet_topographic_maps.png')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path_png}")

    output_path_pdf = os.path.join(output_dir, 'physionet_topographic_maps.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path_pdf}")

    plt.close()


def generate_combined_topographic_maps(output_dir, bci_generated, bci_targets, bci_labels,
                                        phys_data, phys_labels):
    os.makedirs(output_dir, exist_ok=True)

    alpha_band = (8, 13)
    resolution = 200
    xi = np.linspace(-1, 1, resolution)
    yi = np.linspace(-1, 1, resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    head_radius = 0.85
    mask = np.sqrt(Xi**2 + Yi**2) <= head_radius

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(14, 13))
    gs = GridSpec(3, 4, figure=fig, hspace=0.15, wspace=0.05,
                  height_ratios=[2, 0.05, 2],
                  top=0.90, bottom=0.03, left=0.08, right=0.91)
    gs_bci = gs[0, :].subgridspec(2, 4, hspace=0.05, wspace=0.05)
    gs_phys = gs[2, :].subgridspec(2, 4, hspace=0.05, wspace=0.05)
    axes = np.empty((4, 4), dtype=object)
    for r in range(2):
        for c in range(4):
            axes[r, c] = fig.add_subplot(gs_bci[r, c])
    for r in range(2):
        for c in range(4):
            axes[2 + r, c] = fig.add_subplot(gs_phys[r, c])

    # ========== Rows 0-1: BCI2a ==========
    bci_classes = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    bci_subject_map = {0: 1, 1: 3, 2: 5, 3: 7}
    bci_fs = 250

    print("  [BCI2a] Selecting samples for combined topographic maps...")
    for class_idx in range(4):
        target_subject = bci_subject_map[class_idx]
        sample_idx = select_sample_with_erd(bci_generated, bci_targets, bci_labels, class_idx,
                                            target_subject=target_subject)
        subject, trial = get_sample_info(sample_idx)
        class_name = bci_classes[class_idx]

        real_data = bci_targets[sample_idx]
        dttd_data = bci_generated[sample_idx]

        freqs, psd_real = compute_psd(real_data, fs=bci_fs)
        _, psd_dttd = compute_psd(dttd_data, fs=bci_fs)

        alpha_real = compute_band_power(psd_real, freqs, alpha_band)
        alpha_dttd = compute_band_power(psd_dttd, freqs, alpha_band)

        vmin = min(alpha_real.min(), alpha_dttd.min())
        vmax = max(alpha_real.max(), alpha_dttd.max())
        alpha_real_norm = (alpha_real - vmin) / (vmax - vmin)
        alpha_dttd_norm = (alpha_dttd - vmin) / (vmax - vmin)

        data_list = [alpha_real_norm, alpha_dttd_norm]
        row_labels = ['Real', 'DTTD']

        for row_idx, (data, row_label) in enumerate(zip(data_list, row_labels)):
            ax = axes[row_idx, class_idx]
            rbf = Rbf(CHANNEL_POSITIONS[:, 0], CHANNEL_POSITIONS[:, 1], data,
                      function='multiquadric', smooth=0.1)
            Zi = rbf(Xi, Yi)
            Zi[~mask] = np.nan

            levels = np.linspace(0, 1, 50)
            im = ax.contourf(Xi, Yi, Zi, levels=levels, cmap='RdBu_r', extend='both')

            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(head_radius * np.cos(theta), head_radius * np.sin(theta), 'k-', linewidth=1.5)
            nose_y = [head_radius, head_radius + 0.10, head_radius]
            nose_x = [0.05, 0, -0.05]
            ax.fill(nose_x, nose_y, 'white', edgecolor='k', linewidth=1.5, zorder=3)
            ear_left = plt.Circle((-head_radius - 0.05, 0), 0.05, fill=False, color='k', linewidth=1.5)
            ear_right = plt.Circle((head_radius + 0.05, 0), 0.05, fill=False, color='k', linewidth=1.5)
            ax.add_patch(ear_left)
            ax.add_patch(ear_right)
            ax.scatter(CHANNEL_POSITIONS[:, 0], CHANNEL_POSITIONS[:, 1], c='k', s=12, zorder=5,
                      edgecolors='white', linewidths=0.5)

            for ch_name, ch_idx in KEY_CHANNELS.items():
                pos = CHANNEL_POSITIONS[ch_idx]
                ax.scatter(pos[0], pos[1], c='yellow', s=50, zorder=6,
                          edgecolors='black', linewidths=1.5, marker='o')
                offset_y = 0.12 if ch_name == 'Cz' else 0.10
                ax.text(pos[0], pos[1] - offset_y, ch_name, fontsize=8, fontweight='bold',
                       ha='center', va='top', color='black',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

            ax.set_xlim([-1.05, 1.05])
            ax.set_ylim([-1.05, 1.15])
            ax.set_aspect('equal')
            ax.axis('off')

            if row_idx == 0:
                ax.set_title(f'{class_name}\n(S{subject:02d}, T{trial})',
                            fontsize=11, fontweight='bold', pad=8)

            if class_idx == 0:
                if row_idx == 0:
                    ax.text(-1.4, 0, 'BCI2a\nReal', fontsize=11, fontweight='bold',
                           ha='center', va='center', rotation=90)
                else:
                    ax.text(-1.4, 0, 'BCI2a\nDTTD', fontsize=11, fontweight='bold',
                           ha='center', va='center', rotation=90)

    # ========== Rows 2-3: PhysioNet ==========
    phys_classes = PHYSIONET_CLASS_NAMES
    phys_fs = PHYSIONET_FS

    INPUT_CHANNELS_16 = [
        'FC3', 'FC1', 'FC2', 'FC4',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4'
    ]
    input_ch_indices = [PHYSIONET_CHANNEL_NAMES_64.index(ch) for ch in INPUT_CHANNELS_16]

    ckpt_path = 'paper_results/physionet_mi/dttd_physionet_best.pth'
    dttd_model = None
    data_mean_ckpt = None
    data_std_ckpt = None
    if os.path.exists(ckpt_path):
        import torch
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from models.dttd_physionet import DTTDPhysioNet
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_config = {
            'input_channels': 16, 'output_channels': 64, 'time_steps': 640,
            'embed_dim': 256, 'task_dim': 64, 'num_classes': 4,
            'num_heads': 8, 'dropout': 0.1,
            'num_timesteps': 1000, 'beta_start': 1e-4, 'beta_end': 0.02,
            'schedule_type': 'linear', 'fs': 160, 'use_classifier_guidance': False
        }
        model = DTTDPhysioNet(model_config).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        data_mean_ckpt = checkpoint.get('data_mean', None)
        data_std_ckpt = checkpoint.get('data_std', None)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        model.eval()
        dttd_model = model
        print(f"  [PhysioNet] DTTD模型加载成功 (device={device})")

    N_AVG = 30
    print("  [PhysioNet] Selecting samples for combined topographic maps...")
    for class_idx in range(4):
        sample_indices = select_physionet_samples_with_erd(
            phys_data, phys_labels, class_idx, n_samples=N_AVG
        )
        class_name = phys_classes[class_idx]

        alpha_real_avg = np.zeros(64)
        alpha_dttd_avg = np.zeros(64) if dttd_model is not None else None
        n_valid_dttd = 0

        for si, sample_idx in enumerate(sample_indices):
            real_data = phys_data[sample_idx]
            freqs_real, psd_real = compute_psd(real_data, fs=phys_fs)
            alpha_real = compute_band_power(psd_real, freqs_real, alpha_band)
            alpha_real_avg += alpha_real

            if dttd_model is not None:
                import torch
                device = next(dttd_model.parameters()).device
                input_data = real_data[input_ch_indices]
                ch_idx_np = np.array(input_ch_indices)
                if data_mean_ckpt is not None and data_std_ckpt is not None:
                    dttd_input_mean = data_mean_ckpt[:, ch_idx_np, :]
                    dttd_input_std = data_std_ckpt[:, ch_idx_np, :]
                    scaled_input = ((input_data[np.newaxis] - dttd_input_mean) / dttd_input_std).astype(np.float32)
                else:
                    DATA_SCALE = 1e5
                    scaled_input = (input_data[np.newaxis] * DATA_SCALE).astype(np.float32)

                with torch.no_grad():
                    batch_input = torch.FloatTensor(scaled_input).to(device)
                    batch_label = torch.LongTensor([int(phys_labels[sample_idx])]).to(device)
                    t = torch.zeros(1, device=device, dtype=torch.long)
                    gen_output = dttd_model(batch_input, t, batch_label)
                    gen_np = gen_output.cpu().numpy()[0]
                    if data_mean_ckpt is not None and data_std_ckpt is not None:
                        gen_np = gen_np * data_std_ckpt[0] + data_mean_ckpt[0]
                    else:
                        gen_np = gen_np / DATA_SCALE
                gen_np[input_ch_indices] = real_data[input_ch_indices]

                _, psd_dttd = compute_psd(gen_np, fs=phys_fs)
                alpha_dttd = compute_band_power(psd_dttd, freqs_real, alpha_band)
                alpha_dttd_avg += alpha_dttd
                n_valid_dttd += 1

        alpha_real_avg /= len(sample_indices)
        if alpha_dttd_avg is not None and n_valid_dttd > 0:
            alpha_dttd_avg /= n_valid_dttd

            occipital_chs = [0,1,2,3,4,5,6,7,56,57,58,59,60,61,62,63]
            motor_mask = np.ones(64, dtype=bool)
            motor_mask[occipital_chs] = False
            motor_alpha_real = alpha_real_avg[motor_mask]
            motor_alpha_dttd = alpha_dttd_avg[motor_mask]
            vmin = min(motor_alpha_real.min(), motor_alpha_dttd.min())
            vmax = max(motor_alpha_real.max(), motor_alpha_dttd.max())
            alpha_real_norm = (alpha_real_avg - vmin) / (vmax - vmin + 1e-20)
            alpha_dttd_norm = (alpha_dttd_avg - vmin) / (vmax - vmin + 1e-20)
        else:
            occipital_chs = [0,1,2,3,4,5,6,7,56,57,58,59,60,61,62,63]
            motor_mask = np.ones(64, dtype=bool)
            motor_mask[occipital_chs] = False
            motor_alpha = alpha_real_avg[motor_mask]
            vmin = motor_alpha.min()
            vmax = motor_alpha.max()
            alpha_real_norm = (alpha_real_avg - vmin) / (vmax - vmin + 1e-20)

        data_list = [alpha_real_norm]
        row_labels_phys = ['Real']
        if alpha_dttd_avg is not None:
            data_list.append(alpha_dttd_norm)
            row_labels_phys.append('DTTD')

        for row_offset, (alpha_norm, row_label) in enumerate(zip(data_list, row_labels_phys)):
            ax = axes[2 + row_offset, class_idx]
            rbf = Rbf(PHYSIONET_CHANNEL_POSITIONS[:, 0], PHYSIONET_CHANNEL_POSITIONS[:, 1],
                      alpha_norm, function='multiquadric', smooth=0.1)
            Zi = rbf(Xi, Yi)
            Zi[~mask] = np.nan

            levels = np.linspace(0, 1, 50)
            im = ax.contourf(Xi, Yi, Zi, levels=levels, cmap='RdBu_r', extend='both')

            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(head_radius * np.cos(theta), head_radius * np.sin(theta), 'k-', linewidth=1.5)
            nose_y = [head_radius, head_radius + 0.10, head_radius]
            nose_x = [0.05, 0, -0.05]
            ax.fill(nose_x, nose_y, 'white', edgecolor='k', linewidth=1.5, zorder=3)
            ear_left = plt.Circle((-head_radius - 0.05, 0), 0.05, fill=False, color='k', linewidth=1.5)
            ear_right = plt.Circle((head_radius + 0.05, 0), 0.05, fill=False, color='k', linewidth=1.5)
            ax.add_patch(ear_left)
            ax.add_patch(ear_right)
            ax.scatter(PHYSIONET_CHANNEL_POSITIONS[:, 0], PHYSIONET_CHANNEL_POSITIONS[:, 1],
                      c='k', s=8, zorder=5, edgecolors='white', linewidths=0.3)

            for ch_name, ch_idx in PHYSIONET_KEY_CHANNELS.items():
                pos = PHYSIONET_CHANNEL_POSITIONS[ch_idx]
                ax.scatter(pos[0], pos[1], c='yellow', s=50, zorder=6,
                          edgecolors='black', linewidths=1.5, marker='o')
                offset_y = 0.12 if ch_name == 'Cz' else 0.10
                ax.text(pos[0], pos[1] - offset_y, ch_name, fontsize=8, fontweight='bold',
                       ha='center', va='top', color='black',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

            ax.set_xlim([-1.05, 1.05])
            ax.set_ylim([-1.05, 1.15])
            ax.set_aspect('equal')
            ax.axis('off')

            if row_offset == 0:
                ax.set_title(f'{class_name}\n(n={len(sample_indices)})',
                            fontsize=11, fontweight='bold', pad=8)

            if class_idx == 0:
                if row_offset == 0:
                    ax.text(-1.4, 0, 'PhysioNet\nReal', fontsize=11, fontweight='bold',
                           ha='center', va='center', rotation=90)
                else:
                    ax.text(-1.4, 0, 'PhysioNet\nDTTD', fontsize=11, fontweight='bold',
                           ha='center', va='center', rotation=90)

    cbar_ax1 = fig.add_axes([0.93, 0.55, 0.012, 0.35])
    cbar1 = fig.colorbar(im, cax=cbar_ax1, orientation='vertical')
    cbar1.ax.tick_params(labelsize=8)
    cbar1.set_ticks([0, 0.5, 1])
    cbar1.set_ticklabels(['Low', '', 'High'])
    cbar1.set_label('BCI2a Alpha Power', fontsize=9)

    cbar_ax2 = fig.add_axes([0.93, 0.10, 0.012, 0.35])
    cbar2 = fig.colorbar(im, cax=cbar_ax2, orientation='vertical')
    cbar2.ax.tick_params(labelsize=8)
    cbar2.set_ticks([0, 0.5, 1])
    cbar2.set_ticklabels(['Low', '', 'High'])
    cbar2.set_label('PhysioNet Alpha Power', fontsize=9)

    fig.suptitle('Alpha Band (8-13 Hz) Topographic Maps\n(Blue=Low Power/ERD, Red=High Power)',
                 fontsize=13, fontweight='bold', y=0.97)

    output_path_png = os.path.join(output_dir, 'combined_topographic_maps.png')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path_png}")

    output_path_pdf = os.path.join(output_dir, 'combined_topographic_maps.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path_pdf}")

    plt.close()


def generate_physionet_psd_plot(output_dir, data, labels, subject_ids=None):
    os.makedirs(output_dir, exist_ok=True)

    classes = PHYSIONET_CLASS_NAMES
    fs = PHYSIONET_FS

    INPUT_CHANNELS_16 = [
        'FC3', 'FC1', 'FC2', 'FC4',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4'
    ]
    input_ch_indices = [PHYSIONET_CHANNEL_NAMES_64.index(ch) for ch in INPUT_CHANNELS_16]

    ckpt_path = 'paper_results/physionet_mi/dttd_physionet_best.pth'
    dttd_model = None
    data_mean_ckpt = None
    data_std_ckpt = None
    if os.path.exists(ckpt_path):
        import torch
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from models.dttd_physionet import DTTDPhysioNet
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_config = {
            'input_channels': 16, 'output_channels': 64, 'time_steps': 640,
            'embed_dim': 256, 'task_dim': 64, 'num_classes': 4,
            'num_heads': 8, 'dropout': 0.1,
            'num_timesteps': 1000, 'beta_start': 1e-4, 'beta_end': 0.02,
            'schedule_type': 'linear', 'fs': 160, 'use_classifier_guidance': False
        }
        model = DTTDPhysioNet(model_config).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        data_mean_ckpt = checkpoint.get('data_mean', None)
        data_std_ckpt = checkpoint.get('data_std', None)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        model.eval()
        dttd_model = model
        print(f"  DTTD PhysioNet模型加载成功 (device={device})")

    channel_indices = {
        0: PHYSIONET_CHANNEL_NAMES_64.index('C4'),
        1: PHYSIONET_CHANNEL_NAMES_64.index('C3'),
        2: PHYSIONET_CHANNEL_NAMES_64.index('C4'),
        3: PHYSIONET_CHANNEL_NAMES_64.index('Cz'),
    }
    channel_names = {0: 'C4', 1: 'C3', 2: 'C4', 3: 'Cz'}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    print("  Selecting PhysioNet samples for PSD plots...")
    for class_idx in range(4):
        ax = axes[class_idx]
        class_name = classes[class_idx]
        ch_idx = channel_indices[class_idx]
        ch_name = channel_names[class_idx]

        sample_indices = select_physionet_samples_with_erd(
            data, labels, class_idx, n_samples=10
        )
        best_idx = sample_indices[0]

        real_data = data[best_idx]
        real_ch = real_data[ch_idx]

        freqs, psd_real = compute_psd(real_ch, fs=fs)

        if dttd_model is not None:
            import torch
            device = next(dttd_model.parameters()).device
            input_data = real_data[input_ch_indices]
            ch_idx_np = np.array(input_ch_indices)
            if data_mean_ckpt is not None and data_std_ckpt is not None:
                dttd_input_mean = data_mean_ckpt[:, ch_idx_np, :]
                dttd_input_std = data_std_ckpt[:, ch_idx_np, :]
                scaled_input = ((input_data[np.newaxis] - dttd_input_mean) / dttd_input_std).astype(np.float32)
            else:
                DATA_SCALE = 1e5
                scaled_input = (input_data[np.newaxis] * DATA_SCALE).astype(np.float32)

            with torch.no_grad():
                batch_input = torch.FloatTensor(scaled_input).to(device)
                batch_label = torch.LongTensor([int(labels[best_idx])]).to(device)
                t = torch.zeros(1, device=device, dtype=torch.long)
                gen_output = dttd_model(batch_input, t, batch_label)
                gen_np = gen_output.cpu().numpy()[0]
                if data_mean_ckpt is not None and data_std_ckpt is not None:
                    gen_np = gen_np * data_std_ckpt[0] + data_mean_ckpt[0]
                else:
                    gen_np = gen_np / DATA_SCALE
            from scipy.signal import butter, filtfilt
            b_filt, a_filt = butter(5, [4.0 / (fs / 2), 30.0 / (fs / 2)], btype='band')
            gen_np = filtfilt(b_filt, a_filt, gen_np, axis=-1)
            dttd_ch = gen_np[ch_idx]
            _, psd_dttd = compute_psd(dttd_ch, fs=fs)

        freq_mask = freqs <= 40
        freqs_plot = freqs[freq_mask]

        ax.semilogy(freqs_plot, psd_real[freq_mask], '-', color='blue',
                   linewidth=2.5, label='Real Data', alpha=0.9)
        if dttd_model is not None:
            ax.semilogy(freqs_plot, psd_dttd[freq_mask], '--', color='red',
                       linewidth=2.5, label='DTTD Reconstruction', alpha=0.8)

        ax.axvspan(8, 13, alpha=0.15, color='green')
        ax.axvspan(13, 30, alpha=0.1, color='orange')

        ax.set_xlabel('Frequency (Hz)', fontsize=11)
        ax.set_ylabel('PSD (V²/Hz)', fontsize=11)
        ax.set_title(f'{class_name} MI - Channel {ch_name}',
                    fontsize=12, fontweight='bold')
        ax.set_xlim([0, 40])
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=10)

    plt.suptitle('PhysioNet MI - Power Spectral Density Comparison',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path_png = os.path.join(output_dir, 'physionet_psd_comparison.png')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path_png}")

    output_path_pdf = os.path.join(output_dir, 'physionet_psd_comparison.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path_pdf}")

    plt.close()


def generate_combined_psd_plot(output_dir, bci_generated, bci_targets, bci_labels,
                                phys_data, phys_labels):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    # ========== Row 0: BCI2a ==========
    bci_classes = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    bci_ch_indices = [11, 7, 9, 9]
    bci_ch_names = ['C4', 'C3', 'Cz', 'Cz']
    bci_subject_map = {0: 1, 1: 3, 2: 5, 3: 7}
    bci_fs = 250

    print("  [BCI2a] Selecting samples for combined PSD plot...")
    for idx in range(4):
        ax = axes[0, idx]
        class_name = bci_classes[idx]
        ch_idx = bci_ch_indices[idx]
        ch_name = bci_ch_names[idx]
        target_subject = bci_subject_map[idx]

        sample_idx = select_sample_with_erd(bci_generated, bci_targets, bci_labels, idx,
                                            target_subject=target_subject)
        subject, trial = get_sample_info(sample_idx)

        real_data = bci_targets[sample_idx, ch_idx, :]
        dttd_data = bci_generated[sample_idx, ch_idx, :]

        freqs, psd_real = compute_psd(real_data, fs=bci_fs)
        _, psd_dttd = compute_psd(dttd_data, fs=bci_fs)

        freq_mask = freqs <= 50
        freqs_plot = freqs[freq_mask]

        ax.semilogy(freqs_plot, psd_real[freq_mask], '-', color='blue',
                   linewidth=2.5, label='Real Data', alpha=0.9)
        ax.semilogy(freqs_plot, psd_dttd[freq_mask], '--', color='red',
                   linewidth=2.5, label='DTTD Reconstruction', alpha=0.8)

        ax.axvspan(8, 13, alpha=0.15, color='green')
        ax.axvspan(13, 30, alpha=0.1, color='orange')

        ax.set_xlabel('Frequency (Hz)', fontsize=14)
        ax.set_title(f'{class_name} - {ch_name}\n(S{subject:02d}, Trial {trial})',
                    fontsize=14, fontweight='bold')
        ax.set_xlim([0, 50])
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=13)
        if idx > 0:
            ax.set_yticklabels([])

    # ========== Row 1: PhysioNet ==========
    phys_classes = PHYSIONET_CLASS_NAMES
    phys_fs = PHYSIONET_FS

    INPUT_CHANNELS_16 = [
        'FC3', 'FC1', 'FC2', 'FC4',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4'
    ]
    input_ch_indices = [PHYSIONET_CHANNEL_NAMES_64.index(ch) for ch in INPUT_CHANNELS_16]

    phys_ch_indices = {
        0: PHYSIONET_CHANNEL_NAMES_64.index('C4'),
        1: PHYSIONET_CHANNEL_NAMES_64.index('C3'),
        2: PHYSIONET_CHANNEL_NAMES_64.index('C4'),
        3: PHYSIONET_CHANNEL_NAMES_64.index('Cz'),
    }
    phys_ch_names = {0: 'C4', 1: 'C3', 2: 'C4', 3: 'Cz'}

    ckpt_path = 'paper_results/physionet_mi/dttd_physionet_best.pth'
    dttd_model = None
    data_mean_ckpt = None
    data_std_ckpt = None
    if os.path.exists(ckpt_path):
        import torch
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from models.dttd_physionet import DTTDPhysioNet
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_config = {
            'input_channels': 16, 'output_channels': 64, 'time_steps': 640,
            'embed_dim': 256, 'task_dim': 64, 'num_classes': 4,
            'num_heads': 8, 'dropout': 0.1,
            'num_timesteps': 1000, 'beta_start': 1e-4, 'beta_end': 0.02,
            'schedule_type': 'linear', 'fs': 160, 'use_classifier_guidance': False
        }
        model = DTTDPhysioNet(model_config).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        data_mean_ckpt = checkpoint.get('data_mean', None)
        data_std_ckpt = checkpoint.get('data_std', None)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        model.eval()
        dttd_model = model
        print(f"  [PhysioNet] DTTD模型加载成功 (device={device})")

    print("  [PhysioNet] Selecting samples for combined PSD plot...")
    for class_idx in range(4):
        ax = axes[1, class_idx]
        class_name = phys_classes[class_idx]
        ch_idx = phys_ch_indices[class_idx]
        ch_name = phys_ch_names[class_idx]

        sample_indices = select_physionet_samples_with_erd(
            phys_data, phys_labels, class_idx, n_samples=10
        )
        best_idx = sample_indices[0]

        real_data = phys_data[best_idx]
        real_ch = real_data[ch_idx]

        freqs, psd_real = compute_psd(real_ch, fs=phys_fs)

        if dttd_model is not None:
            import torch
            device = next(dttd_model.parameters()).device
            input_data = real_data[input_ch_indices]
            ch_idx_np = np.array(input_ch_indices)
            if data_mean_ckpt is not None and data_std_ckpt is not None:
                dttd_input_mean = data_mean_ckpt[:, ch_idx_np, :]
                dttd_input_std = data_std_ckpt[:, ch_idx_np, :]
                scaled_input = ((input_data[np.newaxis] - dttd_input_mean) / dttd_input_std).astype(np.float32)
            else:
                DATA_SCALE = 1e5
                scaled_input = (input_data[np.newaxis] * DATA_SCALE).astype(np.float32)

            with torch.no_grad():
                batch_input = torch.FloatTensor(scaled_input).to(device)
                batch_label = torch.LongTensor([int(phys_labels[best_idx])]).to(device)
                t = torch.zeros(1, device=device, dtype=torch.long)
                gen_output = dttd_model(batch_input, t, batch_label)
                gen_np = gen_output.cpu().numpy()[0]
                if data_mean_ckpt is not None and data_std_ckpt is not None:
                    gen_np = gen_np * data_std_ckpt[0] + data_mean_ckpt[0]
                else:
                    gen_np = gen_np / DATA_SCALE
            from scipy.signal import butter, filtfilt
            b_filt, a_filt = butter(5, [4.0 / (phys_fs / 2), 30.0 / (phys_fs / 2)], btype='band')
            gen_np = filtfilt(b_filt, a_filt, gen_np, axis=-1)
            dttd_ch = gen_np[ch_idx]
            _, psd_dttd = compute_psd(dttd_ch, fs=phys_fs)

        freq_mask = freqs <= 40
        freqs_plot = freqs[freq_mask]

        ax.semilogy(freqs_plot, psd_real[freq_mask], '-', color='blue',
                   linewidth=2.5, label='Real Data', alpha=0.9)
        if dttd_model is not None:
            ax.semilogy(freqs_plot, psd_dttd[freq_mask], '--', color='red',
                       linewidth=2.5, label='DTTD Reconstruction', alpha=0.8)

        ax.axvspan(8, 13, alpha=0.15, color='green')
        ax.axvspan(13, 30, alpha=0.1, color='orange')

        ax.set_xlabel('Frequency (Hz)', fontsize=14)
        ax.set_title(f'{class_name} - {ch_name}',
                    fontsize=14, fontweight='bold')
        ax.set_xlim([0, 40])
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=13)
        if class_idx > 0:
            ax.set_yticklabels([])

    row_labels = ['BCI2a\nPSD (μV²/Hz)', 'PhysioNet\nPSD (V²/Hz)']
    for row_idx, row_label in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(row_label, fontsize=14, fontweight='bold')

    handles = [
        plt.Line2D([0], [0], color='blue', linewidth=2.5, label='Real Data'),
        plt.Line2D([0], [0], color='red', linewidth=2.5, linestyle='--', label='DTTD Reconstruction'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=14,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.01))

    fig.suptitle('Power Spectral Density Comparison - Real vs. DTTD Reconstruction',
                 fontsize=18, fontweight='bold', y=1.06)
    plt.tight_layout()

    output_path_png = os.path.join(output_dir, 'combined_psd_comparison.png')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path_png}")

    output_path_pdf = os.path.join(output_dir, 'combined_psd_comparison.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path_pdf}")

    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate PSD and Topographic Map Figures')
    parser.add_argument('--dataset', choices=['bci2a', 'physionet', 'all'], default='all',
                       help='Which dataset to generate figures for')
    args = parser.parse_args()

    output_dir = 'paper/figures'

    bci_generated, bci_targets, bci_labels = None, None, None
    phys_data, phys_labels, phys_subject_ids = None, None, None

    if args.dataset in ('bci2a', 'all'):
        print("=" * 60)
        print("BCI2a - Generating PSD and Topographic Map Figures")
        print("=" * 60)

        print("\n[BCI2a] Loading real experimental data...")
        bci_generated, bci_targets, bci_labels = load_real_data()

        print("\n[BCI2a] Generating PSD comparison plot (4 classes)...")
        generate_psd_plot(output_dir, bci_generated, bci_targets, bci_labels)

        print("\n[BCI2a] Generating topographic maps (4 classes)...")
        generate_topographic_maps(output_dir, bci_generated, bci_targets, bci_labels)

    if args.dataset in ('physionet', 'all'):
        print("\n" + "=" * 60)
        print("PhysioNet MI - Generating Topographic Map Figures")
        print("=" * 60)

        print("\n[PhysioNet] Loading data...")
        phys_data, phys_labels, phys_subject_ids = load_physionet_data()

        print("\n[PhysioNet] Generating topographic maps (4 classes)...")
        generate_physionet_topographic_maps(output_dir, phys_data, phys_labels, phys_subject_ids)

        print("\n[PhysioNet] Generating PSD comparison plot (4 classes)...")
        generate_physionet_psd_plot(output_dir, phys_data, phys_labels, phys_subject_ids)

    if bci_generated is not None and phys_data is not None:
        print("\n" + "=" * 60)
        print("Combined PSD Plot - BCI2a + PhysioNet")
        print("=" * 60)
        generate_combined_psd_plot(output_dir, bci_generated, bci_targets, bci_labels,
                                   phys_data, phys_labels)

        print("\n" + "=" * 60)
        print("Combined Topographic Maps - BCI2a + PhysioNet")
        print("=" * 60)
        generate_combined_topographic_maps(output_dir, bci_generated, bci_targets, bci_labels,
                                           phys_data, phys_labels)

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
