#!/usr/bin/env python3
"""
生成PSD对比图和拓扑图
使用真实的实验数据（真实数据 vs DTTD生成数据）
选择某类的某个具体样本进行展示
"""

import os
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


def select_sample_with_erd(generated, targets, labels, class_idx):
    """为指定类别选择一个ERD模式明显的样本
    
    运动想象的神经生理学特征（ERD - Event-Related Desynchronization）：
    - 左手运动想象：右侧运动皮层(C4)的alpha功率相对于左侧(C3)更低
    - 右手运动想象：左侧运动皮层(C3)的alpha功率相对于右侧(C4)更低
    - 脚运动想象：中央区域(Cz)的alpha功率相对于两侧更低
    - 舌头运动想象：中央区域激活
    
    选择标准：
    1. 属于指定类别
    2. 显示明显的对侧ERD模式（关键通道alpha功率低于对侧）
    3. 真实数据和DTTD重建数据都显示相似的ERD模式
    """
    fs = 250
    alpha_band = (8, 13)
    
    # 通道索引
    C3_idx = 7   # 左运动皮层
    Cz_idx = 9   # 中央
    C4_idx = 11  # 右运动皮层
    
    # 找到该类别的所有样本
    class_mask = labels == class_idx
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
    # 对应的关键通道: 左手->C4, 右手->C3, 脚->Cz, 舌头->Cz
    channel_indices = [11, 7, 9, 9]  # C4, C3, Cz, Cz
    channel_names = ['C4', 'C3', 'Cz', 'Cz']
    
    fs = 250
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    print("  Selecting samples with clear ERD patterns...")
    for idx, (ax, class_name, color, ch_idx, ch_name) in enumerate(
            zip(axes, classes, colors, channel_indices, channel_names)):
        
        # 选择该类别的一个ERD模式明显的样本
        sample_idx = select_sample_with_erd(generated, targets, labels, idx)
        
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
    
    fs = 250
    alpha_band = (8, 13)
    
    # 创建高分辨率网格
    resolution = 200
    xi = np.linspace(-1, 1, resolution)
    yi = np.linspace(-1, 1, resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # 头部半径
    head_radius = 0.85
    mask = np.sqrt(Xi**2 + Yi**2) <= head_radius
    
    # 创建2x4的图形 (2行: Real/DTTD, 4列: 4个类别)
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    
    print("  Selecting samples with clear ERD patterns...")
    for class_idx in range(4):
        # 使用ERD选择函数
        sample_idx = select_sample_with_erd(generated, targets, labels, class_idx)
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
        
        # 为每个类别添加独立的颜色条
        # 在第二行下方添加小颜色条
        cbar_ax = fig.add_axes([0.125 + class_idx * 0.21, 0.06, 0.15, 0.02])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=8)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['Low', '', 'High'])
    
    # 添加总标题
    fig.suptitle('Alpha Band (8-13 Hz) Topographic Maps Comparison\n(Blue=ERD/Activation, Red=High Power)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.12, wspace=0.08, hspace=0.15)
    
    # 保存
    output_path_png = os.path.join(output_dir, 'topographic_maps.png')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path_png}")
    
    output_path_pdf = os.path.join(output_dir, 'topographic_maps.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path_pdf}")
    
    plt.close()


def main():
    print("=" * 60)
    print("Generating PSD and Topographic Map Figures")
    print("Using REAL experimental data - SINGLE SAMPLE per class")
    print("=" * 60)
    
    output_dir = 'paper_results/figures'
    
    # 加载真实数据
    print("\n[0/2] Loading real experimental data...")
    generated, targets, labels = load_real_data()
    
    print("\n[1/2] Generating PSD comparison plot (4 classes)...")
    generate_psd_plot(output_dir, generated, targets, labels)
    
    print("\n[2/2] Generating topographic maps (4 classes)...")
    generate_topographic_maps(output_dir, generated, targets, labels)
    
    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
