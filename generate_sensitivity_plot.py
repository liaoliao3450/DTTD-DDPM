#!/usr/bin/env python3
"""生成超参数敏感度分析曲线图"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_sensitivity_data():
    """加载敏感度分析数据"""
    data_path = 'results/sensitivity_cross_session/sensitivity_cross_session.json'
    with open(data_path, 'r') as f:
        return json.load(f)

def plot_sensitivity_curves(data, output_dir='paper_results/figures'):
    """绘制敏感度分析曲线图"""
    os.makedirs(output_dir, exist_ok=True)
    
    baseline_acc = data['baseline']['accuracy'] * 100
    
    # 创建双子图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 噪声尺度敏感性
    ax1 = axes[0]
    noise_scales = []
    noise_accs = []
    for scale, result in data['noise_scale'].items():
        noise_scales.append(float(scale))
        noise_accs.append(result['accuracy'] * 100)
    
    # 排序
    sorted_idx = np.argsort(noise_scales)
    noise_scales = np.array(noise_scales)[sorted_idx]
    noise_accs = np.array(noise_accs)[sorted_idx]
    
    ax1.plot(noise_scales, noise_accs, 'b-o', linewidth=2, markersize=8, label='DTTD')
    ax1.axhline(y=baseline_acc, color='r', linestyle='--', linewidth=2, label=f'Baseline ({baseline_acc:.2f}%)')
    
    # 标记最优点
    best_idx = np.argmax(noise_accs)
    ax1.scatter([noise_scales[best_idx]], [noise_accs[best_idx]], 
                color='green', s=150, zorder=5, marker='*', label=f'Best ({noise_accs[best_idx]:.2f}%)')
    
    ax1.set_xlabel('Noise Scale', fontsize=12)
    ax1.set_ylabel('Cross-Session Accuracy (%)', fontsize=12)
    ax1.set_title('(a) Noise Scale Sensitivity', fontsize=14)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([54, 64])
    
    # 引导尺度敏感性
    ax2 = axes[1]
    guidance_scales = []
    guidance_accs = []
    for scale, result in data['guidance_scale'].items():
        guidance_scales.append(float(scale))
        guidance_accs.append(result['accuracy'] * 100)
    
    # 排序
    sorted_idx = np.argsort(guidance_scales)
    guidance_scales = np.array(guidance_scales)[sorted_idx]
    guidance_accs = np.array(guidance_accs)[sorted_idx]
    
    ax2.plot(guidance_scales, guidance_accs, 'b-o', linewidth=2, markersize=8, label='DTTD')
    ax2.axhline(y=baseline_acc, color='r', linestyle='--', linewidth=2, label=f'Baseline ({baseline_acc:.2f}%)')
    
    # 标记最优点
    best_idx = np.argmax(guidance_accs)
    ax2.scatter([guidance_scales[best_idx]], [guidance_accs[best_idx]], 
                color='green', s=150, zorder=5, marker='*', label=f'Best ({guidance_accs[best_idx]:.2f}%)')
    
    ax2.set_xlabel('Guidance Scale', fontsize=12)
    ax2.set_ylabel('Cross-Session Accuracy (%)', fontsize=12)
    ax2.set_title('(b) Guidance Scale Sensitivity', fontsize=14)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([54, 64])
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, 'sensitivity_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    # 也保存PDF版本用于论文
    output_path_pdf = os.path.join(output_dir, 'sensitivity_analysis.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Saved: {output_path_pdf}")
    
    plt.close()

if __name__ == '__main__':
    data = load_sensitivity_data()
    plot_sensitivity_curves(data)
    print("Sensitivity analysis plots generated successfully!")
