"""
生成各被试分类准确率条形图
用于论文展示被试内、跨会话、跨被试三种场景的结果
使用最新数据: paper_results/classification/classification_results.json
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_results():
    """加载最新的分类结果数据"""
    # 使用 legacy_eeg_classifier_eval 生成的结果
    with open('paper_results/classification_legacy/classification_legacy_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def plot_scenario_barplot(data, scenario_key, scenario_name, output_dir):
    """绘制单个场景的各被试条形图"""
    scenario_data = data[scenario_key]
    
    # 提取各被试数据
    subjects = []
    baseline_9ch = []
    baseline_22ch = []
    dttd_acc = []
    
    for key in sorted(scenario_data.keys()):
        if key.startswith('S'):
            subjects.append(key)
            baseline_9ch.append(scenario_data[key]['baseline_9ch']['accuracy'] * 100)
            baseline_22ch.append(scenario_data[key]['baseline_22ch']['accuracy'] * 100)
            dttd_acc.append(scenario_data[key]['dttd_augmented']['accuracy'] * 100)
    
    x = np.arange(len(subjects))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width, baseline_9ch, width, label='9-ch Baseline', color='#3498DB', alpha=0.8)
    bars2 = ax.bar(x, baseline_22ch, width, label='22-ch Baseline', color='#2ECC71', alpha=0.8)
    bars3 = ax.bar(x + width, dttd_acc, width, label='DTTD (9→22)', color='#E74C3C', alpha=0.8)
    
    ax.set_xlabel('Subject', fontsize=16)
    ax.set_ylabel('Accuracy (%)', fontsize=16)
    ax.set_title(f'{scenario_name} Classification Accuracy', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    # 图例横向排列，放在图表顶部框内右上角
    ax.legend(loc='upper right', fontsize=14, framealpha=0.9, ncol=3)
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)
    for bar in bars3:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)
    
    # 计算统计量
    mean_9ch, std_9ch = np.mean(baseline_9ch), np.std(baseline_9ch)
    mean_22ch, std_22ch = np.mean(baseline_22ch), np.std(baseline_22ch)
    mean_dttd, std_dttd = np.mean(dttd_acc), np.std(dttd_acc)
    
    # 添加平均值线
    ax.axhline(y=mean_9ch, color='#3498DB', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(y=mean_22ch, color='#2ECC71', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(y=mean_dttd, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.7)
    
    plt.tight_layout()
    filename = scenario_name.replace('Within-Subject', 'within_session').replace('Cross-Session', 'cross_session').replace('Cross-Subject', 'cross_subject')
    png_path = os.path.join(output_dir, f'{filename}_per_subject.png')
    pdf_path = os.path.join(output_dir, f'{filename}_per_subject.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")
    plt.close()
    
    return {
        '9ch_mean': mean_9ch, '9ch_std': std_9ch,
        '22ch_mean': mean_22ch, '22ch_std': std_22ch,
        'dttd_mean': mean_dttd, 'dttd_std': std_dttd
    }

def plot_combined_summary(stats_dict, output_dir):
    """绘制三种场景的汇总对比图"""
    scenarios = ['Within-Subject', 'Cross-Session', 'Cross-Subject']
    scenario_keys = ['被试内', '跨会话', '跨被试']
    
    baseline_9ch = [stats_dict[s]['9ch_mean'] for s in scenario_keys]
    baseline_22ch = [stats_dict[s]['22ch_mean'] for s in scenario_keys]
    dttd_means = [stats_dict[s]['dttd_mean'] for s in scenario_keys]
    
    x = np.arange(len(scenarios))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, baseline_9ch, width, label='9-ch Baseline', color='#3498DB', alpha=0.8)
    bars2 = ax.bar(x, baseline_22ch, width, label='22-ch Baseline', color='#2ECC71', alpha=0.8)
    bars3 = ax.bar(x + width, dttd_means, width, label='DTTD (9→22)', color='#E74C3C', alpha=0.8)
    
    ax.set_xlabel('Evaluation Scenario', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Classification Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9, ncol=3)
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars3:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    png_path = os.path.join(output_dir, 'three_scenarios_comparison.png')
    pdf_path = os.path.join(output_dir, 'three_scenarios_comparison.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")
    plt.close()


def main():
    """主函数"""
    print("=" * 60)
    print("Generating per-subject classification bar plots")
    print("=" * 60)
    
    output_dir = 'paper_results/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载最新数据
    data = load_results()
    
    stats_dict = {}
    
    # 1. 被试内
    print("\n[1/4] Generating within-subject bar plot...")
    stats_dict['被试内'] = plot_scenario_barplot(data, 'within_subject', 'Within-Subject', output_dir)
    
    # 2. 跨会话
    print("\n[2/4] Generating cross-session bar plot...")
    stats_dict['跨会话'] = plot_scenario_barplot(data, 'cross_session', 'Cross-Session', output_dir)
    
    # 3. 跨被试
    print("\n[3/4] Generating cross-subject bar plot...")
    stats_dict['跨被试'] = plot_scenario_barplot(data, 'cross_subject', 'Cross-Subject', output_dir)
    
    # 4. 汇总图
    print("\n[4/4] Generating summary comparison plot...")
    plot_combined_summary(stats_dict, output_dir)
    
    print("\n" + "=" * 60)
    print("[OK] All bar plots generated!")
    print("=" * 60)
    
    # 输出论文表格数据
    print("\nPaper table data summary:")
    print("-" * 60)
    for scenario, name in [('被试内', 'Within-Subject'), ('跨会话', 'Cross-Session'), ('跨被试', 'Cross-Subject')]:
        s = stats_dict[scenario]
        print(f"{name}: 9ch {s['9ch_mean']:.2f}±{s['9ch_std']:.2f}%, "
              f"22ch {s['22ch_mean']:.2f}±{s['22ch_std']:.2f}%, "
              f"DTTD {s['dttd_mean']:.2f}±{s['dttd_std']:.2f}%")

if __name__ == '__main__':
    main()
