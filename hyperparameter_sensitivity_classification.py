"""
DTTD Enhanced 超参数敏感度分析 - 基于分类准确率

核心超参数：
1. noise_scale - 输入噪声比例（训练/推理时）
2. guidance_scale - 分类器引导强度
3. lambda_cls - 分类器引导损失权重（训练时）

评估指标：分类准确率（主要）、F1分数、Kappa系数
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.dttd_enhanced_v2 import DTTDEnhanced
from models import get_classifier
from data import get_bci2a_dataloaders
from utils import load_config, set_seed

# 设置matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_model(checkpoint_path, config, device):
    """加载模型"""
    model = DTTDEnhanced(config['model']).to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"模型加载成功")
    
    model.eval()
    return model


def collect_dataset(dataloader):
    """收集数据集"""
    data_list, label_list, ch_indices = [], [], None
    for batch_data in tqdm(dataloader, desc="收集数据"):
        if len(batch_data) == 4:
            target_data, batch_channel_idx, labels, _ = batch_data
        else:
            target_data, batch_channel_idx, labels = batch_data
        data_list.append(target_data)
        label_list.append(labels)
        if ch_indices is None:
            ch_indices = batch_channel_idx[0].tolist() if batch_channel_idx.dim() > 1 else batch_channel_idx.tolist()
    
    data = torch.cat(data_list, dim=0)
    labels = torch.cat(label_list, dim=0)
    return data, labels, ch_indices


def train_classifier(train_data, train_labels, device, num_channels=22, epochs=30):
    """训练分类器"""
    classifier = get_classifier('eegnet', num_channels=num_channels, num_classes=4, time_steps=1000).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    classifier.train()
    for epoch in range(epochs):
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    return classifier


def evaluate_classifier(classifier, data, labels, device):
    """评估分类器"""
    classifier.eval()
    dataset = TensorDataset(data, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_data, batch_labels in loader:
            outputs = classifier(batch_data.to(device))
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    kappa = cohen_kappa_score(all_labels, all_preds)
    return acc, f1, kappa


def reconstruct_with_params(model, data_22ch, ch_idx, labels, device, 
                           noise_scale=0.02, guidance_scale=3.0, use_label=False):
    """
    使用指定参数重建数据
    
    Args:
        use_label: 是否使用标签
                   False = 无条件生成（正确的评估方式）
                   True = 条件生成（会导致信息泄露）
    """
    dataset = TensorDataset(data_22ch, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    recon_list = []
    model.eval()
    
    with torch.no_grad():
        for batch_data, batch_labels in loader:
            input_9ch = batch_data[:, ch_idx, :].to(device)
            
            # 添加指定比例的噪声
            if noise_scale > 0:
                noisy_input = input_9ch + torch.randn_like(input_9ch) * noise_scale
            else:
                noisy_input = input_9ch
            
            t = torch.zeros(input_9ch.size(0), dtype=torch.long, device=device)
            
            # ⭐ 关键修改：默认不使用标签，避免信息泄露
            if use_label:
                batch_labels = batch_labels.to(device)
                # 使用指定的guidance_scale（条件生成）
                if guidance_scale > 1.0:
                    x0_pred_cond = model(noisy_input, t, batch_labels)
                    x0_pred_uncond = model(noisy_input, t, None)
                    reconstructed = x0_pred_uncond + guidance_scale * (x0_pred_cond - x0_pred_uncond)
                else:
                    reconstructed = model(noisy_input, t, batch_labels)
            else:
                # 无条件生成 - 正确的评估方式
                reconstructed = model(noisy_input, t, task_label=None)
            
            recon_list.append(reconstructed.cpu())
    
    return torch.cat(recon_list, dim=0)


def noise_scale_sensitivity(model, classifier, test_data, test_labels, ch_idx, 
                           device, baseline_acc):
    """
    噪声比例敏感度分析
    测试不同输入噪声水平对分类准确率的影响
    
    注意：使用无条件生成，避免信息泄露
    """
    print("\n" + "="*70)
    print("噪声比例 (noise_scale) 敏感度分析")
    print("="*70)
    
    noise_scales = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    results = {}
    
    for noise_scale in noise_scales:
        print(f"\n噪声比例 = {noise_scale*100:.0f}%")
        
        # 重建数据（无条件生成）
        recon_data = reconstruct_with_params(
            model, test_data, ch_idx, test_labels, device,
            noise_scale=noise_scale, guidance_scale=1.0, use_label=False
        )
        
        # 评估分类准确率
        acc, f1, kappa = evaluate_classifier(classifier, recon_data, test_labels, device)
        retention = acc / baseline_acc if baseline_acc > 0 else 0
        
        results[str(noise_scale)] = {
            'accuracy': float(acc),
            'f1': float(f1),
            'kappa': float(kappa),
            'retention': float(retention)
        }
        
        print(f"  准确率: {acc:.4f} (保持率: {retention:.2%})")
        print(f"  F1: {f1:.4f}, Kappa: {kappa:.4f}")
    
    return results


def guidance_scale_sensitivity(model, classifier, test_data, test_labels, ch_idx, 
                              device, baseline_acc):
    """
    分类器引导强度敏感度分析
    
    注意：guidance_scale > 1 时需要使用标签，这会导致信息泄露
    因此这个实验主要用于分析引导强度的影响，而非真实性能评估
    """
    print("\n" + "="*70)
    print("分类器引导强度 (guidance_scale) 敏感度分析")
    print("注意：GS>1时使用标签引导，存在信息泄露，仅供参数分析")
    print("="*70)
    
    guidance_scales = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]
    results = {}
    
    for gs in guidance_scales:
        print(f"\nGuidance Scale = {gs}")
        
        # 当gs <= 1时，使用无条件生成
        # 当gs > 1时，需要使用标签进行引导（存在信息泄露）
        use_label = gs > 1.0
        
        # 重建数据
        recon_data = reconstruct_with_params(
            model, test_data, ch_idx, test_labels, device,
            noise_scale=0.02, guidance_scale=gs, use_label=use_label
        )
        
        # 评估分类准确率
        acc, f1, kappa = evaluate_classifier(classifier, recon_data, test_labels, device)
        retention = acc / baseline_acc if baseline_acc > 0 else 0
        
        results[str(gs)] = {
            'accuracy': float(acc),
            'f1': float(f1),
            'kappa': float(kappa),
            'retention': float(retention),
            'uses_label': use_label  # 标记是否使用了标签
        }
        
        label_warning = " ⚠️(使用标签)" if use_label else ""
        print(f"  准确率: {acc:.4f} (保持率: {retention:.2%}){label_warning}")
        print(f"  F1: {f1:.4f}, Kappa: {kappa:.4f}")
    
    return results


def combined_sensitivity(model, classifier, test_data, test_labels, ch_idx, 
                        device, baseline_acc):
    """
    噪声比例和引导强度的组合敏感度分析
    
    仅测试无条件生成（use_label=False）的情况
    """
    print("\n" + "="*70)
    print("噪声比例敏感度分析（无条件生成）")
    print("="*70)
    
    noise_scales = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]
    
    results = {}
    best_acc = 0
    best_params = None
    
    for noise_scale in noise_scales:
        key = f"noise_{noise_scale}"
        print(f"\n噪声={noise_scale*100:.0f}%")
        
        # 无条件生成
        recon_data = reconstruct_with_params(
            model, test_data, ch_idx, test_labels, device,
            noise_scale=noise_scale, guidance_scale=1.0, use_label=False
        )
        
        # 评估分类准确率
        acc, f1, kappa = evaluate_classifier(classifier, recon_data, test_labels, device)
        retention = acc / baseline_acc if baseline_acc > 0 else 0
        
        results[key] = {
            'noise_scale': noise_scale,
            'accuracy': float(acc),
            'f1': float(f1),
            'kappa': float(kappa),
            'retention': float(retention)
        }
        
        print(f"  准确率: {acc:.4f} (保持率: {retention:.2%})")
        
        if acc > best_acc:
            best_acc = acc
            best_params = noise_scale
    
    print(f"\n最优噪声比例: {best_params}")
    print(f"最优准确率: {best_acc:.4f}")
    
    return results, best_params



def plot_sensitivity_results(results, output_dir, baseline_acc):
    """绘制敏感度分析结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 噪声比例 vs 准确率
    if 'noise_scale' in results:
        noise_data = results['noise_scale']
        noise_values = [float(k) * 100 for k in noise_data.keys()]  # 转换为百分比
        acc_values = [noise_data[k]['accuracy'] for k in noise_data.keys()]
        
        axes[0, 0].plot(noise_values, acc_values, 'o-', color='blue', linewidth=2, markersize=10)
        axes[0, 0].axhline(y=baseline_acc, color='red', linestyle='--', label=f'基准准确率 ({baseline_acc:.4f})')
        axes[0, 0].set_xlabel('噪声比例 (%)', fontsize=12)
        axes[0, 0].set_ylabel('分类准确率', fontsize=12)
        axes[0, 0].set_title('噪声比例 vs 分类准确率', fontsize=14)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 标注最优点
        best_idx = np.argmax(acc_values)
        axes[0, 0].annotate(f'最优: {acc_values[best_idx]:.4f}', 
                           xy=(noise_values[best_idx], acc_values[best_idx]),
                           xytext=(noise_values[best_idx]+5, acc_values[best_idx]+0.02),
                           fontsize=10, color='green')
    
    # 2. 引导强度 vs 准确率
    if 'guidance_scale' in results:
        gs_data = results['guidance_scale']
        gs_values = [float(k) for k in gs_data.keys()]
        acc_values = [gs_data[k]['accuracy'] for k in gs_data.keys()]
        
        axes[0, 1].plot(gs_values, acc_values, 'o-', color='green', linewidth=2, markersize=10)
        axes[0, 1].axhline(y=baseline_acc, color='red', linestyle='--', label=f'基准准确率 ({baseline_acc:.4f})')
        axes[0, 1].set_xlabel('Guidance Scale', fontsize=12)
        axes[0, 1].set_ylabel('分类准确率', fontsize=12)
        axes[0, 1].set_title('分类器引导强度 vs 分类准确率', fontsize=14)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 标注最优点
        best_idx = np.argmax(acc_values)
        axes[0, 1].annotate(f'最优: {acc_values[best_idx]:.4f}', 
                           xy=(gs_values[best_idx], acc_values[best_idx]),
                           xytext=(gs_values[best_idx]+0.5, acc_values[best_idx]+0.02),
                           fontsize=10, color='green')
    
    # 3. 噪声比例 vs 准确率保持率
    if 'noise_scale' in results:
        noise_data = results['noise_scale']
        noise_values = [float(k) * 100 for k in noise_data.keys()]
        retention_values = [noise_data[k]['retention'] * 100 for k in noise_data.keys()]
        
        axes[1, 0].bar(range(len(noise_values)), retention_values, color='steelblue', alpha=0.7)
        axes[1, 0].axhline(y=100, color='red', linestyle='--', label='100% 保持率')
        axes[1, 0].set_xticks(range(len(noise_values)))
        axes[1, 0].set_xticklabels([f'{v:.0f}%' for v in noise_values])
        axes[1, 0].set_xlabel('噪声比例', fontsize=12)
        axes[1, 0].set_ylabel('准确率保持率 (%)', fontsize=12)
        axes[1, 0].set_title('噪声比例 vs 准确率保持率', fontsize=14)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. 引导强度 vs 准确率保持率
    if 'guidance_scale' in results:
        gs_data = results['guidance_scale']
        gs_values = [float(k) for k in gs_data.keys()]
        retention_values = [gs_data[k]['retention'] * 100 for k in gs_data.keys()]
        
        axes[1, 1].bar(range(len(gs_values)), retention_values, color='forestgreen', alpha=0.7)
        axes[1, 1].axhline(y=100, color='red', linestyle='--', label='100% 保持率')
        axes[1, 1].set_xticks(range(len(gs_values)))
        axes[1, 1].set_xticklabels([f'{v}' for v in gs_values])
        axes[1, 1].set_xlabel('Guidance Scale', fontsize=12)
        axes[1, 1].set_ylabel('准确率保持率 (%)', fontsize=12)
        axes[1, 1].set_title('引导强度 vs 准确率保持率', fontsize=14)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sensitivity_classification.png'), dpi=150)
    plt.close()
    print(f"\n敏感度分析图已保存到: {output_dir}/sensitivity_classification.png")
    
    # 绘制热力图（组合敏感度）
    if 'combined' in results:
        plot_combined_heatmap(results['combined'], output_dir)


def plot_combined_heatmap(combined_results, output_dir):
    """绘制组合敏感度热力图"""
    # 提取数据
    noise_scales = sorted(set(r['noise_scale'] for r in combined_results.values()))
    guidance_scales = sorted(set(r['guidance_scale'] for r in combined_results.values()))
    
    # 创建准确率矩阵
    acc_matrix = np.zeros((len(noise_scales), len(guidance_scales)))
    for key, data in combined_results.items():
        i = noise_scales.index(data['noise_scale'])
        j = guidance_scales.index(data['guidance_scale'])
        acc_matrix[i, j] = data['accuracy']
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(acc_matrix, cmap='RdYlGn', aspect='auto')
    
    # 设置坐标轴
    ax.set_xticks(range(len(guidance_scales)))
    ax.set_xticklabels([f'{gs}' for gs in guidance_scales])
    ax.set_yticks(range(len(noise_scales)))
    ax.set_yticklabels([f'{ns*100:.0f}%' for ns in noise_scales])
    
    ax.set_xlabel('Guidance Scale', fontsize=12)
    ax.set_ylabel('噪声比例', fontsize=12)
    ax.set_title('噪声比例 × 引导强度 组合敏感度热力图\n(颜色越绿准确率越高)', fontsize=14)
    
    # 添加数值标注
    for i in range(len(noise_scales)):
        for j in range(len(guidance_scales)):
            text = ax.text(j, i, f'{acc_matrix[i, j]:.3f}',
                          ha='center', va='center', color='black', fontsize=9)
    
    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('分类准确率', rotation=-90, va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sensitivity_heatmap.png'), dpi=150)
    plt.close()
    print(f"组合敏感度热力图已保存到: {output_dir}/sensitivity_heatmap.png")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='DTTD Enhanced 超参数敏感度分析 - 分类准确率')
    parser.add_argument('--config', type=str, default='configs/bci2a_enhanced_config.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/bci2a_enhanced/best_model.pth')
    parser.add_argument('--output-dir', type=str, default='results/sensitivity_classification')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载配置
    config = load_config(args.config)
    
    # 加载数据
    train_loader, _, test_loader = get_bci2a_dataloaders(
        data_path=config['data']['data_path'],
        subject_ids=config['data']['subject_ids'],
        batch_size=32,
        selected_channels=config['data'].get('selected_channels'),
        num_workers=0,
        reconstruction_mode=True
    )
    
    # 收集数据集
    print("\n收集训练集和测试集...")
    train_data, train_labels, ch_idx = collect_dataset(train_loader)
    test_data, test_labels, _ = collect_dataset(test_loader)
    print(f"训练集: {train_data.shape}, 测试集: {test_data.shape}")
    
    # 训练基准分类器
    print("\n训练基准分类器（原始22通道数据）...")
    classifier = train_classifier(train_data, train_labels, device, num_channels=22)
    
    # 评估基准准确率
    baseline_acc, baseline_f1, baseline_kappa = evaluate_classifier(classifier, test_data, test_labels, device)
    print(f"基准准确率 (原始22通道): {baseline_acc:.4f}")
    
    # 加载生成模型
    print("\n加载生成模型...")
    model = load_model(args.checkpoint, config, device)
    
    all_results = {
        'baseline': {
            'accuracy': float(baseline_acc),
            'f1': float(baseline_f1),
            'kappa': float(baseline_kappa)
        },
        'note': '所有评估使用无条件生成（不使用标签），避免信息泄露'
    }
    
    # 1. 噪声比例敏感度分析（无条件生成）
    noise_results = noise_scale_sensitivity(
        model, classifier, test_data, test_labels, ch_idx, device, baseline_acc
    )
    all_results['noise_scale'] = noise_results
    
    # 2. 引导强度敏感度分析（GS>1时会使用标签，仅供参考）
    gs_results = guidance_scale_sensitivity(
        model, classifier, test_data, test_labels, ch_idx, device, baseline_acc
    )
    all_results['guidance_scale'] = gs_results
    
    # 3. 组合敏感度分析（无条件生成）
    combined_results, best_noise = combined_sensitivity(
        model, classifier, test_data, test_labels, ch_idx, device, baseline_acc
    )
    all_results['combined'] = combined_results
    all_results['best_noise_scale'] = best_noise
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'sensitivity_classification.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_path}")
    
    # 绘制图表
    plot_sensitivity_results(all_results, args.output_dir, baseline_acc)
    
    # 打印汇总
    print("\n" + "="*90)
    print("超参数敏感度分析结果汇总（基于分类准确率）")
    print("="*90)
    
    print(f"\n基准准确率（原始22通道）: {baseline_acc:.4f}")
    
    print("\n1. 噪声比例敏感度:")
    print(f"{'噪声比例':<12} {'准确率':<12} {'保持率':<12} {'F1':<12} {'Kappa':<12}")
    print("-"*60)
    for noise, metrics in all_results['noise_scale'].items():
        print(f"{float(noise)*100:>6.0f}%{'':<5} {metrics['accuracy']:<12.4f} {metrics['retention']:<12.2%} {metrics['f1']:<12.4f} {metrics['kappa']:<12.4f}")
    
    print("\n2. 引导强度敏感度:")
    print(f"{'GS':<12} {'准确率':<12} {'保持率':<12} {'F1':<12} {'Kappa':<12}")
    print("-"*60)
    for gs, metrics in all_results['guidance_scale'].items():
        print(f"{gs:<12} {metrics['accuracy']:<12.4f} {metrics['retention']:<12.2%} {metrics['f1']:<12.4f} {metrics['kappa']:<12.4f}")
    
    # 找出最优参数
    print("\n" + "="*90)
    print("最优参数推荐")
    print("="*90)
    
    # 最优噪声比例
    best_noise = max(all_results['noise_scale'].items(), key=lambda x: x[1]['accuracy'])
    print(f"最优噪声比例: {float(best_noise[0])*100:.0f}% (准确率={best_noise[1]['accuracy']:.4f})")
    
    # 最优引导强度
    best_gs = max(all_results['guidance_scale'].items(), key=lambda x: x[1]['accuracy'])
    print(f"最优引导强度: {best_gs[0]} (准确率={best_gs[1]['accuracy']:.4f})")
    
    # 最优组合
    print(f"最优参数组合: noise_scale={best_params[0]}, guidance_scale={best_params[1]}")
    
    # 分析结论
    print("\n" + "="*90)
    print("敏感度分析结论")
    print("="*90)
    
    # 噪声比例分析
    noise_accs = [v['accuracy'] for v in all_results['noise_scale'].values()]
    noise_range = max(noise_accs) - min(noise_accs)
    print(f"\n1. 噪声比例敏感度: 准确率变化范围 {noise_range:.4f}")
    if noise_range > 0.1:
        print("   -> 模型对噪声比例高度敏感，需要仔细调参")
    elif noise_range > 0.05:
        print("   -> 模型对噪声比例中度敏感")
    else:
        print("   -> 模型对噪声比例不敏感，鲁棒性好")
    
    # 引导强度分析
    gs_accs = [v['accuracy'] for v in all_results['guidance_scale'].values()]
    gs_range = max(gs_accs) - min(gs_accs)
    print(f"\n2. 引导强度敏感度: 准确率变化范围 {gs_range:.4f}")
    if gs_range > 0.1:
        print("   -> 模型对引导强度高度敏感，需要仔细调参")
    elif gs_range > 0.05:
        print("   -> 模型对引导强度中度敏感")
    else:
        print("   -> 模型对引导强度不敏感")


if __name__ == '__main__':
    main()
