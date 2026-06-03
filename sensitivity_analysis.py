"""
超参数敏感度分析 (整合版)

分析关键超参数对性能的影响:
1. noise_scale - 输入噪声比例
2. guidance_scale - 分类器引导强度
3. num_augment - 数据增强倍数

支持两种评估模式:
- cross_session: 跨会话评估 (Session T -> Session E)
- cross_subject: 跨被试评估 (LOSO)

使用方法:
    python experiments/sensitivity_analysis.py --mode cross_session
    python experiments/sensitivity_analysis.py --mode cross_subject
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from tqdm import tqdm
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.dttd_enhanced_v2 import DTTDEnhanced
from utils import load_config, get_device, set_seed

# 设置绘图
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 9通道索引
CH_IDX_9 = [7, 9, 11, 1, 3, 5, 13, 15, 17]


def load_raw_bci2a(data_path, subject_id, session='T'):
    """加载原始BCI2a数据"""
    subject_str = f'0{subject_id}' if subject_id < 10 else str(subject_id)
    file_path = os.path.join(data_path, f'A{subject_str}{session}.mat')
    
    mat_data = loadmat(file_path)
    
    if 'data' in mat_data:
        data = mat_data['data']
    elif 'X' in mat_data:
        data = mat_data['X']
    else:
        max_key = max(mat_data.keys(), key=lambda k: mat_data[k].size if isinstance(mat_data[k], np.ndarray) else 0)
        data = mat_data[max_key]
    
    if 'label' in mat_data:
        labels = mat_data['label'].flatten()
    elif 'y' in mat_data:
        labels = mat_data['y'].flatten()
    else:
        labels = mat_data['Y'].flatten()
    
    labels = labels.astype(np.int64)
    if labels.min() > 0:
        labels = labels - labels.min()
    
    return data.astype(np.float32), labels


class EEGNetClassifier(nn.Module):
    """EEGNet分类器"""
    def __init__(self, num_channels=22, num_classes=4, time_steps=1000):
        super().__init__()
        F1, D, F2 = 8, 2, 16
        self.conv1 = nn.Sequential(nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False), nn.BatchNorm2d(F1))
        self.conv2 = nn.Sequential(nn.Conv2d(F1, F1*D, (num_channels, 1), groups=F1, bias=False),
                                   nn.BatchNorm2d(F1*D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(0.5))
        self.conv3 = nn.Sequential(nn.Conv2d(F1*D, F2, (1, 16), padding=(0, 8), bias=False),
                                   nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(0.5))
        self.classifier = nn.Linear(F2 * (time_steps // 32), num_classes)
    
    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1)
        return self.classifier(self.conv3(self.conv2(self.conv1(x))).flatten(1))


def train_classifier(train_data, train_labels, device, epochs=100):
    """训练分类器"""
    clf = EEGNetClassifier(22, 4, train_data.shape[2]).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.FloatTensor(train_data), torch.LongTensor(train_labels)),
                        batch_size=32, shuffle=True, drop_last=True)
    clf.train()
    for _ in range(epochs):
        for data, labels in loader:
            opt.zero_grad()
            nn.functional.cross_entropy(clf(data.to(device)), labels.to(device)).backward()
            opt.step()
    return clf


def evaluate_classifier(clf, test_data, test_labels, device):
    """评估分类器"""
    clf.eval()
    loader = DataLoader(TensorDataset(torch.FloatTensor(test_data), torch.LongTensor(test_labels)), batch_size=32)
    preds, labels_all = [], []
    with torch.no_grad():
        for data, labels in loader:
            preds.extend(torch.argmax(clf(data.to(device)), dim=1).cpu().numpy())
            labels_all.extend(labels.numpy())
    return accuracy_score(labels_all, preds), f1_score(labels_all, preds, average='macro'), cohen_kappa_score(labels_all, preds)


def generate_augmented_data(model, data_22ch, labels, device, data_scale_factor=1e5, 
                           noise_scale=0.02, guidance_scale=3.0, num_augment=1):
    """生成增强数据"""
    model.eval()
    data_9ch = data_22ch[:, CH_IDX_9, :] * data_scale_factor
    
    loader = DataLoader(TensorDataset(torch.FloatTensor(data_9ch), torch.LongTensor(labels)), batch_size=32)
    aug_data_list, aug_labels_list = [], []
    
    with torch.no_grad():
        for _ in range(num_augment):
            for batch_data, batch_labels in loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                # 添加噪声
                if noise_scale > 0:
                    noisy_input = batch_data + torch.randn_like(batch_data) * noise_scale
                else:
                    noisy_input = batch_data
                
                t = torch.zeros(batch_data.size(0), dtype=torch.long, device=device)
                
                # 使用guidance_scale进行条件生成
                if guidance_scale > 1.0 and hasattr(model, 'forward'):
                    x0_cond = model(noisy_input, t, batch_labels)
                    x0_uncond = model(noisy_input, t, None)
                    generated = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
                else:
                    generated = model(noisy_input, t, batch_labels)
                
                aug_data_list.append(generated.cpu() / data_scale_factor)
                aug_labels_list.append(batch_labels.cpu())
    
    return torch.cat(aug_data_list, dim=0).numpy(), torch.cat(aug_labels_list, dim=0).numpy()


def evaluate_with_augmentation(model, train_data, train_labels, test_data, test_labels, 
                               device, data_scale_factor, noise_scale=0.02, 
                               guidance_scale=3.0, num_augment=1):
    """使用指定参数进行数据增强并评估"""
    aug_data, aug_labels = generate_augmented_data(
        model, train_data, train_labels, device, data_scale_factor,
        noise_scale=noise_scale, guidance_scale=guidance_scale, num_augment=num_augment
    )
    
    combined_data = np.concatenate([train_data, aug_data], axis=0)
    combined_labels = np.concatenate([train_labels, aug_labels], axis=0)
    
    classifier = train_classifier(combined_data, combined_labels, device)
    acc, f1, kappa = evaluate_classifier(classifier, test_data, test_labels, device)
    
    del classifier
    torch.cuda.empty_cache()
    
    return acc, f1, kappa


# ==================== 敏感度分析 ====================

def noise_scale_sensitivity(model, train_data, train_labels, test_data, test_labels, 
                           device, data_scale_factor, baseline_acc):
    """噪声比例敏感度分析"""
    print("\n噪声比例 (noise_scale) 敏感度分析")
    print("-" * 50)
    
    noise_scales = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
    results = {}
    
    for noise_scale in noise_scales:
        acc, f1, kappa = evaluate_with_augmentation(
            model, train_data, train_labels, test_data, test_labels,
            device, data_scale_factor, noise_scale=noise_scale, guidance_scale=3.0, num_augment=1
        )
        
        improvement = acc - baseline_acc
        results[str(noise_scale)] = {
            'accuracy': float(acc), 'f1': float(f1), 'kappa': float(kappa),
            'improvement': float(improvement)
        }
        
        print(f"  noise_scale={noise_scale:.2f}: 准确率={acc:.4f} (提升: {improvement:+.4f})")
    
    return results


def guidance_scale_sensitivity(model, train_data, train_labels, test_data, test_labels, 
                              device, data_scale_factor, baseline_acc):
    """引导强度敏感度分析"""
    print("\n引导强度 (guidance_scale) 敏感度分析")
    print("-" * 50)
    
    guidance_scales = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    results = {}
    
    for gs in guidance_scales:
        acc, f1, kappa = evaluate_with_augmentation(
            model, train_data, train_labels, test_data, test_labels,
            device, data_scale_factor, noise_scale=0.02, guidance_scale=gs, num_augment=1
        )
        
        improvement = acc - baseline_acc
        results[str(gs)] = {
            'accuracy': float(acc), 'f1': float(f1), 'kappa': float(kappa),
            'improvement': float(improvement)
        }
        
        print(f"  guidance_scale={gs:.1f}: 准确率={acc:.4f} (提升: {improvement:+.4f})")
    
    return results


def num_augment_sensitivity(model, train_data, train_labels, test_data, test_labels, 
                           device, data_scale_factor, baseline_acc):
    """数据增强倍数敏感度分析"""
    print("\n增强倍数 (num_augment) 敏感度分析")
    print("-" * 50)
    
    num_augments = [1, 2, 3, 5]
    results = {}
    
    for num_aug in num_augments:
        acc, f1, kappa = evaluate_with_augmentation(
            model, train_data, train_labels, test_data, test_labels,
            device, data_scale_factor, noise_scale=0.02, guidance_scale=3.0, num_augment=num_aug
        )
        
        improvement = acc - baseline_acc
        results[str(num_aug)] = {
            'accuracy': float(acc), 'f1': float(f1), 'kappa': float(kappa),
            'improvement': float(improvement)
        }
        
        print(f"  num_augment={num_aug}x: 准确率={acc:.4f} (提升: {improvement:+.4f})")
    
    return results


def plot_sensitivity_results(results, output_dir, baseline_acc):
    """绘制敏感度分析结果（只包含噪声比例和引导强度）"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. 噪声比例
    if 'noise_scale' in results:
        data = results['noise_scale']
        x = [float(k) * 100 for k in data.keys()]
        y = [data[k]['accuracy'] for k in data.keys()]
        axes[0].plot(x, y, 'o-', linewidth=2, markersize=8)
        axes[0].axhline(y=baseline_acc, color='red', linestyle='--', label=f'基准 ({baseline_acc:.4f})')
        axes[0].set_xlabel('噪声比例 (%)')
        axes[0].set_ylabel('准确率')
        axes[0].set_title('(a) 噪声比例敏感度')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # 2. 引导强度
    if 'guidance_scale' in results:
        data = results['guidance_scale']
        x = [float(k) for k in data.keys()]
        y = [data[k]['accuracy'] for k in data.keys()]
        axes[1].plot(x, y, 'o-', linewidth=2, markersize=8, color='green')
        axes[1].axhline(y=baseline_acc, color='red', linestyle='--', label=f'基准 ({baseline_acc:.4f})')
        axes[1].set_xlabel('Guidance Scale')
        axes[1].set_ylabel('准确率')
        axes[1].set_title('(b) 引导强度敏感度')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sensitivity_analysis.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'sensitivity_analysis.pdf'))
    plt.close()
    print(f"\n图表已保存到: {output_dir}/sensitivity_analysis.png")


def run_cross_session_sensitivity(data_path, model, device, data_scale_factor, output_dir):
    """跨会话敏感度分析"""
    print("\n" + "="*60)
    print("跨会话敏感度分析")
    print("="*60)
    
    # 加载数据
    train_data_list, train_labels_list = [], []
    test_data_list, test_labels_list = [], []
    
    for sid in range(1, 10):
        train_data, train_labels = load_raw_bci2a(data_path, sid, 'T')
        test_data, test_labels = load_raw_bci2a(data_path, sid, 'E')
        train_data_list.append(train_data)
        train_labels_list.append(train_labels)
        test_data_list.append(test_data)
        test_labels_list.append(test_labels)
    
    train_data = np.concatenate(train_data_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    test_data = np.concatenate(test_data_list, axis=0)
    test_labels = np.concatenate(test_labels_list, axis=0)
    
    print(f"训练集: {train_data.shape}, 测试集: {test_data.shape}")
    
    # 基准实验
    print("\n基准实验...")
    classifier = train_classifier(train_data, train_labels, device)
    baseline_acc, _, _ = evaluate_classifier(classifier, test_data, test_labels, device)
    print(f"基准准确率: {baseline_acc:.4f}")
    del classifier
    
    all_results = {
        'baseline': {'accuracy': float(baseline_acc)}
    }
    
    # 敏感度分析
    all_results['noise_scale'] = noise_scale_sensitivity(
        model, train_data, train_labels, test_data, test_labels, device, data_scale_factor, baseline_acc)
    
    all_results['guidance_scale'] = guidance_scale_sensitivity(
        model, train_data, train_labels, test_data, test_labels, device, data_scale_factor, baseline_acc)
    
    all_results['num_augment'] = num_augment_sensitivity(
        model, train_data, train_labels, test_data, test_labels, device, data_scale_factor, baseline_acc)
    
    # 绘图
    plot_sensitivity_results(all_results, output_dir, baseline_acc)
    
    return all_results


def run_cross_subject_sensitivity(data_path, model, device, data_scale_factor, output_dir):
    """跨被试敏感度分析 (LOSO)"""
    print("\n" + "="*60)
    print("跨被试敏感度分析 (LOSO)")
    print("="*60)
    
    # 加载所有被试数据（使用Session T）
    all_data = []
    all_labels = []
    
    for sid in range(1, 10):
        data, labels = load_raw_bci2a(data_path, sid, 'T')
        all_data.append(data)
        all_labels.append(labels)
        print(f"加载被试 {sid}: {data.shape}")
    
    # LOSO交叉验证 - 使用前8个被试进行敏感度分析
    # 选择一个代表性的被试作为测试集
    test_sid = 9  # 使用被试9作为测试集
    train_data_list = []
    train_labels_list = []
    
    for sid in range(1, 10):
        if sid != test_sid:
            train_data_list.append(all_data[sid-1])
            train_labels_list.append(all_labels[sid-1])
    
    train_data = np.concatenate(train_data_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    test_data = all_data[test_sid-1]
    test_labels = all_labels[test_sid-1]
    
    print(f"\n训练集 (被试1-8): {train_data.shape}")
    print(f"测试集 (被试{test_sid}): {test_data.shape}")
    
    # 基准实验
    print("\n基准实验...")
    classifier = train_classifier(train_data, train_labels, device)
    baseline_acc, _, _ = evaluate_classifier(classifier, test_data, test_labels, device)
    print(f"基准准确率: {baseline_acc:.4f}")
    del classifier
    
    all_results = {
        'baseline': {'accuracy': float(baseline_acc)},
        'test_subject': test_sid
    }
    
    # 敏感度分析
    all_results['noise_scale'] = noise_scale_sensitivity(
        model, train_data, train_labels, test_data, test_labels, device, data_scale_factor, baseline_acc)
    
    all_results['guidance_scale'] = guidance_scale_sensitivity(
        model, train_data, train_labels, test_data, test_labels, device, data_scale_factor, baseline_acc)
    
    all_results['num_augment'] = num_augment_sensitivity(
        model, train_data, train_labels, test_data, test_labels, device, data_scale_factor, baseline_acc)
    
    # 绘图
    plot_sensitivity_results(all_results, output_dir, baseline_acc)
    
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='超参数敏感度分析')
    parser.add_argument('--config', default='configs/bci2a_enhanced_config.yaml')
    parser.add_argument('--checkpoint', default='checkpoints/dttd/best_model.pth')
    parser.add_argument('--data-path', default='E:/data/BCI2a')
    parser.add_argument('--output-dir', default='paper_results/sensitivity')
    parser.add_argument('--mode', default='cross_session', choices=['cross_session', 'cross_subject'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    print(f"使用设备: {device}")
    
    config = load_config(args.config)
    
    # 加载模型
    model = DTTDEnhanced(config['model']).to(device)
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
        data_scale_factor = checkpoint.get('data_scale_factor', 1e5)
        print(f"[OK] 加载模型: {args.checkpoint}")
    else:
        data_scale_factor = 1e5
        print("[WARN] 未找到checkpoint，使用随机初始化模型")
    model.eval()
    
    # 运行敏感度分析
    if args.mode == 'cross_session':
        results = run_cross_session_sensitivity(args.data_path, model, device, data_scale_factor, args.output_dir)
    else:
        results = run_cross_subject_sensitivity(args.data_path, model, device, data_scale_factor, args.output_dir)
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'sensitivity_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印汇总
    print("\n" + "="*60)
    print("敏感度分析结果汇总")
    print("="*60)
    print(f"基准准确率: {results['baseline']['accuracy']:.4f}")
    
    for param_name, param_label in [('noise_scale', '噪声比例'), ('guidance_scale', '引导强度'), ('num_augment', '增强倍数')]:
        if param_name in results:
            best = max(results[param_name].items(), key=lambda x: x[1]['accuracy'])
            print(f"\n{param_label}最优值: {best[0]}")
            print(f"  准确率: {best[1]['accuracy']:.4f}, 提升: {best[1]['improvement']:+.4f}")
    
    print(f"\n结果已保存到: {args.output_dir}")


if __name__ == '__main__':
    main()
