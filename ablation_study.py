"""
消融实验 (整合版)

评估DTTD各组件对性能的贡献:
1. 完整模型
2. 无拓扑模块 (No Topology)
3. 无频率模块 (No Frequency)
4. 无任务条件 (No Task Conditioning)
5. 无拓扑+无频率 (Baseline)

使用方法:
    python experiments/ablation_study.py --checkpoint checkpoints/dttd/best_model.pth
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.dttd_enhanced_v2 import DTTDEnhanced
from utils import load_config, get_device, set_seed

# 9通道索引
CH_IDX_9 = [7, 9, 11, 1, 3, 5, 13, 15, 17]


# ==================== 消融模型定义 ====================

class DTTDEnhanced_NoTopology(DTTDEnhanced):
    """消融变体：移除动态拓扑模块"""
    
    def __init__(self, config):
        super().__init__(config)
        self.topology_module = None
        num_groups = min(8, self.output_channels)
        while self.output_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.simple_topo_proj = nn.Sequential(
            nn.Conv1d(self.output_channels, self.output_channels, 3, padding=1),
            nn.GroupNorm(num_groups, self.output_channels),
            nn.GELU()
        )
    
    def forward(self, x, t, task_label=None):
        batch_size = x.size(0)
        x = self.input_proj(x)
        t_emb = self.time_mlp(t)
        if task_label is not None:
            task_onehot = F.one_hot(task_label, num_classes=self.num_classes).float()
            task_emb = self.task_encoder(task_onehot)
        else:
            task_emb = torch.zeros(batch_size, self.task_dim, device=x.device)
        
        x_expanded = self.channel_expansion(x)
        x_22ch_rough = self.rough_channel_proj(x_expanded)
        
        x_topo_raw = self.simple_topo_proj(x_22ch_rough)
        x_topo = self.topo_output_proj(x_topo_raw)
        
        x_freq_raw, _, _ = self.frequency_module(x_22ch_rough, fs=self.config.get('fs', 250), task_emb=task_emb)
        x_freq = self.freq_output_proj(x_freq_raw)
        
        gate_input = torch.cat([x_topo, x_freq], dim=1)
        gate_weights = self.gate_conv(gate_input).unsqueeze(1)
        weighted_features = torch.stack([x_topo, x_freq], dim=2) * gate_weights
        fused_features = weighted_features.sum(dim=2)
        
        x_unet = self.denoising_net(fused_features, t_emb, task_emb)
        x_final = x_unet + fused_features
        x0_pred = self.output_proj(x_final)
        return x0_pred


class DTTDEnhanced_NoFrequency(DTTDEnhanced):
    """消融变体：移除频率解耦模块"""
    
    def __init__(self, config):
        super().__init__(config)
        self.frequency_module = None
        num_groups = min(8, self.output_channels)
        while self.output_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.simple_freq_proj = nn.Sequential(
            nn.Conv1d(self.output_channels, self.output_channels, 3, padding=1),
            nn.GroupNorm(num_groups, self.output_channels),
            nn.GELU()
        )
    
    def forward(self, x, t, task_label=None):
        batch_size = x.size(0)
        x = self.input_proj(x)
        t_emb = self.time_mlp(t)
        if task_label is not None:
            task_onehot = F.one_hot(task_label, num_classes=self.num_classes).float()
            task_emb = self.task_encoder(task_onehot)
        else:
            task_emb = torch.zeros(batch_size, self.task_dim, device=x.device)
        
        x_expanded = self.channel_expansion(x)
        x_22ch_rough = self.rough_channel_proj(x_expanded)
        
        x_topo_raw, _ = self.topology_module(x_22ch_rough, task_emb)
        x_topo = self.topo_output_proj(x_topo_raw)
        
        x_freq_raw = self.simple_freq_proj(x_22ch_rough)
        x_freq = self.freq_output_proj(x_freq_raw)
        
        gate_input = torch.cat([x_topo, x_freq], dim=1)
        gate_weights = self.gate_conv(gate_input).unsqueeze(1)
        weighted_features = torch.stack([x_topo, x_freq], dim=2) * gate_weights
        fused_features = weighted_features.sum(dim=2)
        
        x_unet = self.denoising_net(fused_features, t_emb, task_emb)
        x_final = x_unet + fused_features
        x0_pred = self.output_proj(x_final)
        return x0_pred


class DTTDEnhanced_NoTask(DTTDEnhanced):
    """消融变体：移除任务条件编码"""
    
    def forward(self, x, t, task_label=None):
        batch_size = x.size(0)
        x = self.input_proj(x)
        t_emb = self.time_mlp(t)
        task_emb = torch.zeros(batch_size, self.task_dim, device=x.device)
        
        x_expanded = self.channel_expansion(x)
        x_22ch_rough = self.rough_channel_proj(x_expanded)
        
        x_topo_raw, _ = self.topology_module(x_22ch_rough, task_emb)
        x_topo = self.topo_output_proj(x_topo_raw)
        
        x_freq_raw, _, _ = self.frequency_module(x_22ch_rough, fs=self.config.get('fs', 250), task_emb=task_emb)
        x_freq = self.freq_output_proj(x_freq_raw)
        
        gate_input = torch.cat([x_topo, x_freq], dim=1)
        gate_weights = self.gate_conv(gate_input).unsqueeze(1)
        weighted_features = torch.stack([x_topo, x_freq], dim=2) * gate_weights
        fused_features = weighted_features.sum(dim=2)
        
        x_unet = self.denoising_net(fused_features, t_emb, task_emb)
        x_final = x_unet + fused_features
        x0_pred = self.output_proj(x_final)
        return x0_pred


# ==================== 工具函数 ====================

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


def generate_augmented_data(model, data_22ch, labels, device, data_scale_factor=1e5, guidance_scale=1.0):
    """使用模型生成增强数据"""
    model.eval()
    data_9ch = data_22ch[:, CH_IDX_9, :] * data_scale_factor
    
    loader = DataLoader(TensorDataset(torch.FloatTensor(data_9ch), torch.LongTensor(labels)), batch_size=32)
    generated_list = []
    
    with torch.no_grad():
        for batch_data, batch_labels in loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            if hasattr(model, 'sample'):
                generated = model.sample(batch_data, task_label=batch_labels, guidance_scale=guidance_scale)
            else:
                t = torch.zeros(batch_data.size(0), dtype=torch.long, device=device)
                generated = model(batch_data, t, batch_labels)
            
            generated_list.append(generated.cpu() / data_scale_factor)
    
    return torch.cat(generated_list, dim=0).numpy()


def create_ablation_model(model_type, config, device, checkpoint_path=None):
    """创建消融模型"""
    model_classes = {
        'full': DTTDEnhanced,
        'no_topo': DTTDEnhanced_NoTopology,
        'no_freq': DTTDEnhanced_NoFrequency,
        'no_task': DTTDEnhanced_NoTask,
    }
    
    model = model_classes[model_type](config['model']).to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        # 部分加载
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() 
                          if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"  加载了 {len(pretrained_dict)}/{len(model_dict)} 个参数")
    
    return model


# ==================== 消融实验 ====================

def run_ablation_experiment(data_path, config, checkpoint_path, device, data_scale_factor=1e5):
    """运行消融实验"""
    print("\n" + "="*60)
    print("消融实验")
    print("="*60)
    
    # 加载数据 (跨会话: Session T -> Session E)
    print("\n加载数据...")
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
    print("\n[基准] 仅用原始数据训练...")
    classifier = train_classifier(train_data, train_labels, device)
    baseline_acc, baseline_f1, baseline_kappa = evaluate_classifier(classifier, test_data, test_labels, device)
    print(f"  准确率: {baseline_acc:.4f}, F1: {baseline_f1:.4f}, Kappa: {baseline_kappa:.4f}")
    
    results = {
        'baseline': {'accuracy': baseline_acc, 'f1': baseline_f1, 'kappa': baseline_kappa}
    }
    
    # 消融实验
    ablation_configs = [
        ('full', '完整模型'),
        ('no_topo', '无拓扑模块'),
        ('no_freq', '无频率模块'),
        ('no_task', '无任务条件'),
    ]
    
    for model_type, name in ablation_configs:
        print(f"\n[{name}]")
        
        model = create_ablation_model(model_type, config, device, checkpoint_path)
        
        # 生成增强数据
        generated = generate_augmented_data(model, train_data, train_labels, device, data_scale_factor)
        
        # 合并训练数据
        augmented_data = np.concatenate([train_data, generated], axis=0)
        augmented_labels = np.concatenate([train_labels, train_labels], axis=0)
        
        # 训练分类器
        classifier = train_classifier(augmented_data, augmented_labels, device)
        
        # 评估
        acc, f1, kappa = evaluate_classifier(classifier, test_data, test_labels, device)
        
        print(f"  准确率: {acc:.4f} (vs基准: {acc-baseline_acc:+.4f})")
        print(f"  F1: {f1:.4f}, Kappa: {kappa:.4f}")
        
        results[model_type] = {'accuracy': acc, 'f1': f1, 'kappa': kappa, 'name': name}
        
        del model
        torch.cuda.empty_cache()
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='消融实验')
    parser.add_argument('--config', default='configs/bci2a_enhanced_config.yaml')
    parser.add_argument('--checkpoint', default='checkpoints/dttd/best_model.pth')
    parser.add_argument('--data-path', default='E:/data/BCI2a')
    parser.add_argument('--output-dir', default='paper_results/ablation')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    print(f"使用设备: {device}")
    
    config = load_config(args.config)
    
    # 获取数据缩放因子
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        data_scale_factor = ckpt.get('data_scale_factor', 1e5)
    else:
        data_scale_factor = 1e5
    
    # 运行消融实验
    results = run_ablation_experiment(args.data_path, config, args.checkpoint, device, data_scale_factor)
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'ablation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印总结
    print("\n" + "="*80)
    print("消融实验总结")
    print("="*80)
    print(f"{'模型':<20} {'准确率':<12} {'vs基准':<12} {'F1':<12} {'Kappa':<12}")
    print("-"*70)
    
    baseline_acc = results['baseline']['accuracy']
    for key, metrics in results.items():
        name = metrics.get('name', key)
        diff = metrics['accuracy'] - baseline_acc if key != 'baseline' else 0
        diff_str = f"{diff:+.4f}" if key != 'baseline' else "-"
        print(f"{name:<20} {metrics['accuracy']:<12.4f} {diff_str:<12} {metrics['f1']:<12.4f} {metrics['kappa']:<12.4f}")
    
    print(f"\n结果已保存到: {args.output_dir}/ablation_results.json")


if __name__ == '__main__':
    main()
