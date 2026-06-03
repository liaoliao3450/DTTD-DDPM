"""
消融实验 - 跨会话评估 (与分类实验一致)

评估DTTD各组件对跨会话分类性能的贡献
评估方式与classification_eval.py完全一致：
- 每个被试单独评估 (Session T训练 -> Session E测试)
- 最后取9个被试的平均值

使用方法:
    python experiments/ablation_cross_session.py
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
from datetime import datetime

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
        # 强制使用零向量作为任务嵌入
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
    elif 'labels' in mat_data:
        labels = mat_data['labels'].flatten()
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
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (num_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(F2 * (time_steps // 32), num_classes)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv3(self.conv2(self.conv1(x)))
        return self.classifier(x.flatten(1))


def train_classifier(train_data, train_labels, device, num_channels=22, epochs=100):
    """训练分类器"""
    clf = EEGNetClassifier(num_channels, 4, train_data.shape[2]).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    
    loader = DataLoader(TensorDataset(torch.FloatTensor(train_data), torch.LongTensor(train_labels)),
                        batch_size=32, shuffle=True, drop_last=True)
    
    clf.train()
    for _ in range(epochs):
        total_loss = 0.0
        for data, labels in loader:
            opt.zero_grad()
            loss = criterion(clf(data.to(device)), labels.to(device))
            loss.backward()
            opt.step()
            total_loss += loss.item()
        scheduler.step(total_loss / len(loader))
    return clf


def evaluate_classifier(clf, test_data, test_labels, device):
    """评估分类器"""
    clf.eval()
    loader = DataLoader(TensorDataset(torch.FloatTensor(test_data), torch.LongTensor(test_labels)),
                        batch_size=32, shuffle=False)
    preds, labels_all = [], []
    with torch.no_grad():
        for data, labels in loader:
            preds.extend(torch.argmax(clf(data.to(device)), dim=1).cpu().numpy())
            labels_all.extend(labels.numpy())
    
    return {
        'accuracy': accuracy_score(labels_all, preds),
        'f1': f1_score(labels_all, preds, average='macro'),
        'kappa': cohen_kappa_score(labels_all, preds)
    }


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
    
    model.eval()
    return model


def generate_22ch_data(model, data_9ch, labels, device, data_scale_factor=1e5, guidance_scale=3.0):
    """用模型从9通道生成22通道数据"""
    model.eval()
    data_9ch_scaled = data_9ch * data_scale_factor
    
    loader = DataLoader(TensorDataset(torch.FloatTensor(data_9ch_scaled), torch.LongTensor(labels)),
                        batch_size=32, shuffle=False)
    
    generated_list = []
    with torch.no_grad():
        for batch_data, batch_labels in loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            if hasattr(model, 'sample'):
                gen = model.sample(batch_data, task_label=batch_labels, num_steps=10, 
                                   guidance_scale=guidance_scale).cpu().numpy()
            else:
                t = torch.zeros(batch_data.size(0), dtype=torch.long, device=device)
                gen = model(batch_data, t, batch_labels).cpu().numpy()
            
            gen = gen / data_scale_factor
            generated_list.append(gen)
    
    return np.concatenate(generated_list, axis=0)


# ==================== 消融实验主函数 ====================

def cross_session_ablation_eval(data_path, config, checkpoint_path, device, data_scale_factor=1e5, 
                                 guidance_scale=3.0, subject_ids=range(1, 10)):
    """
    跨会话消融实验
    与classification_eval.py的cross_session_eval完全一致的评估方式
    """
    print("\n" + "="*70)
    print("消融实验 - 跨会话评估 (Cross-Session)")
    print("评估方式: 每个被试单独评估，Session T训练 -> Session E测试")
    print("="*70)
    
    ablation_configs = [
        ('full', 'DTTD完整模型'),
        ('no_topo', '无拓扑模块'),
        ('no_freq', '无频率模块'),
        ('no_task', '无任务条件'),
    ]
    
    # 存储所有结果
    all_results = {
        'baseline_9ch': {'per_subject': {}, 'average': None},
        'baseline_22ch': {'per_subject': {}, 'average': None},
    }
    for model_type, _ in ablation_configs:
        all_results[model_type] = {'per_subject': {}, 'average': None}
    
    # 对每个被试进行评估
    for sid in subject_ids:
        print(f"\n{'='*50}")
        print(f"被试 S{sid}")
        print(f"{'='*50}")
        
        # 加载数据
        train_22ch, train_labels = load_raw_bci2a(data_path, sid, 'T')
        test_22ch, test_labels = load_raw_bci2a(data_path, sid, 'E')
        train_9ch = train_22ch[:, CH_IDX_9, :]
        test_9ch = test_22ch[:, CH_IDX_9, :]
        
        # 基线1：9通道
        print(f"\n[基线] 9通道...")
        clf = train_classifier(train_9ch, train_labels, device, 9)
        metrics_9ch = evaluate_classifier(clf, test_9ch, test_labels, device)
        print(f"  准确率: {metrics_9ch['accuracy']*100:.2f}%")
        all_results['baseline_9ch']['per_subject'][f'S{sid}'] = metrics_9ch
        del clf
        
        # 基线2：22通道
        print(f"[基线] 22通道...")
        clf = train_classifier(train_22ch, train_labels, device, 22)
        metrics_22ch = evaluate_classifier(clf, test_22ch, test_labels, device)
        print(f"  准确率: {metrics_22ch['accuracy']*100:.2f}%")
        all_results['baseline_22ch']['per_subject'][f'S{sid}'] = metrics_22ch
        del clf
        
        # 消融实验
        for model_type, name in ablation_configs:
            print(f"\n[{name}]")
            
            # 创建消融模型
            model = create_ablation_model(model_type, config, device, checkpoint_path)
            
            # 生成增强数据
            gen_22ch = generate_22ch_data(model, train_9ch, train_labels, device, 
                                          data_scale_factor, guidance_scale)
            
            # 合并训练数据
            aug_data = np.concatenate([train_22ch, gen_22ch], axis=0)
            aug_labels = np.concatenate([train_labels, train_labels], axis=0)
            
            # 训练分类器
            clf = train_classifier(aug_data, aug_labels, device, 22)
            
            # 评估
            metrics = evaluate_classifier(clf, test_22ch, test_labels, device)
            print(f"  准确率: {metrics['accuracy']*100:.2f}%")
            
            all_results[model_type]['per_subject'][f'S{sid}'] = metrics
            
            del model, clf
            torch.cuda.empty_cache()
    
    # 计算平均值
    print("\n" + "="*70)
    print("计算平均值...")
    print("="*70)
    
    for key in all_results:
        per_subject = all_results[key]['per_subject']
        avg_acc = np.mean([m['accuracy'] for m in per_subject.values()])
        avg_f1 = np.mean([m['f1'] for m in per_subject.values()])
        avg_kappa = np.mean([m['kappa'] for m in per_subject.values()])
        std_acc = np.std([m['accuracy'] for m in per_subject.values()])
        
        all_results[key]['average'] = {
            'accuracy': avg_acc,
            'accuracy_std': std_acc,
            'f1': avg_f1,
            'kappa': avg_kappa
        }
    
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='消融实验 - 跨会话评估')
    parser.add_argument('--data-path', default='E:/data/BCI2a')
    parser.add_argument('--config', default='configs/bci2a_enhanced_config.yaml')
    parser.add_argument('--checkpoint', default='checkpoints/dttd/best_model.pth')
    parser.add_argument('--output-dir', default='ablation_results')
    parser.add_argument('--guidance-scale', type=float, default=3.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
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
    results = cross_session_ablation_eval(
        args.data_path, config, args.checkpoint, device, 
        data_scale_factor, args.guidance_scale
    )
    
    # 添加元信息
    results['metadata'] = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'guidance_scale': args.guidance_scale,
        'evaluation_mode': 'cross_session',
        'description': '跨会话消融实验，与classification_eval.py评估方式一致'
    }
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'ablation_cross_session_new.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 打印总结
    print("\n" + "="*80)
    print("消融实验总结 (跨会话)")
    print("="*80)
    
    baseline_9ch = results['baseline_9ch']['average']['accuracy']
    baseline_22ch = results['baseline_22ch']['average']['accuracy']
    
    print(f"\n{'配置':<20} {'准确率(%)':<12} {'vs 9ch基线':<12} {'vs DTTD':<12}")
    print("-"*60)
    print(f"{'9通道基线':<20} {baseline_9ch*100:<12.2f} {'-':<12} {'-':<12}")
    print(f"{'22通道基线':<20} {baseline_22ch*100:<12.2f} {(baseline_22ch-baseline_9ch)*100:+.2f}{'':>6} {'-':<12}")
    
    full_acc = results['full']['average']['accuracy']
    print(f"{'DTTD完整模型':<20} {full_acc*100:<12.2f} {(full_acc-baseline_9ch)*100:+.2f}{'':>6} {'-':<12}")
    
    for model_type, name in [('no_topo', '无拓扑模块'), ('no_freq', '无频率模块'), ('no_task', '无任务条件')]:
        acc = results[model_type]['average']['accuracy']
        vs_9ch = (acc - baseline_9ch) * 100
        vs_full = (acc - full_acc) * 100
        module_contrib = (full_acc - acc) * 100
        print(f"{name:<20} {acc*100:<12.2f} {vs_9ch:+.2f}{'':>6} {vs_full:+.2f} (模块贡献: {module_contrib:+.2f})")
    
    print(f"\n结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
