"""
DTTD Enhanced V2 消融实验 - 使用新频率模块

评估各组件对跨会话分类准确率的贡献
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from torch.utils.data import DataLoader, TensorDataset

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.dttd_enhanced_v2 import DTTDEnhanced
from data.bci2a import BCI2aDataset
from utils import load_config, set_seed


# ==================== 消融模型定义 ====================

class DTTDEnhanced_NoTopology(DTTDEnhanced):
    """消融变体：移除动态拓扑模块"""
    
    def __init__(self, config):
        super().__init__(config)
        # 移除拓扑模块
        del self.topology_module
        del self.topo_output_proj
        self.topology_module = None
        self.topo_output_proj = None
    
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
        
        # 只使用频率模块
        x_freq_raw, _, _ = self.frequency_module(x_22ch_rough, fs=self.config.get('fs', 250), task_emb=task_emb)
        x_freq = self.freq_output_proj(x_freq_raw)
        
        # 直接使用频率特征
        fused_features = x_freq
        
        x_unet = self.denoising_net(fused_features, t_emb, task_emb)
        x_final = x_unet + fused_features
        x0_pred = self.output_proj(x_final)
        return x0_pred


class DTTDEnhanced_NoFrequency(DTTDEnhanced):
    """消融变体：移除频率解耦模块"""
    
    def __init__(self, config):
        super().__init__(config)
        # 移除频率模块
        del self.frequency_module
        del self.freq_output_proj
        self.frequency_module = None
        self.freq_output_proj = None
    
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
        
        # 只使用拓扑模块
        x_topo_raw, _ = self.topology_module(x_22ch_rough, task_emb)
        x_topo = self.topo_output_proj(x_topo_raw)
        
        # 直接使用拓扑特征
        fused_features = x_topo
        
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
        
        # 不使用任务信息
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


# ==================== 分类器 ====================

# 导入强分类器
from experiments.train_strong_classifier import StrongEEGClassifier


# ==================== 实验函数 ====================

def load_session_data(data_path, session='T'):
    """加载指定session的数据"""
    dataset = BCI2aDataset(
        data_path=data_path,
        subject_ids=list(range(1, 10)),
        normalize=True,
        train=(session == 'T')
    )
    
    # 收集所有数据
    data_list = []
    labels_list = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        data_list.append(sample[0])  # [22, T]
        labels_list.append(sample[2])  # label
    
    data = torch.stack(data_list)  # [N, 22, T]
    labels = torch.stack(labels_list)  # [N]
    ch_idx = torch.tensor(dataset.input_indices)
    
    return data, labels, ch_idx


def generate_augmented_data(model, data, labels, ch_idx, device, noise_scale=0.02, guidance_scale=1.0):
    """使用模型生成增强数据"""
    model.eval()
    generated_list = []
    
    # ⭐ 数据缩放因子（与训练时一致）
    data_scale_factor = 1e5
    
    loader = DataLoader(TensorDataset(data, labels), batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(loader, desc="生成增强数据"):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            # 提取输入通道并缩放
            x_input = batch_data[:, ch_idx, :] * data_scale_factor
            
            # 生成
            generated = model.sample(x_input, task_label=batch_labels, guidance_scale=guidance_scale)
            
            # 缩放回原始尺度
            generated = generated / data_scale_factor
            generated_list.append(generated.cpu())
    
    return torch.cat(generated_list, dim=0)


def train_classifier(train_data, train_labels, device, epochs=100):
    """训练分类器"""
    classifier = StrongEEGClassifier(num_channels=22, num_classes=4, time_steps=1000, dropout=0.5).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=64, shuffle=True)
    
    classifier.train()
    for epoch in range(epochs):
        for batch_data, batch_labels in loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            logits = classifier(batch_data)
            loss = F.cross_entropy(logits, batch_labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    return classifier


def evaluate_classifier(classifier, test_data, test_labels, device):
    """评估分类器"""
    classifier.eval()
    all_preds = []
    
    loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=64, shuffle=False)
    
    with torch.no_grad():
        for batch_data, batch_labels in loader:
            batch_data = batch_data.to(device)
            logits = classifier(batch_data)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    test_labels = test_labels.numpy()
    
    acc = accuracy_score(test_labels, all_preds)
    f1 = f1_score(test_labels, all_preds, average='macro')
    kappa = cohen_kappa_score(test_labels, all_preds)
    
    return acc, f1, kappa


def run_ablation_experiment():
    """运行消融实验"""
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    config = load_config('configs/bci2a_enhanced_config.yaml')
    data_path = config['data'].get('data_path', 'E:/data/BCI2a')
    
    # 加载数据
    print("\n加载数据...")
    train_data, train_labels, ch_idx = load_session_data(data_path, 'T')
    test_data, test_labels, _ = load_session_data(data_path, 'E')
    
    print(f"训练集: {train_data.shape}")
    print(f"测试集: {test_data.shape}")
    
    # 基准实验
    print("\n" + "="*60)
    print("基准实验：仅用原始数据训练")
    print("="*60)
    
    classifier = train_classifier(train_data, train_labels, device)
    baseline_acc, baseline_f1, baseline_kappa = evaluate_classifier(classifier, test_data, test_labels, device)
    print(f"基准准确率: {baseline_acc:.4f}, F1: {baseline_f1:.4f}, Kappa: {baseline_kappa:.4f}")
    
    results = {
        'baseline': {'accuracy': baseline_acc, 'f1': baseline_f1, 'kappa': baseline_kappa}
    }
    
    # 消融实验配置
    ablation_configs = [
        ('完整模型', DTTDEnhanced, 'checkpoints/bci2a_enhanced_v2/best_model.pth'),
        ('无拓扑模块', DTTDEnhanced_NoTopology, None),
        ('无频率模块', DTTDEnhanced_NoFrequency, None),
        ('无任务条件', DTTDEnhanced_NoTask, 'checkpoints/bci2a_enhanced_v2/best_model.pth'),
    ]
    
    for name, model_class, checkpoint_path in ablation_configs:
        print("\n" + "="*60)
        print(f"消融实验: {name}")
        print("="*60)
        
        # 创建模型
        model = model_class(config['model']).to(device)
        
        # 加载权重（如果有）
        if checkpoint_path and os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=device)
            # 部分加载
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"加载了 {len(pretrained_dict)}/{len(model_dict)} 个参数")
        
        # 生成增强数据
        generated = generate_augmented_data(model, train_data, train_labels, ch_idx, device)
        
        # 合并训练数据
        augmented_data = torch.cat([train_data, generated], dim=0)
        augmented_labels = torch.cat([train_labels, train_labels], dim=0)
        
        # 训练分类器
        classifier = train_classifier(augmented_data, augmented_labels, device)
        
        # 评估
        acc, f1, kappa = evaluate_classifier(classifier, test_data, test_labels, device)
        
        print(f"{name}:")
        print(f"  准确率: {acc:.4f} (基准: {baseline_acc:.4f}, 差异: {acc-baseline_acc:+.4f})")
        print(f"  F1: {f1:.4f}, Kappa: {kappa:.4f}")
        
        results[name] = {'accuracy': acc, 'f1': f1, 'kappa': kappa}
    
    # 保存结果
    os.makedirs('ablation_results', exist_ok=True)
    with open('ablation_results/ablation_v2.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印总结
    print("\n" + "="*80)
    print("消融实验总结")
    print("="*80)
    print(f"{'模型':<20} {'准确率':<12} {'vs基准':<12} {'F1':<12} {'Kappa':<12}")
    print("-"*70)
    
    for name, metrics in results.items():
        diff = metrics['accuracy'] - baseline_acc if name != 'baseline' else 0
        diff_str = f"{diff:+.4f}" if name != 'baseline' else "-"
        print(f"{name:<20} {metrics['accuracy']:<12.4f} {diff_str:<12} {metrics['f1']:<12.4f} {metrics['kappa']:<12.4f}")


if __name__ == '__main__':
    run_ablation_experiment()
