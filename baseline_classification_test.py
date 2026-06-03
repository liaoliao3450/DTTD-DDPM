"""
基线分类测试 - 排查预处理问题

这个脚本独立测试不同预处理方式对分类性能的影响：
1. 原始数据（无预处理）
2. 仅带通滤波
3. 带通滤波 + 不同归一化方式
4. 使用MNE标准预处理

目的：确定当前预处理流程是否导致分类性能下降
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class EEGNetClassifier(nn.Module):
    """EEGNet分类器"""
    def __init__(self, num_channels=22, num_classes=4, time_steps=1000, 
                 F1=8, D=2, F2=16, dropout=0.5):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (num_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )
        
        self.fc_input_dim = F2 * (time_steps // 32)
        self.classifier = nn.Linear(self.fc_input_dim, num_classes)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(1)
        return self.classifier(x)


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """带通滤波"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            filtered_data[i, j, :] = filtfilt(b, a, data[i, j, :])
    
    return filtered_data


def load_raw_bci2a(data_path, subject_id, train=True):
    """
    直接加载原始BCI2a数据，不做任何预处理
    """
    subject_str = f'0{subject_id}' if subject_id < 10 else str(subject_id)
    suffix = 'T' if train else 'E'
    file_path = os.path.join(data_path, f'A{subject_str}{suffix}.mat')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    mat_data = loadmat(file_path)
    
    # 获取数据和标签
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
    elif 'Y' in mat_data:
        labels = mat_data['Y'].flatten()
    else:
        raise ValueError("未找到标签")
    
    # 确保标签从0开始
    labels = labels.astype(np.int64)
    if labels.min() > 0:
        labels = labels - labels.min()
    
    return data.astype(np.float32), labels


def preprocess_data(data, mode='none', fs=250):
    """
    不同的预处理方式
    
    Args:
        data: [trials, channels, time]
        mode: 预处理模式
            - 'none': 无预处理
            - 'filter': 仅带通滤波 (4-40Hz)
            - 'filter_norm_sample': 滤波 + 样本级归一化
            - 'filter_norm_channel': 滤波 + 通道级全局归一化
            - 'filter_norm_global': 滤波 + 全局归一化
            - 'filter_scale_norm': 滤波 + 缩放 + 归一化（当前方式）
    """
    if mode == 'none':
        return data
    
    # 带通滤波
    if mode.startswith('filter'):
        data = bandpass_filter(data, 4, 40, fs)
    
    if mode == 'filter':
        return data
    
    if mode == 'filter_norm_sample':
        # 样本级归一化
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True) + 1e-8
        return (data - mean) / std
    
    if mode == 'filter_norm_channel':
        # 通道级全局归一化
        mean = np.mean(data, axis=(0, 2), keepdims=True)
        std = np.std(data, axis=(0, 2), keepdims=True) + 1e-8
        return (data - mean) / std
    
    if mode == 'filter_norm_global':
        # 全局归一化
        mean = np.mean(data)
        std = np.std(data) + 1e-8
        return (data - mean) / std
    
    if mode == 'filter_scale_norm':
        # 当前项目使用的方式：滤波 + 缩放1e6 + 通道级归一化
        data = data * 1e6
        mean = np.mean(data, axis=(0, 2), keepdims=True)
        std = np.std(data, axis=(0, 2), keepdims=True) + 1e-8
        return (data - mean) / std
    
    if mode == 'filter_robust':
        # 鲁棒归一化（使用中位数和MAD）
        median = np.median(data, axis=-1, keepdims=True)
        mad = np.median(np.abs(data - median), axis=-1, keepdims=True) + 1e-8
        return (data - median) / (1.4826 * mad)
    
    return data


def train_classifier(train_data, train_labels, device, num_channels=22, epochs=100, verbose=False):
    """训练分类器"""
    classifier = EEGNetClassifier(
        num_channels=num_channels, num_classes=4, time_steps=train_data.shape[2]
    ).to(device)
    
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(train_data), torch.LongTensor(train_labels)), 
        batch_size=32, shuffle=True, drop_last=True
    )
    
    classifier.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(classifier(data), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step(total_loss / len(train_loader))
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
    
    return classifier


def evaluate_classifier(classifier, test_data, test_labels, device):
    """评估分类器"""
    classifier.eval()
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(test_data), torch.LongTensor(test_labels)), 
        batch_size=32, shuffle=False
    )
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, labels in test_loader:
            preds = torch.argmax(classifier(data.to(device)), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    kappa = cohen_kappa_score(all_labels, all_preds)
    return acc, f1, kappa


def test_within_subject(data_path, subject_id, preprocess_mode, device, num_channels=22):
    """
    Within-subject测试：5折交叉验证
    """
    # 加载两个session的数据
    train_data, train_labels = load_raw_bci2a(data_path, subject_id, train=True)
    test_data, test_labels = load_raw_bci2a(data_path, subject_id, train=False)
    
    # 合并数据
    all_data = np.concatenate([train_data, test_data], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)
    
    # 选择通道
    if num_channels == 9:
        ch_idx = [7, 9, 11, 1, 3, 5, 13, 15, 17]  # C3, C1, C4, FC3, FCz, FC4, CP3, CPz, CP4
        all_data = all_data[:, ch_idx, :]
    
    # 预处理
    all_data = preprocess_data(all_data, preprocess_mode)
    
    # 5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s, kappas = [], [], []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(all_data, all_labels)):
        X_train, X_test = all_data[train_idx], all_data[test_idx]
        y_train, y_test = all_labels[train_idx], all_labels[test_idx]
        
        classifier = train_classifier(X_train, y_train, device, num_channels=num_channels)
        acc, f1, kappa = evaluate_classifier(classifier, X_test, y_test, device)
        accs.append(acc)
        f1s.append(f1)
        kappas.append(kappa)
        del classifier
    
    return np.mean(accs), np.std(accs), np.mean(f1s), np.mean(kappas)


def test_cross_session(data_path, subject_id, preprocess_mode, device, num_channels=22):
    """
    Cross-session测试：Session T训练，Session E测试
    """
    train_data, train_labels = load_raw_bci2a(data_path, subject_id, train=True)
    test_data, test_labels = load_raw_bci2a(data_path, subject_id, train=False)
    
    # 选择通道
    if num_channels == 9:
        ch_idx = [7, 9, 11, 1, 3, 5, 13, 15, 17]
        train_data = train_data[:, ch_idx, :]
        test_data = test_data[:, ch_idx, :]
    
    # 预处理（分别对训练和测试数据）
    train_data = preprocess_data(train_data, preprocess_mode)
    test_data = preprocess_data(test_data, preprocess_mode)
    
    classifier = train_classifier(train_data, train_labels, device, num_channels=num_channels)
    acc, f1, kappa = evaluate_classifier(classifier, test_data, test_labels, device)
    del classifier
    
    return acc, f1, kappa


def main():
    import argparse
    parser = argparse.ArgumentParser(description='基线分类测试')
    parser.add_argument('--data-path', type=str, default='E:/data/BCI2a')
    parser.add_argument('--output-dir', type=str, default='paper_results/baseline_test')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 预处理模式列表
    preprocess_modes = [
        'none',
        'filter',
        'filter_norm_sample',
        'filter_norm_channel',
        'filter_norm_global',
        'filter_scale_norm',
        'filter_robust'
    ]
    
    subject_ids = list(range(1, 10))
    results = {}
    
    print("\n" + "="*80)
    print("基线分类测试 - 不同预处理方式对比")
    print("="*80)
    
    for mode in preprocess_modes:
        print(f"\n{'='*60}")
        print(f"预处理模式: {mode}")
        print(f"{'='*60}")
        
        results[mode] = {
            'within_subject_22ch': {'per_subject': {}, 'mean': 0, 'std': 0},
            'within_subject_9ch': {'per_subject': {}, 'mean': 0, 'std': 0},
            'cross_session_22ch': {'per_subject': {}, 'mean': 0, 'std': 0},
            'cross_session_9ch': {'per_subject': {}, 'mean': 0, 'std': 0}
        }
        
        ws_22_accs, ws_9_accs = [], []
        cs_22_accs, cs_9_accs = [], []
        
        for sid in subject_ids:
            print(f"\n  被试 S{sid}:")
            
            try:
                # Within-subject 22通道
                acc, std, f1, kappa = test_within_subject(
                    args.data_path, sid, mode, device, num_channels=22
                )
                results[mode]['within_subject_22ch']['per_subject'][f'S{sid}'] = {
                    'accuracy': acc, 'std': std, 'f1': f1, 'kappa': kappa
                }
                ws_22_accs.append(acc)
                print(f"    Within-Subject 22ch: {acc*100:.2f}%")
                
                # Within-subject 9通道
                acc, std, f1, kappa = test_within_subject(
                    args.data_path, sid, mode, device, num_channels=9
                )
                results[mode]['within_subject_9ch']['per_subject'][f'S{sid}'] = {
                    'accuracy': acc, 'std': std, 'f1': f1, 'kappa': kappa
                }
                ws_9_accs.append(acc)
                print(f"    Within-Subject 9ch:  {acc*100:.2f}%")
                
                # Cross-session 22通道
                acc, f1, kappa = test_cross_session(
                    args.data_path, sid, mode, device, num_channels=22
                )
                results[mode]['cross_session_22ch']['per_subject'][f'S{sid}'] = {
                    'accuracy': acc, 'f1': f1, 'kappa': kappa
                }
                cs_22_accs.append(acc)
                print(f"    Cross-Session 22ch:  {acc*100:.2f}%")
                
                # Cross-session 9通道
                acc, f1, kappa = test_cross_session(
                    args.data_path, sid, mode, device, num_channels=9
                )
                results[mode]['cross_session_9ch']['per_subject'][f'S{sid}'] = {
                    'accuracy': acc, 'f1': f1, 'kappa': kappa
                }
                cs_9_accs.append(acc)
                print(f"    Cross-Session 9ch:   {acc*100:.2f}%")
                
            except Exception as e:
                print(f"    错误: {e}")
            
            torch.cuda.empty_cache()
        
        # 计算平均值
        results[mode]['within_subject_22ch']['mean'] = float(np.mean(ws_22_accs))
        results[mode]['within_subject_22ch']['std'] = float(np.std(ws_22_accs))
        results[mode]['within_subject_9ch']['mean'] = float(np.mean(ws_9_accs))
        results[mode]['within_subject_9ch']['std'] = float(np.std(ws_9_accs))
        results[mode]['cross_session_22ch']['mean'] = float(np.mean(cs_22_accs))
        results[mode]['cross_session_22ch']['std'] = float(np.std(cs_22_accs))
        results[mode]['cross_session_9ch']['mean'] = float(np.mean(cs_9_accs))
        results[mode]['cross_session_9ch']['std'] = float(np.std(cs_9_accs))
        
        print(f"\n  汇总 ({mode}):")
        print(f"    Within-Subject 22ch: {np.mean(ws_22_accs)*100:.2f}% ± {np.std(ws_22_accs)*100:.2f}%")
        print(f"    Within-Subject 9ch:  {np.mean(ws_9_accs)*100:.2f}% ± {np.std(ws_9_accs)*100:.2f}%")
        print(f"    Cross-Session 22ch:  {np.mean(cs_22_accs)*100:.2f}% ± {np.std(cs_22_accs)*100:.2f}%")
        print(f"    Cross-Session 9ch:   {np.mean(cs_9_accs)*100:.2f}% ± {np.std(cs_9_accs)*100:.2f}%")
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'baseline_classification_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_path}")
    
    # 打印最终对比表
    print("\n" + "="*80)
    print("最终对比表")
    print("="*80)
    print(f"{'预处理模式':<25} {'WS-22ch':<15} {'WS-9ch':<15} {'CS-22ch':<15} {'CS-9ch':<15}")
    print("-"*80)
    for mode in preprocess_modes:
        ws22 = results[mode]['within_subject_22ch']['mean'] * 100
        ws9 = results[mode]['within_subject_9ch']['mean'] * 100
        cs22 = results[mode]['cross_session_22ch']['mean'] * 100
        cs9 = results[mode]['cross_session_9ch']['mean'] * 100
        print(f"{mode:<25} {ws22:>6.2f}%        {ws9:>6.2f}%        {cs22:>6.2f}%        {cs9:>6.2f}%")


if __name__ == '__main__':
    main()
