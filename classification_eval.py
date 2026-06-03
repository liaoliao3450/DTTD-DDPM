"""
分类评估实验 (整合版)

评估DTTD数据增强对分类性能的提升
支持三种评估场景:
1. 被试内 (Within-Subject): 5折交叉验证
2. 跨会话 (Cross-Session): Session T训练 -> Session E测试
3. 跨被试 (Cross-Subject): LOSO留一被试验证

使用方法:
    python experiments/classification_eval.py --mode all
    python experiments/classification_eval.py --mode cross_session
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
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.dttd_enhanced_v2 import DTTDEnhanced
from utils import load_config, get_device

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
    
    # 使用固定随机种子生成器，确保可复现
    generator = torch.Generator()
    generator.manual_seed(42)
    
    loader = DataLoader(TensorDataset(torch.FloatTensor(train_data), torch.LongTensor(train_labels)),
                        batch_size=32, shuffle=True, drop_last=True, generator=generator)
    
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


def load_dttd_model(config_path, checkpoint_path, device):
    """加载DTTD模型"""
    config = load_config(config_path)
    model = DTTDEnhanced(config['model']).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    model.eval()
    data_scale_factor = ckpt.get('data_scale_factor', 1e5)
    print(f"[OK] 加载DTTD模型: {checkpoint_path}")
    return model, data_scale_factor


def generate_22ch_data(model, data_9ch, labels, device, data_scale_factor=1e5, guidance_scale=3.0):
    """用DTTD模型从9通道生成22通道数据"""
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


# ==================== 评估场景 ====================

def within_subject_eval(data_path, model, device, data_scale_factor, subject_ids=range(1, 10), guidance_scale=3.0):
    """被试内测试 - 5折交叉验证"""
    print("\n" + "="*60)
    print("被试内测试 (Within-Subject) - 5折交叉验证")
    print("="*60)
    
    results = {}
    
    for sid in subject_ids:
        print(f"\n被试 S{sid}:")
        
        # 加载并合并两个会话
        data_t, labels_t = load_raw_bci2a(data_path, sid, 'T')
        data_e, labels_e = load_raw_bci2a(data_path, sid, 'E')
        all_data_22ch = np.concatenate([data_t, data_e], axis=0)
        all_labels = np.concatenate([labels_t, labels_e], axis=0)
        all_data_9ch = all_data_22ch[:, CH_IDX_9, :]
        
        # 5折交叉验证
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        accs_22, accs_9, accs_aug = [], [], []
        kappas_22, kappas_9, kappas_aug = [], [], []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(all_data_22ch, all_labels)):
            train_22ch = all_data_22ch[train_idx]
            train_9ch = all_data_9ch[train_idx]
            train_labels = all_labels[train_idx]
            test_22ch = all_data_22ch[test_idx]
            test_9ch = all_data_9ch[test_idx]
            test_labels = all_labels[test_idx]
            
            # 基线1：22通道
            clf = train_classifier(train_22ch, train_labels, device, 22)
            m22 = evaluate_classifier(clf, test_22ch, test_labels, device)
            accs_22.append(m22['accuracy'])
            kappas_22.append(m22['kappa'])
            del clf
            
            # 基线2：9通道
            clf = train_classifier(train_9ch, train_labels, device, 9)
            m9 = evaluate_classifier(clf, test_9ch, test_labels, device)
            accs_9.append(m9['accuracy'])
            kappas_9.append(m9['kappa'])
            del clf
            
            # DTTD增强
            gen_22ch = generate_22ch_data(model, train_9ch, train_labels, device, 
                                          data_scale_factor, guidance_scale)
            aug_data = np.concatenate([train_22ch, gen_22ch], axis=0)
            aug_labels = np.concatenate([train_labels, train_labels], axis=0)
            
            clf = train_classifier(aug_data, aug_labels, device, 22)
            maug = evaluate_classifier(clf, test_22ch, test_labels, device)
            accs_aug.append(maug['accuracy'])
            kappas_aug.append(maug['kappa'])
            del clf
            
            torch.cuda.empty_cache()
        
        mean_22, mean_9, mean_aug = np.mean(accs_22), np.mean(accs_9), np.mean(accs_aug)
        mean_k22, mean_k9, mean_kaug = np.mean(kappas_22), np.mean(kappas_9), np.mean(kappas_aug)
        print(f"  22ch: acc={mean_22*100:.2f}%, kappa={mean_k22:.4f}")
        print(f"  9ch:  acc={mean_9*100:.2f}%, kappa={mean_k9:.4f}")
        print(f"  DTTD: acc={mean_aug*100:.2f}%, kappa={mean_kaug:.4f}")
        
        results[f'S{sid}'] = {
            'baseline_22ch': {'accuracy': mean_22, 'std': np.std(accs_22), 'kappa': mean_k22},
            'baseline_9ch': {'accuracy': mean_9, 'std': np.std(accs_9), 'kappa': mean_k9},
            'dttd_augmented': {'accuracy': mean_aug, 'std': np.std(accs_aug), 'kappa': mean_kaug},
        }
    
    # 平均
    avg_22 = np.mean([r['baseline_22ch']['accuracy'] for r in results.values()])
    avg_9 = np.mean([r['baseline_9ch']['accuracy'] for r in results.values()])
    avg_aug = np.mean([r['dttd_augmented']['accuracy'] for r in results.values()])
    avg_k22 = np.mean([r['baseline_22ch']['kappa'] for r in results.values()])
    avg_k9 = np.mean([r['baseline_9ch']['kappa'] for r in results.values()])
    avg_kaug = np.mean([r['dttd_augmented']['kappa'] for r in results.values()])
    
    results['average'] = {
        'baseline_22ch': {'accuracy': avg_22, 'kappa': avg_k22},
        'baseline_9ch': {'accuracy': avg_9, 'kappa': avg_k9},
        'dttd_augmented': {'accuracy': avg_aug, 'kappa': avg_kaug},
    }
    print(f"\n平均: 22ch acc={avg_22*100:.2f}% kappa={avg_k22:.4f}, "
          f"9ch acc={avg_9*100:.2f}% kappa={avg_k9:.4f}, "
          f"DTTD acc={avg_aug*100:.2f}% kappa={avg_kaug:.4f}")
    
    return results


def cross_session_eval(data_path, model, device, data_scale_factor, subject_ids=range(1, 10), guidance_scale=3.0):
    """跨会话测试：Session T训练 -> Session E测试"""
    print("\n" + "="*60)
    print("跨会话测试 (Cross-Session)")
    print("="*60)
    
    results = {}
    
    for sid in subject_ids:
        print(f"\n被试 S{sid}:")
        
        train_22ch, train_labels = load_raw_bci2a(data_path, sid, 'T')
        test_22ch, test_labels = load_raw_bci2a(data_path, sid, 'E')
        train_9ch = train_22ch[:, CH_IDX_9, :]
        test_9ch = test_22ch[:, CH_IDX_9, :]
        
        # 基线1：22通道
        clf = train_classifier(train_22ch, train_labels, device, 22)
        metrics_22 = evaluate_classifier(clf, test_22ch, test_labels, device)
        print(f"  22ch: acc={metrics_22['accuracy']*100:.2f}%, kappa={metrics_22['kappa']:.4f}")
        del clf
        
        # 基线2：9通道
        clf = train_classifier(train_9ch, train_labels, device, 9)
        metrics_9 = evaluate_classifier(clf, test_9ch, test_labels, device)
        print(f"  9ch:  acc={metrics_9['accuracy']*100:.2f}%, kappa={metrics_9['kappa']:.4f}")
        del clf
        
        # DTTD增强
        gen_22ch = generate_22ch_data(model, train_9ch, train_labels, device, 
                                      data_scale_factor, guidance_scale)
        aug_data = np.concatenate([train_22ch, gen_22ch], axis=0)
        aug_labels = np.concatenate([train_labels, train_labels], axis=0)
        
        clf = train_classifier(aug_data, aug_labels, device, 22)
        metrics_aug = evaluate_classifier(clf, test_22ch, test_labels, device)
        print(f"  DTTD: acc={metrics_aug['accuracy']*100:.2f}%, kappa={metrics_aug['kappa']:.4f}")
        del clf
        
        results[f'S{sid}'] = {
            'baseline_22ch': metrics_22,
            'baseline_9ch': metrics_9,
            'dttd_augmented': metrics_aug,
        }
        torch.cuda.empty_cache()
    
    # 平均
    avg_22 = np.mean([r['baseline_22ch']['accuracy'] for r in results.values()])
    avg_9 = np.mean([r['baseline_9ch']['accuracy'] for r in results.values()])
    avg_aug = np.mean([r['dttd_augmented']['accuracy'] for r in results.values()])
    avg_k22 = np.mean([r['baseline_22ch']['kappa'] for r in results.values()])
    avg_k9 = np.mean([r['baseline_9ch']['kappa'] for r in results.values()])
    avg_kaug = np.mean([r['dttd_augmented']['kappa'] for r in results.values()])
    
    results['average'] = {
        'baseline_22ch': {'accuracy': avg_22, 'kappa': avg_k22},
        'baseline_9ch': {'accuracy': avg_9, 'kappa': avg_k9},
        'dttd_augmented': {'accuracy': avg_aug, 'kappa': avg_kaug},
    }
    print(f"\n平均: 22ch acc={avg_22*100:.2f}% kappa={avg_k22:.4f}, "
          f"9ch acc={avg_9*100:.2f}% kappa={avg_k9:.4f}, "
          f"DTTD acc={avg_aug*100:.2f}% kappa={avg_kaug:.4f}")
    
    return results


def cross_subject_loso_eval(data_path, model, device, data_scale_factor, subject_ids=range(1, 10), guidance_scale=3.0):
    """跨被试测试（LOSO）"""
    print("\n" + "="*60)
    print("跨被试测试 (LOSO)")
    print("="*60)
    
    results = {}
    
    for test_sid in subject_ids:
        print(f"\n测试被试 S{test_sid}:")
        
        # 收集训练数据（只用第一个会话 Session T）
        train_22ch_list, train_labels_list = [], []
        for sid in subject_ids:
            if sid == test_sid:
                continue
            # 只用第一个会话（Session T）作为训练集
            data, labels = load_raw_bci2a(data_path, sid, 'T')
            train_22ch_list.append(data)
            train_labels_list.append(labels)
        
        train_22ch = np.concatenate(train_22ch_list, axis=0)
        train_labels = np.concatenate(train_labels_list, axis=0)
        train_9ch = train_22ch[:, CH_IDX_9, :]
        
        # 测试数据
        test_t, labels_t = load_raw_bci2a(data_path, test_sid, 'T')
        test_e, labels_e = load_raw_bci2a(data_path, test_sid, 'E')
        test_22ch = np.concatenate([test_t, test_e], axis=0)
        test_labels = np.concatenate([labels_t, labels_e], axis=0)
        test_9ch = test_22ch[:, CH_IDX_9, :]
        
        # 基线1：22通道
        clf = train_classifier(train_22ch, train_labels, device, 22)
        metrics_22 = evaluate_classifier(clf, test_22ch, test_labels, device)
        print(f"  22ch: acc={metrics_22['accuracy']*100:.2f}%, kappa={metrics_22['kappa']:.4f}")
        del clf
        
        # 基线2：9通道
        clf = train_classifier(train_9ch, train_labels, device, 9)
        metrics_9 = evaluate_classifier(clf, test_9ch, test_labels, device)
        print(f"  9ch:  acc={metrics_9['accuracy']*100:.2f}%, kappa={metrics_9['kappa']:.4f}")
        del clf
        
        # DTTD增强
        gen_22ch = generate_22ch_data(model, train_9ch, train_labels, device, 
                                      data_scale_factor, guidance_scale)
        aug_data = np.concatenate([train_22ch, gen_22ch], axis=0)
        aug_labels = np.concatenate([train_labels, train_labels], axis=0)
        
        clf = train_classifier(aug_data, aug_labels, device, 22)
        metrics_aug = evaluate_classifier(clf, test_22ch, test_labels, device)
        print(f"  DTTD: acc={metrics_aug['accuracy']*100:.2f}%, kappa={metrics_aug['kappa']:.4f}")
        del clf
        
        results[f'S{test_sid}'] = {
            'baseline_22ch': metrics_22,
            'baseline_9ch': metrics_9,
            'dttd_augmented': metrics_aug,
        }
        torch.cuda.empty_cache()
    
    # 平均
    avg_22 = np.mean([r['baseline_22ch']['accuracy'] for r in results.values()])
    avg_9 = np.mean([r['baseline_9ch']['accuracy'] for r in results.values()])
    avg_aug = np.mean([r['dttd_augmented']['accuracy'] for r in results.values()])
    avg_k22 = np.mean([r['baseline_22ch']['kappa'] for r in results.values()])
    avg_k9 = np.mean([r['baseline_9ch']['kappa'] for r in results.values()])
    avg_kaug = np.mean([r['dttd_augmented']['kappa'] for r in results.values()])
    
    results['average'] = {
        'baseline_22ch': {'accuracy': avg_22, 'kappa': avg_k22},
        'baseline_9ch': {'accuracy': avg_9, 'kappa': avg_k9},
        'dttd_augmented': {'accuracy': avg_aug, 'kappa': avg_kaug},
    }
    print(f"\n平均: 22ch acc={avg_22*100:.2f}% kappa={avg_k22:.4f}, "
          f"9ch acc={avg_9*100:.2f}% kappa={avg_k9:.4f}, "
          f"DTTD acc={avg_aug*100:.2f}% kappa={avg_kaug:.4f}")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='分类评估实验')
    parser.add_argument('--data-path', default='E:/data/BCI2a')
    parser.add_argument('--config', default='configs/bci2a_enhanced_config.yaml')
    parser.add_argument('--checkpoint', default='checkpoints/dttd/best_model.pth')
    parser.add_argument('--output-dir', default='paper_results/classification')
    parser.add_argument('--mode', default='all', choices=['within_subject', 'cross_session', 'cross_subject', 'all'])
    parser.add_argument('--guidance-scale', type=float, default=3.0)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # 设置固定随机种子，确保可复现
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = get_device()
    print(f"随机种子: {args.seed}")
    model, data_scale_factor = load_dttd_model(args.config, args.checkpoint, device)
    
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {'guidance_scale': args.guidance_scale}
    }
    
    if args.mode in ['within_subject', 'all']:
        results['within_subject'] = within_subject_eval(
            args.data_path, model, device, data_scale_factor, guidance_scale=args.guidance_scale)
    
    if args.mode in ['cross_session', 'all']:
        results['cross_session'] = cross_session_eval(
            args.data_path, model, device, data_scale_factor, guidance_scale=args.guidance_scale)
    
    if args.mode in ['cross_subject', 'all']:
        results['cross_subject'] = cross_subject_loso_eval(
            args.data_path, model, device, data_scale_factor, guidance_scale=args.guidance_scale)
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'classification_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到: {args.output_dir}/classification_results.json")


if __name__ == '__main__':
    main()
