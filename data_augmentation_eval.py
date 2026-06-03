"""
数据增强评估实验
使用DTTD-DDPM生成22通道数据进行数据增强，评估分类性能

实验逻辑：
1. 跨会话测试：Session T训练 -> Session E测试
   - 用Session T的9通道数据生成22通道
   - 原始22通道 + 生成22通道 = 增强数据
   - 用增强数据训练分类器，在Session E测试

2. 跨被试测试（LOSO）：8个被试训练 -> 1个被试测试
   - 用8个被试的9通道数据生成22通道
   - 原始22通道 + 生成22通道 = 增强数据
   - 用增强数据训练分类器，在留出被试测试

数据处理：无预处理（原始数据）
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
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models import DTTDEnhanced
from utils import load_config, get_device

# 9通道索引 (C3, C1, Cz, C2, C4, FC3, FCz, FC4, CP3, CPz, CP4相关)
CH_IDX_9 = [7, 9, 11, 1, 3, 5, 13, 15, 17]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_raw_bci2a(data_path, subject_id, session='T'):
    """加载原始BCI2a数据，无预处理"""
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
    
    return (accuracy_score(labels_all, preds), 
            f1_score(labels_all, preds, average='macro'),
            cohen_kappa_score(labels_all, preds))


def load_dttd_model(config_path, checkpoint_path, device):
    """加载DTTD模型"""
    config = load_config(config_path)
    model = DTTDEnhanced(config['model']).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    model.eval()
    print(f"[OK] 加载DTTD模型: {checkpoint_path}")
    return model



def generate_22ch_data(model, data_9ch, labels, device, data_22ch_ref, noise_level=0.02, 
                       use_ddim=False, num_steps=50, guidance_scale=3.0):
    """
    用DTTD模型从9通道生成22通道数据
    
    关键：模型训练时用了 data_scale_factor=1e5，所以需要：
    1. 输入数据放大 1e5
    2. 输出数据缩小 1e5
    
    Args:
        model: DTTD模型
        data_9ch: 9通道输入数据
        labels: 类别标签
        device: 设备
        data_22ch_ref: 22通道参考数据（用于计算统计量）
        noise_level: 噪声水平
        use_ddim: 是否使用DDIM采样（更高质量但更慢）
        num_steps: DDIM采样步数
        guidance_scale: 分类器引导强度
    """
    model.eval()
    
    # ⭐ 训练时使用的缩放因子
    data_scale_factor = 1e5
    
    # 放大输入数据
    data_9ch_scaled = data_9ch * data_scale_factor
    
    loader = DataLoader(TensorDataset(torch.FloatTensor(data_9ch_scaled), torch.LongTensor(labels)),
                        batch_size=32, shuffle=False)
    
    generated_list = []
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(loader, desc="生成22通道"):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            if use_ddim:
                # 使用DDIM采样（更高质量）
                gen = model.sample_ddim(
                    batch_data, 
                    task_label=batch_labels,
                    num_inference_steps=num_steps,
                    eta=0.0,
                    guidance_scale=guidance_scale
                ).cpu().numpy()
            else:
                # 使用快速采样（model.sample方法）
                gen = model.sample(
                    batch_data,
                    task_label=batch_labels,
                    num_steps=10,
                    guidance_scale=guidance_scale,
                    use_full_denoising=False
                ).cpu().numpy()
            
            # 缩小回原始尺度
            gen = gen / data_scale_factor
            generated_list.append(gen)
    
    return np.concatenate(generated_list, axis=0)


def within_subject_eval(data_path, model, device, subject_ids=range(1, 10), 
                        use_ddim=False, num_steps=50, guidance_scale=3.0):
    """
    被试内测试 (Within-Subject) - 5折交叉验证
    
    流程：
    1. 合并两个会话数据
    2. 5折交叉验证：每折用4折训练+生成，1折测试
    3. 用训练折的9通道生成22通道
    4. 原始22通道 + 生成22通道 = 增强数据
    5. 用增强数据训练分类器，在测试折(22通道)上测试
    """
    from sklearn.model_selection import StratifiedKFold
    
    print("\n" + "="*60)
    print("被试内测试 (Within-Subject) - 5折交叉验证")
    print(f"采样方式: {'DDIM' if use_ddim else 'Fast'}, 引导强度: {guidance_scale}")
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
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(all_data_22ch, all_labels)):
            train_22ch = all_data_22ch[train_idx]
            train_9ch = all_data_9ch[train_idx]
            train_labels = all_labels[train_idx]
            test_22ch = all_data_22ch[test_idx]
            test_9ch = all_data_9ch[test_idx]
            test_labels = all_labels[test_idx]
            
            # 基线1：22通道
            clf = train_classifier(train_22ch, train_labels, device, 22)
            acc_22, _, _ = evaluate_classifier(clf, test_22ch, test_labels, device)
            accs_22.append(acc_22)
            del clf
            
            # 基线2：9通道
            clf = train_classifier(train_9ch, train_labels, device, 9)
            acc_9, _, _ = evaluate_classifier(clf, test_9ch, test_labels, device)
            accs_9.append(acc_9)
            del clf
            
            # DTTD增强
            gen_22ch = generate_22ch_data(model, train_9ch, train_labels, device, train_22ch,
                                          use_ddim=use_ddim, num_steps=num_steps, guidance_scale=guidance_scale)
            aug_data = np.concatenate([train_22ch, gen_22ch], axis=0)
            aug_labels = np.concatenate([train_labels, train_labels], axis=0)
            
            clf = train_classifier(aug_data, aug_labels, device, 22)
            acc_aug, _, _ = evaluate_classifier(clf, test_22ch, test_labels, device)
            accs_aug.append(acc_aug)
            del clf
            
            torch.cuda.empty_cache()
        
        mean_22 = np.mean(accs_22)
        mean_9 = np.mean(accs_9)
        mean_aug = np.mean(accs_aug)
        
        print(f"  22ch: {mean_22*100:.2f}%, 9ch: {mean_9*100:.2f}%, DTTD: {mean_aug*100:.2f}%")
        
        results[f'S{sid}'] = {
            'baseline_22ch': {'accuracy': mean_22, 'std': np.std(accs_22)},
            'baseline_9ch': {'accuracy': mean_9, 'std': np.std(accs_9)},
            'dttd_augmented': {'accuracy': mean_aug, 'std': np.std(accs_aug)},
        }
    
    # 平均
    avg_22 = np.mean([r['baseline_22ch']['accuracy'] for r in results.values()])
    avg_9 = np.mean([r['baseline_9ch']['accuracy'] for r in results.values()])
    avg_aug = np.mean([r['dttd_augmented']['accuracy'] for r in results.values()])
    
    results['average'] = {'baseline_22ch': avg_22, 'baseline_9ch': avg_9, 'dttd_augmented': avg_aug}
    
    print(f"\n平均: 22ch={avg_22*100:.2f}%, 9ch={avg_9*100:.2f}%, DTTD={avg_aug*100:.2f}%")
    return results


def cross_session_eval(data_path, model, device, subject_ids=range(1, 10),
                       use_ddim=False, num_steps=50, guidance_scale=3.0):
    """跨会话测试：Session T训练 -> Session E测试"""
    print("\n" + "="*60)
    print("跨会话测试 (Cross-Session)")
    print(f"采样方式: {'DDIM' if use_ddim else 'Fast'}, 引导强度: {guidance_scale}")
    print("="*60)
    
    results = {}
    
    for sid in subject_ids:
        print(f"\n被试 S{sid}:")
        
        train_22ch, train_labels = load_raw_bci2a(data_path, sid, 'T')
        test_22ch, test_labels = load_raw_bci2a(data_path, sid, 'E')
        train_9ch = train_22ch[:, CH_IDX_9, :]
        test_9ch = test_22ch[:, CH_IDX_9, :]
        
        # 基线1：22通道
        print("  [1] 基线22通道...")
        clf = train_classifier(train_22ch, train_labels, device, 22)
        acc_22, f1_22, kappa_22 = evaluate_classifier(clf, test_22ch, test_labels, device)
        print(f"      准确率: {acc_22*100:.2f}%")
        del clf
        
        # 基线2：9通道
        print("  [2] 基线9通道...")
        clf = train_classifier(train_9ch, train_labels, device, 9)
        acc_9, f1_9, kappa_9 = evaluate_classifier(clf, test_9ch, test_labels, device)
        print(f"      准确率: {acc_9*100:.2f}%")
        del clf
        
        # DTTD增强
        print("  [3] DTTD增强...")
        gen_22ch = generate_22ch_data(model, train_9ch, train_labels, device, train_22ch,
                                      use_ddim=use_ddim, num_steps=num_steps, guidance_scale=guidance_scale)
        aug_data = np.concatenate([train_22ch, gen_22ch], axis=0)
        aug_labels = np.concatenate([train_labels, train_labels], axis=0)
        
        clf = train_classifier(aug_data, aug_labels, device, 22)
        acc_aug, f1_aug, kappa_aug = evaluate_classifier(clf, test_22ch, test_labels, device)
        print(f"      准确率: {acc_aug*100:.2f}%")
        del clf
        
        results[f'S{sid}'] = {
            'baseline_22ch': {'accuracy': acc_22, 'f1': f1_22, 'kappa': kappa_22},
            'baseline_9ch': {'accuracy': acc_9, 'f1': f1_9, 'kappa': kappa_9},
            'dttd_augmented': {'accuracy': acc_aug, 'f1': f1_aug, 'kappa': kappa_aug},
        }
        torch.cuda.empty_cache()
    
    # 平均
    avg_22 = np.mean([r['baseline_22ch']['accuracy'] for r in results.values()])
    avg_9 = np.mean([r['baseline_9ch']['accuracy'] for r in results.values()])
    avg_aug = np.mean([r['dttd_augmented']['accuracy'] for r in results.values()])
    
    results['average'] = {'baseline_22ch': avg_22, 'baseline_9ch': avg_9, 'dttd_augmented': avg_aug}
    
    print(f"\n平均: 22ch={avg_22*100:.2f}%, 9ch={avg_9*100:.2f}%, DTTD={avg_aug*100:.2f}%")
    return results


def cross_subject_loso_eval(data_path, model, device, subject_ids=range(1, 10), output_dir='paper_results/augmentation_eval',
                            use_ddim=False, num_steps=50, guidance_scale=3.0):
    """跨被试测试（LOSO）- 增量保存"""
    print("\n" + "="*60)
    print("跨被试测试 (LOSO)")
    print(f"采样方式: {'DDIM' if use_ddim else 'Fast'}, 引导强度: {guidance_scale}")
    print("="*60)
    
    # 尝试加载已有结果
    results_file = os.path.join(output_dir, 'cross_subject_partial.json')
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"[OK] 加载已有结果: {len(results)} 个被试")
    else:
        results = {}
    
    for test_sid in subject_ids:
        # 跳过已完成的被试
        if f'S{test_sid}' in results:
            print(f"\n被试 S{test_sid}: 已完成，跳过")
            continue
        print(f"\n测试被试 S{test_sid}:")
        
        # 收集训练数据
        train_22ch_list, train_labels_list = [], []
        for sid in subject_ids:
            if sid == test_sid:
                continue
            for sess in ['T', 'E']:
                data, labels = load_raw_bci2a(data_path, sid, sess)
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
        print("  [1] 基线22通道...")
        clf = train_classifier(train_22ch, train_labels, device, 22)
        acc_22, f1_22, kappa_22 = evaluate_classifier(clf, test_22ch, test_labels, device)
        print(f"      准确率: {acc_22*100:.2f}%")
        del clf
        
        # 基线2：9通道
        print("  [2] 基线9通道...")
        clf = train_classifier(train_9ch, train_labels, device, 9)
        acc_9, f1_9, kappa_9 = evaluate_classifier(clf, test_9ch, test_labels, device)
        print(f"      准确率: {acc_9*100:.2f}%")
        del clf
        
        # DTTD增强
        print("  [3] DTTD增强...")
        gen_22ch = generate_22ch_data(model, train_9ch, train_labels, device, train_22ch,
                                      use_ddim=use_ddim, num_steps=num_steps, guidance_scale=guidance_scale)
        aug_data = np.concatenate([train_22ch, gen_22ch], axis=0)
        aug_labels = np.concatenate([train_labels, train_labels], axis=0)
        
        clf = train_classifier(aug_data, aug_labels, device, 22)
        acc_aug, f1_aug, kappa_aug = evaluate_classifier(clf, test_22ch, test_labels, device)
        print(f"      准确率: {acc_aug*100:.2f}%")
        del clf
        
        results[f'S{test_sid}'] = {
            'baseline_22ch': {'accuracy': acc_22, 'f1': f1_22, 'kappa': kappa_22},
            'baseline_9ch': {'accuracy': acc_9, 'f1': f1_9, 'kappa': kappa_9},
            'dttd_augmented': {'accuracy': acc_aug, 'f1': f1_aug, 'kappa': kappa_aug},
        }
        
        # 增量保存
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  [OK] 结果已保存")
        
        torch.cuda.empty_cache()
    
    avg_22 = np.mean([r['baseline_22ch']['accuracy'] for r in results.values()])
    avg_9 = np.mean([r['baseline_9ch']['accuracy'] for r in results.values()])
    avg_aug = np.mean([r['dttd_augmented']['accuracy'] for r in results.values()])
    
    results['average'] = {'baseline_22ch': avg_22, 'baseline_9ch': avg_9, 'dttd_augmented': avg_aug}
    
    print(f"\n平均: 22ch={avg_22*100:.2f}%, 9ch={avg_9*100:.2f}%, DTTD={avg_aug*100:.2f}%")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='E:/data/BCI2a')
    parser.add_argument('--config', default='configs/bci2a_enhanced_config.yaml')
    parser.add_argument('--checkpoint', default='checkpoints/bci2a_enhanced/best_model.pth')
    parser.add_argument('--output-dir', default='paper_results/augmentation_eval')
    parser.add_argument('--mode', default='all', choices=['within_subject', 'cross_session', 'cross_subject', 'all'])
    parser.add_argument('--use-ddim', action='store_true', help='使用DDIM采样（更高质量但更慢）')
    parser.add_argument('--num-steps', type=int, default=50, help='DDIM采样步数')
    parser.add_argument('--guidance-scale', type=float, default=3.0, help='分类器引导强度')
    args = parser.parse_args()
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    device = get_device()
    model = load_dttd_model(args.config, args.checkpoint, device)
    
    results = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
               'sampling': {'use_ddim': args.use_ddim, 'num_steps': args.num_steps, 'guidance_scale': args.guidance_scale}}
    
    sampling_params = {'use_ddim': args.use_ddim, 'num_steps': args.num_steps, 'guidance_scale': args.guidance_scale}
    
    if args.mode in ['within_subject', 'all']:
        results['within_subject'] = within_subject_eval(args.data_path, model, device, **sampling_params)
    
    if args.mode in ['cross_session', 'all']:
        results['cross_session'] = cross_session_eval(args.data_path, model, device, **sampling_params)
    
    if args.mode in ['cross_subject', 'all']:
        results['cross_subject'] = cross_subject_loso_eval(args.data_path, model, device, 
                                                           output_dir=args.output_dir, **sampling_params)
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到: {args.output_dir}/results.json")


if __name__ == '__main__':
    main()
