"""
全基线分类评估实验

对所有基线方法（CVAE, cGAN, Simple-DDPM, EEGDiff, BrainDiff）在三个数据集上
进行重建分类和增强分类评估。

评估维度：
- 数据集: BCI2a, HGD, PhysioNet MI
- 场景: 被试内(5-fold CV), 跨session, 跨被试(LOSO)
- 模式: 仅重建(recon), 增强(augmented=real+generated)
- 方法: CVAE, cGAN, Simple-DDPM, EEGDiff, BrainDiff

使用方法:
    python experiments/baseline_classification_all.py --dataset bci2a --mode all
    python experiments/baseline_classification_all.py --dataset hgd --mode cross_subject
    python experiments/baseline_classification_all.py --dataset physionet --mode within_subject
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.baselines import CVAE, ConditionalGAN, SimpleDDPM, EEGDiff, BrainDiff
from utils import get_device


# ==================== EEGNet分类器 ====================

class EEGNetClassifier(nn.Module):
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
        flat_size = F2 * (time_steps // 32)
        self.classifier = nn.Linear(flat_size, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv3(self.conv2(self.conv1(x)))
        return self.classifier(x.flatten(1))


def train_classifier(train_data, train_labels, device, num_channels=22, epochs=100):
    time_steps = train_data.shape[2]
    clf = EEGNetClassifier(num_channels, 4, time_steps).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)

    generator = torch.Generator()
    generator.manual_seed(42)

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(train_data), torch.LongTensor(train_labels)),
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
    clf.eval()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(test_data), torch.LongTensor(test_labels)),
        batch_size=32, shuffle=False)
    preds, labels_all = [], []
    with torch.no_grad():
        for data, labels in loader:
            preds.extend(torch.argmax(clf(data.to(device)), dim=1).cpu().numpy())
            labels_all.extend(labels.numpy())
    return {
        'accuracy': float(accuracy_score(labels_all, preds)),
        'f1': float(f1_score(labels_all, preds, average='macro')),
        'kappa': float(cohen_kappa_score(labels_all, preds))
    }


# ==================== 模型加载 ====================

CHECKPOINT_MAP = {
    'bci2a': {
        'CVAE': 'checkpoints/bci2a/baseline_cvae/best_model.pth',
        'cGAN': 'checkpoints/bci2a_enhanced/baseline_cgan/best_model.pth',
        'Simple-DDPM': 'checkpoints/bci2a/baseline_simple_ddpm/best_model.pth',
        'EEGDiff': 'checkpoints/bci2a/baseline_eegdiff/best_model.pth',
        'BrainDiff': 'checkpoints/bci2a/baseline_braindiff/best_model.pth',
    },
    'hgd': {
        'CVAE': 'checkpoints/hgd/baseline_cvae/best_model.pth',
        'cGAN': 'checkpoints/hgd/baseline_cgan/best_model.pth',
        'EEGDiff': 'checkpoints/hgd/baseline_eegdiff/best_model.pth',
        'BrainDiff': 'checkpoints/hgd/baseline_braindiff/best_model.pth',
    },
    'physionet': {
        'CVAE': 'checkpoints/physionet/baseline_cvae/best_model.pth',
        'cGAN': 'checkpoints/physionet/baseline_cgan/best_model.pth',
        'EEGDiff': 'checkpoints/physionet/baseline_eegdiff/best_model.pth',
        'BrainDiff': 'checkpoints/physionet/baseline_braindiff/best_model.pth',
    }
}

# 数据集配置: (input_channels, output_channels, time_steps, num_classes)
DATASET_CONFIG = {
    'bci2a': (9, 22, 1000, 4),
    'hgd': (16, 128, 1000, 4),
    'physionet': (16, 64, 640, 4),
}


def load_baseline_model(model_name, dataset, device):
    """加载基线模型，返回 (model, scale_info)

    scale_info 决定推理时的数据缩放方式:
      - BCI2a CVAE/cGAN/Simple-DDPM: 使用 data_scale_factor=1e5
      - BCI2a EEGDiff/BrainDiff: 使用 z-score (data_mean, data_std)
      - HGD/PhysioNet 所有模型: 使用 z-score (data_mean, data_std)
    """
    ckpt_path = CHECKPOINT_MAP[dataset].get(model_name)
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print(f"[SKIP] {model_name} checkpoint not found: {ckpt_path}")
        return None, None

    in_ch, out_ch, t_steps, n_cls = DATASET_CONFIG[dataset]

    try:
        if model_name == 'CVAE':
            model = CVAE(in_ch, out_ch, t_steps, n_cls, latent_dim=128).to(device)
        elif model_name == 'cGAN':
            model = ConditionalGAN(in_ch, out_ch, t_steps, n_cls, latent_dim=128).to(device)
        elif model_name == 'Simple-DDPM':
            model = SimpleDDPM(in_ch, out_ch, t_steps, n_cls).to(device)
        elif model_name == 'EEGDiff':
            model = EEGDiff(in_ch, out_ch, t_steps, n_cls).to(device)
        elif model_name == 'BrainDiff':
            model = BrainDiff(in_ch, out_ch, t_steps, n_cls).to(device)
        else:
            return None, None

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # 确定缩放方式：只根据checkpoint中实际保存的缩放信息
        scale_info = {'method': 'none'}
        if 'data_scale_factor' in ckpt:
            scale_info = {
                'method': 'scale_factor',
                'data_scale_factor': float(ckpt['data_scale_factor'])
            }
        elif 'data_mean' in ckpt and 'data_std' in ckpt:
            scale_info = {
                'method': 'zscore',
                'data_mean': ckpt['data_mean'],
                'data_std': ckpt['data_std']
            }
        # 否则 method='none'（CVAE/cGAN/Simple-DDPM训练时没缩放）

        print(f"[OK] Loaded {model_name} from {ckpt_path}, scale={scale_info['method']}")
        return model, scale_info
    except Exception as e:
        print(f"[ERROR] Failed to load {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ==================== 重建生成 ====================

@torch.no_grad()
def generate_reconstruction(model, model_name, data_input, labels, device,
                            scale_info=None, input_ch_indices=None, batch_size=32):
    """用基线模型从低通道输入生成高通道重建

    关键：推理时必须与训练时使用相同的数据缩放方式:
    - scale_factor: 输入 * scale_factor -> 模型 -> 输出 / scale_factor
    - zscore: (输入 - mean) / std -> 模型 -> 输出 * std + mean
      注意：mean/std 是全通道的，输入只需取对应通道的子集
    """
    model.eval()
    generated_list = []

    # 预处理：根据缩放方式转换输入
    if scale_info is None:
        scale_info = {'method': 'none'}

    if scale_info['method'] == 'scale_factor':
        sf = scale_info['data_scale_factor']
        data_scaled = data_input * sf
    elif scale_info['method'] == 'zscore':
        mean = scale_info['data_mean']
        std = scale_info['data_std']
        if isinstance(mean, torch.Tensor):
            mean = mean.cpu().numpy()
            std = std.cpu().numpy()
        # mean/std 是全通道的，输入只需取对应通道
        if input_ch_indices is not None and mean.shape[-2] != data_input.shape[1]:
            input_mean = mean[:, input_ch_indices, :]
            input_std = std[:, input_ch_indices, :]
        else:
            input_mean = mean
            input_std = std
        data_scaled = (data_input - input_mean) / input_std
    else:
        data_scaled = data_input

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(data_scaled), torch.LongTensor(labels)),
        batch_size=batch_size, shuffle=False)

    for batch_data, batch_labels in loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        try:
            if model_name == 'CVAE':
                recon, _, _ = model(batch_data, batch_labels)
            elif model_name == 'cGAN':
                z = torch.randn(batch_data.size(0), model.latent_dim, device=device)
                recon = model(z, batch_labels, input_data=batch_data)
            elif model_name == 'Simple-DDPM':
                recon = model.reconstruct(batch_data, batch_labels,
                                          num_inference_steps=1, noise_level=0.02)
            elif model_name in ('EEGDiff', 'BrainDiff'):
                recon = model.sample(batch_data, task_label=batch_labels, num_steps=1,
                                     noise_scale=0.02, guidance_scale=1.0)
            else:
                recon = batch_data
        except Exception as e:
            print(f"[WARN] Generation failed for {model_name}: {e}")
            recon = torch.zeros(batch_data.size(0), data_input.shape[1], data_input.shape[2],
                                device=device)

        generated_list.append(recon.cpu().numpy())

    generated = np.concatenate(generated_list, axis=0)

    # 后处理：反缩放回原始数据范围
    if scale_info['method'] == 'scale_factor':
        generated = generated / sf
    elif scale_info['method'] == 'zscore':
        mean = scale_info['data_mean']
        std = scale_info['data_std']
        if isinstance(mean, torch.Tensor):
            mean = mean.cpu().numpy()
            std = std.cpu().numpy()
        generated = generated * std + mean

    return generated.astype(np.float32)


# ==================== BCI2a 数据加载 ====================

CH_IDX_9 = [7, 9, 11, 1, 3, 5, 13, 15, 17]


def load_raw_bci2a(data_path, subject_id, session='T'):
    from scipy.io import loadmat
    subject_str = f'0{subject_id}' if subject_id < 10 else str(subject_id)
    file_path = os.path.join(data_path, f'A{subject_str}{session}.mat')
    mat_data = loadmat(file_path)

    if 'data' in mat_data:
        data = mat_data['data']
    elif 'X' in mat_data:
        data = mat_data['X']
    else:
        max_key = max(mat_data.keys(),
                      key=lambda k: mat_data[k].size if isinstance(mat_data[k], np.ndarray) else 0)
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


# ==================== BCI2a 评估场景 ====================

def bci2a_within_subject(data_path, models, device, subject_ids=range(1, 10)):
    """BCI2a被试内 - 5折交叉验证"""
    print("\n" + "=" * 60)
    print("BCI2a 被试内 (5-fold CV)")
    print("=" * 60)

    results = {}

    for sid in subject_ids:
        print(f"\n--- Subject S{sid} ---")
        data_t, labels_t = load_raw_bci2a(data_path, sid, 'T')
        data_e, labels_e = load_raw_bci2a(data_path, sid, 'E')
        all_22ch = np.concatenate([data_t, data_e], axis=0)
        all_labels = np.concatenate([labels_t, labels_e], axis=0)
        all_9ch = all_22ch[:, CH_IDX_9, :]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # 基线
        accs_22, accs_9 = [], []
        for fold, (train_idx, test_idx) in enumerate(skf.split(all_22ch, all_labels)):
            clf = train_classifier(all_22ch[train_idx], all_labels[train_idx], device, 22)
            m = evaluate_classifier(clf, all_22ch[test_idx], all_labels[test_idx], device)
            accs_22.append(m['accuracy'])
            del clf

            clf = train_classifier(all_9ch[train_idx], all_labels[train_idx], device, 9)
            m = evaluate_classifier(clf, all_9ch[test_idx], all_labels[test_idx], device)
            accs_9.append(m['accuracy'])
            del clf

        subj_result = {
            'baseline_22ch': {'accuracy': float(np.mean(accs_22)), 'std': float(np.std(accs_22))},
            'baseline_9ch': {'accuracy': float(np.mean(accs_9)), 'std': float(np.std(accs_9))},
        }
        print(f"  22ch: {np.mean(accs_22)*100:.2f}%, 9ch: {np.mean(accs_9)*100:.2f}%")

        # 各基线方法
        for model_name, (model, scale_info) in models.items():
            if model is None:
                continue
            accs_recon, accs_aug = [], []
            for fold, (train_idx, test_idx) in enumerate(skf.split(all_22ch, all_labels)):
                train_9ch = all_9ch[train_idx]
                train_22ch = all_22ch[train_idx]
                train_labels = all_labels[train_idx]
                test_22ch = all_22ch[test_idx]
                test_labels = all_labels[test_idx]

                # 生成重建
                gen_22ch = generate_reconstruction(model, model_name, train_9ch, train_labels,
                                                   device, scale_info=scale_info,
                                                   input_ch_indices=CH_IDX_9)

                # 仅重建分类
                clf = train_classifier(gen_22ch, train_labels, device, 22)
                m = evaluate_classifier(clf, test_22ch, test_labels, device)
                accs_recon.append(m['accuracy'])
                del clf

                # 增强分类 (real + generated)
                aug_data = np.concatenate([train_22ch, gen_22ch], axis=0)
                aug_labels = np.concatenate([train_labels, train_labels], axis=0)
                clf = train_classifier(aug_data, aug_labels, device, 22)
                m = evaluate_classifier(clf, test_22ch, test_labels, device)
                accs_aug.append(m['accuracy'])
                del clf

            subj_result[f'{model_name}_recon'] = {
                'accuracy': float(np.mean(accs_recon)), 'std': float(np.std(accs_recon))
            }
            subj_result[f'{model_name}_aug'] = {
                'accuracy': float(np.mean(accs_aug)), 'std': float(np.std(accs_aug))
            }
            print(f"  {model_name}: recon={np.mean(accs_recon)*100:.2f}%, aug={np.mean(accs_aug)*100:.2f}%")
            torch.cuda.empty_cache()

        results[f'S{sid}'] = subj_result

    # 计算平均
    avg_result = {}
    for key in results['S1'].keys():
        vals = [r[key]['accuracy'] for r in results.values() if key in r]
        stds = [r[key].get('std', 0) for r in results.values() if key in r]
        avg_result[key] = {
            'accuracy': float(np.mean(vals)),
            'std': float(np.std(vals))
        }
    results['average'] = avg_result
    print(f"\n平均结果:")
    for key, val in avg_result.items():
        print(f"  {key}: {val['accuracy']*100:.2f}±{val['std']*100:.2f}%")

    return results


def bci2a_cross_session(data_path, models, device, subject_ids=range(1, 10)):
    """BCI2a跨session: T训练 -> E测试"""
    print("\n" + "=" * 60)
    print("BCI2a 跨session (T->E)")
    print("=" * 60)

    results = {}

    for sid in subject_ids:
        print(f"\n--- Subject S{sid} ---")
        train_22ch, train_labels = load_raw_bci2a(data_path, sid, 'T')
        test_22ch, test_labels = load_raw_bci2a(data_path, sid, 'E')
        train_9ch = train_22ch[:, CH_IDX_9, :]
        test_9ch = test_22ch[:, CH_IDX_9, :]

        # 基线
        clf = train_classifier(train_22ch, train_labels, device, 22)
        m22 = evaluate_classifier(clf, test_22ch, test_labels, device)
        del clf

        clf = train_classifier(train_9ch, train_labels, device, 9)
        m9 = evaluate_classifier(clf, test_9ch, test_labels, device)
        del clf

        subj_result = {
            'baseline_22ch': m22,
            'baseline_9ch': m9,
        }
        print(f"  22ch: {m22['accuracy']*100:.2f}%, 9ch: {m9['accuracy']*100:.2f}%")

        for model_name, (model, scale_info) in models.items():
            if model is None:
                continue
            gen_22ch = generate_reconstruction(model, model_name, train_9ch, train_labels,
                                               device, scale_info=scale_info,
                                               input_ch_indices=CH_IDX_9)

            # 仅重建
            clf = train_classifier(gen_22ch, train_labels, device, 22)
            m_recon = evaluate_classifier(clf, test_22ch, test_labels, device)
            del clf

            # 增强
            aug_data = np.concatenate([train_22ch, gen_22ch], axis=0)
            aug_labels = np.concatenate([train_labels, train_labels], axis=0)
            clf = train_classifier(aug_data, aug_labels, device, 22)
            m_aug = evaluate_classifier(clf, test_22ch, test_labels, device)
            del clf

            subj_result[f'{model_name}_recon'] = m_recon
            subj_result[f'{model_name}_aug'] = m_aug
            print(f"  {model_name}: recon={m_recon['accuracy']*100:.2f}%, aug={m_aug['accuracy']*100:.2f}%")
            torch.cuda.empty_cache()

        results[f'S{sid}'] = subj_result

    # 平均
    avg_result = {}
    for key in results['S1'].keys():
        vals = [r[key]['accuracy'] for r in results.values() if key in r]
        avg_result[key] = {
            'accuracy': float(np.mean(vals)),
            'std': float(np.std(vals))
        }
    results['average'] = avg_result
    print(f"\n平均结果:")
    for key, val in avg_result.items():
        print(f"  {key}: {val['accuracy']*100:.2f}±{val.get('std', 0)*100:.2f}%")

    return results


def bci2a_cross_subject(data_path, models, device, subject_ids=range(1, 10)):
    """BCI2a跨被试 LOSO"""
    print("\n" + "=" * 60)
    print("BCI2a 跨被试 (LOSO)")
    print("=" * 60)

    results = {}

    for test_sid in subject_ids:
        print(f"\n--- Test Subject S{test_sid} ---")
        train_22ch_list, train_labels_list = [], []
        for sid in subject_ids:
            if sid == test_sid:
                continue
            data, labels = load_raw_bci2a(data_path, sid, 'T')
            train_22ch_list.append(data)
            train_labels_list.append(labels)

        train_22ch = np.concatenate(train_22ch_list, axis=0)
        train_labels = np.concatenate(train_labels_list, axis=0)
        train_9ch = train_22ch[:, CH_IDX_9, :]

        test_t, labels_t = load_raw_bci2a(data_path, test_sid, 'T')
        test_e, labels_e = load_raw_bci2a(data_path, test_sid, 'E')
        test_22ch = np.concatenate([test_t, test_e], axis=0)
        test_labels = np.concatenate([labels_t, labels_e], axis=0)
        test_9ch = test_22ch[:, CH_IDX_9, :]

        # 基线
        clf = train_classifier(train_22ch, train_labels, device, 22)
        m22 = evaluate_classifier(clf, test_22ch, test_labels, device)
        del clf

        clf = train_classifier(train_9ch, train_labels, device, 9)
        m9 = evaluate_classifier(clf, test_9ch, test_labels, device)
        del clf

        subj_result = {
            'baseline_22ch': m22,
            'baseline_9ch': m9,
        }
        print(f"  22ch: {m22['accuracy']*100:.2f}%, 9ch: {m9['accuracy']*100:.2f}%")

        for model_name, (model, scale_info) in models.items():
            if model is None:
                continue
            gen_22ch = generate_reconstruction(model, model_name, train_9ch, train_labels,
                                               device, scale_info=scale_info,
                                               input_ch_indices=CH_IDX_9)

            clf = train_classifier(gen_22ch, train_labels, device, 22)
            m_recon = evaluate_classifier(clf, test_22ch, test_labels, device)
            del clf

            aug_data = np.concatenate([train_22ch, gen_22ch], axis=0)
            aug_labels = np.concatenate([train_labels, train_labels], axis=0)
            clf = train_classifier(aug_data, aug_labels, device, 22)
            m_aug = evaluate_classifier(clf, test_22ch, test_labels, device)
            del clf

            subj_result[f'{model_name}_recon'] = m_recon
            subj_result[f'{model_name}_aug'] = m_aug
            print(f"  {model_name}: recon={m_recon['accuracy']*100:.2f}%, aug={m_aug['accuracy']*100:.2f}%")
            torch.cuda.empty_cache()

        results[f'S{test_sid}'] = subj_result

    # 平均
    avg_result = {}
    for key in results['S1'].keys():
        vals = [r[key]['accuracy'] for r in results.values() if key in r]
        avg_result[key] = {
            'accuracy': float(np.mean(vals)),
            'std': float(np.std(vals))
        }
    results['average'] = avg_result
    print(f"\n平均结果:")
    for key, val in avg_result.items():
        print(f"  {key}: {val['accuracy']*100:.2f}±{val.get('std', 0)*100:.2f}%")

    return results


# ==================== HGD 评估 ====================

def load_hgd_data(data_path='E:/data/HGD'):
    """加载HGD数据"""
    from data.high_gamma_dataset import HighGammaDataset
    dataset = HighGammaDataset(data_path, subject_ids=list(range(1, 15)),
                               sessions='both', fs_target=250)
    return dataset


def hgd_cross_subject(data_path, models, device):
    """HGD跨被试 LOSO"""
    print("\n" + "=" * 60)
    print("HGD 跨被试 (LOSO)")
    print("=" * 60)

    dataset = load_hgd_data(data_path)
    input_indices = dataset.input_ch_indices
    n_output = dataset.num_output_channels

    results = {}

    for test_sid in range(1, 15):
        print(f"\n--- Test Subject S{test_sid} ---")
        train_data_list, train_labels_list = [], []
        test_data_list, test_labels_list = [], []

        for sid in range(1, 15):
            mask = dataset.subject_ids == sid
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            data = dataset.data[idx]
            labels = dataset.labels[idx]
            if sid == test_sid:
                test_data_list.append(data)
                test_labels_list.append(labels)
            else:
                train_data_list.append(data)
                train_labels_list.append(labels)

        if not train_data_list or not test_data_list:
            continue

        train_full = np.concatenate(train_data_list, axis=0).astype(np.float32)
        train_labels = np.concatenate(train_labels_list, axis=0).astype(np.int64)
        test_full = np.concatenate(test_data_list, axis=0).astype(np.float32)
        test_labels = np.concatenate(test_labels_list, axis=0).astype(np.int64)

        train_input = train_full[:, input_indices, :]
        test_input = test_full[:, input_indices, :]

        # 基线
        clf = train_classifier(train_full, train_labels, device, n_output)
        m_full = evaluate_classifier(clf, test_full, test_labels, device)
        del clf

        clf = train_classifier(train_input, train_labels, device, len(input_indices))
        m_input = evaluate_classifier(clf, test_input, test_labels, device)
        del clf

        subj_result = {
            'baseline_full': m_full,
            'baseline_input': m_input,
        }
        print(f"  {n_output}ch: {m_full['accuracy']*100:.2f}%, {len(input_indices)}ch: {m_input['accuracy']*100:.2f}%")

        for model_name, (model, scale_info) in models.items():
            if model is None:
                continue
            gen_full = generate_reconstruction(model, model_name, train_input, train_labels,
                                               device, scale_info=scale_info,
                                               input_ch_indices=input_indices)

            clf = train_classifier(gen_full, train_labels, device, n_output)
            m_recon = evaluate_classifier(clf, test_full, test_labels, device)
            del clf

            aug_data = np.concatenate([train_full, gen_full], axis=0)
            aug_labels = np.concatenate([train_labels, train_labels], axis=0)
            clf = train_classifier(aug_data, aug_labels, device, n_output)
            m_aug = evaluate_classifier(clf, test_full, test_labels, device)
            del clf

            subj_result[f'{model_name}_recon'] = m_recon
            subj_result[f'{model_name}_aug'] = m_aug
            print(f"  {model_name}: recon={m_recon['accuracy']*100:.2f}%, aug={m_aug['accuracy']*100:.2f}%")
            torch.cuda.empty_cache()

        results[f'S{test_sid}'] = subj_result

    # 平均
    avg_result = {}
    first_key = list(results.keys())[0]
    for key in results[first_key].keys():
        vals = [r[key]['accuracy'] for r in results.values() if key in r]
        avg_result[key] = {
            'accuracy': float(np.mean(vals)),
            'std': float(np.std(vals))
        }
    results['average'] = avg_result
    print(f"\n平均结果:")
    for key, val in avg_result.items():
        print(f"  {key}: {val['accuracy']*100:.2f}±{val.get('std', 0)*100:.2f}%")

    return results


# ==================== PhysioNet MI 评估 ====================

def load_physionet_data():
    """加载PhysioNet MI数据"""
    from data.physionet_mi import PhysioNetMIDataset, INPUT_CHANNEL_INDICES_16
    cache_path = 'paper_results/physionet_mi/physionet_mi_all_subjects.npz'
    if os.path.exists(cache_path):
        cache = np.load(cache_path)
        data = cache['data'].astype(np.float32)
        labels = cache['labels'].astype(np.int64)
        subject_ids = cache.get('subject_ids', None)
        if subject_ids is not None:
            subject_ids = subject_ids.astype(np.int64)
    else:
        dataset = PhysioNetMIDataset()
        data = dataset.data.astype(np.float32)
        labels = dataset.labels.astype(np.int64)
        subject_ids = getattr(dataset, 'subject_ids', None)

    return data, labels, subject_ids, INPUT_CHANNEL_INDICES_16


def physionet_within_subject(models, device):
    """PhysioNet MI被试内评估"""
    print("\n" + "=" * 60)
    print("PhysioNet MI 被试内 (5-fold CV)")
    print("=" * 60)

    data, labels, subject_ids, input_indices = load_physionet_data()
    n_output = data.shape[1]
    n_input = len(input_indices)

    unique_subjects = np.unique(subject_ids) if subject_ids is not None else [0]
    results = {}

    for subj in unique_subjects:
        if subject_ids is not None:
            mask = subject_ids == subj
            subj_data = data[mask]
            subj_labels = labels[mask]
        else:
            subj_data = data
            subj_labels = labels

        print(f"\n--- Subject {subj} ---")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs_full, accs_input = [], []

        for fold, (train_idx, test_idx) in enumerate(skf.split(subj_data, subj_labels)):
            train_full = subj_data[train_idx]
            train_labels = subj_labels[train_idx]
            test_full = subj_data[test_idx]
            test_labels = subj_labels[test_idx]

            train_input = train_full[:, input_indices, :]
            test_input = test_full[:, input_indices, :]

            clf = train_classifier(train_full, train_labels, device, n_output)
            m = evaluate_classifier(clf, test_full, test_labels, device)
            accs_full.append(m['accuracy'])
            del clf

            clf = train_classifier(train_input, train_labels, device, n_input)
            m = evaluate_classifier(clf, test_input, test_labels, device)
            accs_input.append(m['accuracy'])
            del clf

        subj_result = {
            'baseline_full': {'accuracy': float(np.mean(accs_full)), 'std': float(np.std(accs_full))},
            'baseline_input': {'accuracy': float(np.mean(accs_input)), 'std': float(np.std(accs_input))},
        }
        print(f"  {n_output}ch: {np.mean(accs_full)*100:.2f}%, {n_input}ch: {np.mean(accs_input)*100:.2f}%")

        for model_name, (model, scale_info) in models.items():
            if model is None:
                continue
            accs_recon, accs_aug = [], []
            for fold, (train_idx, test_idx) in enumerate(skf.split(subj_data, subj_labels)):
                train_full = subj_data[train_idx]
                train_labels = subj_labels[train_idx]
                test_full = subj_data[test_idx]
                test_labels = subj_labels[test_idx]
                train_input = train_full[:, input_indices, :]

                gen_full = generate_reconstruction(model, model_name, train_input, train_labels,
                                                   device, scale_info=scale_info,
                                                   input_ch_indices=input_indices)

                clf = train_classifier(gen_full, train_labels, device, n_output)
                m = evaluate_classifier(clf, test_full, test_labels, device)
                accs_recon.append(m['accuracy'])
                del clf

                aug_data = np.concatenate([train_full, gen_full], axis=0)
                aug_labels = np.concatenate([train_labels, train_labels], axis=0)
                clf = train_classifier(aug_data, aug_labels, device, n_output)
                m = evaluate_classifier(clf, test_full, test_labels, device)
                accs_aug.append(m['accuracy'])
                del clf

            subj_result[f'{model_name}_recon'] = {
                'accuracy': float(np.mean(accs_recon)), 'std': float(np.std(accs_recon))
            }
            subj_result[f'{model_name}_aug'] = {
                'accuracy': float(np.mean(accs_aug)), 'std': float(np.std(accs_aug))
            }
            print(f"  {model_name}: recon={np.mean(accs_recon)*100:.2f}%, aug={np.mean(accs_aug)*100:.2f}%")
            torch.cuda.empty_cache()

        results[f'S{subj}'] = subj_result

    # 平均
    avg_result = {}
    first_key = list(results.keys())[0]
    for key in results[first_key].keys():
        vals = [r[key]['accuracy'] for r in results.values() if key in r]
        avg_result[key] = {
            'accuracy': float(np.mean(vals)),
            'std': float(np.std(vals))
        }
    results['average'] = avg_result
    print(f"\n平均结果:")
    for key, val in avg_result.items():
        print(f"  {key}: {val['accuracy']*100:.2f}±{val.get('std', 0)*100:.2f}%")

    return results


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='全基线分类评估')
    parser.add_argument('--dataset', type=str, default='bci2a',
                        choices=['bci2a', 'hgd', 'physionet', 'all'])
    parser.add_argument('--mode', type=str, default='all',
                        choices=['within_subject', 'cross_session', 'cross_subject', 'all'])
    parser.add_argument('--bci2a-data-path', default='E:/data/BCI2a')
    parser.add_argument('--hgd-data-path', default='E:/data/HGD')
    parser.add_argument('--output-dir', default='paper_results/baseline_classification')
    parser.add_argument('--methods', type=str, default='all',
                        help='Comma-separated methods: CVAE,cGAN,Simple-DDPM,EEGDiff,BrainDiff')
    args = parser.parse_args()

    device = get_device()
    np.random.seed(42)
    torch.manual_seed(42)

    os.makedirs(args.output_dir, exist_ok=True)

    # 确定要评估的方法
    if args.methods == 'all':
        method_names = ['CVAE', 'cGAN', 'Simple-DDPM', 'EEGDiff', 'BrainDiff']
    else:
        method_names = [m.strip() for m in args.methods.split(',')]

    all_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'methods': method_names,
    }

    datasets = ['bci2a', 'hgd', 'physionet'] if args.dataset == 'all' else [args.dataset]

    for dataset in datasets:
        print(f"\n{'#'*60}")
        print(f"# 数据集: {dataset}")
        print(f"{'#'*60}")

        # 加载模型
        models = {}
        for name in method_names:
            if dataset == 'hgd' and name == 'Simple-DDPM':
                print(f"[SKIP] Simple-DDPM not available for HGD")
                continue
            if dataset == 'physionet' and name == 'Simple-DDPM':
                print(f"[SKIP] Simple-DDPM not available for PhysioNet")
                continue
            model, scale_info = load_baseline_model(name, dataset, device)
            if model is not None:
                models[name] = (model, scale_info)

        dataset_results = {}

        if dataset == 'bci2a':
            if args.mode in ('within_subject', 'all'):
                dataset_results['within_subject'] = bci2a_within_subject(
                    args.bci2a_data_path, models, device)
            if args.mode in ('cross_session', 'all'):
                dataset_results['cross_session'] = bci2a_cross_session(
                    args.bci2a_data_path, models, device)
            if args.mode in ('cross_subject', 'all'):
                dataset_results['cross_subject'] = bci2a_cross_subject(
                    args.bci2a_data_path, models, device)
        elif dataset == 'hgd':
            if args.mode in ('cross_subject', 'all'):
                dataset_results['cross_subject'] = hgd_cross_subject(
                    args.hgd_data_path, models, device)
        elif dataset == 'physionet':
            if args.mode in ('within_subject', 'all'):
                dataset_results['within_subject'] = physionet_within_subject(
                    models, device)

        all_results[dataset] = dataset_results

        # 保存中间结果
        out_path = os.path.join(args.output_dir, f'{dataset}_baseline_classification.json')
        with open(out_path, 'w') as f:
            json.dump(dataset_results, f, indent=2, default=str)
        print(f"\n结果已保存到: {out_path}")

    # 保存总结果
    out_path = os.path.join(args.output_dir, 'all_baseline_classification.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n总结果已保存到: {out_path}")


if __name__ == '__main__':
    main()
