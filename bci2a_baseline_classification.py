"""
BCI2a 基线分类评估

对基线方法在BCI2a数据集上进行重建分类和增强分类评估。
场景：被试内(5-fold CV)、跨session(T->E)、跨被试(LOSO)

使用方法:
    python experiments/bci2a_baseline_classification.py
    python experiments/bci2a_baseline_classification.py --mode cross_session
    python experiments/bci2a_baseline_classification.py --methods CVAE,EEGDiff
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

# BCI2a 9通道索引 (从22通道中选取)
CH_IDX_9 = [7, 9, 11, 1, 3, 5, 13, 15, 17]

from models.traditional_baselines import SplineInterpolation, KrigingInterpolation

TRADITIONAL_METHODS = {'Spline', 'Kriging'}

CHECKPOINT_MAP = {
    'CVAE': 'checkpoints/bci2a/baseline_cvae/best_model.pth',
    'cGAN': 'checkpoints/bci2a_enhanced/baseline_cgan/best_model.pth',
    'Simple-DDPM': 'checkpoints/bci2a/baseline_simple_ddpm/best_model.pth',
    'EEGDiff': 'checkpoints/bci2a/baseline_eegdiff/best_model.pth',
    'BrainDiff': 'checkpoints/bci2a/baseline_braindiff/best_model.pth',
}


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

def load_baseline_model(model_name, device):
    """加载基线模型，返回 (model, scale_info)"""
    ckpt_path = CHECKPOINT_MAP.get(model_name)
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print(f"[SKIP] {model_name} checkpoint not found: {ckpt_path}")
        return None, None

    in_ch, out_ch, t_steps, n_cls = 9, 22, 1000, 4

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
            # 只有checkpoint明确保存了scale_factor才使用
            scale_info = {
                'method': 'scale_factor',
                'data_scale_factor': float(ckpt['data_scale_factor'])
            }
        elif 'data_mean' in ckpt and 'data_std' in ckpt:
            # checkpoint保存了z-score参数
            scale_info = {
                'method': 'zscore',
                'data_mean': ckpt['data_mean'],
                'data_std': ckpt['data_std']
            }
        # 否则 method='none'，不做任何缩放（CVAE/cGAN/Simple-DDPM训练时没缩放）

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
                            scale_info=None, batch_size=32):
    """用基线模型从9通道输入生成22通道重建

    缩放逻辑:
    - scale_factor: 输入 * sf -> 模型 -> 输出 / sf
    - zscore: 输入用9通道mean/std标准化 -> 模型 -> 输出用22通道mean/std反标准化
    """
    model.eval()
    generated_list = []

    if scale_info is None:
        scale_info = {'method': 'none'}

    # 预处理输入
    if scale_info['method'] == 'scale_factor':
        sf = scale_info['data_scale_factor']
        data_scaled = data_input * sf
    elif scale_info['method'] == 'zscore':
        mean = scale_info['data_mean']
        std = scale_info['data_std']
        if isinstance(mean, torch.Tensor):
            mean = mean.cpu().numpy()
            std = std.cpu().numpy()
        # mean/std 是 (1,22,1000)，输入是 (N,9,1000)，只取9通道
        input_mean = mean[:, CH_IDX_9, :]
        input_std = std[:, CH_IDX_9, :]
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
            recon = torch.zeros(batch_data.size(0), 22, data_input.shape[2], device=device)

        generated_list.append(recon.cpu().numpy())

    generated = np.concatenate(generated_list, axis=0)

    # 反缩放输出（22通道）
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


# ==================== 数据加载 ====================

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


# ==================== 被试内 ====================

def within_subject(data_path, models, device, subject_ids=range(1, 10)):
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
        accs_22, kappas_22, accs_9, kappas_9 = [], [], [], []
        for fold, (train_idx, test_idx) in enumerate(skf.split(all_22ch, all_labels)):
            clf = train_classifier(all_22ch[train_idx], all_labels[train_idx], device, 22)
            m = evaluate_classifier(clf, all_22ch[test_idx], all_labels[test_idx], device)
            accs_22.append(m['accuracy'])
            kappas_22.append(m['kappa'])
            del clf

            clf = train_classifier(all_9ch[train_idx], all_labels[train_idx], device, 9)
            m = evaluate_classifier(clf, all_9ch[test_idx], all_labels[test_idx], device)
            accs_9.append(m['accuracy'])
            kappas_9.append(m['kappa'])
            del clf

        subj_result = {
            'baseline_22ch': {
                'accuracy': float(np.mean(accs_22)), 'std': float(np.std(accs_22)),
                'kappa': float(np.mean(kappas_22)), 'kappa_std': float(np.std(kappas_22)),
            },
            'baseline_9ch': {
                'accuracy': float(np.mean(accs_9)), 'std': float(np.std(accs_9)),
                'kappa': float(np.mean(kappas_9)), 'kappa_std': float(np.std(kappas_9)),
            },
        }
        print(f"  22ch: {np.mean(accs_22)*100:.2f}% (κ={np.mean(kappas_22):.3f}), 9ch: {np.mean(accs_9)*100:.2f}% (κ={np.mean(kappas_9):.3f})")

        # 各基线方法
        for model_name, (model, scale_info) in models.items():
            if model is None:
                continue
            accs_recon, kappas_recon, accs_aug, kappas_aug = [], [], [], []
            for fold, (train_idx, test_idx) in enumerate(skf.split(all_22ch, all_labels)):
                train_9ch = all_9ch[train_idx]
                train_22ch = all_22ch[train_idx]
                train_labels = all_labels[train_idx]
                test_22ch = all_22ch[test_idx]
                test_labels = all_labels[test_idx]

                # 生成重建
                gen_22ch = generate_reconstruction(model, model_name, train_9ch, train_labels,
                                                   device, scale_info=scale_info)

                # 仅重建分类：训练用重建数据，测试用原始数据
                clf = train_classifier(gen_22ch, train_labels, device, 22)
                m = evaluate_classifier(clf, test_22ch, test_labels, device)
                accs_recon.append(m['accuracy'])
                kappas_recon.append(m['kappa'])
                del clf

                # 增强分类：训练用原始+重建，测试用原始数据
                aug_data = np.concatenate([train_22ch, gen_22ch], axis=0)
                aug_labels = np.concatenate([train_labels, train_labels], axis=0)
                clf = train_classifier(aug_data, aug_labels, device, 22)
                m = evaluate_classifier(clf, test_22ch, test_labels, device)
                accs_aug.append(m['accuracy'])
                kappas_aug.append(m['kappa'])
                del clf

            subj_result[f'{model_name}_recon'] = {
                'accuracy': float(np.mean(accs_recon)), 'std': float(np.std(accs_recon)),
                'kappa': float(np.mean(kappas_recon)), 'kappa_std': float(np.std(kappas_recon)),
            }
            subj_result[f'{model_name}_aug'] = {
                'accuracy': float(np.mean(accs_aug)), 'std': float(np.std(accs_aug)),
                'kappa': float(np.mean(kappas_aug)), 'kappa_std': float(np.std(kappas_aug)),
            }
            print(f"  {model_name}: recon={np.mean(accs_recon)*100:.2f}% (κ={np.mean(kappas_recon):.3f}), aug={np.mean(accs_aug)*100:.2f}% (κ={np.mean(kappas_aug):.3f})")
            torch.cuda.empty_cache()

        # 传统插值方法
        for trad_name in ['Spline', 'Kriging']:
            accs_aug, kappas_aug = [], []
            for fold, (train_idx, test_idx) in enumerate(skf.split(all_22ch, all_labels)):
                train_9ch = all_9ch[train_idx]
                train_22ch = all_22ch[train_idx]
                train_labels = all_labels[train_idx]
                test_22ch = all_22ch[test_idx]
                test_labels = all_labels[test_idx]

                if trad_name == 'Spline':
                    interp = SplineInterpolation(input_channels=9, output_channels=22)
                else:
                    interp = KrigingInterpolation(input_channels=9, output_channels=22)
                gen_22ch = interp.reconstruct(train_9ch)
                if torch.is_tensor(gen_22ch):
                    gen_22ch = gen_22ch.cpu().numpy()
                gen_22ch = gen_22ch.astype(np.float32)

                aug_data = np.concatenate([train_22ch, gen_22ch], axis=0)
                aug_labels = np.concatenate([train_labels, train_labels], axis=0)
                clf = train_classifier(aug_data, aug_labels, device, 22)
                m = evaluate_classifier(clf, test_22ch, test_labels, device)
                accs_aug.append(m['accuracy'])
                kappas_aug.append(m['kappa'])
                del clf

            subj_result[f'{trad_name}_aug'] = {
                'accuracy': float(np.mean(accs_aug)), 'std': float(np.std(accs_aug)),
                'kappa': float(np.mean(kappas_aug)), 'kappa_std': float(np.std(kappas_aug)),
            }
            print(f"  {trad_name}: aug={np.mean(accs_aug)*100:.2f}% (κ={np.mean(kappas_aug):.3f})")

        results[f'S{sid}'] = subj_result

    # 计算平均
    avg_result = {}
    for key in results['S1'].keys():
        vals = [r[key]['accuracy'] for r in results.values() if key in r]
        kappas = [r[key].get('kappa', 0) for r in results.values() if key in r]
        avg_result[key] = {
            'accuracy': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'kappa': float(np.mean(kappas)) if kappas else 0,
            'kappa_std': float(np.std(kappas)) if kappas else 0,
        }
    results['average'] = avg_result
    print(f"\n平均结果:")
    for key, val in avg_result.items():
        print(f"  {key}: {val['accuracy']*100:.2f}±{val['std']*100:.2f}% (κ={val['kappa']:.3f})")

    return results


# ==================== 跨session ====================

def cross_session(data_path, models, device, subject_ids=range(1, 10)):
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
        print(f"  22ch: {m22['accuracy']*100:.2f}% (κ={m22['kappa']:.3f}), 9ch: {m9['accuracy']*100:.2f}% (κ={m9['kappa']:.3f})")

        for model_name, (model, scale_info) in models.items():
            if model is None:
                continue
            gen_22ch = generate_reconstruction(model, model_name, train_9ch, train_labels,
                                               device, scale_info=scale_info)

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
            print(f"  {model_name}: recon={m_recon['accuracy']*100:.2f}% (κ={m_recon['kappa']:.3f}), aug={m_aug['accuracy']*100:.2f}% (κ={m_aug['kappa']:.3f})")
            torch.cuda.empty_cache()

        # 传统插值方法
        for trad_name in ['Spline', 'Kriging']:
            if trad_name == 'Spline':
                interp = SplineInterpolation(input_channels=9, output_channels=22)
            else:
                interp = KrigingInterpolation(input_channels=9, output_channels=22)
            gen_22ch = interp.reconstruct(train_9ch)
            if torch.is_tensor(gen_22ch):
                gen_22ch = gen_22ch.cpu().numpy()
            gen_22ch = gen_22ch.astype(np.float32)

            aug_data = np.concatenate([train_22ch, gen_22ch], axis=0)
            aug_labels = np.concatenate([train_labels, train_labels], axis=0)
            clf = train_classifier(aug_data, aug_labels, device, 22)
            m_aug = evaluate_classifier(clf, test_22ch, test_labels, device)
            del clf

            subj_result[f'{trad_name}_aug'] = m_aug
            print(f"  {trad_name}: aug={m_aug['accuracy']*100:.2f}% (κ={m_aug['kappa']:.3f})")

        results[f'S{sid}'] = subj_result

    # 平均
    avg_result = {}
    for key in results['S1'].keys():
        vals = [r[key]['accuracy'] for r in results.values() if key in r]
        kappas = [r[key].get('kappa', 0) for r in results.values() if key in r]
        avg_result[key] = {
            'accuracy': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'kappa': float(np.mean(kappas)) if kappas else 0,
            'kappa_std': float(np.std(kappas)) if kappas else 0,
        }
    results['average'] = avg_result
    print(f"\n平均结果:")
    for key, val in avg_result.items():
        print(f"  {key}: {val['accuracy']*100:.2f}±{val['std']*100:.2f}% (κ={val['kappa']:.3f})")

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
        print(f"  22ch: {m22['accuracy']*100:.2f}% (κ={m22['kappa']:.3f}), 9ch: {m9['accuracy']*100:.2f}% (κ={m9['kappa']:.3f})")

        for model_name, (model, scale_info) in models.items():
            if model is None:
                continue
            gen_22ch = generate_reconstruction(model, model_name, train_9ch, train_labels,
                                               device, scale_info=scale_info)

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
            print(f"  {model_name}: recon={m_recon['accuracy']*100:.2f}% (κ={m_recon['kappa']:.3f}), aug={m_aug['accuracy']*100:.2f}% (κ={m_aug['kappa']:.3f})")
            torch.cuda.empty_cache()

        # 传统插值方法
        for trad_name in ['Spline', 'Kriging']:
            if trad_name == 'Spline':
                interp = SplineInterpolation(input_channels=9, output_channels=22)
            else:
                interp = KrigingInterpolation(input_channels=9, output_channels=22)
            gen_22ch = interp.reconstruct(train_9ch)
            if torch.is_tensor(gen_22ch):
                gen_22ch = gen_22ch.cpu().numpy()
            gen_22ch = gen_22ch.astype(np.float32)

            aug_data = np.concatenate([train_22ch, gen_22ch], axis=0)
            aug_labels = np.concatenate([train_labels, train_labels], axis=0)
            clf = train_classifier(aug_data, aug_labels, device, 22)
            m_aug = evaluate_classifier(clf, test_22ch, test_labels, device)
            del clf

            subj_result[f'{trad_name}_aug'] = m_aug
            print(f"  {trad_name}: aug={m_aug['accuracy']*100:.2f}% (κ={m_aug['kappa']:.3f})")

        results[f'S{test_sid}'] = subj_result

    # 平均
    avg_result = {}
    for key in results['S1'].keys():
        vals = [r[key]['accuracy'] for r in results.values() if key in r]
        kappas = [r[key].get('kappa', 0) for r in results.values() if key in r]
        avg_result[key] = {
            'accuracy': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'kappa': float(np.mean(kappas)) if kappas else 0,
            'kappa_std': float(np.std(kappas)) if kappas else 0,
        }
    results['average'] = avg_result
    print(f"\n平均结果:")
    for key, val in avg_result.items():
        print(f"  {key}: {val['accuracy']*100:.2f}±{val['std']*100:.2f}% (κ={val['kappa']:.3f})")

    return results


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='BCI2a基线分类评估')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['within_subject', 'cross_session', 'cross_subject', 'all'])
    parser.add_argument('--data-path', default='E:/data/BCI2a')
    parser.add_argument('--output-dir', default='paper_results/baseline_classification')
    parser.add_argument('--methods', type=str, default='all',
                        help='Comma-separated: CVAE,cGAN,Simple-DDPM,EEGDiff,BrainDiff')
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

    # 加载模型
    print("加载模型...")
    models = {}
    for name in method_names:
        model, scale_info = load_baseline_model(name, device)
        if model is not None:
            models[name] = (model, scale_info)

    if not models and not any(m in method_names for m in TRADITIONAL_METHODS):
        print("[ERROR] 没有可用的模型，退出")
        return

    all_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'methods': list(models.keys()),
    }

    if args.mode in ('within_subject', 'all'):
        all_results['within_subject'] = within_subject(args.data_path, models, device)
    if args.mode in ('cross_session', 'all'):
        all_results['cross_session'] = cross_session(args.data_path, models, device)
    if args.mode in ('cross_subject', 'all'):
        all_results['cross_subject'] = bci2a_cross_subject(args.data_path, models, device)

    # 保存结果
    out_path = os.path.join(args.output_dir, 'bci2a_baseline_classification.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存到 {out_path}")


if __name__ == '__main__':
    main()
