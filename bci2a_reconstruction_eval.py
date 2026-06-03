"""
BCI Competition IV 2a DTTD重建与增强分类实验

与PhysioNet实验对齐的完整评估：
1. 9ch基线
2. 22ch基线
3. DTTD重建（9ch→22ch，t=0确定性）
4. DTTD增强-单步（随机t+噪声）
5. DTTD增强-DDIM（多步去噪+CFG）
6. 传统加噪增强

评估方式：
- 跨会话：Session T训练 → Session E测试
- 跨被试5折：所有被试数据混合，5折交叉验证
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.dttd_enhanced_v2 import DTTDEnhanced
from utils import load_config, get_device

CH_IDX_9 = [7, 9, 11, 1, 3, 5, 13, 15, 17]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_raw_bci2a(data_path, subject_id, session='T'):
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


def train_classifier(train_data, train_labels, num_channels, device, epochs=100):
    time_steps = train_data.shape[-1]
    channel_mean = train_data.mean(axis=(0, 2), keepdims=True)
    channel_std = train_data.std(axis=(0, 2), keepdims=True) + 1e-8
    normed_data = ((train_data - channel_mean) / channel_std).astype(np.float32)

    clf = EEGNetClassifier(num_channels, 4, time_steps).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(normed_data), torch.LongTensor(train_labels)),
        batch_size=32, shuffle=True, drop_last=True
    )

    clf.train()
    for _ in range(epochs):
        for data, labels in loader:
            opt.zero_grad()
            loss = criterion(clf(data.to(device)), labels.to(device))
            loss.backward()
            opt.step()

    return clf, channel_mean, channel_std


def evaluate_classifier(clf, test_data, test_labels, device, channel_mean=None, channel_std=None):
    clf.eval()
    if channel_mean is not None and channel_std is not None:
        test_data = ((test_data - channel_mean) / channel_std).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.FloatTensor(test_data)), batch_size=32, shuffle=False)

    preds = []
    with torch.no_grad():
        for batch in loader:
            output = clf(batch[0].to(device))
            preds.extend(torch.argmax(output, dim=1).cpu().numpy())

    return accuracy_score(test_labels, preds)


def compute_reconstruction_quality(recon_data, target_data, labels=None, channel_names=None):
    """
    计算全面的重建质量指标
    
    Args:
        recon_data: 重建数据 [N, C, T]
        target_data: 目标数据 [N, C, T]
        labels: 标签 [N]（可选，用于逐类别分析）
        channel_names: 通道名称列表（可选）
    
    Returns:
        dict: 包含各项重建质量指标
    """
    metrics = {}
    
    diff = recon_data - target_data
    
    mse = np.mean(diff ** 2)
    metrics['mse'] = float(mse)
    
    target_var = np.var(target_data)
    metrics['nmse'] = float(mse / (target_var + 1e-12))
    
    metrics['snr_db'] = float(10 * np.log10(target_var / (mse + 1e-12)))
    
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((target_data - np.mean(target_data)) ** 2)
    metrics['r_squared'] = float(1 - ss_res / (ss_tot + 1e-12))
    
    n_samples = min(100, len(recon_data))
    corrs = []
    for i in range(n_samples):
        r = recon_data[i].flatten()
        t = target_data[i].flatten()
        if np.std(r) > 1e-12 and np.std(t) > 1e-12:
            corrs.append(np.corrcoef(r, t)[0, 1])
    metrics['corr'] = float(np.mean(corrs)) if corrs else 0.0
    
    mae = np.mean(np.abs(diff))
    metrics['mae'] = float(mae)
    metrics['nmae'] = float(mae / (np.mean(np.abs(target_data)) + 1e-12))
    
    per_ch_mse = np.mean(diff ** 2, axis=(0, 2))
    per_ch_var = np.var(target_data, axis=(0, 2))
    per_ch_nmse = per_ch_mse / (per_ch_var + 1e-12)
    per_ch_snr = 10 * np.log10(per_ch_var / (per_ch_mse + 1e-12))
    
    n_ch = recon_data.shape[1]
    if channel_names is None:
        channel_names = [f'ch{i}' for i in range(n_ch)]
    
    metrics['per_channel'] = {}
    for i in range(n_ch):
        ch_name = channel_names[i] if i < len(channel_names) else f'ch{i}'
        metrics['per_channel'][ch_name] = {
            'mse': float(per_ch_mse[i]),
            'nmse': float(per_ch_nmse[i]),
            'snr_db': float(per_ch_snr[i])
        }
    
    input_ch_snr = np.mean(per_ch_snr[CH_IDX_9]) if all(i < n_ch for i in CH_IDX_9) else 0
    missing_mask = np.ones(n_ch, dtype=bool)
    missing_mask[CH_IDX_9] = False
    missing_ch_snr = np.mean(per_ch_snr[missing_mask]) if missing_mask.any() else 0
    metrics['input_channels_avg_snr'] = float(input_ch_snr)
    metrics['missing_channels_avg_snr'] = float(missing_ch_snr)
    
    if labels is not None:
        unique_labels = np.unique(labels)
        metrics['per_class'] = {}
        for lbl in unique_labels:
            mask = labels == lbl
            ch_mse = np.mean(diff[mask] ** 2)
            ch_var = np.var(target_data[mask])
            ch_nmse = ch_mse / (ch_var + 1e-12)
            ch_snr = 10 * np.log10(ch_var / (ch_mse + 1e-12))
            metrics['per_class'][f'class_{lbl}'] = {
                'nmse': float(ch_nmse),
                'snr_db': float(ch_snr),
                'count': int(mask.sum())
            }
    
    return metrics


def generate_data(model, input_data, labels, device, data_scale_factor=1e5,
                  mode='reconstruct', timestep_range=(50, 300),
                  train_noise_scale=0.2, use_ddim=False,
                  num_inference_steps=50, guidance_scale=3.0, eta=0.0,
                  batch_size=32):
    model.eval()
    generated_list = []

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(input_data * data_scale_factor),
                      torch.LongTensor(labels)),
        batch_size=batch_size, shuffle=False
    )

    with torch.no_grad():
        for batch_data, batch_labels in loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            if mode == 'augment' and use_ddim:
                gen = model.sample_ddim(
                    batch_data,
                    task_label=batch_labels,
                    num_inference_steps=num_inference_steps,
                    eta=eta,
                    guidance_scale=guidance_scale
                )
            elif mode == 'augment':
                t = torch.randint(timestep_range[0], timestep_range[1] + 1,
                                  (batch_data.size(0),), device=device)
                noise_level = train_noise_scale * model.scheduler.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
                noisy_input = batch_data + torch.randn_like(batch_data) * noise_level
                gen = model.forward(noisy_input, t, batch_labels)
            else:
                t = torch.zeros(batch_data.size(0), dtype=torch.long, device=device)
                gen = model.forward(batch_data, t, batch_labels)

            generated_list.append(gen.cpu().numpy() / data_scale_factor)
            del batch_data, batch_labels, gen
            torch.cuda.empty_cache()

    return np.concatenate(generated_list, axis=0)


def cross_session_eval(data_path, model, device, subject_ids=range(1, 10)):
    print("\n" + "=" * 60)
    print("BCI 2a 跨会话评估 (Session T → E)")
    print("=" * 60)

    ALL_CHANNELS = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
        'P1', 'Pz', 'P2', 'POz'
    ]

    results = {
        'baseline_9ch': [],
        'baseline_22ch': [],
        'dttd_reconstructed': [],
        'dttd_augmented': [],
        'dttd_ddim_augmented': [],
        'noise_augmented': [],
        'reconstruction_quality': []
    }

    for sid in subject_ids:
        print(f"\n--- 被试 S{sid} ---")

        train_22ch, train_labels = load_raw_bci2a(data_path, sid, 'T')
        test_22ch, test_labels = load_raw_bci2a(data_path, sid, 'E')
        train_9ch = train_22ch[:, CH_IDX_9, :]
        test_9ch = test_22ch[:, CH_IDX_9, :]

        # 1. 9ch基线
        clf_9, m9, s9 = train_classifier(train_9ch, train_labels, 9, device)
        acc_9 = evaluate_classifier(clf_9, test_9ch, test_labels, device, m9, s9)
        results['baseline_9ch'].append(acc_9)
        print(f"9ch基线: {acc_9:.4f}")
        del clf_9
        torch.cuda.empty_cache()

        # 2. 22ch基线
        clf_22, m22, s22 = train_classifier(train_22ch, train_labels, 22, device)
        acc_22 = evaluate_classifier(clf_22, test_22ch, test_labels, device, m22, s22)
        results['baseline_22ch'].append(acc_22)
        print(f"22ch基线: {acc_22:.4f}")
        del clf_22
        torch.cuda.empty_cache()

        # 3. DTTD重建: 在原始22ch上训练，在重建的测试数据上测试
        print("DTTD重建...")
        recon_test = generate_data(model, test_9ch, test_labels, device, mode='reconstruct')
        rq = compute_reconstruction_quality(recon_test, test_22ch, test_labels, ALL_CHANNELS)
        results['reconstruction_quality'].append(rq)
        print(f"  重建质量: NMSE={rq['nmse']:.4f}, SNR={rq['snr_db']:.1f}dB, R2={rq['r_squared']:.4f}, Corr={rq['corr']:.4f}")

        recon_test[:, CH_IDX_9, :] = test_9ch

        clf_22_for_recon, m22r, s22r = train_classifier(train_22ch, train_labels, 22, device)
        acc_recon = evaluate_classifier(clf_22_for_recon, recon_test, test_labels, device, m22r, s22r)
        results['dttd_reconstructed'].append(acc_recon)
        print(f"DTTD重建准确率: {acc_recon:.4f}")
        del clf_22_for_recon
        torch.cuda.empty_cache()

        # 4. DTTD增强(单步)
        print("DTTD增强(单步)...")
        aug_data = generate_data(model, train_9ch, train_labels, device,
                                 mode='augment', timestep_range=(50, 300),
                                 train_noise_scale=0.2)
        aug_count = len(train_22ch) // 2
        aug_indices = np.random.choice(len(aug_data), aug_count, replace=False)
        aug_subset = aug_data[aug_indices]
        aug_labels_subset = train_labels[aug_indices]

        combined = np.concatenate([train_22ch, aug_subset], axis=0).astype(np.float32)
        combined_labels = np.concatenate([train_labels, aug_labels_subset], axis=0)

        clf_aug, ma, sa = train_classifier(combined, combined_labels, 22, device)
        acc_aug = evaluate_classifier(clf_aug, test_22ch, test_labels, device, ma, sa)
        results['dttd_augmented'].append(acc_aug)
        print(f"DTTD增强(单步)准确率: {acc_aug:.4f}")
        del clf_aug
        torch.cuda.empty_cache()

        # 5. DTTD增强(DDIM)
        print("DTTD增强(DDIM)...")
        ddim_data = generate_data(model, train_9ch, train_labels, device,
                                  mode='augment', use_ddim=True,
                                  num_inference_steps=50,
                                  guidance_scale=3.0, eta=0.0,
                                  batch_size=8)
        ddim_count = len(train_22ch) // 2
        ddim_indices = np.random.choice(len(ddim_data), ddim_count, replace=False)
        ddim_subset = ddim_data[ddim_indices]
        ddim_labels_subset = train_labels[ddim_indices]

        ddim_combined = np.concatenate([train_22ch, ddim_subset], axis=0).astype(np.float32)
        ddim_combined_labels = np.concatenate([train_labels, ddim_labels_subset], axis=0)

        clf_ddim, md, sd = train_classifier(ddim_combined, ddim_combined_labels, 22, device)
        acc_ddim = evaluate_classifier(clf_ddim, test_22ch, test_labels, device, md, sd)
        results['dttd_ddim_augmented'].append(acc_ddim)
        print(f"DTTD增强(DDIM)准确率: {acc_ddim:.4f}")
        del clf_ddim
        torch.cuda.empty_cache()

        # 6. 传统加噪增强
        print("传统加噪增强...")
        noise_std = np.std(train_22ch, axis=(0, 2), keepdims=True).astype(np.float32)
        noised_data = (train_22ch + np.random.randn(*train_22ch.shape).astype(np.float32) * noise_std * 0.2).astype(np.float32)
        noise_combined = np.concatenate([train_22ch, noised_data], axis=0)
        noise_labels = np.concatenate([train_labels, train_labels], axis=0)

        clf_noise, mn, sn = train_classifier(noise_combined, noise_labels, 22, device)
        acc_noise = evaluate_classifier(clf_noise, test_22ch, test_labels, device, mn, sn)
        results['noise_augmented'].append(acc_noise)
        print(f"传统加噪增强准确率: {acc_noise:.4f}")
        del clf_noise
        torch.cuda.empty_cache()

    return results


def cross_subject_5fold_eval(data_path, model, device, subject_ids=range(1, 10)):
    print("\n" + "=" * 60)
    print("BCI 2a 跨被试5折交叉验证（按被试分组）")
    print("=" * 60)

    ALL_CHANNELS = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
        'P1', 'Pz', 'P2', 'POz'
    ]

    subject_ids = list(subject_ids)
    n_subjects = len(subject_ids)
    np.random.seed(42)
    shuffled_ids = np.random.permutation(subject_ids).tolist()

    fold_size = n_subjects // 5
    remainder = n_subjects % 5
    folds = []
    start = 0
    for i in range(5):
        size = fold_size + (1 if i < remainder else 0)
        folds.append(shuffled_ids[start:start + size])
        start += size

    for i, fold_subjects in enumerate(folds):
        print(f"  Fold {i+1} 测试被试: {fold_subjects}")

    results = {
        'baseline_9ch': [],
        'baseline_22ch': [],
        'dttd_reconstructed': [],
        'dttd_augmented': [],
        'dttd_ddim_augmented': [],
        'noise_augmented': [],
        'fold_test_subjects': [fold for fold in folds],
        'reconstruction_quality': []
    }

    for fold_idx, test_subjects in enumerate(folds):
        train_subjects = [s for s in subject_ids if s not in test_subjects]
        print(f"\n--- Fold {fold_idx+1}/5 (测试被试: {test_subjects}) ---")

        train_22ch_list, train_labels_list = [], []
        for sid in train_subjects:
            data_t, labels_t = load_raw_bci2a(data_path, sid, 'T')
            data_e, labels_e = load_raw_bci2a(data_path, sid, 'E')
            train_22ch_list.append(np.concatenate([data_t, data_e], axis=0))
            train_labels_list.append(np.concatenate([labels_t, labels_e], axis=0))
        train_22ch = np.concatenate(train_22ch_list, axis=0)
        train_labels = np.concatenate(train_labels_list, axis=0)

        test_22ch_list, test_labels_list = [], []
        for sid in test_subjects:
            data_t, labels_t = load_raw_bci2a(data_path, sid, 'T')
            data_e, labels_e = load_raw_bci2a(data_path, sid, 'E')
            test_22ch_list.append(np.concatenate([data_t, data_e], axis=0))
            test_labels_list.append(np.concatenate([labels_t, labels_e], axis=0))
        test_22ch = np.concatenate(test_22ch_list, axis=0)
        test_labels = np.concatenate(test_labels_list, axis=0)

        train_9ch = train_22ch[:, CH_IDX_9, :]
        test_9ch = test_22ch[:, CH_IDX_9, :]

        print(f"训练集: {len(train_22ch)} trials (被试{train_subjects}), 测试集: {len(test_22ch)} trials (被试{test_subjects})")

        # 1. 9ch基线
        clf_9, m9, s9 = train_classifier(train_9ch, train_labels, 9, device)
        acc_9 = evaluate_classifier(clf_9, test_9ch, test_labels, device, m9, s9)
        results['baseline_9ch'].append(acc_9)
        print(f"9ch基线: {acc_9:.4f}")
        del clf_9
        torch.cuda.empty_cache()

        # 2. 22ch基线
        clf_22, m22, s22 = train_classifier(train_22ch, train_labels, 22, device)
        acc_22 = evaluate_classifier(clf_22, test_22ch, test_labels, device, m22, s22)
        results['baseline_22ch'].append(acc_22)
        print(f"22ch基线: {acc_22:.4f}")
        del clf_22
        torch.cuda.empty_cache()

        # 3. DTTD重建: 在原始22ch上训练，在重建的测试数据上测试
        print("DTTD重建...")
        recon_test = generate_data(model, test_9ch, test_labels, device, mode='reconstruct')
        rq = compute_reconstruction_quality(recon_test, test_22ch, test_labels, ALL_CHANNELS)
        results['reconstruction_quality'].append(rq)
        print(f"  重建质量: NMSE={rq['nmse']:.4f}, SNR={rq['snr_db']:.1f}dB, R2={rq['r_squared']:.4f}, Corr={rq['corr']:.4f}")

        recon_test[:, CH_IDX_9, :] = test_9ch

        clf_22_for_recon, m22r, s22r = train_classifier(train_22ch, train_labels, 22, device)
        acc_recon = evaluate_classifier(clf_22_for_recon, recon_test, test_labels, device, m22r, s22r)
        results['dttd_reconstructed'].append(acc_recon)
        print(f"DTTD重建准确率: {acc_recon:.4f}")
        del clf_22_for_recon
        torch.cuda.empty_cache()

        # 4. DTTD增强(单步)
        print("DTTD增强(单步)...")
        aug_data = generate_data(model, train_9ch, train_labels, device,
                                 mode='augment', timestep_range=(50, 300),
                                 train_noise_scale=0.2)
        aug_count = len(train_22ch) // 2
        aug_indices = np.random.choice(len(aug_data), aug_count, replace=False)
        aug_subset = aug_data[aug_indices]
        aug_labels_subset = train_labels[aug_indices]

        combined = np.concatenate([train_22ch, aug_subset], axis=0).astype(np.float32)
        combined_labels = np.concatenate([train_labels, aug_labels_subset], axis=0)

        clf_aug, ma, sa = train_classifier(combined, combined_labels, 22, device)
        acc_aug = evaluate_classifier(clf_aug, test_22ch, test_labels, device, ma, sa)
        results['dttd_augmented'].append(acc_aug)
        print(f"DTTD增强(单步)准确率: {acc_aug:.4f}")
        del clf_aug
        torch.cuda.empty_cache()

        # 5. DTTD增强(DDIM)
        print("DTTD增强(DDIM)...")
        ddim_data = generate_data(model, train_9ch, train_labels, device,
                                  mode='augment', use_ddim=True,
                                  num_inference_steps=50,
                                  guidance_scale=3.0, eta=0.0,
                                  batch_size=8)
        ddim_count = len(train_22ch) // 2
        ddim_indices = np.random.choice(len(ddim_data), ddim_count, replace=False)
        ddim_subset = ddim_data[ddim_indices]
        ddim_labels_subset = train_labels[ddim_indices]

        ddim_combined = np.concatenate([train_22ch, ddim_subset], axis=0).astype(np.float32)
        ddim_combined_labels = np.concatenate([train_labels, ddim_labels_subset], axis=0)

        clf_ddim, md, sd = train_classifier(ddim_combined, ddim_combined_labels, 22, device)
        acc_ddim = evaluate_classifier(clf_ddim, test_22ch, test_labels, device, md, sd)
        results['dttd_ddim_augmented'].append(acc_ddim)
        print(f"DTTD增强(DDIM)准确率: {acc_ddim:.4f}")
        del clf_ddim
        torch.cuda.empty_cache()

        # 6. 传统加噪增强
        print("传统加噪增强...")
        noise_std = np.std(train_22ch, axis=(0, 2), keepdims=True).astype(np.float32)
        noised_data = (train_22ch + np.random.randn(*train_22ch.shape).astype(np.float32) * noise_std * 0.2).astype(np.float32)
        noise_combined = np.concatenate([train_22ch, noised_data], axis=0)
        noise_labels = np.concatenate([train_labels, train_labels], axis=0)

        clf_noise, mn, sn = train_classifier(noise_combined, noise_labels, 22, device)
        acc_noise = evaluate_classifier(clf_noise, test_22ch, test_labels, device, mn, sn)
        results['noise_augmented'].append(acc_noise)
        print(f"传统加噪增强准确率: {acc_noise:.4f}")
        del clf_noise
        torch.cuda.empty_cache()

    return results


def save_results(results, output_dir, filename, title):
    os.makedirs(output_dir, exist_ok=True)
    save_data = {}
    for k, v in results.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
            valid = [x for x in v if x is not None]
            save_data[k] = {
                'values': [float(x) for x in v],
                'mean': float(np.mean(valid)) if valid else None,
                'std': float(np.std(valid)) if valid else None
            }
        else:
            save_data[k] = v

    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"[OK] {title}结果已保存至: {filepath}")


def print_summary(results, title):
    print(f"\n{'=' * 60}")
    print(f"{title} 结果汇总")
    print(f"{'=' * 60}")
    for k, v in results.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
            valid = [x for x in v if x is not None]
            if valid:
                print(f"{k}: {np.mean(valid)*100:.2f}% ± {np.std(valid)*100:.2f}%")
    if 'reconstruction_quality' in results and results['reconstruction_quality']:
        rq_list = results['reconstruction_quality']
        avg_nmse = np.mean([r.get('nmse', 0) for r in rq_list])
        avg_snr = np.mean([r.get('snr_db', 0) for r in rq_list])
        avg_r2 = np.mean([r.get('r_squared', 0) for r in rq_list])
        avg_corr = np.mean([r.get('corr', 0) for r in rq_list])
        print(f"重建质量: NMSE={avg_nmse:.4f}, SNR={avg_snr:.1f}dB, R2={avg_r2:.4f}, Corr={avg_corr:.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='BCI 2a DTTD重建与增强分类实验')
    parser.add_argument('--data-path', default='E:/data/BCI2a')
    parser.add_argument('--config', default='configs/bci2a_enhanced_config.yaml')
    parser.add_argument('--checkpoint', default='checkpoints/bci2a_enhanced_v2/best_model.pth')
    parser.add_argument('--output-dir', default='paper_results/bci2a_recon_eval')
    parser.add_argument('--mode', default='all', choices=['cross_session', 'cross_subject', 'all'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"使用设备: {device}")

    config = load_config(args.config)
    model = DTTDEnhanced(config['model']).to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
        print(f"[OK] 加载模型: {args.checkpoint}")
    else:
        print(f"[ERROR] 模型文件不存在: {args.checkpoint}")
        return

    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode in ['cross_session', 'all']:
        results_cs = cross_session_eval(args.data_path, model, device)
        print_summary(results_cs, "跨会话")
        save_results(results_cs, args.output_dir, 'cross_session_results.json', '跨会话')

    if args.mode in ['cross_subject', 'all']:
        results_cv = cross_subject_5fold_eval(args.data_path, model, device)
        print_summary(results_cv, "跨被试5折")
        save_results(results_cv, args.output_dir, 'cross_subject_5fold_results.json', '跨被试5折')


if __name__ == '__main__':
    main()
