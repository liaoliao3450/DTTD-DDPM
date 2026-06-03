"""
PhysioNet MI 数据集分类实验（独立版本）

使用专用的 DTTDPhysioNet 模型，不修改 BCI2a 的任何代码。

论文中提到的实验：
- 使用16通道作为输入，重建到64通道
- 评估跨被试分类性能
- 对比16通道基线、64通道基线和DTTD增强结果
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, cohen_kappa_score

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.physionet_mi import PhysioNetMIDataset, INPUT_CHANNEL_INDICES_16
from models.dttd_physionet import DTTDPhysioNet
from utils import get_device, set_seed


class EEGNetClassifier(nn.Module):
    def __init__(self, num_channels=64, num_classes=4, time_steps=640):
        super().__init__()
        F1, D, F2 = 8, 2, 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (num_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, time_steps)
            dummy = self.conv1(dummy)
            dummy = self.conv2(dummy)
            dummy = self.conv3(dummy)
            flat_size = dummy.flatten(1).shape[1]
        self.classifier = nn.Linear(flat_size, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.classifier(x.flatten(1))


def train_classifier(train_data, train_labels, num_channels, device, epochs=300,
                     extra_data=None, extra_labels=None, extra_epochs=100):
    time_steps = train_data.shape[-1]

    channel_mean = train_data.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    channel_std = train_data.std(axis=(0, 2), keepdims=True).astype(np.float32) + 1e-8

    clf = EEGNetClassifier(num_channels=num_channels, num_classes=4, time_steps=time_steps).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    # 分批标准化+训练
    n_samples = len(train_data)
    batch_sz = 32
    clf.train()
    for ep in range(epochs):
        indices = np.random.permutation(n_samples)
        for start in range(0, n_samples, batch_sz):
            idx = indices[start:start+batch_sz]
            if len(idx) < batch_sz:
                continue
            batch = ((train_data[idx] - channel_mean) / channel_std).astype(np.float32)
            data_t = torch.FloatTensor(batch).to(device)
            labels_t = torch.LongTensor(train_labels[idx]).to(device)
            opt.zero_grad()
            loss = criterion(clf(data_t), labels_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
            opt.step()
        scheduler.step()

    if extra_data is not None and len(extra_data) > 0:
        n_extra = len(extra_data)
        opt2 = torch.optim.Adam(clf.parameters(), lr=5e-4, weight_decay=5e-4)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=extra_epochs, eta_min=1e-5)
        clf.train()
        for _ in range(extra_epochs):
            indices = np.random.permutation(n_extra)
            for start in range(0, n_extra, batch_sz):
                idx = indices[start:start+batch_sz]
                if len(idx) < batch_sz:
                    continue
                batch = ((extra_data[idx] - channel_mean) / channel_std).astype(np.float32)
                data_t = torch.FloatTensor(batch).to(device)
                labels_t = torch.LongTensor(extra_labels[idx]).to(device)
                opt2.zero_grad()
                loss = criterion(clf(data_t), labels_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
                opt2.step()
            scheduler2.step()

    return clf, channel_mean, channel_std


def evaluate_classifier(clf, test_data, test_labels, device, channel_mean=None, channel_std=None):
    clf.eval()
    # 分批标准化+推理，避免内存爆炸
    n_test = len(test_data)
    batch_sz = 32
    preds = []
    with torch.no_grad():
        for start in range(0, n_test, batch_sz):
            batch = test_data[start:start+batch_sz]
            if channel_mean is not None and channel_std is not None:
                batch = ((batch - channel_mean) / channel_std).astype(np.float32)
            output = clf(torch.FloatTensor(batch).to(device))
            preds.extend(torch.argmax(output, dim=1).cpu().numpy())

    return {
        'accuracy': accuracy_score(test_labels, preds),
        'kappa': cohen_kappa_score(test_labels, preds)
    }


def compute_reconstruction_quality(recon_data, target_data, labels=None, channel_names=None):
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
    input_ch_indices = INPUT_CHANNEL_INDICES_16
    input_ch_snr = np.mean(per_ch_snr[input_ch_indices]) if all(i < n_ch for i in input_ch_indices) else 0
    missing_mask = np.ones(n_ch, dtype=bool)
    missing_mask[input_ch_indices] = False
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


def generate_augmented_data(model, input_data, labels, device, data_scale_factor=1e5,
                           mode='reconstruct', timestep_range=(50, 300),
                           train_noise_scale=0.1, use_ddim=False,
                           num_inference_steps=50, guidance_scale=3.0, eta=0.0,
                           batch_size=4):
    """
    使用DTTD模型生成数据

    mode='reconstruct': t=0, 无噪声 (确定性重建)
    mode='augment':
        - use_ddim=False: 随机t∈timestep_range, 按训练匹配噪声 (单步增强)
        - use_ddim=True: DDIM多步去噪 + CFG引导 (高质量增强)
    """
    model.eval()
    generated_list = []

    ch_indices = torch.tensor(INPUT_CHANNEL_INDICES_16, device=device)

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
                    guidance_scale=guidance_scale,
                    channel_indices=ch_indices
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


def train_dttd_model(model, data_path, device, epochs=200, batch_size=32, lr=1e-4,
                     save_path='checkpoints/physionet_mi/best_model.pth'):
    """训练DTTDPhysioNet模型"""
    print("\n" + "=" * 60)
    print("训练 DTTDPhysioNet 模型")
    print("=" * 60)

    # 从npz缓存加载，避免读EDF
    valid_subject_ids, subject_data_map = load_all_subjects_with_cache(data_path)
    data_list, label_list = [], []
    for sid in valid_subject_ids:
        d, l = subject_data_map[sid]
        data_list.append(d)
        label_list.append(l)
    all_data = np.concatenate(data_list, axis=0).astype(np.float32)
    all_labels = np.concatenate(label_list, axis=0)
    num_channels = all_data.shape[1]

    data_scale_factor = 1e5
    print(f"数据缩放因子: {data_scale_factor}")
    print(f"训练样本数: {len(all_data)}, 通道数: {num_channels}")

    scaled_data = all_data * data_scale_factor

    train_tensor = torch.FloatTensor(scaled_data)
    labels_tensor = torch.LongTensor(all_labels)
    ch_idx = torch.tensor(INPUT_CHANNEL_INDICES_16, device=device)

    loader = DataLoader(
        TensorDataset(train_tensor, labels_tensor),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for data_batch, label_batch in loader:
            data_batch = data_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()

            loss = model.compute_loss(
                x_target=data_batch,
                channel_indices=ch_idx,
                task_label=label_batch,
                loss_type='l2',
                current_epoch=epoch,
                noise_scale=0.1
            )

            if isinstance(loss, dict):
                loss_value = loss['total_loss']
            else:
                loss_value = loss

            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss_value.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / num_batches

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}, LR = {scheduler.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
            if (epoch + 1) % 10 == 0:
                print(f"  -> 保存最佳模型 (loss: {best_loss:.6f})")

    print(f"\n训练完成! 最佳loss: {best_loss:.6f}, 模型保存至: {save_path}")

    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def load_all_subjects_with_cache(data_path, cache_path='paper_results/physionet_mi/physionet_mi_all_subjects.npz'):
    """加载所有被试数据，优先从npy预提取文件加载"""
    # 优先从预提取的npy文件加载（1秒）
    # 搜索多个可能的位置
    npy_candidates = [
        os.path.join(data_path, 'physionet_mi_preprocessed.npz'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_cache', 'physionet_mi_preprocessed.npz'),
    ]
    npy_path = None
    for p in npy_candidates:
        if os.path.exists(p):
            npy_path = p
            break
    if npy_path is not None:
        print(f"[预提取] 从npy加载: {npy_path}")
        t0 = __import__('time').time()
        npz = np.load(npy_path, allow_pickle=True)
        data_arr = npz['data']
        labels_arr = npz['labels']
        sid_arr = npz['subject_ids']
        subject_data_map = {}
        valid_subject_ids = []
        for i, sid_str in enumerate(sid_arr):
            sid_int = int(sid_str[1:])  # 'S001' -> 1
            subject_data_map[sid_int] = (data_arr[i], labels_arr[i])
            valid_subject_ids.append(sid_int)
        print(f"[预提取] 加载完成: {len(valid_subject_ids)} 个被试, 耗时 {__import__('time').time()-t0:.1f}s")
        return valid_subject_ids, subject_data_map

    # 其次从缓存加载
    if os.path.exists(cache_path):
        print(f"[缓存] 从缓存加载: {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        subject_data_map = dict(cached['subject_data_map'].item())
        valid_subject_ids = list(subject_data_map.keys())
        print(f"[缓存] 加载完成: {len(valid_subject_ids)} 个被试")
        return valid_subject_ids, subject_data_map

    print("[缓存] 未找到缓存，从EDF文件加载...")
    all_subject_ids = list(range(1, 110))
    valid_subject_ids = []
    subject_data_map = {}

    for sid in all_subject_ids:
        try:
            dataset = PhysioNetMIDataset(
                data_path=data_path,
                subject_ids=[sid],
                runs={4, 6, 8, 10, 12, 14},
                reconstruction_mode=True
            )
            if len(dataset.data) > 0:
                subject_data_map[sid] = (dataset.data, dataset.labels)
                valid_subject_ids.append(sid)
                print(f"  被试{sid}: {len(dataset.data)} trials")
        except Exception as e:
            pass

    # 保存缓存
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, subject_data_map=subject_data_map)
    print(f"[缓存] 已保存至: {cache_path} ({len(valid_subject_ids)} 个被试)")

    return valid_subject_ids, subject_data_map


def run_within_subject_experiment(data_path, model, device, output_dir, model_trained=False):
    print("\n" + "=" * 60)
    print("PhysioNet MI 跨被试10折交叉验证（按被试分组）")
    print("=" * 60)

    print("加载数据集...")
    valid_subject_ids, subject_data_map = load_all_subjects_with_cache(data_path)

    print(f"成功加载 {len(valid_subject_ids)} 个被试")

    np.random.seed(42)
    shuffled_ids = np.random.permutation(valid_subject_ids).tolist()
    n_subjects = len(shuffled_ids)
    n_folds = 10
    fold_size = n_subjects // n_folds
    remainder = n_subjects % n_folds
    folds = []
    start = 0
    for i in range(n_folds):
        size = fold_size + (1 if i < remainder else 0)
        folds.append(shuffled_ids[start:start + size])
        start += size

    for i, fold_subjects in enumerate(folds):
        print(f"  Fold {i+1} 测试被试({len(fold_subjects)}): {fold_subjects[:5]}...")

    num_input_channels = len(INPUT_CHANNEL_INDICES_16)

    results = {
        'baseline_16ch': [],
        'baseline_64ch': [],
        'dttd_reconstructed': [],
        'dttd_augmented': [],
        'dttd_ddim_augmented': [],
        'noise_augmented': [],
        'fold_test_subjects': [fold for fold in folds],
        'reconstruction_quality': []
    }

    for fold_idx, test_subjects in enumerate(folds):
        train_subjects = [s for s in valid_subject_ids if s not in test_subjects]
        print(f"\n--- Fold {fold_idx+1}/{n_folds} (测试被试: {len(test_subjects)}, 训练被试: {len(train_subjects)}) ---")

        train_data_list, train_labels_list = [], []
        for sid in train_subjects:
            d, l = subject_data_map[sid]
            train_data_list.append(d)
            train_labels_list.append(l)
        train_data_64 = np.concatenate(train_data_list, axis=0).astype(np.float32)
        train_labels = np.concatenate(train_labels_list, axis=0)

        test_data_list, test_labels_list = [], []
        for sid in test_subjects:
            d, l = subject_data_map[sid]
            test_data_list.append(d)
            test_labels_list.append(l)
        test_data_64 = np.concatenate(test_data_list, axis=0).astype(np.float32)
        test_labels = np.concatenate(test_labels_list, axis=0)

        train_data_16 = train_data_64[:, INPUT_CHANNEL_INDICES_16, :]
        test_data_16 = test_data_64[:, INPUT_CHANNEL_INDICES_16, :]

        print(f"训练集: {len(train_data_64)} trials, 测试集: {len(test_data_64)} trials")

        # 1. 16通道基线
        print("训练16通道基线分类器...")
        clf_16, m16, s16 = train_classifier(train_data_16, train_labels, num_input_channels, device)
        acc_16 = evaluate_classifier(clf_16, test_data_16, test_labels, device, m16, s16)
        results['baseline_16ch'].append(acc_16)
        print(f"16通道准确率: {acc_16['accuracy']:.4f}, kappa: {acc_16['kappa']:.4f}")
        del clf_16
        torch.cuda.empty_cache()

        # 2. 64通道基线
        print("训练64通道基线分类器...")
        clf_64, m64, s64 = train_classifier(train_data_64, train_labels, 64, device)
        acc_64 = evaluate_classifier(clf_64, test_data_64, test_labels, device, m64, s64)
        results['baseline_64ch'].append(acc_64)
        print(f"64通道准确率: {acc_64['accuracy']:.4f}, kappa: {acc_64['kappa']:.4f}")
        del clf_64
        torch.cuda.empty_cache()

        # 3. DTTD重建: 在原始64ch上训练，在重建的测试数据上测试
        if model_trained:
            print("DTTD重建...")
            recon_test = generate_augmented_data(model, test_data_16, test_labels, device, mode='reconstruct')

            rq = compute_reconstruction_quality(recon_test, test_data_64, test_labels)
            results['reconstruction_quality'].append(rq)
            print(f"  重建质量: NMSE={rq['nmse']:.4f}, SNR={rq['snr_db']:.1f}dB, R2={rq['r_squared']:.4f}, Corr={rq['corr']:.4f}")

            recon_test[:, INPUT_CHANNEL_INDICES_16, :] = test_data_16

            clf_64_for_recon, m64r, s64r = train_classifier(train_data_64, train_labels, 64, device)
            acc_recon = evaluate_classifier(clf_64_for_recon, recon_test, test_labels, device, m64r, s64r)
            results['dttd_reconstructed'].append(acc_recon)
            print(f"DTTD重建准确率: {acc_recon['accuracy']:.4f}, kappa: {acc_recon['kappa']:.4f}")
            del clf_64_for_recon
        else:
            print("DTTD重建: 跳过（模型未训练）")
            results['dttd_reconstructed'].append(None)

        torch.cuda.empty_cache()

        # 4. DTTD增强（单步）
        if model_trained:
            print("DTTD增强(单步)...")
            aug_data = generate_augmented_data(model, train_data_16, train_labels, device,
                                               mode='augment', timestep_range=(50, 300),
                                               train_noise_scale=0.1)

            aug_count = len(train_data_64) // 2
            aug_indices = np.random.choice(len(aug_data), aug_count, replace=False)
            aug_subset = aug_data[aug_indices]
            aug_labels_subset = train_labels[aug_indices]

            combined_data = np.concatenate([train_data_64, aug_subset], axis=0)
            combined_labels = np.concatenate([train_labels, aug_labels_subset], axis=0)

            clf_aug, ma, sa = train_classifier(combined_data, combined_labels, 64, device, epochs=300)
            acc_aug = evaluate_classifier(clf_aug, test_data_64, test_labels, device, ma, sa)
            results['dttd_augmented'].append(acc_aug)
            print(f"DTTD增强(单步)准确率: {acc_aug['accuracy']:.4f}, kappa: {acc_aug['kappa']:.4f}")
            del clf_aug
        else:
            print("DTTD增强(单步): 跳过（模型未训练）")
            results['dttd_augmented'].append(None)

        torch.cuda.empty_cache()

        # 5. DTTD增强（DDIM多步）
        if model_trained:
            print("DTTD增强(DDIM)...")
            ddim_data = generate_augmented_data(model, train_data_16, train_labels, device,
                                                mode='augment', use_ddim=True,
                                                num_inference_steps=50,
                                                guidance_scale=3.0, eta=0.0,
                                                batch_size=2)

            ddim_count = len(train_data_64) // 2
            ddim_indices = np.random.choice(len(ddim_data), ddim_count, replace=False)
            ddim_subset = ddim_data[ddim_indices]
            ddim_labels_subset = train_labels[ddim_indices]

            ddim_combined = np.concatenate([train_data_64, ddim_subset], axis=0)
            ddim_combined_labels = np.concatenate([train_labels, ddim_labels_subset], axis=0)

            clf_ddim, md, sd = train_classifier(ddim_combined, ddim_combined_labels, 64, device, epochs=300)
            acc_ddim = evaluate_classifier(clf_ddim, test_data_64, test_labels, device, md, sd)
            results['dttd_ddim_augmented'].append(acc_ddim)
            print(f"DTTD增强(DDIM)准确率: {acc_ddim['accuracy']:.4f}, kappa: {acc_ddim['kappa']:.4f}")
            del clf_ddim
        else:
            print("DTTD增强(DDIM): 跳过（模型未训练）")
            results['dttd_ddim_augmented'].append(None)

        torch.cuda.empty_cache()

        # 6. 传统加噪增强（对比基线）
        print("传统加噪增强...")
        noise_std = np.std(train_data_64, axis=(0, 2), keepdims=True).astype(np.float32)
        noised_data = (train_data_64 + np.random.randn(*train_data_64.shape).astype(np.float32) * noise_std * 0.2).astype(np.float32)
        noise_combined = np.concatenate([train_data_64, noised_data], axis=0)
        noise_labels = np.concatenate([train_labels, train_labels], axis=0)

        clf_noise, mn, sn = train_classifier(noise_combined, noise_labels, 64, device, epochs=300)
        acc_noise = evaluate_classifier(clf_noise, test_data_64, test_labels, device, mn, sn)
        results['noise_augmented'].append(acc_noise)
        print(f"传统加噪增强准确率: {acc_noise['accuracy']:.4f}, kappa: {acc_noise['kappa']:.4f}")
        del clf_noise

        torch.cuda.empty_cache()

    save_data = {}
    for k, v in results.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            valid = [x for x in v if x is not None]
            save_data[k] = {
                'values': v,
                'mean_accuracy': float(np.mean([x['accuracy'] for x in valid])) if valid else None,
                'std_accuracy': float(np.std([x['accuracy'] for x in valid])) if valid else None,
                'mean_kappa': float(np.mean([x['kappa'] for x in valid])) if valid else None,
                'std_kappa': float(np.std([x['kappa'] for x in valid])) if valid else None,
            }
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
            valid = [x for x in v if x is not None]
            save_data[k] = {
                'values': [float(x) for x in v],
                'mean': float(np.mean(valid)) if valid else None,
                'std': float(np.std(valid)) if valid else None
            }
        else:
            save_data[k] = v

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'physionet_mi_cross_subject_10fold.json')
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    for k in ['baseline_16ch', 'baseline_64ch', 'dttd_reconstructed',
              'dttd_augmented', 'dttd_ddim_augmented', 'noise_augmented']:
        sd = save_data[k]
        if sd.get('mean_accuracy') is not None:
            print(f"{k}: acc={sd['mean_accuracy']*100:.2f}% ± {sd['std_accuracy']*100:.2f}%, "
                  f"kappa={sd['mean_kappa']:.4f} ± {sd['std_kappa']:.4f}")
        else:
            print(f"{k}: 未执行")
    if results['reconstruction_quality']:
        rq_list = results['reconstruction_quality']
        avg_nmse = np.mean([r.get('nmse', 0) for r in rq_list])
        avg_snr = np.mean([r.get('snr_db', 0) for r in rq_list])
        avg_r2 = np.mean([r.get('r_squared', 0) for r in rq_list])
        avg_corr = np.mean([r.get('corr', 0) for r in rq_list])
        print(f"重建质量: NMSE={avg_nmse:.4f}, SNR={avg_snr:.1f}dB, R2={avg_r2:.4f}, Corr={avg_corr:.4f}")

    print(f"\n[OK] 结果已保存至: {filepath}")

    return save_data


def main():
    import argparse
    parser = argparse.ArgumentParser(description='PhysioNet MI分类实验（独立版本）')
    parser.add_argument('--data-path', default='E:/data/PhysioNetMI')
    parser.add_argument('--output-dir', default='paper_results/physionet_mi')
    parser.add_argument('--checkpoint', default='paper_results/physionet_mi/dttd_physionet_best.pth', help='DTTDPhysioNet模型checkpoint路径')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='训练批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"使用设备: {device}")

    model_config = {
        'input_channels': 16,
        'output_channels': 64,
        'time_steps': 640,
        'embed_dim': 256,
        'task_dim': 64,
        'num_classes': 4,
        'num_heads': 8,
        'dropout': 0.1,
        'num_timesteps': 1000,
        'beta_start': 1e-4,
        'beta_end': 0.02,
        'schedule_type': 'linear',
        'fs': 160,
        'use_classifier_guidance': False
    }

    model = DTTDPhysioNet(model_config).to(device)
    print(f"[OK] DTTDPhysioNet模型创建成功 (输入:{model_config['input_channels']}ch -> 输出:{model_config['output_channels']}ch)")

    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"[OK] 加载模型: {args.checkpoint}")
        model_trained = True
    else:
        print("[INFO] 未提供checkpoint，开始训练模型...")
        save_path = os.path.join(args.output_dir, 'dttd_physionet_best.pth')
        model = train_dttd_model(
            model=model,
            data_path=args.data_path,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_path=save_path
        )
        model_trained = True

    model.eval()

    run_within_subject_experiment(args.data_path, model, device, args.output_dir, model_trained)


if __name__ == '__main__':
    main()
