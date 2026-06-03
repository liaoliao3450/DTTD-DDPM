"""
High Gamma Dataset DTTD重建与增强分类实验

与PhysioNet实验对齐的完整评估：
1. 16ch基线 (标准10-20运动皮层通道)
2. 128ch基线 (全EEG通道)
3. DTTD重建 (16ch→128ch, t=0确定性)
4. DTTD增强-单步 (随机t+噪声)
5. DTTD增强-DDIM (多步去噪+CFG)
6. 传统加噪增强

评估方式: 跨被试loso证

数据下载:
  从GIN下载: https://web.gin.g-node.org/robintibor/high-gamma-dataset
  目录结构: data_path/train/{1..14}.edf, data_path/test/{1..14}.edf
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

from data.high_gamma_dataset import HighGammaDataset
from models.dttd_physionet import DTTDPhysioNet
from utils import get_device


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EEGNetClassifier(nn.Module):
    def __init__(self, num_channels, num_classes=4, time_steps=1000):
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


def train_classifier(train_data, train_labels, num_channels, device, epochs=100,
                     extra_data=None, extra_labels=None, extra_epochs=30):
    time_steps = train_data.shape[-1]
    channel_mean = train_data.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    channel_std = train_data.std(axis=(0, 2), keepdims=True).astype(np.float32) + 1e-8

    clf = EEGNetClassifier(num_channels, 4, time_steps).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 分批标准化+训练，避免内存爆炸
    n_samples = len(train_data)
    batch_sz = 32
    clf.train()
    for _ in range(epochs):
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
            opt.step()

    if extra_data is not None and len(extra_data) > 0:
        n_extra = len(extra_data)
        opt2 = torch.optim.Adam(clf.parameters(), lr=1e-4, weight_decay=1e-4)
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
                opt2.step()

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


def compute_reconstruction_quality(recon_data, target_data, labels=None, channel_names=None,
                                   input_ch_indices=None):
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
    if input_ch_indices is not None and all(i < n_ch for i in input_ch_indices):
        input_ch_snr = np.mean(per_ch_snr[input_ch_indices])
        missing_mask = np.ones(n_ch, dtype=bool)
        missing_mask[input_ch_indices] = False
        missing_ch_snr = np.mean(per_ch_snr[missing_mask]) if missing_mask.any() else 0
    else:
        input_ch_snr = 0
        missing_ch_snr = 0
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


def generate_augmented_data(model, input_data, labels, device, data_mean=None, data_std=None,
                            mode='reconstruct', timestep_range=(50, 300),
                            train_noise_scale=0.1, use_ddim=False,
                            num_inference_steps=50, guidance_scale=3.0, eta=0.0,
                            channel_indices=None, batch_size=2):
    model.eval()
    generated_list = []

    if channel_indices is None:
        channel_indices = torch.arange(input_data.shape[1], device=device)
    elif not isinstance(channel_indices, torch.Tensor):
        channel_indices = torch.tensor(channel_indices, device=device)

    if data_mean is not None and data_std is not None:
        ch_idx_np = channel_indices.cpu().numpy() if isinstance(channel_indices, torch.Tensor) else np.array(channel_indices)
        input_mean = data_mean[:, ch_idx_np, :]
        input_std = data_std[:, ch_idx_np, :]
        normed_input = ((input_data - input_mean) / input_std).astype(np.float32)
    else:
        normed_input = input_data

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(normed_input),
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
                    channel_indices=channel_indices
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

            gen_np = gen.cpu().numpy()
            if data_mean is not None:
                gen_np = gen_np * data_std + data_mean
            generated_list.append(gen_np)
            del batch_data, batch_labels, gen
            torch.cuda.empty_cache()

    return np.concatenate(generated_list, axis=0)


def train_dttd_model(model, dataset, device, epochs=200, batch_size=16, lr=1e-4,
                     save_path='checkpoints/hgd/best_model.pth'):
    print("\n" + "=" * 60)
    print("训练 DTTD 模型 (High Gamma Dataset)")
    print("=" * 60)

    all_data = dataset.data
    all_labels = dataset.labels
    ch_idx = torch.tensor(dataset.input_ch_indices, device=device)

    data_mean = all_data.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    data_std = all_data.std(axis=(0, 2), keepdims=True).astype(np.float32)
    data_std = np.maximum(data_std, 1e-6)

    print(f"训练样本数: {len(all_data)}, 输入通道: {dataset.num_input_channels}, 输出通道: {dataset.num_output_channels}")
    print(f"数据归一化: mean={data_mean.mean():.2f}, std={data_std.mean():.2f}")

    normed_data = ((all_data - data_mean) / data_std).astype(np.float32)
    train_tensor = torch.FloatTensor(normed_data)
    labels_tensor = torch.LongTensor(all_labels)

    loader = DataLoader(
        TensorDataset(train_tensor, labels_tensor),
        batch_size=batch_size, shuffle=True, drop_last=True
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
                'data_mean': data_mean,
                'data_std': data_std,
            }, save_path)
            if (epoch + 1) % 10 == 0:
                print(f"  -> 保存最佳模型 (loss: {best_loss:.6f})")

    print(f"\n训练完成! 最佳loss: {best_loss:.6f}, 模型保存至: {save_path}")

    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, data_mean, data_std


def run_experiment(data_path, model, device, output_dir, model_trained=False,
                   data_mean=None, data_std=None,
                   input_ch_indices=None, num_input_channels=16, num_output_channels=128):
    print("\n" + "=" * 60)
    print("High Gamma Dataset LOSO交叉验证 (Leave-One-Subject-Out)")
    print("=" * 60)

    print("按被试加载数据集...")
    all_subject_ids = list(range(1, 15))
    valid_subject_ids = []
    subject_data_map = {}

    for sid in all_subject_ids:
        try:
            dataset = HighGammaDataset(data_path, subject_ids=[sid], sessions='both', fs_target=250)
            if len(dataset.data) > 0:
                subject_data_map[sid] = {
                    'data': dataset.data,
                    'labels': dataset.labels,
                    'input_ch_indices': dataset.input_ch_indices,
                    'num_input_channels': dataset.num_input_channels,
                    'num_output_channels': dataset.num_output_channels
                }
                valid_subject_ids.append(sid)
                print(f"  被试{sid}: {len(dataset.data)} trials, {dataset.num_output_channels}ch")
        except Exception as e:
            print(f"  被试{sid}: 加载失败 - {e}")

    if not valid_subject_ids:
        raise RuntimeError("没有成功加载任何被试数据！")

    num_input_channels = subject_data_map[valid_subject_ids[0]]['num_input_channels']
    num_output_channels = subject_data_map[valid_subject_ids[0]]['num_output_channels']
    input_ch_indices = subject_data_map[valid_subject_ids[0]]['input_ch_indices']

    n_folds = len(valid_subject_ids)
    folds = [[sid] for sid in valid_subject_ids]
    print(f"LOSO: {n_folds}折, 每折留1个被试测试")

    results = {
        'baseline_16ch': [],
        'baseline_128ch': [],
        'dttd_reconstructed': [],
        'dttd_augmented': [],
        'dttd_ddim_augmented': [],
        'noise_augmented': [],
        'fold_test_subjects': folds,
        'reconstruction_quality': []
    }

    os.makedirs(output_dir, exist_ok=True)
    resume_path = os.path.join(output_dir, 'hgd_loso_resume.json')
    start_fold = 0
    if os.path.exists(resume_path):
        with open(resume_path, 'r') as f:
            saved = json.load(f)
        completed = saved.get('completed_folds', 0)
        if completed > 0:
            print(f"[INFO] 发现断点记录: 已完成{completed}折, 从第{completed+1}折继续")
            for k in ['baseline_16ch', 'baseline_128ch', 'dttd_reconstructed',
                      'dttd_augmented', 'dttd_ddim_augmented', 'noise_augmented',
                      'reconstruction_quality']:
                results[k] = saved.get(k, [])
            start_fold = completed

    ch_indices_tensor = torch.tensor(input_ch_indices, device=device)
    DS = 4

    for fold_idx, test_subjects in enumerate(folds):
        if fold_idx < start_fold:
            continue
        train_subjects = [s for s in valid_subject_ids if s not in test_subjects]
        print(f"\n--- Fold {fold_idx+1}/{n_folds} (测试被试: {test_subjects}) ---")

        train_data_list, train_labels_list = [], []
        for sid in train_subjects:
            train_data_list.append(subject_data_map[sid]['data'][:, :, ::DS])
            train_labels_list.append(subject_data_map[sid]['labels'])
        train_data_full = np.concatenate(train_data_list, axis=0)
        del train_data_list
        train_labels = np.concatenate(train_labels_list, axis=0)
        del train_labels_list

        test_data_list, test_labels_list = [], []
        for sid in test_subjects:
            test_data_list.append(subject_data_map[sid]['data'][:, :, ::DS])
            test_labels_list.append(subject_data_map[sid]['labels'])
        test_data_full = np.concatenate(test_data_list, axis=0)
        del test_data_list
        test_labels = np.concatenate(test_labels_list, axis=0)
        del test_labels_list

        train_data_in = train_data_full[:, input_ch_indices, :]
        test_data_in = test_data_full[:, input_ch_indices, :]

        num_time_cls = train_data_full.shape[-1]
        print(f"训练集: {len(train_data_full)} trials, 测试集: {len(test_data_full)} trials, 分类时间步: {num_time_cls}")

        # 1. 16ch基线
        print("训练16ch基线分类器...")
        clf_16, m16, s16 = train_classifier(train_data_in, train_labels, num_input_channels, device)
        acc_16 = evaluate_classifier(clf_16, test_data_in, test_labels, device, m16, s16)
        results['baseline_16ch'].append(acc_16)
        print(f"16ch准确率: {acc_16['accuracy']:.4f}, kappa: {acc_16['kappa']:.4f}")
        del clf_16, train_data_in
        torch.cuda.empty_cache()

        # 2. 128ch基线
        print(f"训练128ch基线分类器({num_output_channels}ch)...")
        clf_full, mfull, sfull = train_classifier(train_data_full, train_labels, num_output_channels, device)
        acc_full = evaluate_classifier(clf_full, test_data_full, test_labels, device, mfull, sfull)
        results['baseline_128ch'].append(acc_full)
        print(f"128ch准确率: {acc_full['accuracy']:.4f}, kappa: {acc_full['kappa']:.4f}")
        del clf_full
        torch.cuda.empty_cache()

        # 3. DTTD重建: 在原始128ch上训练→在重建的测试数据上测试（与BCI2a/PhysioNet对齐）
        if model_trained:
            print("DTTD重建...")
            test_data_full_hr = np.concatenate([subject_data_map[sid]['data'] for sid in test_subjects], axis=0)
            test_data_in_full = test_data_full_hr[:, input_ch_indices, :]
            recon_test = generate_augmented_data(
                model, test_data_in_full, test_labels, device,
                data_mean=data_mean, data_std=data_std,
                mode='reconstruct', channel_indices=ch_indices_tensor, batch_size=1
            )
            rq = compute_reconstruction_quality(
                recon_test, test_data_full_hr, test_labels,
                input_ch_indices=input_ch_indices
            )
            results['reconstruction_quality'].append(rq)
            print(f"  重建质量: NMSE={rq['nmse']:.4f}, SNR={rq['snr_db']:.1f}dB, R2={rq['r_squared']:.4f}, Corr={rq['corr']:.4f}")
            print(f"  输入通道SNR: {rq['input_channels_avg_snr']:.1f}dB, 缺失通道SNR: {rq['missing_channels_avg_snr']:.1f}dB")
            print(f"  重建数据范围: [{recon_test.min():.4f}, {recon_test.max():.4f}], 原始数据范围: [{test_data_full_hr.min():.4f}, {test_data_full_hr.max():.4f}]")
            print(f"  重建数据均值: {recon_test.mean():.4f}, 原始数据均值: {test_data_full_hr.mean():.4f}")

            recon_test_ds = recon_test[:, :, ::DS]
            del recon_test, test_data_in_full, test_data_full_hr
            recon_test_ds[:, input_ch_indices, :] = test_data_full[:, input_ch_indices, :]

            clf_recon, mr, sr = train_classifier(train_data_full, train_labels, num_output_channels, device)
            acc_recon = evaluate_classifier(clf_recon, recon_test_ds, test_labels, device, mr, sr)
            results['dttd_reconstructed'].append(acc_recon)
            print(f"DTTD重建准确率: {acc_recon['accuracy']:.4f}, kappa: {acc_recon['kappa']:.4f}")
            del clf_recon, recon_test_ds
        else:
            print("DTTD重建: 跳过（模型未训练）")
            results['dttd_reconstructed'].append(None)

        torch.cuda.empty_cache()

        # 4. DTTD增强(单步)
        if model_trained:
            print("DTTD增强(单步)...")
            train_data_in_full = np.concatenate([subject_data_map[sid]['data'][:, input_ch_indices, :] for sid in train_subjects], axis=0)
            aug_count = min(len(train_data_full) // 4, 2000)
            # 只对需要增强的样本进行推理，避免内存爆炸
            aug_indices = np.random.choice(len(train_data_in_full), aug_count, replace=False)
            aug_input = train_data_in_full[aug_indices]
            aug_labels_in = train_labels[aug_indices]
            del train_data_in_full
            aug_data = generate_augmented_data(
                model, aug_input, aug_labels_in, device,
                data_mean=data_mean, data_std=data_std,
                mode='augment', timestep_range=(50, 300),
                train_noise_scale=0.1, channel_indices=ch_indices_tensor, batch_size=1
            )
            del aug_input
            aug_subset = aug_data[:, :, ::DS].astype(np.float32)
            aug_labels_subset = aug_labels_in
            del aug_data

            clf_aug, ma, sa = train_classifier(
                train_data_full, train_labels, num_output_channels, device,
                extra_data=aug_subset, extra_labels=aug_labels_subset
            )
            acc_aug = evaluate_classifier(clf_aug, test_data_full, test_labels, device, ma, sa)
            results['dttd_augmented'].append(acc_aug)
            print(f"DTTD增强(单步)准确率: {acc_aug['accuracy']:.4f}, kappa: {acc_aug['kappa']:.4f}")
            del clf_aug, aug_subset
        else:
            print("DTTD增强(单步): 跳过（模型未训练）")
            results['dttd_augmented'].append(None)

        torch.cuda.empty_cache()

        # 5. DTTD增强(DDIM)
        if model_trained:
            print("DTTD增强(DDIM)...")
            train_data_in_full = np.concatenate([subject_data_map[sid]['data'][:, input_ch_indices, :] for sid in train_subjects], axis=0)
            ddim_count = min(len(train_data_full) // 4, 2000)
            # 只对需要增强的样本进行推理，避免内存爆炸
            ddim_indices = np.random.choice(len(train_data_in_full), ddim_count, replace=False)
            ddim_input = train_data_in_full[ddim_indices]
            ddim_labels_in = train_labels[ddim_indices]
            del train_data_in_full
            ddim_data = generate_augmented_data(
                model, ddim_input, ddim_labels_in, device,
                data_mean=data_mean, data_std=data_std,
                mode='augment', use_ddim=True,
                num_inference_steps=20, guidance_scale=3.0, eta=0.0,
                channel_indices=ch_indices_tensor, batch_size=1
            )
            del ddim_input
            ddim_subset = ddim_data[:, :, ::DS].astype(np.float32)
            ddim_labels_subset = ddim_labels_in
            del ddim_data

            clf_ddim, md, sd = train_classifier(
                train_data_full, train_labels, num_output_channels, device,
                extra_data=ddim_subset, extra_labels=ddim_labels_subset
            )
            acc_ddim = evaluate_classifier(clf_ddim, test_data_full, test_labels, device, md, sd)
            results['dttd_ddim_augmented'].append(acc_ddim)
            print(f"DTTD增强(DDIM)准确率: {acc_ddim['accuracy']:.4f}, kappa: {acc_ddim['kappa']:.4f}")
            del clf_ddim, ddim_subset
        else:
            print("DTTD增强(DDIM): 跳过（模型未训练）")
            results['dttd_ddim_augmented'].append(None)

        torch.cuda.empty_cache()

        # 6. 传统加噪增强
        print("传统加噪增强...")
        noise_std = np.std(train_data_full, axis=(0, 2), keepdims=True).astype(np.float32)
        # 原位加噪（train_data_full不再需要），然后训练
        chunk_size = 2000
        for i in range(0, len(train_data_full), chunk_size):
            chunk = train_data_full[i:i+chunk_size].astype(np.float32)
            train_data_full[i:i+chunk_size] = chunk + np.random.randn(*chunk.shape).astype(np.float32) * noise_std * 0.2
        del noise_std
        clf_noise, mn, sn = train_classifier(
            train_data_full, train_labels, num_output_channels, device
        )
        acc_noise = evaluate_classifier(clf_noise, test_data_full, test_labels, device, mn, sn)
        results['noise_augmented'].append(acc_noise)
        print(f"传统加噪增强准确率: {acc_noise['accuracy']:.4f}, kappa: {acc_noise['kappa']:.4f}")
        del clf_noise, train_data_full, test_data_full, test_data_in, train_labels, test_labels
        torch.cuda.empty_cache()

        resume_data = {'completed_folds': fold_idx + 1}
        for k in ['baseline_16ch', 'baseline_128ch', 'dttd_reconstructed',
                  'dttd_augmented', 'dttd_ddim_augmented', 'noise_augmented',
                  'reconstruction_quality']:
            resume_data[k] = results[k]
        with open(resume_path, 'w') as f:
            json.dump(resume_data, f)
        print(f"  [已保存] Fold {fold_idx+1}/{n_folds} 完成, 断点已记录")

    save_data = {}
    for k, v in results.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            valid = [x for x in v if x is not None]
            if 'accuracy' in valid[0]:
                save_data[k] = {
                    'values': v,
                    'mean_accuracy': float(np.mean([x['accuracy'] for x in valid])) if valid else None,
                    'std_accuracy': float(np.std([x['accuracy'] for x in valid])) if valid else None,
                    'mean_kappa': float(np.mean([x['kappa'] for x in valid])) if valid else None,
                    'std_kappa': float(np.std([x['kappa'] for x in valid])) if valid else None,
                }
            else:
                # reconstruction_quality等非分类结果
                save_data[k] = {
                    'values': v,
                    'mean': {mk: float(np.mean([x[mk] for x in valid if mk in x])) for mk in valid[0].keys() if isinstance(valid[0][mk], (int, float))},
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
    filepath = os.path.join(output_dir, 'hgd_loso_results.json')
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    for k in ['baseline_16ch', 'baseline_128ch', 'dttd_reconstructed',
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
    parser = argparse.ArgumentParser(description='High Gamma Dataset DTTD重建与增强分类实验')
    parser.add_argument('--data-path', default='E:/data/HGD', help='HGD数据根目录')
    parser.add_argument('--output-dir', default='paper_results/hgd')
    parser.add_argument('--checkpoint', default=None, help='DTTD模型checkpoint路径')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=8, help='训练批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"使用设备: {device}")

    print("预加载数据集以确定通道数...")
    dataset = HighGammaDataset(args.data_path, subject_ids=[1], sessions='train', fs_target=250)
    num_input_ch = dataset.num_input_channels
    num_output_ch = dataset.num_output_channels
    time_steps = dataset.data.shape[2]
    del dataset

    model_config = {
        'input_channels': num_input_ch,
        'output_channels': num_output_ch,
        'time_steps': time_steps,
        'embed_dim': 256,
        'task_dim': 64,
        'num_classes': 4,
        'num_heads': 8,
        'dropout': 0.1,
        'num_timesteps': 1000,
        'beta_start': 1e-4,
        'beta_end': 0.02,
        'schedule_type': 'linear',
        'fs': 250,
        'use_classifier_guidance': False
    }

    model = DTTDPhysioNet(model_config).to(device)
    print(f"[OK] DTTD模型创建成功 (输入:{num_input_ch}ch -> 输出:{num_output_ch}ch, 时间步:{time_steps})")

    data_mean = None
    data_std = None

    save_path = os.path.join(args.output_dir, 'dttd_hgd_best.pth')
    ckpt_path = args.checkpoint if args.checkpoint else save_path

    if os.path.exists(ckpt_path):
        print(f"[INFO] 加载已有模型: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            data_mean = checkpoint.get('data_mean', None)
            data_std = checkpoint.get('data_std', None)
        else:
            model.load_state_dict(checkpoint, strict=False)
        model_trained = True
    else:
        print("[INFO] 未找到checkpoint，开始训练模型...")
        full_dataset = HighGammaDataset(args.data_path, sessions='both', fs_target=250)
        model, data_mean, data_std = train_dttd_model(model, full_dataset, device,
                                                       epochs=args.epochs, batch_size=args.batch_size,
                                                       lr=args.lr, save_path=save_path)
        model_trained = True

    run_experiment(args.data_path, model, device, args.output_dir, model_trained,
                   data_mean=data_mean, data_std=data_std)


if __name__ == '__main__':
    main()
