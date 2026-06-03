"""
High Gamma Dataset 跨会话评估实验

实验设计：
- 训练集：所有被试的train session（前11个run）
- 测试集：每个被试独立的test session（后2个run）

评估指标：
1. 16ch基线
2. 128ch基线  
3. DTTD重建（16ch→128ch）
4. DTTD增强
5. 传统加噪增强

输出：每个被试的详细结果 + 平均值
支持断点续跑，跳过已完成的评估项
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
    normed_data = (train_data - channel_mean)
    normed_data /= channel_std

    clf = EEGNetClassifier(num_channels, 4, time_steps).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(normed_data), torch.LongTensor(train_labels)),
        batch_size=32, shuffle=True, drop_last=True
    )
    del normed_data

    clf.train()
    for _ in range(epochs):
        for data, labels in loader:
            opt.zero_grad()
            loss = criterion(clf(data.to(device)), labels.to(device))
            loss.backward()
            opt.step()

    if extra_data is not None and len(extra_data) > 0:
        extra_normed = (extra_data - channel_mean)
        extra_normed /= channel_std
        extra_loader = DataLoader(
            TensorDataset(torch.FloatTensor(extra_normed), torch.LongTensor(extra_labels)),
            batch_size=32, shuffle=True, drop_last=True
        )
        del extra_normed
        opt2 = torch.optim.Adam(clf.parameters(), lr=1e-4, weight_decay=1e-4)
        clf.train()
        for _ in range(extra_epochs):
            for data, labels in extra_loader:
                opt2.zero_grad()
                loss = criterion(clf(data.to(device)), labels.to(device))
                loss.backward()
                opt2.step()

    return clf, channel_mean, channel_std


def evaluate_classifier(clf, test_data, test_labels, device, channel_mean=None, channel_std=None):
    clf.eval()
    if channel_mean is not None and channel_std is not None:
        test_data = (test_data - channel_mean)
        test_data = (test_data / channel_std).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.FloatTensor(test_data)), batch_size=32, shuffle=False)

    preds = []
    with torch.no_grad():
        for batch in loader:
            output = clf(batch[0].to(device))
            preds.extend(torch.argmax(output, dim=1).cpu().numpy())

    return {
        'accuracy': accuracy_score(test_labels, preds),
        'kappa': cohen_kappa_score(test_labels, preds)
    }


def compute_reconstruction_quality(recon_data, target_data, input_ch_indices=None):
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
    
    if input_ch_indices is not None:
        n_ch = recon_data.shape[1]
        per_ch_var = np.var(target_data, axis=(0, 2))
        per_ch_mse = np.mean(diff ** 2, axis=(0, 2))
        per_ch_snr = 10 * np.log10(per_ch_var / (per_ch_mse + 1e-12))
        input_ch_snr = np.mean(per_ch_snr[input_ch_indices]) if all(i < n_ch for i in input_ch_indices) else 0
        missing_mask = np.ones(n_ch, dtype=bool)
        missing_mask[input_ch_indices] = False
        missing_ch_snr = np.mean(per_ch_snr[missing_mask]) if missing_mask.any() else 0
        metrics['input_channels_avg_snr'] = float(input_ch_snr)
        metrics['missing_channels_avg_snr'] = float(missing_ch_snr)
    
    return metrics


def generate_augmented_data(model, input_data, labels, device, data_mean=None, data_std=None,
                            mode='reconstruct', timestep_range=(50, 300),
                            train_noise_scale=0.1, channel_indices=None, batch_size=4,
                            guidance_scale=1.0):
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
        TensorDataset(torch.FloatTensor(normed_input), torch.LongTensor(labels)),
        batch_size=batch_size, shuffle=False
    )

    with torch.no_grad():
        for batch_data, batch_labels in loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            if mode == 'reconstruct':
                t = torch.zeros(batch_data.size(0), dtype=torch.long, device=device)
                gen = model.forward(batch_data, t, batch_labels)
            elif mode == 'augment':
                t = torch.randint(timestep_range[0], timestep_range[1] + 1,
                                  (batch_data.size(0),), device=device)
                noise_level = train_noise_scale * model.scheduler.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
                noisy_input = batch_data + torch.randn_like(batch_data) * noise_level
                if guidance_scale > 1.0 and batch_labels is not None:
                    x0_cond = model.forward(noisy_input, t, batch_labels)
                    x0_uncond = model.forward(noisy_input, t, None)
                    gen = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
                else:
                    gen = model.forward(noisy_input, t, batch_labels)

            gen_np = gen.cpu().numpy()
            if data_mean is not None:
                gen_np = gen_np * data_std + data_mean
            generated_list.append(gen_np)
            del batch_data, batch_labels, gen
            torch.cuda.empty_cache()

    return np.concatenate(generated_list, axis=0)


def train_dttd_model_cross_session(model, train_data, train_labels, device, 
                                    input_ch_indices=None,
                                    epochs=200, batch_size=8, lr=1e-4, save_path=None):
    """在跨会话场景下训练DTTD模型"""
    print("\n" + "=" * 60)
    print("训练 DTTD 模型 (跨会话场景)")
    print("=" * 60)

    data_mean = train_data.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    data_std = train_data.std(axis=(0, 2), keepdims=True).astype(np.float32)
    data_std = np.maximum(data_std, 1e-6)

    print(f"训练样本数: {len(train_data)}, 通道数: {train_data.shape[1]}")
    print(f"数据归一化: mean={data_mean.mean():.2f}, std={data_std.mean():.2f}")

    normed_data = ((train_data - data_mean) / data_std).astype(np.float32)
    train_tensor = torch.FloatTensor(normed_data)
    labels_tensor = torch.LongTensor(train_labels)

    loader = DataLoader(
        TensorDataset(train_tensor, labels_tensor),
        batch_size=batch_size, shuffle=True, drop_last=True
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if save_path else None

    if input_ch_indices is not None:
        ch_indices_tensor = torch.tensor(input_ch_indices, device=device)
        print(f"输入通道索引: {input_ch_indices}")
    else:
        ch_indices_tensor = torch.tensor(range(16), device=device)
        print("[WARN] 未指定输入通道索引，使用前16通道")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for data_batch, label_batch in loader:
            data_batch = data_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()

            if torch.rand(1).item() < 0.1:
                task_label = None
            else:
                task_label = label_batch

            loss = model.compute_loss(
                x_target=data_batch,
                channel_indices=ch_indices_tensor,
                task_label=task_label,
                current_epoch=epoch
            )

            if isinstance(loss, dict):
                loss['total_loss'].backward()
                total_loss += loss['reconstruction_loss']
            else:
                loss.backward()
                total_loss += loss.item()

            optimizer.step()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / num_batches

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

        if avg_loss < best_loss and save_path:
            best_loss = avg_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'data_mean': data_mean,
                'data_std': data_std,
                'epoch': epoch,
                'loss': avg_loss
            }
            torch.save(checkpoint, save_path)

    print(f"[OK] DTTD模型训练完成，最佳Loss: {best_loss:.6f}")
    return model, data_mean, data_std


def run_cross_session_experiment(data_path, model, device, output_dir, 
                                 data_mean=None, data_std=None, model_trained=False):
    print("\n" + "=" * 60)
    print("High Gamma Dataset 跨会话评估")
    print("对齐BCI2a: 每个被试独立 train session→test session")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    resume_path = os.path.join(output_dir, 'cs_resume.json')
    if os.path.exists(resume_path):
        with open(resume_path, 'r') as f:
            resume_data = json.load(f)
        print(f"[INFO] 检测到断点，已完成: {resume_data.get('completed_tasks', [])}")
    else:
        resume_data = {'completed_tasks': [], 'saved_results': {}}

    all_subject_ids = list(range(1, 15))
    subject_data_map = {}
    input_ch_indices = None
    num_input_channels = 0
    num_output_channels = 0

    print("\n加载所有被试数据...")
    for sid in all_subject_ids:
        try:
            train_ds = HighGammaDataset(data_path, subject_ids=[sid], sessions='train', fs_target=250)
            test_ds = HighGammaDataset(data_path, subject_ids=[sid], sessions='test', fs_target=250)
            
            if len(train_ds.data) > 0 and len(test_ds.data) > 0:
                subject_data_map[sid] = {
                    'train_data': train_ds.data,
                    'train_labels': train_ds.labels,
                    'test_data': test_ds.data,
                    'test_labels': test_ds.labels
                }
                if input_ch_indices is None:
                    input_ch_indices = train_ds.input_ch_indices
                    num_input_channels = train_ds.num_input_channels
                    num_output_channels = train_ds.num_output_channels
                print(f"  被试{sid}: train={len(train_ds.data)} trials, test={len(test_ds.data)} trials")
        except Exception as e:
            print(f"  被试{sid}: 加载失败 - {e}")

    if not subject_data_map:
        raise RuntimeError("没有成功加载任何被试数据！")

    DS = 2

    results = {
        'per_subject': {},
        'summary': {
            'baseline_16ch': {'values': [], 'mean': None, 'std': None},
            'baseline_128ch': {'values': [], 'mean': None, 'std': None},
            'dttd_reconstructed': {'values': [], 'mean': None, 'std': None},
            'dttd_augmented': {'values': [], 'mean': None, 'std': None},
            'noise_augmented': {'values': [], 'mean': None, 'std': None},
            'reconstruction_quality': []
        }
    }

    # ========== 每个被试独立评估（对齐BCI2a） ==========
    for sid in sorted(subject_data_map.keys()):
        task_key = f'subject_{sid}'
        if task_key in resume_data['completed_tasks']:
            print(f"\n[SKIP] 被试{sid}已完成，恢复结果")
            saved = resume_data.get('saved_results', {})
            results['per_subject'][sid] = saved.get(task_key, {})
            continue

        print(f"\n{'='*50}")
        print(f"被试 {sid}")
        print(f"{'='*50}")

        train_128ch = subject_data_map[sid]['train_data'][:, :, ::DS]
        train_labels = subject_data_map[sid]['train_labels']
        test_128ch = subject_data_map[sid]['test_data'][:, :, ::DS]
        test_labels = subject_data_map[sid]['test_labels']
        train_16ch = train_128ch[:, input_ch_indices, :]
        test_16ch = test_128ch[:, input_ch_indices, :]

        subj_result = {}

        # 1. 128ch基线
        clf_128, m128, s128 = train_classifier(train_128ch, train_labels, num_output_channels, device)
        acc_128 = evaluate_classifier(clf_128, test_128ch, test_labels, device, m128, s128)
        subj_result['baseline_128ch'] = acc_128
        results['summary']['baseline_128ch']['values'].append(acc_128)
        print(f"  128ch基线: acc={acc_128['accuracy']:.4f}, kappa={acc_128['kappa']:.4f}")
        del clf_128

        # 2. 16ch基线
        clf_16, m16, s16 = train_classifier(train_16ch, train_labels, num_input_channels, device)
        acc_16 = evaluate_classifier(clf_16, test_16ch, test_labels, device, m16, s16)
        subj_result['baseline_16ch'] = acc_16
        results['summary']['baseline_16ch']['values'].append(acc_16)
        print(f"  16ch基线: acc={acc_16['accuracy']:.4f}, kappa={acc_16['kappa']:.4f}")
        del clf_16

        # 3. DTTD重建: 对齐BCI2a - 重建测试数据, 原始训练数据训练分类器, 重建测试数据上测试
        if model_trained and data_mean is not None and data_std is not None:
            recon_test_128ch = generate_augmented_data(
                model, test_16ch, test_labels, device,
                data_mean=data_mean, data_std=data_std,
                mode='reconstruct', channel_indices=torch.tensor(input_ch_indices, device=device),
                guidance_scale=1.0
            )
            recon_test_128ch[:, input_ch_indices, :] = test_128ch[:, input_ch_indices, :]

            rq = compute_reconstruction_quality(recon_test_128ch, test_128ch, input_ch_indices=input_ch_indices)
            results['summary']['reconstruction_quality'].append(rq)
            print(f"  重建质量: NMSE={rq['nmse']:.4f}, SNR={rq['snr_db']:.1f}dB")

            clf_orig, mo, so = train_classifier(train_128ch, train_labels, num_output_channels, device)
            acc_recon = evaluate_classifier(clf_orig, recon_test_128ch, test_labels, device, mo, so)
            subj_result['dttd_reconstructed'] = acc_recon
            results['summary']['dttd_reconstructed']['values'].append(acc_recon)
            print(f"  DTTD重建: acc={acc_recon['accuracy']:.4f}, kappa={acc_recon['kappa']:.4f}")
            del clf_orig, recon_test_128ch

            # 4. DTTD增强: 两阶段训练 - 先原始数据训练, 再增强数据微调
            gen_128ch = generate_augmented_data(
                model, train_16ch, train_labels, device,
                data_mean=data_mean, data_std=data_std,
                mode='augment', channel_indices=torch.tensor(input_ch_indices, device=device),
                timestep_range=(50, 300), train_noise_scale=0.2,
                guidance_scale=1.0
            )
            gen_128ch[:, input_ch_indices, :] = train_128ch[:, input_ch_indices, :]

            aug_count = len(train_128ch) // 2
            aug_indices = np.random.choice(len(gen_128ch), aug_count, replace=False)
            aug_subset = gen_128ch[aug_indices]
            aug_labels_subset = train_labels[aug_indices]
            del gen_128ch

            clf_aug, ma, sa = train_classifier(
                train_128ch, train_labels, num_output_channels, device,
                extra_data=aug_subset, extra_labels=aug_labels_subset, extra_epochs=50
            )
            del aug_subset
            acc_aug = evaluate_classifier(clf_aug, test_128ch, test_labels, device, ma, sa)
            subj_result['dttd_augmented'] = acc_aug
            results['summary']['dttd_augmented']['values'].append(acc_aug)
            print(f"  DTTD增强: acc={acc_aug['accuracy']:.4f}, kappa={acc_aug['kappa']:.4f}")
            del clf_aug
        else:
            subj_result['dttd_reconstructed'] = None
            subj_result['dttd_augmented'] = None
            results['summary']['dttd_reconstructed']['values'].append(None)
            results['summary']['dttd_augmented']['values'].append(None)

        # 5. 传统加噪增强: 两阶段训练
        noise_std = np.std(train_128ch, axis=(0, 2), keepdims=True).astype(np.float32)
        noise_count = len(train_128ch) // 2
        noise_idx = np.random.choice(len(train_128ch), noise_count, replace=False)
        noised_subset = (train_128ch[noise_idx] + np.random.randn(noise_count, num_output_channels, train_128ch.shape[2]).astype(np.float32) * noise_std * 0.2).astype(np.float32)
        noise_labels_subset = train_labels[noise_idx]

        clf_noise, mn, sn = train_classifier(
            train_128ch, train_labels, num_output_channels, device,
            extra_data=noised_subset, extra_labels=noise_labels_subset, extra_epochs=50
        )
        del noised_subset
        acc_noise = evaluate_classifier(clf_noise, test_128ch, test_labels, device, mn, sn)
        subj_result['noise_augmented'] = acc_noise
        results['summary']['noise_augmented']['values'].append(acc_noise)
        print(f"  传统加噪: acc={acc_noise['accuracy']:.4f}, kappa={acc_noise['kappa']:.4f}")
        del clf_noise

        results['per_subject'][sid] = subj_result
        resume_data['completed_tasks'].append(task_key)
        resume_data['saved_results'][task_key] = subj_result
        with open(resume_path, 'w') as f:
            json.dump(resume_data, f)

        torch.cuda.empty_cache()

    # 计算平均值和标准差
    for key in ['baseline_16ch', 'baseline_128ch', 'dttd_reconstructed', 'dttd_augmented', 'noise_augmented']:
        values = [v for v in results['summary'][key]['values'] if v is not None]
        if values:
            results['summary'][key]['mean_accuracy'] = float(np.mean([v['accuracy'] for v in values]))
            results['summary'][key]['std_accuracy'] = float(np.std([v['accuracy'] for v in values]))
            results['summary'][key]['mean_kappa'] = float(np.mean([v['kappa'] for v in values]))
            results['summary'][key]['std_kappa'] = float(np.std([v['kappa'] for v in values]))

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'hgd_cross_session_results.json')
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 打印汇总
    def safe_print(label, key):
        m = results['summary'][key].get('mean_accuracy')
        s = results['summary'][key].get('std_accuracy')
        mk = results['summary'][key].get('mean_kappa')
        sk = results['summary'][key].get('std_kappa')
        if m is not None:
            print(f"{label}: acc={m*100:.2f}% ± {s*100:.2f}%, kappa={mk:.4f} ± {sk:.4f}")
        else:
            print(f"{label}: 无数据")

    print("\n" + "=" * 60)
    print("跨会话评估结果汇总")
    print("=" * 60)
    safe_print("16ch基线", 'baseline_16ch')
    safe_print("128ch基线", 'baseline_128ch')
    safe_print("DTTD重建", 'dttd_reconstructed')
    safe_print("DTTD增强", 'dttd_augmented')
    safe_print("传统加噪", 'noise_augmented')
    
    if results['summary']['reconstruction_quality']:
        avg_nmse = np.mean([r['nmse'] for r in results['summary']['reconstruction_quality']])
        avg_snr = np.mean([r['snr_db'] for r in results['summary']['reconstruction_quality']])
        print(f"重建质量(平均): NMSE={avg_nmse:.4f}, SNR={avg_snr:.1f}dB")
    
    print(f"\n详细结果已保存至: {filepath}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='High Gamma Dataset 跨会话评估')
    parser.add_argument('--data-path', default='E:/data/HGD', help='HGD数据根目录')
    parser.add_argument('--output-dir', default='paper_results/hgd_cross_session')
    parser.add_argument('--checkpoint', default=None, help='预训练DTTD模型路径（可选）')
    parser.add_argument('--train-dttd', action='store_true', help='是否在跨会话场景下训练DTTD模型')
    parser.add_argument('--dttd-epochs', type=int, default=200, help='DTTD训练轮数')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"使用设备: {device}")

    # 预加载数据集确定通道数
    dataset = HighGammaDataset(args.data_path, subject_ids=[1], sessions='train', fs_target=250)
    num_input_ch = dataset.num_input_channels
    num_output_ch = dataset.num_output_channels
    time_steps = dataset.data.shape[2] // 2  # 考虑降采样
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
        'use_classifier_guidance': True,
        'lambda_cls': 0.1,
        'warmup_epochs': 30
    }

    model = DTTDPhysioNet(model_config).to(device)
    print(f"[OK] DTTD模型创建成功 (输入:{num_input_ch}ch -> 输出:{num_output_ch}ch)")

    data_mean = None
    data_std = None
    model_trained = False

    save_path = os.path.join(args.output_dir, 'dttd_hgd_cs_best.pth')
    ckpt_path = args.checkpoint if args.checkpoint else save_path

    if os.path.exists(ckpt_path) and not args.train_dttd:
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
        if args.train_dttd:
            print("[INFO] 强制重新训练DTTD模型（--train-dttd）...")
        else:
            print("[INFO] 未找到预训练模型，自动开始训练DTTD模型...")
        os.makedirs(args.output_dir, exist_ok=True)
        all_train_data = []
        all_train_labels = []
        input_ch_indices = None
        for sid in range(1, 15):
            try:
                ds = HighGammaDataset(args.data_path, subject_ids=[sid], sessions='train', fs_target=250)
                if len(ds.data) > 0:
                    downsampled = ds.data[:, :, ::2]
                    all_train_data.append(downsampled)
                    all_train_labels.append(ds.labels)
                    if input_ch_indices is None:
                        input_ch_indices = ds.input_ch_indices
                    print(f"  被试{sid}: {len(ds.data)} trials (降采样后 {downsampled.shape})")
                    del ds, downsampled
            except Exception as e:
                print(f"  被试{sid}: 跳过 - {e}")
        train_data = np.concatenate(all_train_data, axis=0)
        train_labels = np.concatenate(all_train_labels, axis=0)
        del all_train_data, all_train_labels
        print(f"总训练数据: {len(train_data)} trials, shape={train_data.shape}")
        model, data_mean, data_std = train_dttd_model_cross_session(
            model, train_data, train_labels, device,
            input_ch_indices=input_ch_indices,
            epochs=args.dttd_epochs, batch_size=8, lr=1e-4, save_path=save_path
        )
        model_trained = True

    run_cross_session_experiment(args.data_path, model, device, args.output_dir,
                                  data_mean=data_mean, data_std=data_std, 
                                  model_trained=model_trained)


if __name__ == '__main__':
    main()