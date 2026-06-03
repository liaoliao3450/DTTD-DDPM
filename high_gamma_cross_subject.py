"""
High Gamma Dataset 跨被试评估实验 (LOSO - Leave-One-Subject-Out)

基于 high_gamma_cross_session.py 的评估流程，改为跨被试评估：
- 每折留1个被试作为测试集
- 其余被试的所有数据（train+test session）作为训练集
- 保持与跨会话实验完全相同的数据处理和评估流程

评估指标：
1. 16ch基线
2. 128ch基线
3. DTTD重建（16ch→128ch）
4. DTTD增强
5. 传统加噪增强

输出：每个被试的详细结果 + 平均值
支持断点续跑
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


class OnlineNormDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, channel_mean, channel_std):
        self.data = data
        self.labels = labels
        self.channel_mean = channel_mean
        self.channel_std = channel_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = ((self.data[idx] - self.channel_mean) / self.channel_std).astype(np.float32)
        return torch.FloatTensor(x), torch.LongTensor([self.labels[idx]])[0]


def train_classifier(train_data, train_labels, num_channels, device, epochs=100,
                     extra_data=None, extra_labels=None, extra_epochs=50):
    time_steps = train_data.shape[-1]
    n_samples = train_data.shape[0]
    channel_mean = np.mean(train_data, axis=2, keepdims=True).astype(np.float32)
    channel_mean = np.mean(channel_mean, axis=0, keepdims=True)
    diff_sq_sum = np.zeros((1, num_channels, 1), dtype=np.float64)
    for i in range(n_samples):
        d = train_data[i:i+1].astype(np.float64) - channel_mean
        diff_sq_sum += np.mean(d ** 2, axis=2, keepdims=True)
    channel_var = (diff_sq_sum / n_samples).astype(np.float32)
    channel_std = np.sqrt(channel_var) + 1e-8
    del diff_sq_sum

    clf = EEGNetClassifier(num_channels, 4, time_steps).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    dataset = OnlineNormDataset(train_data, train_labels, channel_mean, channel_std)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=0)

    clf.train()
    for _ in range(epochs):
        for data, labels in loader:
            opt.zero_grad()
            loss = criterion(clf(data.to(device)), labels.to(device))
            loss.backward()
            opt.step()

    if extra_data is not None and len(extra_data) > 0:
        extra_dataset = OnlineNormDataset(extra_data, extra_labels, channel_mean, channel_std)
        extra_loader = DataLoader(extra_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=0)
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
        eval_dataset = OnlineNormDataset(test_data, test_labels, channel_mean, channel_std)
    else:
        eval_dataset = TensorDataset(torch.FloatTensor(test_data), torch.LongTensor(test_labels))
    loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=0)

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
        gen_dataset = OnlineNormDataset(input_data, labels, input_mean, input_std)
    else:
        gen_dataset = TensorDataset(torch.FloatTensor(input_data), torch.LongTensor(labels))

    loader = DataLoader(gen_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

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


def train_dttd_model_loso(model, train_data, train_labels, device,
                          input_ch_indices=None,
                          epochs=200, batch_size=8, lr=1e-4, save_path=None):
    print("\n" + "=" * 60)
    print("训练 DTTD 模型 (跨被试场景 - LOSO)")
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


def run_loso_experiment(data_path, model, device, output_dir,
                        data_mean=None, data_std=None, model_trained=False,
                        input_ch_indices=None, num_input_channels=16, num_output_channels=128):
    print("\n" + "=" * 60)
    print("High Gamma Dataset 跨被试评估 (LOSO)")
    print("对齐cross_session: 保持完全相同的数据处理和评估流程")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    resume_path = os.path.join(output_dir, 'loso_resume.json')
    if os.path.exists(resume_path):
        with open(resume_path, 'r') as f:
            resume_data = json.load(f)
        print(f"[INFO] 检测到断点，已完成: {resume_data.get('completed_tasks', [])}")
    else:
        resume_data = {'completed_tasks': [], 'saved_results': {}}

    all_subject_ids = list(range(1, 15))
    subject_data_map = {}

    print("\n加载所有被试数据 (sessions='both', 合并train+test)...")
    print("每个被试的所有session数据合并后用于LOSO评估")
    DS = 2
    for sid in all_subject_ids:
        try:
            dataset = HighGammaDataset(data_path, subject_ids=[sid], sessions='both', fs_target=250)
            if len(dataset.data) > 0:
                downsampled = dataset.data[:, :, ::DS].astype(np.float32)
                subject_data_map[sid] = {
                    'data': downsampled,
                    'labels': dataset.labels,
                    'input_ch_indices': dataset.input_ch_indices,
                    'num_input_channels': dataset.num_input_channels,
                    'num_output_channels': dataset.num_output_channels
                }
                if input_ch_indices is None:
                    input_ch_indices = dataset.input_ch_indices
                    num_input_channels = dataset.num_input_channels
                    num_output_channels = dataset.num_output_channels
                print(f"  [OK] 被试{sid}: {len(dataset.data)} trials → 降采样后 {downsampled.shape}, 128ch EEG")
                del dataset, downsampled
        except Exception as e:
            print(f"  [FAIL] 被试{sid}: 加载失败 - {e}")

    if not subject_data_map:
        raise RuntimeError("没有成功加载任何被试数据！")

    results = {
        'per_subject': {},
        'summary': {
            'baseline_128ch': {'values': [], 'mean': None, 'std': None},
            'dttd_augmented': {'values': [], 'mean': None, 'std': None},
        }
    }

    for test_sid in sorted(subject_data_map.keys()):
        task_key = f'subject_{test_sid}'
        if task_key in resume_data['completed_tasks']:
            print(f"\n[SKIP] 被试{test_sid}已完成，恢复结果")
            saved = resume_data.get('saved_results', {})
            results['per_subject'][test_sid] = saved.get(task_key, {})
            continue

        print(f"\n{'='*50}")
        print(f"LOSO Fold: 测试被试 {test_sid}")
        print(f"{'='*50}")

        train_subjects = [s for s in sorted(subject_data_map.keys()) if s != test_sid]

        train_data_list, train_labels_list = [], []
        for sid in train_subjects:
            train_data_list.append(subject_data_map[sid]['data'])
            train_labels_list.append(subject_data_map[sid]['labels'])
        train_128ch = np.concatenate(train_data_list, axis=0)
        del train_data_list
        train_labels = np.concatenate(train_labels_list, axis=0)
        del train_labels_list

        test_128ch = subject_data_map[test_sid]['data']
        test_labels = subject_data_map[test_sid]['labels']

        train_16ch = train_128ch[:, input_ch_indices, :]

        print(f"  训练集: {len(train_128ch)} trials, 测试集: {len(test_128ch)} trials")

        subj_result = {}

        # 1. 128ch基线
        clf_128, m128, s128 = train_classifier(train_128ch, train_labels, num_output_channels, device)
        acc_128 = evaluate_classifier(clf_128, test_128ch, test_labels, device, m128, s128)
        subj_result['baseline_128ch'] = acc_128
        results['summary']['baseline_128ch']['values'].append(acc_128)
        print(f"  128ch基线: acc={acc_128['accuracy']:.4f}, kappa={acc_128['kappa']:.4f}")
        del clf_128

        # 2. DTTD增强
        if model_trained and data_mean is not None and data_std is not None:
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

        results['per_subject'][test_sid] = subj_result
        resume_data['completed_tasks'].append(task_key)
        resume_data['saved_results'][task_key] = subj_result
        with open(resume_path, 'w') as f:
            json.dump(resume_data, f)

        torch.cuda.empty_cache()

    # 计算平均值和标准差
    for key in ['baseline_128ch', 'dttd_augmented']:
        values = [v for v in results['summary'][key]['values'] if v is not None]
        if values:
            results['summary'][key]['mean_accuracy'] = float(np.mean([v['accuracy'] for v in values]))
            results['summary'][key]['std_accuracy'] = float(np.std([v['accuracy'] for v in values]))
            results['summary'][key]['mean_kappa'] = float(np.mean([v['kappa'] for v in values]))
            results['summary'][key]['std_kappa'] = float(np.std([v['kappa'] for v in values]))

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'hgd_loso_results.json')
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
    print("跨被试评估结果汇总 (LOSO)")
    print("=" * 60)
    safe_print("128ch基线", 'baseline_128ch')
    safe_print("DTTD增强", 'dttd_augmented')

    print(f"\n详细结果已保存至: {filepath}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='High Gamma Dataset 跨被试评估 (LOSO)')
    parser.add_argument('--data-path', default='E:/data/HGD', help='HGD数据根目录')
    parser.add_argument('--output-dir', default='paper_results/hgd_loso')
    parser.add_argument('--checkpoint', default=None, help='预训练DTTD模型路径（可选）')
    parser.add_argument('--train-dttd', action='store_true', help='是否训练DTTD模型')
    parser.add_argument('--dttd-epochs', type=int, default=200, help='DTTD训练轮数')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"使用设备: {device}")

    dataset = HighGammaDataset(args.data_path, subject_ids=[1], sessions='both', fs_target=250)
    num_input_ch = dataset.num_input_channels
    num_output_ch = dataset.num_output_channels
    time_steps = dataset.data.shape[2] // 2
    input_ch_indices = dataset.input_ch_indices
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

    save_path = os.path.join(args.output_dir, 'dttd_hgd_loso_best.pth')
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
        for sid in range(1, 15):
            try:
                ds = HighGammaDataset(args.data_path, subject_ids=[sid], sessions='both', fs_target=250)
                if len(ds.data) > 0:
                    downsampled = ds.data[:, :, ::2]
                    all_train_data.append(downsampled)
                    all_train_labels.append(ds.labels)
                    print(f"  被试{sid}: {len(ds.data)} trials (降采样后 {downsampled.shape})")
                    del ds, downsampled
            except Exception as e:
                print(f"  被试{sid}: 跳过 - {e}")
        train_data = np.concatenate(all_train_data, axis=0)
        train_labels = np.concatenate(all_train_labels, axis=0)
        del all_train_data, all_train_labels
        print(f"总训练数据: {len(train_data)} trials, shape={train_data.shape}")
        model, data_mean, data_std = train_dttd_model_loso(
            model, train_data, train_labels, device,
            input_ch_indices=input_ch_indices,
            epochs=args.dttd_epochs, batch_size=8, lr=1e-4, save_path=save_path
        )
        model_trained = True

    run_loso_experiment(args.data_path, model, device, args.output_dir,
                        data_mean=data_mean, data_std=data_std,
                        model_trained=model_trained,
                        input_ch_indices=input_ch_indices,
                        num_input_channels=num_input_ch,
                        num_output_channels=num_output_ch)


if __name__ == '__main__':
    main()
