"""
HGD跨session消融实验 - v9 (推理时消融)

核心设计：
1. 所有变体使用同一个Full Model checkpoint
2. 通过ablation_mode在推理时禁用对应模块（而非训练独立模型）
3. 确保消融比较的公平性：唯一变量是被禁用的模块

消融配置：
1. 完整DTTD (Full) - 所有模块正常工作
2. 无拓扑模块 (No Topology) - 推理时跳过topology_module
3. 无频率模块 (No Frequency) - 推理时跳过frequency_module
4. 无任务条件 (No Task) - 推理时task_emb置零

使用方法:
    python experiments/hgd_ablation_v3.py
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
                     extra_data=None, extra_labels=None):
    time_steps = train_data.shape[-1]
    channel_mean = train_data.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    channel_std = train_data.std(axis=(0, 2), keepdims=True).astype(np.float32) + 1e-8
    normed_data = (train_data - channel_mean)
    normed_data /= channel_std

    torch.manual_seed(42)
    np.random.seed(42)

    clf = EEGNetClassifier(num_channels, 4, time_steps).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)

    generator = torch.Generator()
    generator.manual_seed(42)

    if extra_data is not None and len(extra_data) > 0:
        extra_normed = ((extra_data - channel_mean) / channel_std).astype(np.float32)
        ds1 = TensorDataset(torch.FloatTensor(normed_data), torch.LongTensor(train_labels))
        ds2 = TensorDataset(torch.FloatTensor(extra_normed), torch.LongTensor(extra_labels))
        dataset = torch.utils.data.ConcatDataset([ds1, ds2])
        del extra_normed
    else:
        dataset = TensorDataset(torch.FloatTensor(normed_data), torch.LongTensor(train_labels))

    loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False, generator=generator)
    del normed_data

    clf.train()
    best_loss = float('inf')
    best_state = None
    for epoch in range(epochs):
        total_loss = 0.0
        for data, labels in loader:
            opt.zero_grad()
            loss = criterion(clf(data.to(device)), labels.to(device))
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in clf.state_dict().items()}

    if best_state is not None:
        clf.load_state_dict(best_state)

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
        'accuracy': float(accuracy_score(test_labels, preds)),
        'kappa': float(cohen_kappa_score(test_labels, preds))
    }


class DTTDPhysioNet_NoTopology(DTTDPhysioNet):
    def __init__(self, config):
        super().__init__(config)
        self.topology_module = None
        num_groups = min(8, self.output_channels)
        while self.output_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.simple_topo_proj = nn.Sequential(
            nn.Conv1d(self.output_channels, self.output_channels, 3, padding=1),
            nn.GroupNorm(num_groups, self.output_channels),
            nn.GELU()
        )

    def forward(self, x, t, task_label=None):
        batch_size = x.size(0)
        x = self.input_proj(x)
        t_emb = self.time_mlp(t)
        if task_label is not None:
            task_onehot = F.one_hot(task_label, num_classes=self.num_classes).float()
            task_emb = self.task_encoder(task_onehot)
        else:
            task_emb = torch.zeros(batch_size, self.task_dim, device=x.device)

        x_expanded = self.channel_expansion(x)
        x_128ch_rough = self.rough_channel_proj(x_expanded)

        x_topo_raw = self.simple_topo_proj(x_128ch_rough)
        x_topo = self.topo_output_proj(x_topo_raw)

        x_freq_raw, _, _ = self.frequency_module(x_128ch_rough, fs=self.config.get('fs', 250), task_emb=task_emb)
        x_freq = self.freq_output_proj(x_freq_raw)

        gate_input = torch.cat([x_topo, x_freq], dim=1)
        gate_logits = self.gate_conv(gate_input) + self.freq_bias.view(1, 2, 1)
        gate_weights = F.softmax(gate_logits, dim=1).unsqueeze(1)
        weighted_features = torch.stack([x_topo, x_freq], dim=2) * gate_weights
        fused_features = weighted_features.sum(dim=2)

        x_unet = self.denoising_net(fused_features, t_emb, task_emb)
        x_final = x_unet + fused_features
        x0_pred = self.output_proj(x_final)
        return x0_pred


class DTTDPhysioNet_NoFrequency(DTTDPhysioNet):
    def __init__(self, config):
        super().__init__(config)
        self.frequency_module = None
        num_groups = min(8, self.output_channels)
        while self.output_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.simple_freq_proj = nn.Sequential(
            nn.Conv1d(self.output_channels, self.output_channels, 3, padding=1),
            nn.GroupNorm(num_groups, self.output_channels),
            nn.GELU()
        )

    def forward(self, x, t, task_label=None):
        batch_size = x.size(0)
        x = self.input_proj(x)
        t_emb = self.time_mlp(t)
        if task_label is not None:
            task_onehot = F.one_hot(task_label, num_classes=self.num_classes).float()
            task_emb = self.task_encoder(task_onehot)
        else:
            task_emb = torch.zeros(batch_size, self.task_dim, device=x.device)

        x_expanded = self.channel_expansion(x)
        x_128ch_rough = self.rough_channel_proj(x_expanded)

        x_topo_raw, _ = self.topology_module(x_128ch_rough, task_emb)
        x_topo = self.topo_output_proj(x_topo_raw)

        x_freq_raw = self.simple_freq_proj(x_128ch_rough)
        x_freq = self.freq_output_proj(x_freq_raw)

        gate_input = torch.cat([x_topo, x_freq], dim=1)
        gate_logits = self.gate_conv(gate_input) + self.freq_bias.view(1, 2, 1)
        gate_weights = F.softmax(gate_logits, dim=1).unsqueeze(1)
        weighted_features = torch.stack([x_topo, x_freq], dim=2) * gate_weights
        fused_features = weighted_features.sum(dim=2)

        x_unet = self.denoising_net(fused_features, t_emb, task_emb)
        x_final = x_unet + fused_features
        x0_pred = self.output_proj(x_final)
        return x0_pred


class DTTDPhysioNet_NoTask(DTTDPhysioNet):
    def forward(self, x, t, task_label=None):
        batch_size = x.size(0)
        x = self.input_proj(x)
        t_emb = self.time_mlp(t)
        task_emb = torch.zeros(batch_size, self.task_dim, device=x.device)

        x_expanded = self.channel_expansion(x)
        x_128ch_rough = self.rough_channel_proj(x_expanded)

        x_topo_raw, _ = self.topology_module(x_128ch_rough, task_emb)
        x_topo = self.topo_output_proj(x_topo_raw)

        x_freq_raw, _, _ = self.frequency_module(x_128ch_rough, fs=self.config.get('fs', 250), task_emb=task_emb)
        x_freq = self.freq_output_proj(x_freq_raw)

        gate_input = torch.cat([x_topo, x_freq], dim=1)
        gate_logits = self.gate_conv(gate_input) + self.freq_bias.view(1, 2, 1)
        gate_weights = F.softmax(gate_logits, dim=1).unsqueeze(1)
        weighted_features = torch.stack([x_topo, x_freq], dim=2) * gate_weights
        fused_features = weighted_features.sum(dim=2)

        x_unet = self.denoising_net(fused_features, t_emb, task_emb)
        x_final = x_unet + fused_features
        x0_pred = self.output_proj(x_final)
        return x0_pred


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


def train_dttd_model(model, train_data, train_labels, device, input_ch_indices=None,
                     epochs=200, batch_size=8, lr=1e-4, save_path=None):
    data_mean = train_data.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    data_std = train_data.std(axis=(0, 2), keepdims=True).astype(np.float32)
    data_std = np.maximum(data_std, 1e-6)

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
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if input_ch_indices is not None:
        ch_indices_tensor = torch.tensor(input_ch_indices, device=device)
    else:
        ch_indices_tensor = torch.tensor(range(16), device=device)

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

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

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

    print(f"  Training done. Best loss: {best_loss:.6f}")
    return model, data_mean, data_std


def evaluate_single_subject(model, sid, subject_data_map, input_ch_indices,
                            num_output_channels, num_input_channels, device,
                            data_mean, data_std, DS=2):
    train_128ch_hr = subject_data_map[sid]['train_data']
    train_labels = subject_data_map[sid]['train_labels']
    test_128ch_hr = subject_data_map[sid]['test_data']
    test_labels = subject_data_map[sid]['test_labels']

    train_128ch = train_128ch_hr[:, :, ::DS].astype(np.float32)
    test_128ch = test_128ch_hr[:, :, ::DS].astype(np.float32)
    train_16ch = train_128ch[:, input_ch_indices, :]
    test_16ch = test_128ch[:, input_ch_indices, :]

    clf_128, m128, s128 = train_classifier(train_128ch, train_labels, num_output_channels, device)
    acc_128 = evaluate_classifier(clf_128, test_128ch, test_labels, device, m128, s128)
    del clf_128

    clf_16, m16, s16 = train_classifier(train_16ch, train_labels, num_input_channels, device)
    acc_16 = evaluate_classifier(clf_16, test_16ch, test_labels, device, m16, s16)
    del clf_16

    train_16ch_hr = train_128ch_hr[:, input_ch_indices, :].astype(np.float32)
    gen_128ch_hr = generate_augmented_data(
        model, train_16ch_hr, train_labels, device,
        data_mean=data_mean, data_std=data_std,
        mode='augment', channel_indices=torch.tensor(input_ch_indices, device=device),
        timestep_range=(50, 300), train_noise_scale=0.2,
        guidance_scale=1.0
    )
    gen_128ch_hr[:, input_ch_indices, :] = train_128ch_hr[:, input_ch_indices, :]
    gen_128ch = gen_128ch_hr[:, :, ::DS].astype(np.float32)
    del gen_128ch_hr, train_16ch_hr

    aug_ratio = 0.5
    n_aug = max(1, int(len(train_labels) * aug_ratio))
    rng = np.random.RandomState(42)
    aug_idx = rng.choice(len(gen_128ch), n_aug, replace=False)
    aug_subset = gen_128ch[aug_idx]
    aug_labels_subset = train_labels[aug_idx]
    del gen_128ch

    clf_aug, ma, sa = train_classifier(
        train_128ch, train_labels, num_output_channels, device,
        extra_data=aug_subset, extra_labels=aug_labels_subset
    )
    del aug_subset
    acc_aug = evaluate_classifier(clf_aug, test_128ch, test_labels, device, ma, sa)
    del clf_aug

    torch.cuda.empty_cache()

    return {
        'baseline_128ch': acc_128['accuracy'],
        'baseline_16ch': acc_16['accuracy'],
        'dttd_augmented': acc_aug['accuracy'],
        'kappa_128ch': acc_128['kappa'],
        'kappa_16ch': acc_16['kappa'],
        'kappa_aug': acc_aug['kappa']
    }


def load_main_experiment_results(output_dir):
    """直接读取主实验的结果JSON作为Full Model结果，优先使用summary"""
    result_paths = [
        os.path.join(project_root, 'paper_results', 'hgd_cross_session', 'hgd_cross_session_results.json'),
        os.path.join(project_root, 'paper_results', 'hgd_cross_session_kappa', 'hgd_cross_session_results.json'),
    ]

    for rp in result_paths:
        if os.path.exists(rp):
            print(f"  Loading main experiment results from: {rp}")
            with open(rp, 'r') as f:
                main_results = json.load(f)

            if 'summary' in main_results and 'dttd_augmented' in main_results['summary']:
                summary = main_results['summary']
                aug_values = summary['dttd_augmented']['values']
                baseline128_values = summary['baseline_128ch']['values']
                baseline16_values = summary['baseline_16ch']['values']
                per_subject = {}
                for i, v in enumerate(aug_values):
                    per_subject[str(i + 1)] = float(v)
                return {
                    'mean': float(np.mean(aug_values)),
                    'std': float(np.std(aug_values)),
                    'baseline_128ch_mean': float(np.mean(baseline128_values)),
                    'baseline_16ch_mean': float(np.mean(baseline16_values)),
                    'per_subject': per_subject,
                    'source': rp
                }

            aug_values = []
            baseline128_values = []
            baseline16_values = []
            per_subject = {}

            for sid_str, subj in main_results['per_subject'].items():
                aug_val = subj.get('dttd_augmented')
                if aug_val is not None:
                    if isinstance(aug_val, dict):
                        aug_values.append(aug_val['accuracy'])
                        per_subject[sid_str] = aug_val['accuracy']
                    else:
                        aug_values.append(float(aug_val))
                        per_subject[sid_str] = float(aug_val)
                b128 = subj.get('baseline_128ch')
                if b128 is not None:
                    if isinstance(b128, dict):
                        baseline128_values.append(b128['accuracy'])
                    else:
                        baseline128_values.append(float(b128))
                b16 = subj.get('baseline_16ch')
                if b16 is not None:
                    if isinstance(b16, dict):
                        baseline16_values.append(b16['accuracy'])
                    else:
                        baseline16_values.append(float(b16))

            return {
                'mean': float(np.mean(aug_values)) if aug_values else 0,
                'std': float(np.std(aug_values)) if aug_values else 0,
                'baseline_128ch_mean': float(np.mean(baseline128_values)) if baseline128_values else 0,
                'baseline_16ch_mean': float(np.mean(baseline16_values)) if baseline16_values else 0,
                'per_subject': per_subject,
                'source': rp
            }

    return None


def run_hgd_ablation(data_path, config, device, output_dir, base_seed=42, num_runs=5, dttd_epochs=200):
    print("\n" + "=" * 60)
    print(f"HGD Ablation Study (Cross-Session) - v7")
    print(f"Full Model: 使用主实验checkpoint，统一评估流程")
    print(f"消融变体: 训练{dttd_epochs} epochs, 评估1次")
    print(f"评估流程: DS=2, 单阶段训练100 epochs, 50%增强, best model + scheduler")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    resume_path = os.path.join(output_dir, 'ablation_v9_resume.json')
    if os.path.exists(resume_path):
        with open(resume_path, 'r') as f:
            resume_data = json.load(f)
        print(f"[INFO] Resume file found, completed: {resume_data.get('completed_configs', [])}")
    else:
        resume_data = {'completed_configs': [], 'saved_results': {}}

    all_subject_ids = list(range(1, 15))
    subject_data_map = {}
    input_ch_indices = None
    num_input_channels = 0
    num_output_channels = 0

    print("\nLoading all subject data...")
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
                print(f"  Subject {sid}: train={len(train_ds.data)} trials, test={len(test_ds.data)} trials")
        except Exception as e:
            print(f"  Subject {sid}: Failed - {e}")

    if not subject_data_map:
        raise RuntimeError("No subject data loaded!")

    DS = 2
    sids = sorted(subject_data_map.keys())

    print("\nPreparing training data (all subjects, train sessions)...")
    all_train_data = []
    all_train_labels = []
    for sid in sids:
        data = subject_data_map[sid]['train_data'].astype(np.float32)
        labels = subject_data_map[sid]['train_labels'].astype(np.int64)
        all_train_data.append(data)
        all_train_labels.append(labels)
    all_train_data = np.concatenate(all_train_data, axis=0)
    all_train_labels = np.concatenate(all_train_labels, axis=0)
    print(f"  Total training data: {all_train_data.shape[0]} trials, {all_train_data.shape[1]}ch, {all_train_data.shape[2]} timepoints")

    run_seeds = [base_seed + i * 1000 for i in range(num_runs)]

    ablation_configs = [
        ('full', 'Full Model'),
        ('no_topo', 'No Topology'),
        ('no_freq', 'No Frequency'),
        ('no_task', 'No Task'),
    ]

    all_results = {}

    for model_type, model_name in ablation_configs:
        config_key = model_type
        if config_key in resume_data.get('completed_configs', []):
            print(f"\n[SKIP] {model_name} already completed, loading saved results")
            all_results[model_type] = resume_data['saved_results'].get(config_key, {})
            continue

        print(f"\n{'='*50}")
        print(f"Ablation: {model_name}")
        print(f"{'='*50}")

        model = DTTDPhysioNet(config).to(device)

        main_ckpt_path = os.path.join(project_root, 'paper_results', 'hgd_cross_session', 'dttd_hgd_cs_best.pth')
        if not os.path.exists(main_ckpt_path):
            main_ckpt_path = os.path.join(project_root, 'paper_results', 'hgd_cross_session_kappa', 'dttd_hgd_cs_best.pth')

        if os.path.exists(main_ckpt_path):
            print(f"  Loading MAIN experiment checkpoint: {main_ckpt_path}")
            checkpoint = torch.load(main_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            data_mean = checkpoint['data_mean']
            data_std = checkpoint['data_std']
        else:
            print(f"  ERROR: Main experiment checkpoint not found at {main_ckpt_path}")
            print(f"  Please run the main experiment first!")
            continue

        if model_type != 'full':
            model.ablation_mode = model_type
            print(f"  Ablation mode: {model_type} (disabling module at inference time)")
        else:
            model.ablation_mode = None
            print(f"  Full model (all modules active)")

        all_runs_accs = {sid: [] for sid in sids}
        all_runs_kappas = {sid: [] for sid in sids}
        all_runs_baseline128 = {sid: [] for sid in sids}
        all_runs_baseline16 = {sid: [] for sid in sids}

        for run_idx, seed in enumerate(run_seeds):
            print(f"\n  --- Eval Run {run_idx+1}/{num_runs} (seed={seed}) ---")
            set_seed(seed)

            for sid in sids:
                result = evaluate_single_subject(
                    model, sid, subject_data_map, input_ch_indices,
                    num_output_channels, num_input_channels, device,
                    data_mean, data_std, DS=DS
                )
                all_runs_accs[sid].append(result['dttd_augmented'])
                all_runs_kappas[sid].append(result['kappa_aug'])
                all_runs_baseline128[sid].append(result['baseline_128ch'])
                all_runs_baseline16[sid].append(result['baseline_16ch'])
                print(f"    S{sid}: aug={result['dttd_augmented']*100:.2f}%(k={result['kappa_aug']:.4f}), 128ch={result['baseline_128ch']*100:.2f}%(k={result['kappa_128ch']:.4f}), 16ch={result['baseline_16ch']*100:.2f}%(k={result['kappa_16ch']:.4f})")

        per_subject_mean = {sid: float(np.mean(all_runs_accs[sid])) if all_runs_accs[sid] else 0 for sid in sids}
        per_subject_std = {sid: float(np.std(all_runs_accs[sid])) if len(all_runs_accs[sid]) > 1 else 0 for sid in sids}
        mean_acc = float(np.mean(list(per_subject_mean.values())))
        std_acc = float(np.std(list(per_subject_mean.values())))

        per_subject_kappa_mean = {sid: float(np.mean(all_runs_kappas[sid])) if all_runs_kappas[sid] else 0 for sid in sids}
        kappa_mean = float(np.mean(list(per_subject_kappa_mean.values())))

        baseline128_mean = {sid: float(np.mean(all_runs_baseline128[sid])) if all_runs_baseline128[sid] else 0 for sid in sids}
        baseline128_overall = float(np.mean(list(baseline128_mean.values())))

        result = {
            'name': model_name,
            'per_subject_mean': {f'S{sid}': per_subject_mean[sid] for sid in sids},
            'per_subject_std': {f'S{sid}': per_subject_std[sid] for sid in sids},
            'per_subject_all_runs': {f'S{sid}': all_runs_accs[sid] for sid in sids},
            'mean': mean_acc,
            'std': std_acc,
            'kappa_mean': kappa_mean,
            'num_runs': num_runs,
            'seeds': run_seeds,
            'baseline_128ch_mean': baseline128_overall,
            'use_main_checkpoint': True
        }

        all_results[model_type] = result

        print(f"\n  {model_name} DTTD Augmented ({num_runs} runs): {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")
        print(f"  Baseline 128ch ({num_runs} runs): {baseline128_overall*100:.2f}%")
        for sid in sids:
            accs = all_runs_accs[sid]
            if accs:
                print(f"    S{sid}: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%  (runs: {[f'{a*100:.1f}' for a in accs]})")

        resume_data['completed_configs'].append(config_key)
        resume_data['saved_results'][config_key] = result
        with open(resume_path, 'w') as f:
            json.dump(resume_data, f)

        del model
        torch.cuda.empty_cache()

    filepath = os.path.join(output_dir, 'hgd_ablation_v9_results.json')
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("HGD Ablation Study Summary (v9 - Inference-time Ablation)")
    print("=" * 60)
    print(f"{'Configuration':<25} {'Accuracy (%)':<20} {'Kappa':<10} {'vs 128ch Baseline':<20} {'Module Contrib.'}")
    print("-" * 90)

    full_mean = all_results['full']['mean']
    baseline_mean = all_results['full'].get('baseline_128ch_mean', 0)

    for key in ['full', 'no_topo', 'no_freq', 'no_task']:
        r = all_results[key]
        m = r['mean']
        s = r['std']
        k = r.get('kappa_mean', 0)
        vs_baseline = f"{(m - baseline_mean)*100:+.2f}"
        if key == 'full':
            contrib = "-"
        else:
            contrib = f"{(full_mean - m)*100:+.2f}"
        print(f"{r['name']:<25} {m*100:.2f}+/-{s*100:.2f}       {k:.4f}     {vs_baseline:<20} {contrib}")

    print(f"\nResults saved to: {filepath}")
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='HGD Ablation Study (v7)')
    parser.add_argument('--data-path', default='E:/data/HGD')
    parser.add_argument('--output-dir', default='paper_results/hgd_ablation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-runs', type=int, default=1)
    parser.add_argument('--dttd-epochs', type=int, default=200)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    dataset = HighGammaDataset(args.data_path, subject_ids=[1], sessions='train', fs_target=250)
    num_input_ch = dataset.num_input_channels
    num_output_ch = dataset.num_output_channels
    time_steps = dataset.data.shape[2] // 2
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

    run_hgd_ablation(args.data_path, model_config, device, args.output_dir,
                     base_seed=args.seed, num_runs=args.num_runs, dttd_epochs=args.dttd_epochs)


if __name__ == '__main__':
    main()
