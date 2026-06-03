"""
HGD消融实验 (High Gamma Dataset Ablation Study)

评估DTTD各组件在HGD跨session场景下对分类性能的贡献:
1. 完整模型 (Full Model)
2. 无拓扑模块 (No Topology)
3. 无频率模块 (No Frequency)
4. 无任务条件 (No Task Conditioning)

评估方式与high_gamma_cross_session.py一致：
- 训练集：所有被试的train session（前11个run）
- 测试集：每个被试独立的test session（后2个run）
- 使用EEGNet分类器，两阶段训练（原始数据+增强数据微调）

使用方法:
    python experiments/hgd_ablation.py --data-path E:/data/HGD --checkpoint paper_results/hgd_cross_session/dttd_hgd_cs_best.pth
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
from sklearn.metrics import accuracy_score

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.high_gamma_dataset import HighGammaDataset
from models.dttd_physionet import DTTDPhysioNet
from models.scheduler import SinusoidalPositionEmbeddings
from models.topology import DynamicTopologyModule
from models.frequency import FrequencyAttentionModule
from models.dttd_x0pred import SimpleClassifier
from models.dttd_enhanced_v2 import EnhancedUNet
from utils import get_device


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    return accuracy_score(test_labels, preds)


def generate_augmented_data(model, input_data, labels, device, data_mean=None, data_std=None,
                            mode='augment', timestep_range=(50, 300),
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


def create_ablation_model(model_type, config, device, checkpoint_path=None):
    model_classes = {
        'full': DTTDPhysioNet,
        'no_topo': DTTDPhysioNet_NoTopology,
        'no_freq': DTTDPhysioNet_NoFrequency,
        'no_task': DTTDPhysioNet_NoTask,
    }

    model = model_classes[model_type](config).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                          if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"  Loaded {len(pretrained_dict)}/{len(model_dict)} parameters")

    return model


def run_hgd_ablation(data_path, config, checkpoint_path, device, output_dir,
                     data_mean=None, data_std=None):
    print("\n" + "=" * 60)
    print("HGD Ablation Study (Cross-Session)")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    resume_path = os.path.join(output_dir, 'ablation_resume.json')
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

        model = create_ablation_model(model_type, config, device, checkpoint_path)

        per_subject_accs = []

        for sid in sorted(subject_data_map.keys()):
            train_128ch = subject_data_map[sid]['train_data'][:, :, ::DS]
            train_labels = subject_data_map[sid]['train_labels']
            test_128ch = subject_data_map[sid]['test_data'][:, :, ::DS]
            test_labels = subject_data_map[sid]['test_labels']
            train_16ch = train_128ch[:, input_ch_indices, :]
            test_16ch = test_128ch[:, input_ch_indices, :]

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
            per_subject_accs.append(acc_aug)
            print(f"  Subject {sid}: {acc_aug:.4f}")
            del clf_aug
            torch.cuda.empty_cache()

        mean_acc = float(np.mean(per_subject_accs))
        std_acc = float(np.std(per_subject_accs))

        all_results[model_type] = {
            'name': model_name,
            'per_subject': {f'S{sid}': acc for sid, acc in zip(sorted(subject_data_map.keys()), per_subject_accs)},
            'mean': mean_acc,
            'std': std_acc
        }

        print(f"\n  {model_name} Average: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

        resume_data['completed_configs'].append(config_key)
        resume_data['saved_results'][config_key] = all_results[model_type]
        with open(resume_path, 'w') as f:
            json.dump(resume_data, f, indent=2)

        del model
        torch.cuda.empty_cache()

    baseline_16ch_accs = []
    baseline_128ch_accs = []
    for sid in sorted(subject_data_map.keys()):
        train_128ch = subject_data_map[sid]['train_data'][:, :, ::DS]
        train_labels = subject_data_map[sid]['train_labels']
        test_128ch = subject_data_map[sid]['test_data'][:, :, ::DS]
        test_labels = subject_data_map[sid]['test_labels']
        train_16ch = train_128ch[:, input_ch_indices, :]
        test_16ch = test_128ch[:, input_ch_indices, :]

        clf_16, m16, s16 = train_classifier(train_16ch, train_labels, num_input_channels, device)
        acc_16 = evaluate_classifier(clf_16, test_16ch, test_labels, device, m16, s16)
        baseline_16ch_accs.append(acc_16)
        del clf_16

        clf_128, m128, s128 = train_classifier(train_128ch, train_labels, num_output_channels, device)
        acc_128 = evaluate_classifier(clf_128, test_128ch, test_labels, device, m128, s128)
        baseline_128ch_accs.append(acc_128)
        del clf_128

    all_results['baseline_16ch'] = {
        'name': '16-ch Baseline',
        'per_subject': {f'S{sid}': acc for sid, acc in zip(sorted(subject_data_map.keys()), baseline_16ch_accs)},
        'mean': float(np.mean(baseline_16ch_accs)),
        'std': float(np.std(baseline_16ch_accs))
    }
    all_results['baseline_128ch'] = {
        'name': '128-ch Baseline',
        'per_subject': {f'S{sid}': acc for sid, acc in zip(sorted(subject_data_map.keys()), baseline_128ch_accs)},
        'mean': float(np.mean(baseline_128ch_accs)),
        'std': float(np.std(baseline_128ch_accs))
    }

    filepath = os.path.join(output_dir, 'hgd_ablation_results.json')
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("HGD Ablation Study Summary")
    print("=" * 60)
    print(f"{'Configuration':<25} {'Accuracy (%)':<20} {'vs 16-ch Baseline':<20} {'Module Contrib.'}")
    print("-" * 80)

    baseline_16_mean = all_results['baseline_16ch']['mean']
    full_mean = all_results['full']['mean']

    for key in ['baseline_16ch', 'baseline_128ch', 'full', 'no_topo', 'no_freq', 'no_task']:
        r = all_results[key]
        m = r['mean']
        s = r['std']
        vs_baseline = f"{(m - baseline_16_mean)*100:+.2f}"
        if key == 'full':
            contrib = "-"
        elif key in ['no_topo', 'no_freq', 'no_task']:
            contrib = f"{(full_mean - m)*100:+.2f}"
        else:
            contrib = "-"
        print(f"{r['name']:<25} {m*100:.2f}±{s*100:.2f}       {vs_baseline:<20} {contrib}")

    print(f"\nResults saved to: {filepath}")
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='HGD Ablation Study')
    parser.add_argument('--data-path', default='E:/data/HGD', help='HGD data root directory')
    parser.add_argument('--output-dir', default='paper_results/hgd_ablation')
    parser.add_argument('--checkpoint', default='paper_results/hgd_cross_session/dttd_hgd_cs_best.pth',
                        help='Pretrained DTTD model checkpoint')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
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

    data_mean = None
    data_std = None

    if os.path.exists(args.checkpoint):
        print(f"[INFO] Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'data_mean' in checkpoint:
            data_mean = checkpoint['data_mean']
            data_std = checkpoint['data_std']
            print(f"[OK] Data normalization stats loaded from checkpoint")
    else:
        print(f"[WARN] Checkpoint not found: {args.checkpoint}")
        print("[WARN] Will proceed without data normalization stats")

    run_hgd_ablation(args.data_path, model_config, args.checkpoint, device, args.output_dir,
                     data_mean=data_mean, data_std=data_std)


if __name__ == '__main__':
    main()
