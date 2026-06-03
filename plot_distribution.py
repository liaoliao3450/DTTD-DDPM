#!/usr/bin/env python3
"""
UMAP分布对比图
- generate_combined_tsne: BCI2a + PhysioNet 联合 UMAP (Real vs DTTD)
- generate_method_comparison_tsne: 多方法对比 UMAP
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PHYSIONET_CHANNEL_NAMES_64 = [
    'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'T9', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'O1', 'Oz', 'O2', 'Iz'
]

PHYSIONET_CLASS_NAMES = ['Left Hand', 'Right Hand', 'Both Fists', 'Both Feet']


def load_real_data():
    data_path = 'results/generated_samples_test.npz'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    data = np.load(data_path)
    generated = data['generated']
    targets = data['targets']
    labels = data['labels']
    print(f"加载数据: generated={generated.shape}, targets={targets.shape}, labels={labels.shape}")
    print(f"类别分布: {np.bincount(labels.astype(int))}")
    return generated, targets, labels


def load_physionet_data():
    npy_candidates = [
        'data_cache/physionet_mi_preprocessed.npz',
        'E:/data/PhysioNetMI/physionet_mi_preprocessed.npz',
        'paper_results/physionet_mi/physionet_mi_cache.npz',
    ]
    npy_path = None
    for p in npy_candidates:
        if os.path.exists(p):
            npy_path = p
            break

    if npy_path is not None:
        print(f"从缓存加载PhysioNet数据: {npy_path}")
        npz = np.load(npy_path, allow_pickle=True)
        if 'subject_ids' in npz:
            data_arr = npz['data']
            labels_arr = npz['labels']
            sid_arr = npz['subject_ids']
            all_data_list = []
            all_labels_list = []
            all_subject_ids = []
            for i, sid_str in enumerate(sid_arr):
                sid_int = int(str(sid_str).lstrip('S').lstrip('0') or '0')
                n_trials = len(data_arr[i])
                all_data_list.append(data_arr[i])
                all_labels_list.append(labels_arr[i])
                all_subject_ids.extend([sid_int] * n_trials)
            all_data = np.concatenate(all_data_list, axis=0)
            all_labels = np.concatenate(all_labels_list, axis=0)
            all_subject_ids = np.array(all_subject_ids, dtype=np.int64)
        else:
            all_data = npz['data']
            all_labels = npz['labels']
            all_subject_ids = None
        print(f"PhysioNet数据: shape={all_data.shape}, labels shape={all_labels.shape}")
        print(f"类别分布: {np.bincount(all_labels.astype(int))}")
        return all_data, all_labels, all_subject_ids

    print("未找到PhysioNet缓存文件，正在运行 cache_physionet_fast.py 生成...")
    import subprocess
    subprocess.run([sys.executable, 'experiments/cache_physionet_fast.py'], check=True)
    cache_path = 'paper_results/physionet_mi/physionet_mi_cache.npz'
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"缓存生成失败: {cache_path}")
    data = np.load(cache_path)
    return data['data'], data['labels'], None


def generate_combined_tsne(output_dir, bci_generated, bci_targets, bci_labels,
                            phys_data, phys_labels):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    import torch
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    from sklearn.decomposition import PCA
    import umap
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    N_PER_CLASS = 200
    DATA_SCALE = 1e5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def flatten_data(data, max_time=200):
        n = data.shape[0]
        n_ch = data.shape[1]
        t = data.shape[2]
        if t > max_time:
            step = t // max_time
            data_ds = data[:, :, ::step][:, :, :max_time]
        else:
            data_ds = data
        flat = data_ds.reshape(n, -1)
        return flat

    print("  [BCI2a] Preparing raw data for UMAP...")
    bci_real_list, bci_gen_list, bci_label_list, bci_source_list = [], [], [], []
    for cls in range(4):
        mask = bci_labels == cls
        indices = np.where(mask)[0]
        if len(indices) > N_PER_CLASS:
            indices = np.random.choice(indices, N_PER_CLASS, replace=False)
        bci_real_list.append(bci_targets[indices])
        bci_gen_list.append(bci_generated[indices])
        bci_label_list.extend([cls] * len(indices))
        bci_source_list.extend(['Real'] * len(indices))
        bci_label_list.extend([cls] * len(indices))
        bci_source_list.extend(['DTTD'] * len(indices))

    bci_real_all = np.concatenate(bci_real_list)
    bci_gen_all = np.concatenate(bci_gen_list)
    bci_all_labels = np.array(bci_label_list)
    bci_all_source = np.array(bci_source_list)

    bci_real_flat = flatten_data(bci_real_all)
    bci_gen_flat = flatten_data(bci_gen_all)
    bci_all_flat = np.concatenate([bci_real_flat, bci_gen_flat])

    bci_ch_mean = bci_all_flat.mean(axis=0, keepdims=True)
    bci_ch_std = bci_all_flat.std(axis=0, keepdims=True) + 1e-8
    bci_all_norm = (bci_all_flat - bci_ch_mean) / bci_ch_std

    print("  [BCI2a] Computing PCA + UMAP on raw data...")
    bci_pca = PCA(n_components=min(50, bci_all_norm.shape[1] - 1), random_state=42)
    bci_all_pca = bci_pca.fit_transform(bci_all_norm)
    bci_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    bci_emb = bci_reducer.fit_transform(bci_all_pca)

    print("  [PhysioNet] Preparing raw data for UMAP...")
    phys_real_list, phys_gen_list, phys_label_list, phys_source_list = [], [], [], []

    ckpt_path = 'paper_results/physionet_mi/dttd_physionet_best.pth'
    dttd_model = None
    data_mean_ckpt = None
    data_std_ckpt = None
    if os.path.exists(ckpt_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from models.dttd_physionet import DTTDPhysioNet
        model_config = {
            'input_channels': 16, 'output_channels': 64, 'time_steps': 640,
            'embed_dim': 256, 'task_dim': 64, 'num_classes': 4,
            'num_heads': 8, 'dropout': 0.1,
            'num_timesteps': 1000, 'beta_start': 1e-4, 'beta_end': 0.02,
            'schedule_type': 'linear', 'fs': 160, 'use_classifier_guidance': False
        }
        model = DTTDPhysioNet(model_config).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        data_mean_ckpt = checkpoint.get('data_mean', None)
        data_std_ckpt = checkpoint.get('data_std', None)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        model.eval()
        dttd_model = model
        print(f"  [PhysioNet] DTTD模型加载成功 (device={device})")

    INPUT_CHANNELS_16 = [
        'FC3', 'FC1', 'FC2', 'FC4',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4'
    ]
    input_ch_indices = [PHYSIONET_CHANNEL_NAMES_64.index(ch) for ch in INPUT_CHANNELS_16]

    phys_gen_raw_list = []
    for cls in range(4):
        mask = phys_labels == cls
        indices = np.where(mask)[0]
        if len(indices) > N_PER_CLASS:
            indices = np.random.choice(indices, N_PER_CLASS, replace=False)

        real_data = phys_data[indices]
        phys_real_list.append(real_data)
        phys_label_list.extend([cls] * len(indices))
        phys_source_list.extend(['Real'] * len(indices))

        if dttd_model is not None:
            gen_list = []
            batch_inputs = []
            batch_labels = []
            for idx in indices:
                real_sample = phys_data[idx:idx+1]
                label_sample = phys_labels[idx:idx+1]
                input_data = real_sample[0, input_ch_indices, :]
                ch_idx_np = np.array(input_ch_indices)
                if data_mean_ckpt is not None and data_std_ckpt is not None:
                    dttd_input_mean = data_mean_ckpt[:, ch_idx_np, :]
                    dttd_input_std = data_std_ckpt[:, ch_idx_np, :]
                    scaled_input = ((input_data[np.newaxis] - dttd_input_mean) / dttd_input_std).astype(np.float32)
                else:
                    scaled_input = (input_data[np.newaxis] * DATA_SCALE).astype(np.float32)
                batch_inputs.append(scaled_input[0])
                batch_labels.append(int(label_sample[0]))

            batch_input_t = torch.FloatTensor(np.stack(batch_inputs)).to(device)
            batch_label_t = torch.LongTensor(batch_labels).to(device)
            ch_indices_t = torch.LongTensor(input_ch_indices).to(device)

            gen_chunks = []
            chunk_size = 32
            for c_start in range(0, len(batch_input_t), chunk_size):
                c_end = min(c_start + chunk_size, len(batch_input_t))
                c_input = batch_input_t[c_start:c_end]
                c_label = batch_label_t[c_start:c_end]
                with torch.no_grad():
                    c_gen = dttd_model.sample_ddim(
                        c_input, task_label=c_label,
                        num_inference_steps=50, eta=0.0,
                        guidance_scale=3.0, channel_indices=ch_indices_t
                    )
                    gen_chunks.append(c_gen.cpu().numpy())
                torch.cuda.empty_cache()
            gen_all = np.concatenate(gen_chunks, axis=0)

            if data_mean_ckpt is not None and data_std_ckpt is not None:
                gen_all = gen_all * data_std_ckpt[0] + data_mean_ckpt[0]
            else:
                gen_all = gen_all / DATA_SCALE

            for i, idx in enumerate(indices):
                gen_np = gen_all[i].copy()
                gen_np[input_ch_indices] = phys_data[idx, input_ch_indices, :]
                gen_list.append(gen_np)

            if gen_list:
                gen_stack = np.stack(gen_list)
                phys_gen_raw_list.append(gen_stack)
                phys_label_list.extend([cls] * len(gen_list))
                phys_source_list.extend(['DTTD'] * len(gen_list))

    phys_real_all = np.concatenate(phys_real_list)
    if phys_gen_raw_list:
        phys_gen_all = np.concatenate(phys_gen_raw_list)
    else:
        phys_gen_all = None
    phys_all_labels = np.array(phys_label_list)
    phys_all_source = np.array(phys_source_list)

    phys_real_flat = flatten_data(phys_real_all)
    if phys_gen_all is not None:
        phys_gen_flat = flatten_data(phys_gen_all)
        phys_all_flat = np.concatenate([phys_real_flat, phys_gen_flat])
    else:
        phys_all_flat = phys_real_flat

    phys_ch_mean = phys_all_flat.mean(axis=0, keepdims=True)
    phys_ch_std = phys_all_flat.std(axis=0, keepdims=True) + 1e-8
    phys_all_norm = (phys_all_flat - phys_ch_mean) / phys_ch_std

    print("  [PhysioNet] Computing PCA + UMAP on raw data...")
    phys_pca = PCA(n_components=min(50, phys_all_norm.shape[1] - 1), random_state=42)
    phys_all_pca = phys_pca.fit_transform(phys_all_norm)
    phys_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    phys_emb = phys_reducer.fit_transform(phys_all_pca)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))

    class_colors = ['#D32F2F', '#1565C0', '#2E7D32', '#F57F17']
    class_names_bci = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    class_names_phys = PHYSIONET_CLASS_NAMES

    for cls in range(4):
        real_mask = (bci_all_labels == cls) & (bci_all_source == 'Real')
        gen_mask = (bci_all_labels == cls) & (bci_all_source == 'DTTD')
        axes[0].scatter(bci_emb[real_mask, 0], bci_emb[real_mask, 1],
                       c=class_colors[cls], marker='o', alpha=0.2, s=25, edgecolors='none')
        axes[0].scatter(bci_emb[gen_mask, 0], bci_emb[gen_mask, 1],
                       c=class_colors[cls], marker='^', alpha=0.4, s=35, edgecolors='none')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=9, label='Real'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=9, label='DTTD'),
    ]
    for cls in range(4):
        legend_elements.append(
            Line2D([0], [0], marker='s', color='w', markerfacecolor=class_colors[cls],
                   markersize=9, label=class_names_bci[cls])
        )
    axes[0].legend(handles=legend_elements, fontsize=13, loc='best', framealpha=0.9, ncol=2)
    axes[0].set_title('BCI2a', fontsize=18, fontweight='bold')
    axes[0].set_xlabel('UMAP 1', fontsize=15)
    axes[0].set_ylabel('UMAP 2', fontsize=15)
    axes[0].tick_params(labelsize=14)
    axes[0].grid(True, alpha=0.15)

    for cls in range(4):
        real_mask = (phys_all_labels == cls) & (phys_all_source == 'Real')
        gen_mask = (phys_all_labels == cls) & (phys_all_source == 'DTTD')
        axes[1].scatter(phys_emb[real_mask, 0], phys_emb[real_mask, 1],
                       c=class_colors[cls], marker='o', alpha=0.2, s=25, edgecolors='none')
        if np.any(gen_mask):
            axes[1].scatter(phys_emb[gen_mask, 0], phys_emb[gen_mask, 1],
                           c=class_colors[cls], marker='^', alpha=0.4, s=35, edgecolors='none')

    phys_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=9, label='Real'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=9, label='DTTD'),
    ]
    for cls in range(4):
        phys_legend.append(
            Line2D([0], [0], marker='s', color='w', markerfacecolor=class_colors[cls],
                   markersize=9, label=class_names_phys[cls])
        )
    axes[1].legend(handles=phys_legend, fontsize=13, loc='best', framealpha=0.9, ncol=2)
    axes[1].set_title('PhysioNet', fontsize=18, fontweight='bold')
    axes[1].set_xlabel('UMAP 1', fontsize=15)
    axes[1].set_ylabel('UMAP 2', fontsize=15)
    axes[1].tick_params(labelsize=14)
    axes[1].grid(True, alpha=0.15)

    fig.suptitle('UMAP Distribution: Real vs DTTD', fontsize=20, fontweight='bold')
    plt.tight_layout()

    output_path_png = os.path.join(output_dir, 'combined_umap.png')
    output_path_pdf = os.path.join(output_dir, 'combined_umap.pdf')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path_png}")
    plt.close()


def generate_method_comparison_tsne(output_dir, bci_targets, bci_labels, bci_generated=None):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    import torch
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import umap

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    class_colors = ['#D32F2F', '#1565C0', '#2E7D32', '#F57F17']
    INPUT_CH = [7, 9, 11, 1, 3, 5, 13, 15, 17]
    DATA_SCALE = 1e5

    METHODS_CONFIG = {
        'DTTD': {
            'checkpoint': 'checkpoints/bci2a_enhanced/best_model.pth',
            'color': '#E53935',
        },
        'CVAE': {
            'checkpoint': 'checkpoints/bci2a/baseline_cvae/best_model.pth',
            'color': '#7B1FA2',
        },
        'cGAN': {
            'checkpoint': 'checkpoints/bci2a_enhanced/baseline_cgan/best_model.pth',
            'color': '#1E88E5',
        },
        'EEGDiff': {
            'checkpoint': 'checkpoints/bci2a/baseline_eegdiff/best_model.pth',
            'color': '#FF8F00',
        },
        'BrainDiff': {
            'checkpoint': 'checkpoints/bci2a/baseline_braindiff/best_model.pth',
            'color': '#00838F',
        },
        'Spline': {
            'color': '#AB47BC',
            'type': 'interpolation',
        },
        'Kriging': {
            'color': '#78909C',
            'type': 'interpolation',
        },
    }

    N_PER_CLASS = 150

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from models import DTTDEnhanced
    from models.baselines import CVAE, Generator, EEGDiff, BrainDiff
    from models.classifier import EEGNet as EEGNetClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score

    class CVAE_Large(nn.Module):
        def __init__(self, input_channels, output_channels, time_steps, num_classes, latent_dim=128, pool_size=256):
            super().__init__()
            self.input_channels = input_channels
            self.output_channels = output_channels
            self.time_steps = time_steps
            self.num_classes = num_classes
            self.latent_dim = latent_dim
            self.encoder = nn.Sequential(
                nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(128), nn.ReLU(),
                nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(256), nn.ReLU(),
                nn.AdaptiveAvgPool1d(pool_size)
            )
            self.condition_embed = nn.Embedding(num_classes, 64)
            enc_out_dim = 256 * pool_size
            self.fc_mu = nn.Linear(enc_out_dim + 64, latent_dim)
            self.fc_logvar = nn.Linear(enc_out_dim + 64, latent_dim)
            self.decoder_fc = nn.Linear(latent_dim + 64, 256 * 64)
            self.decoder_conv = nn.Sequential(
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256), nn.ReLU(),
                nn.ConvTranspose1d(256, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.BatchNorm1d(256), nn.ReLU(),
                nn.ConvTranspose1d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.BatchNorm1d(128), nn.ReLU(),
                nn.Conv1d(128, output_channels, kernel_size=3, padding=1),
            )

        def forward(self, x, condition):
            h = self.encoder(x)
            h = h.view(h.size(0), -1)
            c = self.condition_embed(condition)
            h = torch.cat([h, c], dim=1)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            z_c = torch.cat([z, c], dim=1)
            h = self.decoder_fc(z_c)
            h = h.view(h.size(0), 256, -1)
            h = self.decoder_conv(h)
            if h.size(2) > self.time_steps:
                h = h[:, :, :self.time_steps]
            elif h.size(2) < self.time_steps:
                h = F.pad(h, (0, self.time_steps - h.size(2)))
            return h, mu, logvar

    class Generator_Large(nn.Module):
        def __init__(self, input_channels, output_channels, time_steps, num_classes, latent_dim=128):
            super().__init__()
            self.input_channels = input_channels
            self.output_channels = output_channels
            self.time_steps = time_steps
            self.latent_dim = latent_dim
            self.input_encoder = nn.Sequential(
                nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(128), nn.ReLU(),
                nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(256), nn.ReLU(),
                nn.AdaptiveAvgPool1d(16)
            )
            self.condition_embed = nn.Embedding(num_classes, 64)
            input_feature_dim = 256 * 16
            self.fc = nn.Linear(latent_dim + 64 + input_feature_dim, 256 * 64)
            self.deconv = nn.Sequential(
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256), nn.ReLU(),
                nn.ConvTranspose1d(256, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.BatchNorm1d(256), nn.ReLU(),
                nn.ConvTranspose1d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.BatchNorm1d(128), nn.ReLU(),
                nn.Conv1d(128, output_channels, kernel_size=3, padding=1),
                nn.Tanh()
            )

        def forward(self, z, condition, input_data=None):
            batch_size = z.size(0)
            c = self.condition_embed(condition)
            if input_data is not None:
                input_features = self.input_encoder(input_data)
                input_features = input_features.view(batch_size, -1)
            else:
                input_features = torch.zeros(batch_size, 256 * 16).to(z.device)
            x = torch.cat([z, c, input_features], dim=1)
            x = self.fc(x)
            x = x.view(batch_size, 256, -1)
            x = self.deconv(x)
            if x.size(2) > self.time_steps:
                x = x[:, :, :self.time_steps]
            elif x.size(2) < self.time_steps:
                x = F.pad(x, (0, self.time_steps - x.size(2)))
            return x

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("  Training EEGNet classifier on BCI2a...")
    bci_clf = EEGNetClassifier(num_channels=22, num_classes=4, time_steps=1000).to(device)
    ch_mean = bci_targets.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    ch_std = bci_targets.std(axis=(0, 2), keepdims=True).astype(np.float32) + 1e-8
    normed = ((bci_targets - ch_mean) / ch_std).astype(np.float32)
    opt = torch.optim.Adam(bci_clf.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
    criterion = nn.CrossEntropyLoss()
    n = len(bci_targets)
    bci_clf.train()
    for ep in range(100):
        perm = np.random.permutation(n)
        for start in range(0, n, 64):
            idx = perm[start:start+64]
            data_t = torch.FloatTensor(normed[idx]).to(device)
            labels_t = torch.LongTensor(bci_labels[idx]).to(device)
            opt.zero_grad()
            loss = criterion(bci_clf(data_t), labels_t)
            loss.backward()
            opt.step()
        scheduler.step()
    bci_clf.eval()
    with torch.no_grad():
        all_pred = []
        for start in range(0, n, 256):
            data_t = torch.FloatTensor(normed[start:start+256]).to(device)
            all_pred.append(bci_clf(data_t).argmax(dim=1).cpu().numpy())
    acc = (np.concatenate(all_pred) == bci_labels).mean()
    print(f"  EEGNet accuracy: {acc:.4f}")

    def extract_features(clf, data, ch_m, ch_s, batch_size=128):
        feats = []
        for i in range(0, len(data), batch_size):
            batch = ((data[i:i+batch_size] - ch_m) / ch_s).astype(np.float32)
            data_t = torch.FloatTensor(batch).to(device)
            with torch.no_grad():
                x = data_t.unsqueeze(1)
                x = clf.conv1(x)
                x = clf.batchnorm1(x)
                x = clf.depthwise(x)
                x = clf.batchnorm2(x)
                x = clf.activation1(x)
                x = clf.pooling1(x)
                x = clf.dropout1(x)
                x = clf.separable(x)
                x = clf.batchnorm3(x)
                x = clf.activation2(x)
                x = clf.pooling2(x)
                x = clf.dropout2(x)
                x = x.view(x.size(0), -1)
                feats.append(x.cpu().numpy())
        return np.concatenate(feats)

    real_list, real_label_list = [], []
    for cls in range(4):
        mask = bci_labels == cls
        indices = np.where(mask)[0]
        if len(indices) > N_PER_CLASS:
            indices = np.random.choice(indices, N_PER_CLASS, replace=False)
        real_feat = extract_features(bci_clf, bci_targets[indices], ch_mean, ch_std)
        real_list.append(real_feat)
        real_label_list.extend([cls] * len(indices))
    real_feat = np.concatenate(real_list)
    real_labels = np.array(real_label_list)

    gen_raw_dict = {}
    gen_labels_dict = {}

    for method_name, cfg in METHODS_CONFIG.items():
        method_type = cfg.get('type', 'model')
        ckpt_path = cfg.get('checkpoint', None)

        if method_type == 'model' and ckpt_path and not os.path.exists(ckpt_path):
            print(f"  [跳过] {method_name}: checkpoint不存在 ({ckpt_path})")
            continue

        if method_type == 'interpolation':
            try:
                BCI2A_CHANNELS = [
                    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                    'P1', 'Pz', 'P2', 'POz'
                ]
                electrode_2d = {
                    'Fz': (0.0, 0.3), 'FC3': (-0.3, 0.2), 'FC1': (-0.1, 0.2),
                    'FCz': (0.0, 0.2), 'FC2': (0.1, 0.2), 'FC4': (0.3, 0.2),
                    'C5': (-0.4, 0.0), 'C3': (-0.3, 0.0), 'C1': (-0.1, 0.0),
                    'Cz': (0.0, 0.0), 'C2': (0.1, 0.0), 'C4': (0.3, 0.0),
                    'C6': (0.4, 0.0), 'CP3': (-0.3, -0.2), 'CP1': (-0.1, -0.2),
                    'CPz': (0.0, -0.2), 'CP2': (0.1, -0.2), 'CP4': (0.3, -0.2),
                    'P1': (-0.1, -0.3), 'Pz': (0.0, -0.3), 'P2': (0.1, -0.3),
                    'POz': (0.0, -0.4),
                }

                all_indices = []
                all_lbl = []
                for cls in range(4):
                    mask = bci_labels == cls
                    indices = np.where(mask)[0][:N_PER_CLASS]
                    all_indices.extend(indices)
                    all_lbl.extend([cls] * len(indices))
                sel_data = bci_targets[all_indices]
                n_samples, n_total_ch, n_time = sel_data.shape
                input_data = sel_data[:, INPUT_CH, :]

                if method_name == 'Spline':
                    from scipy.interpolate import CubicSpline
                    sorted_order = np.argsort(INPUT_CH)
                    sorted_indices = np.array(INPUT_CH)[sorted_order]
                    all_positions = np.arange(n_total_ch, dtype=np.float64)
                    result = np.zeros((n_samples, n_total_ch, n_time), dtype=np.float32)
                    for i in range(n_samples):
                        for t in range(n_time):
                            values = input_data[i, sorted_order, t]
                            cs = CubicSpline(sorted_indices.astype(np.float64), values, bc_type='natural')
                            result[i, :, t] = cs(all_positions)
                    gen_np = result

                elif method_name == 'Kriging':
                    all_coords = np.array([electrode_2d[BCI2A_CHANNELS[i]] for i in range(n_total_ch)])
                    input_coords = np.array([electrode_2d[BCI2A_CHANNELS[i]] for i in INPUT_CH])
                    n_input = len(INPUT_CH)
                    sigma = 0.3
                    dists = np.zeros((n_total_ch, n_input))
                    for i in range(n_total_ch):
                        for j in range(n_input):
                            dists[i, j] = np.sum((all_coords[i] - input_coords[j]) ** 2)
                    weights = np.exp(-dists / (2 * sigma ** 2))
                    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-10)
                    result = np.zeros((n_samples, n_total_ch, n_time), dtype=np.float32)
                    for i in range(n_samples):
                        for t in range(0, n_time, 50):
                            t_end = min(t + 50, n_time)
                            vals = input_data[i, :, t:t_end]
                            result[i, :, t:t_end] = weights @ vals
                    gen_np = result

                gen_lbl = all_lbl
                gen_raw_dict[method_name] = gen_np
                gen_labels_dict[method_name] = np.array(gen_lbl)
                print(f"  [OK] {method_name}: {gen_np.shape}")
            except Exception as e:
                import traceback
                print(f"  [错误] {method_name}: {e}")
                traceback.print_exc()
            continue

        try:
            if method_name == 'DTTD':
                gen_list = []
                gen_lbl = []
                real_indices_list = []
                for cls in range(4):
                    mask = bci_labels == cls
                    indices = np.where(mask)[0][:N_PER_CLASS]
                    for idx in indices:
                        gen = bci_generated[idx].copy()
                        gen[INPUT_CH, :] = bci_targets[idx, INPUT_CH, :]
                        gen_list.append(gen)
                        gen_lbl.append(cls)
                gen_np = np.stack(gen_list)

            elif method_name == 'CVAE':
                checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                model = CVAE(input_channels=9, output_channels=22, time_steps=1000,
                             num_classes=4, latent_dim=128, pool_size=256).to(device)
                model.decoder_fc = nn.Sequential(
                    nn.Linear(128 + 64, 512),
                    nn.ReLU(),
                    nn.Linear(512, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 22000),
                ).to(device)
                model.decoder_conv = None
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                model.eval()

                gen_list = []
                gen_lbl = []
                for cls in range(4):
                    mask = bci_labels == cls
                    indices = np.where(mask)[0][:N_PER_CLASS]
                    for idx in indices:
                        real_22 = bci_targets[idx:idx+1]
                        input_9ch = real_22[:, INPUT_CH, :]
                        batch_input = torch.tensor(input_9ch, dtype=torch.float32).to(device)
                        batch_label = torch.tensor([cls], dtype=torch.long).to(device)
                        with torch.no_grad():
                            gen_output, _, _ = model(batch_input, batch_label)
                            gen_np = gen_output.cpu().numpy()[0]
                        gen_np[INPUT_CH, :] = real_22[0, INPUT_CH, :]
                        gen_list.append(gen_np)
                        gen_lbl.append(cls)
                gen_np = np.stack(gen_list)

            elif method_name == 'cGAN':
                checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                model = Generator(input_channels=9, output_channels=22, time_steps=1000,
                                  num_classes=4, latent_dim=128).to(device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                model.eval()

                gen_list = []
                gen_lbl = []
                for cls in range(4):
                    mask = bci_labels == cls
                    indices = np.where(mask)[0][:N_PER_CLASS]
                    for idx in indices:
                        real_22 = bci_targets[idx:idx+1]
                        input_9ch = real_22[:, INPUT_CH, :]
                        batch_input = torch.tensor(input_9ch, dtype=torch.float32).to(device)
                        batch_label = torch.tensor([cls], dtype=torch.long).to(device)
                        z = torch.randn(1, 128).to(device)
                        with torch.no_grad():
                            gen_output = model(z, batch_label, batch_input)
                            gen_np = gen_output.cpu().numpy()[0]
                        gen_np[INPUT_CH, :] = real_22[0, INPUT_CH, :]
                        gen_list.append(gen_np)
                        gen_lbl.append(cls)
                gen_np = np.stack(gen_list)

            elif method_name == 'EEGDiff':
                checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                model = EEGDiff(input_channels=9, output_channels=22, time_steps=1000,
                                num_classes=4).to(device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                model.eval()

                gen_list = []
                gen_lbl = []
                for cls in range(4):
                    mask = bci_labels == cls
                    indices = np.where(mask)[0][:N_PER_CLASS]
                    for idx in indices:
                        real_22 = bci_targets[idx:idx+1]
                        input_9ch = real_22[:, INPUT_CH, :]
                        batch_input = torch.tensor(input_9ch, dtype=torch.float32).to(device)
                        batch_label = torch.tensor([cls], dtype=torch.long).to(device)
                        with torch.no_grad():
                            gen_output = model.sample(batch_input, task_label=batch_label,
                                                      num_steps=1, noise_scale=0.02, guidance_scale=3.0)
                            gen_np = gen_output.cpu().numpy()[0]
                        gen_np[INPUT_CH, :] = real_22[0, INPUT_CH, :]
                        gen_list.append(gen_np)
                        gen_lbl.append(cls)
                gen_np = np.stack(gen_list)

            elif method_name == 'BrainDiff':
                checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                model = BrainDiff(input_channels=9, output_channels=22, time_steps=1000,
                                  num_classes=4).to(device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                model.eval()

                gen_list = []
                gen_lbl = []
                for cls in range(4):
                    mask = bci_labels == cls
                    indices = np.where(mask)[0][:N_PER_CLASS]
                    for idx in indices:
                        real_22 = bci_targets[idx:idx+1]
                        input_9ch = real_22[:, INPUT_CH, :]
                        batch_input = torch.tensor(input_9ch, dtype=torch.float32).to(device)
                        with torch.no_grad():
                            gen_output = model.sample(batch_input, num_steps=1, noise_scale=0.02)
                            gen_np = gen_output.cpu().numpy()[0]
                        gen_np[INPUT_CH, :] = real_22[0, INPUT_CH, :]
                        gen_list.append(gen_np)
                        gen_lbl.append(cls)
                gen_np = np.stack(gen_list)

            gen_raw_dict[method_name] = gen_np
            gen_labels_dict[method_name] = np.array(gen_lbl)
            print(f"  [OK] {method_name}: {gen_np.shape}")
        except Exception as e:
            import traceback
            print(f"  [错误] {method_name}: {e}")
            traceback.print_exc()

    if not gen_raw_dict:
        print("  [警告] 没有可用的生成方法，跳过t-SNE对比图")
        return

    available_methods = list(gen_raw_dict.keys())
    n_methods = len(available_methods)

    print("  Extracting EEGNet features for generated data...")
    gen_feat_dict = {}
    for m in available_methods:
        gen_feat_dict[m] = extract_features(bci_clf, gen_raw_dict[m], ch_mean, ch_std)
        print(f"    {m}: {gen_feat_dict[m].shape}")

    n_pca = min(50, real_feat.shape[1] - 1)
    print("  Computing PCA + UMAP (joint)...")
    pca = PCA(n_components=n_pca, random_state=42)

    all_feat_list = [real_feat]
    all_labels_list = [real_labels]
    all_method_list = [['Real'] * len(real_labels)]
    for m in available_methods:
        all_feat_list.append(gen_feat_dict[m])
        all_labels_list.append(gen_labels_dict[m])
        all_method_list.append([m] * len(gen_labels_dict[m]))

    all_feat = np.concatenate(all_feat_list)
    all_labels = np.concatenate(all_labels_list)
    all_methods = np.concatenate(all_method_list)

    all_pca = pca.fit_transform(all_feat)
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    emb = reducer.fit_transform(all_pca)

    n_cols = min(n_methods + 1, 4)
    n_rows = (n_methods + n_cols) // n_cols

    def compute_knn_accuracy(embeddings, labels, n_neighbors=5):
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        scores = cross_val_score(knn, embeddings, labels, cv=3, scoring='accuracy')
        return scores.mean()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    real_emb = emb[all_methods == 'Real']
    real_lbl = all_labels[all_methods == 'Real']

    for idx, method_name in enumerate(['Real'] + available_methods):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        for cls in range(4):
            real_cls_mask = (real_lbl == cls)
            rx, ry = real_emb[real_cls_mask, 0], real_emb[real_cls_mask, 1]
            ax.scatter(rx, ry, c=class_colors[cls], marker='o', alpha=0.15, s=20, edgecolors='none')
            r_cx, r_cy = np.mean(rx), np.mean(ry)
            ax.scatter([r_cx], [r_cy], c=class_colors[cls], marker='*', s=150,
                      edgecolors='black', linewidths=0.5, zorder=5)

        if method_name != 'Real':
            method_emb = emb[all_methods == method_name]
            method_lbl = all_labels[all_methods == method_name]
            for cls in range(4):
                cls_mask = (method_lbl == cls)
                mx, my = method_emb[cls_mask, 0], method_emb[cls_mask, 1]
                ax.scatter(mx, my, c=class_colors[cls], marker='^', alpha=0.4, s=30,
                          edgecolors='none')
                m_cx, m_cy = np.mean(mx), np.mean(my)
                ax.scatter([m_cx], [m_cy], c=class_colors[cls], marker='P', s=100,
                          edgecolors='black', linewidths=0.5, zorder=5)
                r_cls_mask = (real_lbl == cls)
                r_cx = np.mean(real_emb[r_cls_mask, 0])
                r_cy = np.mean(real_emb[r_cls_mask, 1])
                ax.plot([r_cx, m_cx], [r_cy, m_cy], color=class_colors[cls],
                       linewidth=1.5, linestyle='--', alpha=0.5)

            def compute_cross_eegnet(clf, gen_data, gen_labels, ch_m, ch_s, batch_size=128):
                all_pred = []
                for i in range(0, len(gen_data), batch_size):
                    batch = ((gen_data[i:i+batch_size] - ch_m) / ch_s).astype(np.float32)
                    data_t = torch.FloatTensor(batch).to(device)
                    with torch.no_grad():
                        pred = clf(data_t).argmax(dim=1).cpu().numpy()
                        all_pred.append(pred)
                return (np.concatenate(all_pred) == gen_labels).mean()

            cross_acc = compute_cross_eegnet(bci_clf, gen_raw_dict[method_name], method_lbl, ch_mean, ch_std)
            print(f'    {method_name} Cross EEGNet: {cross_acc:.4f}')
            ax.text(0.02, 0.98, f'Cross EEGNet: {cross_acc:.3f}', transform=ax.transAxes,
                   fontsize=14, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8, edgecolor='gray'))
        else:
            ax.text(0.02, 0.98, f'EEGNet Acc: {acc:.3f}', transform=ax.transAxes,
                   fontsize=14, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8, edgecolor='gray'))

        ax.set_title(method_name, fontsize=18, fontweight='bold')
        ax.set_xlabel('UMAP 1', fontsize=15)
        ax.set_ylabel('UMAP 2', fontsize=15)
        ax.grid(True, alpha=0.15)
        ax.tick_params(labelsize=13)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Real'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, label='Generated'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markersize=10, label='Real Center'),
        Line2D([0], [0], marker='P', color='w', markerfacecolor='gray', markersize=8, label='Gen Center'),
    ]
    for cls in range(4):
        legend_elements.append(
            Line2D([0], [0], marker='s', color='w', markerfacecolor=class_colors[cls],
                   markersize=8, label=class_names[cls])
        )
    axes[0, 0].legend(handles=legend_elements, fontsize=12, loc='best', framealpha=0.9, ncol=2)

    for idx in range(n_methods + 1, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle('UMAP in EEGNet Feature Space', fontsize=22, fontweight='bold')
    plt.tight_layout()

    output_path_png = os.path.join(output_dir, 'method_comparison_umap.png')
    output_path_pdf = os.path.join(output_dir, 'method_comparison_umap.pdf')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path_png}")
    plt.close()


def main():
    import argparse
    np.random.seed(42)
    import torch
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    parser = argparse.ArgumentParser(description='Generate UMAP Distribution Figures')
    parser.add_argument('--dataset', choices=['bci2a', 'physionet', 'all'], default='all',
                       help='Which dataset to generate figures for')
    args = parser.parse_args()

    output_dir = 'paper/figures'
    os.makedirs(output_dir, exist_ok=True)

    bci_generated, bci_targets, bci_labels = None, None, None
    phys_data, phys_labels, phys_subject_ids = None, None, None

    if args.dataset in ('bci2a', 'all'):
        print("=" * 60)
        print("Loading BCI2a data...")
        print("=" * 60)
        bci_generated, bci_targets, bci_labels = load_real_data()

    if args.dataset in ('physionet', 'all'):
        print("\n" + "=" * 60)
        print("Loading PhysioNet data...")
        print("=" * 60)
        phys_data, phys_labels, phys_subject_ids = load_physionet_data()

    if bci_generated is not None and phys_data is not None:
        print("\n" + "=" * 60)
        print("Combined t-SNE - BCI2a + PhysioNet")
        print("=" * 60)
        generate_combined_tsne(output_dir, bci_generated, bci_targets, bci_labels,
                               phys_data, phys_labels)

    if bci_targets is not None:
        print("\n" + "=" * 60)
        print("Method Comparison t-SNE - BCI2a")
        print("=" * 60)
        generate_method_comparison_tsne(output_dir, bci_targets, bci_labels, bci_generated)

    print("\n" + "=" * 60)
    print("All UMAP figures generated successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
