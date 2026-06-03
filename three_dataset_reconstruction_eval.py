"""
三数据集重建质量评估 (BCI2a, HGD, PhysioNet MI)

每个数据集随机选取200个样本，评估DTTD和基线方法的重建质量指标：
- 拓扑相似性 (Topology Similarity)
- NMSE (Normalized MSE)
- PSD相关 (PSD Correlation)
- 频带相似性 (Frequency Band Similarity)

并进行统计比较（Wilcoxon signed-rank test vs DTTD）。

Usage:
    python experiments/three_dataset_reconstruction_eval.py --dataset bci2a --n-samples 200
    python experiments/three_dataset_reconstruction_eval.py --dataset hgd --n-samples 200
    python experiments/three_dataset_reconstruction_eval.py --dataset physionet --n-samples 200
    python experiments/three_dataset_reconstruction_eval.py --dataset all --n-samples 200
"""
import os
import sys
import argparse
import json
import numpy as np
import torch
from scipy import signal as scipy_signal
from scipy.stats import pearsonr, wilcoxon
from scipy.interpolate import CubicSpline, Rbf, RBFInterpolator

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import get_device


ELECTRODE_POSITIONS_10_10 = {
    # Standard 10-10 system
    'FP1': (-0.309, 0.951, 0.0), 'FPZ': (0.0, 0.988, 0.0), 'FP2': (0.309, 0.951, 0.0),
    'AF7': (-0.588, 0.809, 0.0), 'AF3': (-0.309, 0.809, 0.5), 'AFZ': (0.0, 0.844, 0.5),
    'AF4': (0.309, 0.809, 0.5), 'AF8': (0.588, 0.809, 0.0),
    'F7': (-0.809, 0.588, 0.0), 'F5': (-0.588, 0.588, 0.559),
    'F3': (-0.309, 0.588, 0.743), 'F1': (-0.156, 0.588, 0.743),
    'FZ': (0.0, 0.588, 0.809), 'F2': (0.156, 0.588, 0.743),
    'F4': (0.309, 0.588, 0.743), 'F6': (0.588, 0.588, 0.559),
    'F8': (0.809, 0.588, 0.0),
    'FT7': (-0.951, 0.309, 0.0), 'FC5': (-0.809, 0.309, 0.5),
    'FC3': (-0.454, 0.309, 0.835), 'FC1': (-0.156, 0.309, 0.936),
    'FCZ': (0.0, 0.309, 0.951), 'FC2': (0.156, 0.309, 0.936),
    'FC4': (0.454, 0.309, 0.835), 'FC6': (0.809, 0.309, 0.5),
    'FT8': (0.951, 0.309, 0.0),
    'T7': (-1.0, 0.0, 0.0), 'C5': (-0.809, 0.0, 0.588),
    'C3': (-0.454, 0.0, 0.891), 'C1': (-0.156, 0.0, 0.988),
    'CZ': (0.0, 0.0, 1.0), 'C2': (0.156, 0.0, 0.988),
    'C4': (0.454, 0.0, 0.891), 'C6': (0.809, 0.0, 0.588),
    'T8': (1.0, 0.0, 0.0),
    'TP7': (-0.951, -0.309, 0.0), 'CP5': (-0.809, -0.309, 0.5),
    'CP3': (-0.454, -0.309, 0.835), 'CP1': (-0.156, -0.309, 0.936),
    'CPZ': (0.0, -0.309, 0.951), 'CP2': (0.156, -0.309, 0.936),
    'CP4': (0.454, -0.309, 0.835), 'CP6': (0.809, -0.309, 0.5),
    'TP8': (0.951, -0.309, 0.0),
    'P7': (-0.809, -0.588, 0.0), 'P5': (-0.588, -0.588, 0.559),
    'P3': (-0.309, -0.588, 0.743), 'P1': (-0.156, -0.588, 0.743),
    'PZ': (0.0, -0.588, 0.809), 'P2': (0.156, -0.588, 0.743),
    'P4': (0.309, -0.588, 0.743), 'P6': (0.588, -0.588, 0.559),
    'P8': (0.809, -0.588, 0.0),
    'PO7': (-0.588, -0.809, 0.0), 'PO3': (-0.309, -0.809, 0.5),
    'POZ': (0.0, -0.809, 0.588), 'PO4': (0.309, -0.809, 0.5),
    'PO8': (0.588, -0.809, 0.0),
    'O1': (-0.309, -0.951, 0.0), 'OZ': (0.0, -0.951, 0.309),
    'O2': (0.309, -0.951, 0.0),
    'IZ': (0.0, -1.0, 0.0),
    'T9': (-1.0, -0.156, 0.0), 'T10': (1.0, -0.156, 0.0),
    # 10-5 system high-density channels (HGD dataset)
    # FFC = between F and FC, FCC = between FC and C, CCP = between C and CP, CPP = between CP and P
    # 'h' suffix = halfway between two 10-10 electrodes
    'FFC5H': (-0.809, 0.449, 0.380), 'FFC3H': (-0.454, 0.449, 0.766),
    'FFC1H': (-0.156, 0.449, 0.890), 'FFC2H': (0.156, 0.449, 0.890),
    'FFC4H': (0.454, 0.449, 0.766), 'FFC6H': (0.809, 0.449, 0.380),
    'FCC5H': (-0.809, 0.156, 0.530), 'FCC3H': (-0.454, 0.156, 0.863),
    'FCC1H': (-0.156, 0.156, 0.962), 'FCC2H': (0.156, 0.156, 0.962),
    'FCC4H': (0.454, 0.156, 0.863), 'FCC6H': (0.809, 0.156, 0.530),
    'CCP5H': (-0.809, -0.156, 0.530), 'CCP3H': (-0.454, -0.156, 0.863),
    'CCP1H': (-0.156, -0.156, 0.962), 'CCP2H': (0.156, -0.156, 0.962),
    'CCP4H': (0.454, -0.156, 0.863), 'CCP6H': (0.809, -0.156, 0.530),
    'CPP5H': (-0.809, -0.449, 0.380), 'CPP3H': (-0.454, -0.449, 0.766),
    'CPP1H': (-0.156, -0.449, 0.890), 'CPP2H': (0.156, -0.449, 0.890),
    'CPP4H': (0.454, -0.449, 0.766), 'CPP6H': (0.809, -0.449, 0.380),
}


def get_electrode_positions(channel_names):
    positions = []
    for ch in channel_names:
        # Clean channel name: remove dots, EEG prefix, reference suffix
        ch_key = ch.rstrip('.').upper()
        # Remove common EDF prefixes like 'EEG ', 'EEG-'
        if ch_key.startswith('EEG'):
            ch_key = ch_key[3:].lstrip(' -_')
        # Remove reference suffix like '-0', '-REF', '-LE', '-AVG'
        for suffix in ['-REF', '-AVG', '-LE']:
            if ch_key.endswith(suffix):
                ch_key = ch_key[:-len(suffix)]
        # Handle trailing dash+digit like 'FP1-0'
        if '-' in ch_key:
            parts = ch_key.split('-')
            if parts[-1].isdigit():
                ch_key = '-'.join(parts[:-1])

        if ch_key in ELECTRODE_POSITIONS_10_10:
            positions.append(ELECTRODE_POSITIONS_10_10[ch_key])
        else:
            positions.append((0.0, 0.0, 0.0))
    return np.array(positions)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(generated, target, fs=250):
    assert generated.shape == target.shape, f"Shape mismatch: {generated.shape} vs {target.shape}"
    n_samples, n_channels, n_times = generated.shape

    mse_raw = np.mean((generated - target) ** 2)
    target_var = np.var(target)
    nmse = mse_raw / (target_var + 1e-10)

    topo_sims = []
    for i in range(n_samples):
        corr_gen = np.corrcoef(generated[i])
        corr_target = np.corrcoef(target[i])
        if np.isnan(corr_gen).any() or np.isnan(corr_target).any():
            continue
        mask = np.triu(np.ones((n_channels, n_channels), dtype=bool), k=1)
        gen_vec = corr_gen[mask]
        target_vec = corr_target[mask]
        if np.std(gen_vec) < 1e-12 or np.std(target_vec) < 1e-12:
            continue
        corr, _ = pearsonr(gen_vec, target_vec)
        if not np.isnan(corr):
            topo_sims.append(corr)
    topo_sim = np.mean(topo_sims) if topo_sims else 0.0

    psd_corrs = []
    for i in range(n_samples):
        for ch in range(n_channels):
            nperseg = min(256, n_times)
            f_gen, psd_gen = scipy_signal.welch(generated[i, ch], fs=fs, nperseg=nperseg)
            f_tgt, psd_tgt = scipy_signal.welch(target[i, ch], fs=fs, nperseg=nperseg)
            if np.std(psd_gen) < 1e-12 or np.std(psd_tgt) < 1e-12:
                continue
            corr, _ = pearsonr(psd_gen, psd_tgt)
            if not np.isnan(corr):
                psd_corrs.append(corr)
    psd_corr = np.mean(psd_corrs) if psd_corrs else 0.0

    freq_bands = {
        'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
        'beta': (13, 30), 'gamma': (30, min(50, fs / 2))
    }
    band_sims = []
    for band_name, (f_low, f_high) in freq_bands.items():
        if f_high > fs / 2:
            continue
        band_corrs = []
        for i in range(n_samples):
            for ch in range(n_channels):
                nperseg = min(256, n_times)
                f_gen, psd_gen = scipy_signal.welch(generated[i, ch], fs=fs, nperseg=nperseg)
                f_tgt, psd_tgt = scipy_signal.welch(target[i, ch], fs=fs, nperseg=nperseg)
                band_mask = (f_gen >= f_low) & (f_gen <= f_high)
                if band_mask.sum() < 2:
                    continue
                b_gen = psd_gen[band_mask]
                b_tgt = psd_tgt[band_mask]
                if np.std(b_gen) < 1e-12 or np.std(b_tgt) < 1e-12:
                    continue
                corr, _ = pearsonr(b_gen, b_tgt)
                if not np.isnan(corr):
                    band_corrs.append(corr)
        if band_corrs:
            band_sims.append(np.mean(band_corrs))
    freq_sim = np.mean(band_sims) if band_sims else 0.0

    return {
        'topology_similarity': float(topo_sim),
        'nmse': float(nmse),
        'psd_correlation': float(psd_corr),
        'frequency_similarity': float(freq_sim)
    }


def compute_per_sample_topo(generated, target):
    n_samples = generated.shape[0]
    n_channels = generated.shape[1]
    per_sample = []
    for i in range(n_samples):
        corr_gen = np.corrcoef(generated[i])
        corr_target = np.corrcoef(target[i])
        if np.isnan(corr_gen).any() or np.isnan(corr_target).any():
            per_sample.append(np.nan)
            continue
        mask = np.triu(np.ones((n_channels, n_channels), dtype=bool), k=1)
        gen_vec = corr_gen[mask]
        target_vec = corr_target[mask]
        if np.std(gen_vec) < 1e-12 or np.std(target_vec) < 1e-12:
            per_sample.append(np.nan)
            continue
        corr, _ = pearsonr(gen_vec, target_vec)
        per_sample.append(corr if not np.isnan(corr) else np.nan)
    return np.array(per_sample)


def compute_per_sample_psd(generated, target, fs=250):
    n_samples = generated.shape[0]
    n_channels = generated.shape[1]
    n_times = generated.shape[2]
    per_sample = []
    nperseg = min(256, n_times)
    for i in range(n_samples):
        corrs = []
        for ch in range(n_channels):
            f_gen, psd_gen = scipy_signal.welch(generated[i, ch], fs=fs, nperseg=nperseg)
            f_tgt, psd_tgt = scipy_signal.welch(target[i, ch], fs=fs, nperseg=nperseg)
            if np.std(psd_gen) < 1e-12 or np.std(psd_tgt) < 1e-12:
                continue
            corr, _ = pearsonr(psd_gen, psd_tgt)
            if not np.isnan(corr):
                corrs.append(corr)
        per_sample.append(np.mean(corrs) if corrs else np.nan)
    return np.array(per_sample)


def spline_interpolation(input_data, ch_indices, n_output_channels, channel_names=None):
    n_samples, _, n_times = input_data.shape
    output = np.zeros((n_samples, n_output_channels, n_times))

    if channel_names is not None and len(channel_names) == n_output_channels:
        all_positions = get_electrode_positions(channel_names)
        input_positions = all_positions[ch_indices]
        
        dist_matrix = np.linalg.norm(all_positions[:, np.newaxis] - input_positions, axis=2)
        dist_matrix[dist_matrix < 1e-6] = 1e-6
        weights = 1.0 / dist_matrix**2
        weights /= weights.sum(axis=1, keepdims=True)
        
        for i in range(n_samples):
            output[i] = weights @ input_data[i]
        return output

    sorted_idx = sorted(ch_indices)
    for i in range(n_samples):
        known_vals = input_data[i, [ch_indices.index(idx) for idx in sorted_idx], :]
        for t in range(n_times):
            cs = CubicSpline(sorted_idx, known_vals[:, t], extrapolate=True)
            output[i, :, t] = cs(np.arange(n_output_channels))
    return output


def kriging_interpolation(input_data, ch_indices, n_output_channels, channel_names=None):
    n_samples, _, n_times = input_data.shape
    output = np.zeros((n_samples, n_output_channels, n_times))

    if channel_names is not None and len(channel_names) == n_output_channels:
        all_positions = get_electrode_positions(channel_names)
        input_positions = all_positions[ch_indices]
        
        dist_matrix = np.linalg.norm(all_positions[:, np.newaxis] - input_positions, axis=2)
        dist_matrix[dist_matrix < 1e-6] = 1e-6
        weights = np.exp(-dist_matrix**2 / (2 * 0.5**2))
        weights /= weights.sum(axis=1, keepdims=True)
        
        for i in range(n_samples):
            output[i] = weights @ input_data[i]
        return output

    sorted_idx = sorted(ch_indices)
    for i in range(n_samples):
        known_vals = input_data[i, [ch_indices.index(idx) for idx in sorted_idx], :]
        for t in range(n_times):
            try:
                rbf = Rbf(sorted_idx, known_vals[:, t], function='multiquadric')
                output[i, :, t] = rbf(np.arange(n_output_channels))
            except Exception:
                output[i, :, t] = np.interp(np.arange(n_output_channels), sorted_idx, known_vals[:, t])
    return output


def eval_bci2a(data_path, n_samples, device, seed):
    print("\n" + "=" * 70, flush=True)
    print("BCI Competition IV 2a 重建质量评估", flush=True)
    print("=" * 70, flush=True)

    set_seed(seed)

    from models import DTTDEnhanced
    from models.baselines import CVAE, Generator as cGANGenerator
    from data import get_bci2a_dataloaders
    from utils import load_config

    config = load_config('configs/bci2a_enhanced_config.yaml')
    DATA_SCALE = 1e5

    print("[1/6] 加载BCI2a数据...", flush=True)
    _, _, test_loader = get_bci2a_dataloaders(
        data_path=data_path, batch_size=32, subject_ids=list(range(1, 10)),
        num_workers=0, reconstruction_mode=True
    )

    all_data, all_labels, ch_indices = [], [], None
    for batch in test_loader:
        target_data, channel_indices, labels, _ = batch
        all_data.append(target_data)
        all_labels.append(labels)
        if ch_indices is None:
            ch_indices = channel_indices[0].tolist()

    all_data = torch.cat(all_data, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    n_total = len(all_data)
    if n_samples < n_total:
        idx = np.random.choice(n_total, n_samples, replace=False)
        all_data = all_data[idx]
        all_labels = all_labels[idx]

    input_9ch = all_data[:, ch_indices, :].numpy()
    target_22ch = all_data.numpy()
    labels_np = all_labels.numpy()

    print(f"  样本数: {len(all_data)}, 输入通道索引: {ch_indices}", flush=True)

    results = {}

    # DTTD
    print("[2/6] 评估 DTTD...", flush=True)
    model = DTTDEnhanced(config['model']).to(device)
    ckpt = torch.load('checkpoints/bci2a_enhanced/best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    gen_list = []
    with torch.no_grad():
        for i in range(0, len(all_data), 32):
            batch_data = all_data[i:i+32].to(device)
            batch_labels = all_labels[i:i+32].to(device)
            input_data = batch_data[:, ch_indices, :]
            scaled_input = input_data * DATA_SCALE
            noisy_input = scaled_input + torch.randn_like(scaled_input) * 0.02
            t = torch.zeros(input_data.size(0), device=device, dtype=torch.long)
            generated = model(noisy_input, t, batch_labels)
            generated = generated / DATA_SCALE
            gen_list.append(generated.cpu().numpy())
    dttd_gen = np.concatenate(gen_list, axis=0)
    results['DTTD'] = compute_metrics(dttd_gen, target_22ch, fs=250)
    results['DTTD']['per_sample_topo'] = compute_per_sample_topo(dttd_gen, target_22ch).tolist()
    results['DTTD']['per_sample_psd'] = compute_per_sample_psd(dttd_gen, target_22ch, fs=250).tolist()
    print(f"  DTTD: topo={results['DTTD']['topology_similarity']:.4f}, "
          f"nmse={results['DTTD']['nmse']:.4f}, psd={results['DTTD']['psd_correlation']:.4f}, "
          f"freq={results['DTTD']['frequency_similarity']:.4f}", flush=True)

    # CVAE
    cvae_path = 'checkpoints/bci2a/baseline_cvae/best_model.pth'
    if os.path.exists(cvae_path):
        print("[3/6] 评估 CVAE...", flush=True)
        cvae = CVAE(input_channels=9, output_channels=22, time_steps=1000,
                    num_classes=4, latent_dim=128).to(device)
        cvae_ckpt = torch.load(cvae_path, map_location=device, weights_only=False)
        cvae.load_state_dict(cvae_ckpt['model_state_dict'], strict=False)
        cvae.eval()
        gen_list = []
        with torch.no_grad():
            for i in range(0, len(all_data), 32):
                batch_data = all_data[i:i+32].to(device)
                batch_labels = all_labels[i:i+32].to(device)
                input_data = batch_data[:, ch_indices, :]
                scaled_input = input_data * DATA_SCALE
                generated, _, _ = cvae(scaled_input, batch_labels)
                generated = generated / DATA_SCALE
                gen_list.append(generated.cpu().numpy())
        cvae_gen = np.concatenate(gen_list, axis=0)
        results['CVAE'] = compute_metrics(cvae_gen, target_22ch, fs=250)
        results['CVAE']['per_sample_topo'] = compute_per_sample_topo(cvae_gen, target_22ch).tolist()
        results['CVAE']['per_sample_psd'] = compute_per_sample_psd(cvae_gen, target_22ch, fs=250).tolist()
        print(f"  CVAE: topo={results['CVAE']['topology_similarity']:.4f}, "
              f"nmse={results['CVAE']['nmse']:.4f}, psd={results['CVAE']['psd_correlation']:.4f}, "
              f"freq={results['CVAE']['frequency_similarity']:.4f}", flush=True)

    # cGAN
    cgan_path = 'checkpoints/bci2a_enhanced/baseline_cgan/best_model.pth'
    if os.path.exists(cgan_path):
        print("[4/6] 评估 cGAN...", flush=True)
        cgan = cGANGenerator(input_channels=9, output_channels=22, time_steps=1000,
                             num_classes=4, latent_dim=128).to(device)
        cgan_ckpt = torch.load(cgan_path, map_location=device, weights_only=False)
        cgan.load_state_dict(cgan_ckpt['model_state_dict'], strict=False)
        cgan.eval()
        gen_list = []
        with torch.no_grad():
            for i in range(0, len(all_data), 32):
                batch_data = all_data[i:i+32].to(device)
                batch_labels = all_labels[i:i+32].to(device)
                input_data = batch_data[:, ch_indices, :]
                batch_size = input_data.size(0)
                z = torch.randn(batch_size, 128).to(device)
                scaled_input = input_data * DATA_SCALE
                generated = cgan(z, batch_labels, scaled_input)
                generated = generated / DATA_SCALE
                gen_list.append(generated.cpu().numpy())
        cgan_gen = np.concatenate(gen_list, axis=0)
        results['cGAN'] = compute_metrics(cgan_gen, target_22ch, fs=250)
        results['cGAN']['per_sample_topo'] = compute_per_sample_topo(cgan_gen, target_22ch).tolist()
        results['cGAN']['per_sample_psd'] = compute_per_sample_psd(cgan_gen, target_22ch, fs=250).tolist()
        print(f"  cGAN: topo={results['cGAN']['topology_similarity']:.4f}, "
              f"nmse={results['cGAN']['nmse']:.4f}, psd={results['cGAN']['psd_correlation']:.4f}, "
              f"freq={results['cGAN']['frequency_similarity']:.4f}", flush=True)

    # EEGDiff
    eegdiff_path = 'checkpoints/bci2a/baseline_eegdiff/best_model.pth'
    if os.path.exists(eegdiff_path):
        print("[5/8] 评估 EEGDiff...", flush=True)
        from models.baselines import EEGDiff
        eegdiff_ckpt = torch.load(eegdiff_path, map_location=device, weights_only=False)
        eegdiff = EEGDiff(
            input_channels=eegdiff_ckpt['n_input_ch'],
            output_channels=eegdiff_ckpt['n_output_ch'],
            time_steps=eegdiff_ckpt['time_steps'],
            num_classes=eegdiff_ckpt['num_classes'],
            num_timesteps=1000, embed_dim=128
        ).to(device)
        eegdiff.load_state_dict(eegdiff_ckpt['model_state_dict'], strict=False)
        eegdiff.eval()

        eegdiff_mean = eegdiff_ckpt.get('data_mean', None)
        eegdiff_std = eegdiff_ckpt.get('data_std', None)
        if eegdiff_mean is not None and eegdiff_std is not None:
            eegdiff_input = ((input_9ch - eegdiff_mean[:, ch_indices, :]) / eegdiff_std[:, ch_indices, :]).astype(np.float32)
        else:
            eegdiff_input = input_9ch * DATA_SCALE

        gen_list = []
        with torch.no_grad():
            for i in range(0, len(all_data), 32):
                batch_input = torch.FloatTensor(eegdiff_input[i:i+32]).to(device)
                batch_labels = all_labels[i:i+32].to(device)
                generated = eegdiff.sample(batch_input, task_label=batch_labels, noise_scale=0.02)
                gen_np = generated.cpu().numpy()
                if eegdiff_mean is not None and eegdiff_std is not None:
                    gen_np = gen_np * eegdiff_std + eegdiff_mean
                else:
                    gen_np = gen_np / DATA_SCALE
                gen_list.append(gen_np)
        eegdiff_gen = np.concatenate(gen_list, axis=0)
        results['EEGDiff'] = compute_metrics(eegdiff_gen, target_22ch, fs=250)
        results['EEGDiff']['per_sample_topo'] = compute_per_sample_topo(eegdiff_gen, target_22ch).tolist()
        results['EEGDiff']['per_sample_psd'] = compute_per_sample_psd(eegdiff_gen, target_22ch, fs=250).tolist()
        print(f"  EEGDiff: topo={results['EEGDiff']['topology_similarity']:.4f}, "
              f"nmse={results['EEGDiff']['nmse']:.4f}, psd={results['EEGDiff']['psd_correlation']:.4f}, "
              f"freq={results['EEGDiff']['frequency_similarity']:.4f}", flush=True)
    else:
        print(f"  [SKIP] EEGDiff checkpoint not found: {eegdiff_path}", flush=True)

    # BrainDiff
    braindiff_path = 'checkpoints/bci2a/baseline_braindiff/best_model.pth'
    if os.path.exists(braindiff_path):
        print("[6/8] 评估 BrainDiff...", flush=True)
        from models.baselines import BrainDiff
        bdiff_ckpt = torch.load(braindiff_path, map_location=device, weights_only=False)
        braindiff = BrainDiff(
            input_channels=bdiff_ckpt['n_input_ch'],
            output_channels=bdiff_ckpt['n_output_ch'],
            time_steps=bdiff_ckpt['time_steps'],
            num_classes=bdiff_ckpt['num_classes'],
            num_timesteps=1000, embed_dim=128
        ).to(device)
        braindiff.load_state_dict(bdiff_ckpt['model_state_dict'], strict=False)
        braindiff.eval()

        bdiff_mean = bdiff_ckpt.get('data_mean', None)
        bdiff_std = bdiff_ckpt.get('data_std', None)
        if bdiff_mean is not None and bdiff_std is not None:
            bdiff_input = ((input_9ch - bdiff_mean[:, ch_indices, :]) / bdiff_std[:, ch_indices, :]).astype(np.float32)
        else:
            bdiff_input = input_9ch * DATA_SCALE

        gen_list = []
        with torch.no_grad():
            for i in range(0, len(all_data), 32):
                batch_input = torch.FloatTensor(bdiff_input[i:i+32]).to(device)
                generated = braindiff.sample(batch_input, noise_scale=0.02)
                gen_np = generated.cpu().numpy()
                if bdiff_mean is not None and bdiff_std is not None:
                    gen_np = gen_np * bdiff_std + bdiff_mean
                else:
                    gen_np = gen_np / DATA_SCALE
                gen_list.append(gen_np)
        braindiff_gen = np.concatenate(gen_list, axis=0)
        results['BrainDiff'] = compute_metrics(braindiff_gen, target_22ch, fs=250)
        results['BrainDiff']['per_sample_topo'] = compute_per_sample_topo(braindiff_gen, target_22ch).tolist()
        results['BrainDiff']['per_sample_psd'] = compute_per_sample_psd(braindiff_gen, target_22ch, fs=250).tolist()
        print(f"  BrainDiff: topo={results['BrainDiff']['topology_similarity']:.4f}, "
              f"nmse={results['BrainDiff']['nmse']:.4f}, psd={results['BrainDiff']['psd_correlation']:.4f}, "
              f"freq={results['BrainDiff']['frequency_similarity']:.4f}", flush=True)
    else:
        print(f"  [SKIP] BrainDiff checkpoint not found: {braindiff_path}", flush=True)

    # Spline
    print("[7/8] 评估 Spline...", flush=True)
    spline_gen = spline_interpolation(input_9ch, ch_indices, 22)
    results['Spline'] = compute_metrics(spline_gen, target_22ch, fs=250)
    results['Spline']['per_sample_topo'] = compute_per_sample_topo(spline_gen, target_22ch).tolist()
    results['Spline']['per_sample_psd'] = compute_per_sample_psd(spline_gen, target_22ch, fs=250).tolist()
    print(f"  Spline: topo={results['Spline']['topology_similarity']:.4f}, "
          f"nmse={results['Spline']['nmse']:.4f}, psd={results['Spline']['psd_correlation']:.4f}, "
          f"freq={results['Spline']['frequency_similarity']:.4f}", flush=True)

    # Kriging
    print("[8/8] 评估 Kriging...", flush=True)
    kriging_gen = kriging_interpolation(input_9ch, ch_indices, 22)
    results['Kriging'] = compute_metrics(kriging_gen, target_22ch, fs=250)
    results['Kriging']['per_sample_topo'] = compute_per_sample_topo(kriging_gen, target_22ch).tolist()
    results['Kriging']['per_sample_psd'] = compute_per_sample_psd(kriging_gen, target_22ch, fs=250).tolist()
    print(f"  Kriging: topo={results['Kriging']['topology_similarity']:.4f}, "
          f"nmse={results['Kriging']['nmse']:.4f}, psd={results['Kriging']['psd_correlation']:.4f}, "
          f"freq={results['Kriging']['frequency_similarity']:.4f}", flush=True)

    return results


def eval_hgd(data_path, n_samples, device, seed):
    print("\n" + "=" * 70, flush=True)
    print("High Gamma Dataset 重建质量评估", flush=True)
    print("=" * 70, flush=True)

    set_seed(seed)

    from models.dttd_physionet import DTTDPhysioNet
    from data.high_gamma_dataset import HighGammaDataset
    from experiments.train_baselines_multidataset import CVAE_Large, Generator_Large

    print("[1/6] 加载HGD数据（仅加载2个被试以获取200+样本）...", flush=True)
    dataset = HighGammaDataset(data_path, subject_ids=[1, 2],
                               sessions='test', fs_target=250)
    all_data = dataset.data
    all_labels = dataset.labels
    input_ch_indices = dataset.input_ch_indices
    n_output_ch = dataset.num_output_channels

    n_total = len(all_data)
    if n_samples < n_total:
        idx = np.random.choice(n_total, n_samples, replace=False)
        all_data = all_data[idx]
        all_labels = all_labels[idx]

    target_data = all_data
    input_data = all_data[:, input_ch_indices, :]
    labels_np = all_labels

    print(f"  样本数: {len(target_data)}, 输入通道: {len(input_ch_indices)}, 输出通道: {n_output_ch}", flush=True)

    # 标准化（与训练时一致）
    data_mean = all_data.mean(axis=0, keepdims=True)
    data_std = all_data.std(axis=0, keepdims=True) + 1e-8
    normed_data = (all_data - data_mean) / data_std
    normed_input = normed_data[:, input_ch_indices, :]

    time_steps = target_data.shape[2]
    model_config = {
        'input_channels': len(input_ch_indices),
        'output_channels': n_output_ch,
        'time_steps': time_steps,
        'embed_dim': 256, 'task_dim': 64, 'num_classes': 4,
        'num_heads': 8, 'dropout': 0.1,
        'num_timesteps': 1000, 'beta_start': 1e-4, 'beta_end': 0.02,
        'schedule_type': 'linear', 'fs': 250,
        'use_classifier_guidance': True, 'lambda_cls': 0.1, 'warmup_epochs': 30
    }

    results = {}

    # DTTD
    ckpt_path = 'paper_results/hgd/dttd_hgd_best.pth'
    if not os.path.exists(ckpt_path):
        ckpt_path = 'paper_results/hgd_cross_session/dttd_hgd_cs_best.pth'
    if os.path.exists(ckpt_path):
        print(f"[2/6] 评估 DTTD (checkpoint: {ckpt_path})...", flush=True)
        model = DTTDPhysioNet(model_config).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        dttd_data_mean = checkpoint.get('data_mean', None)
        dttd_data_std = checkpoint.get('data_std', None)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        model.eval()

        ch_idx_np = np.array(input_ch_indices)
        if dttd_data_mean is not None and dttd_data_std is not None:
            input_mean = dttd_data_mean[:, ch_idx_np, :]
            input_std = dttd_data_std[:, ch_idx_np, :]
            dttd_input = ((input_data - input_mean) / input_std).astype(np.float32)
        else:
            dttd_input = normed_input

        gen_list = []
        hgd_batch_size = 2
        with torch.no_grad():
            for i in range(0, len(target_data), hgd_batch_size):
                batch_input = torch.FloatTensor(dttd_input[i:i+hgd_batch_size]).to(device)
                batch_labels = torch.LongTensor(labels_np[i:i+hgd_batch_size]).to(device)
                noisy_input = batch_input + torch.randn_like(batch_input) * 0.02
                t = torch.zeros(batch_input.size(0), device=device, dtype=torch.long)
                generated = model(noisy_input, t, batch_labels)
                gen_np = generated.cpu().numpy()
                if dttd_data_mean is not None:
                    gen_np = gen_np * dttd_data_std + dttd_data_mean
                gen_list.append(gen_np)
        dttd_gen = np.concatenate(gen_list, axis=0)
        dttd_gen[:, input_ch_indices, :] = target_data[:, input_ch_indices, :]
        results['DTTD'] = compute_metrics(dttd_gen, target_data, fs=250)
        results['DTTD']['per_sample_topo'] = compute_per_sample_topo(dttd_gen, target_data).tolist()
        results['DTTD']['per_sample_psd'] = compute_per_sample_psd(dttd_gen, target_data, fs=250).tolist()
        print(f"  DTTD: topo={results['DTTD']['topology_similarity']:.4f}, "
              f"nmse={results['DTTD']['nmse']:.4f}, psd={results['DTTD']['psd_correlation']:.4f}, "
              f"freq={results['DTTD']['frequency_similarity']:.4f}", flush=True)
    else:
        print(f"  [SKIP] DTTD checkpoint not found", flush=True)

    # CVAE
    cvae_path = 'checkpoints/hgd/baseline_cvae/best_model.pth'
    if os.path.exists(cvae_path):
        print("[3/6] 评估 CVAE...", flush=True)
        cvae_ckpt = torch.load(cvae_path, map_location=device, weights_only=False)
        cvae = CVAE_Large(
            input_channels=cvae_ckpt['n_input_ch'],
            output_channels=cvae_ckpt['n_output_ch'],
            time_steps=cvae_ckpt['time_steps'],
            num_classes=cvae_ckpt['num_classes'],
            latent_dim=cvae_ckpt.get('latent_dim', 256)
        ).to(device)
        cvae.load_state_dict(cvae_ckpt['model_state_dict'], strict=False)
        cvae.eval()

        cvae_mean = cvae_ckpt.get('data_mean', data_mean)
        cvae_std = cvae_ckpt.get('data_std', data_std)
        cvae_input = ((input_data - cvae_mean[:, input_ch_indices, :]) / cvae_std[:, input_ch_indices, :]).astype(np.float32)

        gen_list = []
        with torch.no_grad():
            for i in range(0, len(target_data), 16):
                batch_input = torch.FloatTensor(cvae_input[i:i+16]).to(device)
                batch_labels = torch.LongTensor(labels_np[i:i+16]).to(device)
                generated, _, _ = cvae(batch_input, batch_labels)
                gen_np = generated.cpu().numpy() * cvae_std + cvae_mean
                gen_list.append(gen_np)
        cvae_gen = np.concatenate(gen_list, axis=0)
        cvae_gen[:, input_ch_indices, :] = target_data[:, input_ch_indices, :]
        results['CVAE'] = compute_metrics(cvae_gen, target_data, fs=250)
        results['CVAE']['per_sample_topo'] = compute_per_sample_topo(cvae_gen, target_data).tolist()
        results['CVAE']['per_sample_psd'] = compute_per_sample_psd(cvae_gen, target_data, fs=250).tolist()
        print(f"  CVAE: topo={results['CVAE']['topology_similarity']:.4f}, "
              f"nmse={results['CVAE']['nmse']:.4f}, psd={results['CVAE']['psd_correlation']:.4f}, "
              f"freq={results['CVAE']['frequency_similarity']:.4f}", flush=True)
    else:
        print(f"  [SKIP] CVAE checkpoint not found: {cvae_path}", flush=True)

    # cGAN
    cgan_path = 'checkpoints/hgd/baseline_cgan/best_model.pth'
    if os.path.exists(cgan_path):
        print("[4/6] 评估 cGAN...", flush=True)
        cgan_ckpt = torch.load(cgan_path, map_location=device, weights_only=False)
        cgan = Generator_Large(
            input_channels=cgan_ckpt['n_input_ch'],
            output_channels=cgan_ckpt['n_output_ch'],
            time_steps=cgan_ckpt['time_steps'],
            num_classes=cgan_ckpt['num_classes'],
            latent_dim=cgan_ckpt.get('latent_dim', 128)
        ).to(device)
        cgan.load_state_dict(cgan_ckpt['model_state_dict'], strict=False)
        cgan.eval()

        cgan_mean = cgan_ckpt.get('data_mean', data_mean)
        cgan_std = cgan_ckpt.get('data_std', data_std)
        cgan_input = ((input_data - cgan_mean[:, input_ch_indices, :]) / cgan_std[:, input_ch_indices, :]).astype(np.float32)

        gen_list = []
        with torch.no_grad():
            for i in range(0, len(target_data), 16):
                batch_input = torch.FloatTensor(cgan_input[i:i+16]).to(device)
                batch_labels = torch.LongTensor(labels_np[i:i+16]).to(device)
                batch_size = batch_input.size(0)
                z = torch.randn(batch_size, cgan_ckpt.get('latent_dim', 128)).to(device)
                generated = cgan(z, batch_labels, batch_input)
                gen_np = generated.cpu().numpy() * cgan_std + cgan_mean
                gen_list.append(gen_np)
        cgan_gen = np.concatenate(gen_list, axis=0)
        cgan_gen[:, input_ch_indices, :] = target_data[:, input_ch_indices, :]
        results['cGAN'] = compute_metrics(cgan_gen, target_data, fs=250)
        results['cGAN']['per_sample_topo'] = compute_per_sample_topo(cgan_gen, target_data).tolist()
        results['cGAN']['per_sample_psd'] = compute_per_sample_psd(cgan_gen, target_data, fs=250).tolist()
        print(f"  cGAN: topo={results['cGAN']['topology_similarity']:.4f}, "
              f"nmse={results['cGAN']['nmse']:.4f}, psd={results['cGAN']['psd_correlation']:.4f}, "
              f"freq={results['cGAN']['frequency_similarity']:.4f}", flush=True)
    else:
        print(f"  [SKIP] cGAN checkpoint not found: {cgan_path}", flush=True)

    # EEGDiff
    eegdiff_path = 'checkpoints/hgd/baseline_eegdiff/best_model.pth'
    if os.path.exists(eegdiff_path):
        print("[5/8] 评估 EEGDiff...", flush=True)
        from models.baselines import EEGDiff
        eegdiff_ckpt = torch.load(eegdiff_path, map_location=device, weights_only=False)
        eegdiff = EEGDiff(
            input_channels=eegdiff_ckpt['n_input_ch'],
            output_channels=eegdiff_ckpt['n_output_ch'],
            time_steps=eegdiff_ckpt['time_steps'],
            num_classes=eegdiff_ckpt['num_classes'],
            num_timesteps=1000, embed_dim=128
        ).to(device)
        eegdiff.load_state_dict(eegdiff_ckpt['model_state_dict'], strict=False)
        eegdiff.eval()

        eegdiff_mean = eegdiff_ckpt.get('data_mean', data_mean)
        eegdiff_std = eegdiff_ckpt.get('data_std', data_std)
        eegdiff_input = ((input_data - eegdiff_mean[:, input_ch_indices, :]) / eegdiff_std[:, input_ch_indices, :]).astype(np.float32)

        gen_list = []
        with torch.no_grad():
            for i in range(0, len(target_data), 16):
                batch_input = torch.FloatTensor(eegdiff_input[i:i+16]).to(device)
                batch_labels = torch.LongTensor(labels_np[i:i+16]).to(device)
                generated = eegdiff.sample(batch_input, task_label=batch_labels, noise_scale=0.02)
                gen_np = generated.cpu().numpy() * eegdiff_std + eegdiff_mean
                gen_list.append(gen_np)
        eegdiff_gen = np.concatenate(gen_list, axis=0)
        eegdiff_gen[:, input_ch_indices, :] = target_data[:, input_ch_indices, :]
        results['EEGDiff'] = compute_metrics(eegdiff_gen, target_data, fs=250)
        results['EEGDiff']['per_sample_topo'] = compute_per_sample_topo(eegdiff_gen, target_data).tolist()
        results['EEGDiff']['per_sample_psd'] = compute_per_sample_psd(eegdiff_gen, target_data, fs=250).tolist()
        print(f"  EEGDiff: topo={results['EEGDiff']['topology_similarity']:.4f}, "
              f"nmse={results['EEGDiff']['nmse']:.4f}, psd={results['EEGDiff']['psd_correlation']:.4f}, "
              f"freq={results['EEGDiff']['frequency_similarity']:.4f}", flush=True)
    else:
        print(f"  [SKIP] EEGDiff checkpoint not found: {eegdiff_path}", flush=True)

    # BrainDiff
    braindiff_path = 'checkpoints/hgd/baseline_braindiff/best_model.pth'
    if os.path.exists(braindiff_path):
        print("[6/8] 评估 BrainDiff...", flush=True)
        from models.baselines import BrainDiff
        bdiff_ckpt = torch.load(braindiff_path, map_location=device, weights_only=False)
        braindiff = BrainDiff(
            input_channels=bdiff_ckpt['n_input_ch'],
            output_channels=bdiff_ckpt['n_output_ch'],
            time_steps=bdiff_ckpt['time_steps'],
            num_classes=bdiff_ckpt['num_classes'],
            num_timesteps=1000, embed_dim=128
        ).to(device)
        braindiff.load_state_dict(bdiff_ckpt['model_state_dict'], strict=False)
        braindiff.eval()

        bdiff_mean = bdiff_ckpt.get('data_mean', data_mean)
        bdiff_std = bdiff_ckpt.get('data_std', data_std)
        bdiff_input = ((input_data - bdiff_mean[:, input_ch_indices, :]) / bdiff_std[:, input_ch_indices, :]).astype(np.float32)

        gen_list = []
        with torch.no_grad():
            for i in range(0, len(target_data), 16):
                batch_input = torch.FloatTensor(bdiff_input[i:i+16]).to(device)
                generated = braindiff.sample(batch_input, noise_scale=0.02)
                gen_np = generated.cpu().numpy() * bdiff_std + bdiff_mean
                gen_list.append(gen_np)
        braindiff_gen = np.concatenate(gen_list, axis=0)
        braindiff_gen[:, input_ch_indices, :] = target_data[:, input_ch_indices, :]
        results['BrainDiff'] = compute_metrics(braindiff_gen, target_data, fs=250)
        results['BrainDiff']['per_sample_topo'] = compute_per_sample_topo(braindiff_gen, target_data).tolist()
        results['BrainDiff']['per_sample_psd'] = compute_per_sample_psd(braindiff_gen, target_data, fs=250).tolist()
        print(f"  BrainDiff: topo={results['BrainDiff']['topology_similarity']:.4f}, "
              f"nmse={results['BrainDiff']['nmse']:.4f}, psd={results['BrainDiff']['psd_correlation']:.4f}, "
              f"freq={results['BrainDiff']['frequency_similarity']:.4f}", flush=True)
    else:
        print(f"  [SKIP] BrainDiff checkpoint not found: {braindiff_path}", flush=True)

    # Spline
    print("[7/8] 评估 Spline...", flush=True)
    spline_gen = spline_interpolation(input_data, input_ch_indices, n_output_ch,
                                       channel_names=dataset.ch_names)
    results['Spline'] = compute_metrics(spline_gen, target_data, fs=250)
    results['Spline']['per_sample_topo'] = compute_per_sample_topo(spline_gen, target_data).tolist()
    results['Spline']['per_sample_psd'] = compute_per_sample_psd(spline_gen, target_data, fs=250).tolist()
    print(f"  Spline: topo={results['Spline']['topology_similarity']:.4f}, "
          f"nmse={results['Spline']['nmse']:.4f}, psd={results['Spline']['psd_correlation']:.4f}, "
          f"freq={results['Spline']['frequency_similarity']:.4f}", flush=True)

    # Kriging
    print("[8/8] 评估 Kriging...", flush=True)
    kriging_gen = kriging_interpolation(input_data, input_ch_indices, n_output_ch,
                                         channel_names=dataset.ch_names)
    results['Kriging'] = compute_metrics(kriging_gen, target_data, fs=250)
    results['Kriging']['per_sample_topo'] = compute_per_sample_topo(kriging_gen, target_data).tolist()
    results['Kriging']['per_sample_psd'] = compute_per_sample_psd(kriging_gen, target_data, fs=250).tolist()
    print(f"  Kriging: topo={results['Kriging']['topology_similarity']:.4f}, "
          f"nmse={results['Kriging']['nmse']:.4f}, psd={results['Kriging']['psd_correlation']:.4f}, "
          f"freq={results['Kriging']['frequency_similarity']:.4f}", flush=True)

    return results


def eval_physionet(data_path, n_samples, device, seed):
    print("\n" + "=" * 70, flush=True)
    print("PhysioNet MI 重建质量评估", flush=True)
    print("=" * 70, flush=True)

    set_seed(seed)

    from models.dttd_physionet import DTTDPhysioNet
    from experiments.train_baselines_multidataset import CVAE_Large, Generator_Large

    CHANNEL_NAMES_64 = [
        'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
        'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
        'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
        'T9', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'T10',
        'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
        'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
        'PO7', 'PO3', 'POz', 'PO4', 'PO8',
        'O1', 'Oz', 'O2', 'Iz'
    ]
    INPUT_CHANNELS_16 = [
        'FC3', 'FC1', 'FC2', 'FC4',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4'
    ]
    input_ch_indices = [CHANNEL_NAMES_64.index(ch) for ch in INPUT_CHANNELS_16]

    cache_path = 'paper_results/physionet_mi/physionet_mi_cache.npz'
    if os.path.exists(cache_path):
        print("[1/6] 从缓存加载PhysioNet MI数据...", flush=True)
        cache = np.load(cache_path)
        all_data = cache['data']
        all_labels = cache['labels']
    else:
        print("[1/6] 缓存不存在，运行cache_physionet_fast.py生成...", flush=True)
        import subprocess
        subprocess.run([sys.executable, 'experiments/cache_physionet_fast.py'], check=True)
        cache = np.load(cache_path)
        all_data = cache['data']
        all_labels = cache['labels']

    n_total = len(all_data)
    if n_samples < n_total:
        idx = np.random.choice(n_total, n_samples, replace=False)
        all_data = all_data[idx]
        all_labels = all_labels[idx]

    n_output_ch = all_data.shape[1]
    target_data = all_data
    input_data = all_data[:, input_ch_indices, :]

    # 标准化
    data_mean = all_data.mean(axis=0, keepdims=True)
    data_std = all_data.std(axis=0, keepdims=True) + 1e-8
    normed_data = (all_data - data_mean) / data_std
    normed_input = normed_data[:, input_ch_indices, :]

    print(f"  样本数: {len(target_data)}, 输入通道: {len(input_ch_indices)}, 输出通道: {n_output_ch}", flush=True)

    model_config = {
        'input_channels': 16,
        'output_channels': 64,
        'time_steps': 640,
        'embed_dim': 256, 'task_dim': 64, 'num_classes': 4,
        'num_heads': 8, 'dropout': 0.1,
        'num_timesteps': 1000, 'beta_start': 1e-4, 'beta_end': 0.02,
        'schedule_type': 'linear', 'fs': 160,
        'use_classifier_guidance': False
    }

    results = {}

    # DTTD
    ckpt_path = 'paper_results/physionet_mi/dttd_physionet_best.pth'
    if os.path.exists(ckpt_path):
        print(f"[2/6] 评估 DTTD (checkpoint: {ckpt_path})...", flush=True)
        model = DTTDPhysioNet(model_config).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        data_mean_dttd = checkpoint.get('data_mean', None)
        data_std_dttd = checkpoint.get('data_std', None)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        model.eval()

        ch_idx_np = np.array(input_ch_indices)

        # 使用checkpoint中保存的mean/std做标准化（新模型）
        if data_mean_dttd is not None and data_std_dttd is not None:
            dttd_input_mean = data_mean_dttd[:, ch_idx_np, :]
            dttd_input_std = data_std_dttd[:, ch_idx_np, :]
            scaled_input = ((input_data - dttd_input_mean) / dttd_input_std).astype(np.float32)
        else:
            # 兼容旧模型（使用1e5缩放）
            DATA_SCALE = 1e5
            UV_TO_V = 1e-6
            scaled_input = (input_data * UV_TO_V * DATA_SCALE).astype(np.float32)

        gen_list = []
        with torch.no_grad():
            for i in range(0, len(target_data), 16):
                batch_input = torch.FloatTensor(scaled_input[i:i+16]).to(device)
                batch_labels = torch.LongTensor(all_labels[i:i+16]).to(device)
                t = torch.zeros(batch_input.size(0), device=device, dtype=torch.long)
                generated = model(batch_input, t, batch_labels)
                gen_np = generated.cpu().numpy()
                # 反标准化
                if data_mean_dttd is not None and data_std_dttd is not None:
                    gen_np = gen_np * data_std_dttd + data_mean_dttd
                else:
                    gen_np = gen_np / DATA_SCALE / UV_TO_V
                gen_list.append(gen_np)
        dttd_gen = np.concatenate(gen_list, axis=0)
        dttd_gen[:, input_ch_indices, :] = target_data[:, input_ch_indices, :]
        results['DTTD'] = compute_metrics(dttd_gen, target_data, fs=160)
        results['DTTD']['per_sample_topo'] = compute_per_sample_topo(dttd_gen, target_data).tolist()
        results['DTTD']['per_sample_psd'] = compute_per_sample_psd(dttd_gen, target_data, fs=160).tolist()
        print(f"  DTTD: topo={results['DTTD']['topology_similarity']:.4f}, "
              f"nmse={results['DTTD']['nmse']:.4f}, psd={results['DTTD']['psd_correlation']:.4f}, "
              f"freq={results['DTTD']['frequency_similarity']:.4f}", flush=True)
    else:
        print(f"  [SKIP] DTTD checkpoint not found", flush=True)

    # CVAE (训练时使用二分类: T1 vs T2, num_classes=2)
    cvae_path = 'checkpoints/physionet/baseline_cvae/best_model.pth'
    if os.path.exists(cvae_path):
        print("[3/6] 评估 CVAE...", flush=True)
        cvae_ckpt = torch.load(cvae_path, map_location=device, weights_only=False)
        cvae = CVAE_Large(
            input_channels=cvae_ckpt['n_input_ch'],
            output_channels=cvae_ckpt['n_output_ch'],
            time_steps=cvae_ckpt['time_steps'],
            num_classes=cvae_ckpt['num_classes'],
            latent_dim=cvae_ckpt.get('latent_dim', 256)
        ).to(device)
        cvae.load_state_dict(cvae_ckpt['model_state_dict'], strict=False)
        cvae.eval()

        cvae_mean = cvae_ckpt.get('data_mean', data_mean)
        cvae_std = cvae_ckpt.get('data_std', data_std)
        # 只使用标签0和1的样本（与训练一致）
        binary_mask = all_labels <= 1
        cvae_target = target_data[binary_mask]
        cvae_input_raw = input_data[binary_mask]
        cvae_labels = all_labels[binary_mask]
        cvae_input = ((cvae_input_raw - cvae_mean[:, input_ch_indices, :]) / cvae_std[:, input_ch_indices, :]).astype(np.float32)

        gen_list = []
        with torch.no_grad():
            for i in range(0, len(cvae_target), 16):
                batch_input = torch.FloatTensor(cvae_input[i:i+16]).to(device)
                batch_labels = torch.LongTensor(cvae_labels[i:i+16]).to(device)
                generated, _, _ = cvae(batch_input, batch_labels)
                gen_np = generated.cpu().numpy() * cvae_std + cvae_mean
                gen_list.append(gen_np)
        cvae_gen = np.concatenate(gen_list, axis=0)
        cvae_gen[:, input_ch_indices, :] = cvae_target[:, input_ch_indices, :]
        results['CVAE'] = compute_metrics(cvae_gen, cvae_target, fs=160)
        results['CVAE']['per_sample_topo'] = compute_per_sample_topo(cvae_gen, cvae_target).tolist()
        results['CVAE']['per_sample_psd'] = compute_per_sample_psd(cvae_gen, cvae_target, fs=160).tolist()
        print(f"  CVAE: topo={results['CVAE']['topology_similarity']:.4f}, "
              f"nmse={results['CVAE']['nmse']:.4f}, psd={results['CVAE']['psd_correlation']:.4f}, "
              f"freq={results['CVAE']['frequency_similarity']:.4f}", flush=True)
    else:
        print(f"  [SKIP] CVAE checkpoint not found: {cvae_path}", flush=True)

    # cGAN (训练时使用二分类: T1 vs T2, num_classes=2)
    cgan_path = 'checkpoints/physionet/baseline_cgan/best_model.pth'
    if os.path.exists(cgan_path):
        print("[4/6] 评估 cGAN...", flush=True)
        cgan_ckpt = torch.load(cgan_path, map_location=device, weights_only=False)
        cgan = Generator_Large(
            input_channels=cgan_ckpt['n_input_ch'],
            output_channels=cgan_ckpt['n_output_ch'],
            time_steps=cgan_ckpt['time_steps'],
            num_classes=cgan_ckpt['num_classes'],
            latent_dim=cgan_ckpt.get('latent_dim', 128)
        ).to(device)
        cgan.load_state_dict(cgan_ckpt['model_state_dict'], strict=False)
        cgan.eval()

        cgan_mean = cgan_ckpt.get('data_mean', data_mean)
        cgan_std = cgan_ckpt.get('data_std', data_std)
        # 只使用标签0和1的样本（与训练一致）
        binary_mask = all_labels <= 1
        cgan_target = target_data[binary_mask]
        cgan_input_raw = input_data[binary_mask]
        cgan_labels = all_labels[binary_mask]
        cgan_input = ((cgan_input_raw - cgan_mean[:, input_ch_indices, :]) / cgan_std[:, input_ch_indices, :]).astype(np.float32)

        gen_list = []
        with torch.no_grad():
            for i in range(0, len(cgan_target), 16):
                batch_input = torch.FloatTensor(cgan_input[i:i+16]).to(device)
                batch_labels = torch.LongTensor(cgan_labels[i:i+16]).to(device)
                batch_size = batch_input.size(0)
                z = torch.randn(batch_size, cgan_ckpt.get('latent_dim', 128)).to(device)
                generated = cgan(z, batch_labels, batch_input)
                gen_np = generated.cpu().numpy() * cgan_std + cgan_mean
                gen_list.append(gen_np)
        cgan_gen = np.concatenate(gen_list, axis=0)
        cgan_gen[:, input_ch_indices, :] = cgan_target[:, input_ch_indices, :]
        results['cGAN'] = compute_metrics(cgan_gen, cgan_target, fs=160)
        results['cGAN']['per_sample_topo'] = compute_per_sample_topo(cgan_gen, cgan_target).tolist()
        results['cGAN']['per_sample_psd'] = compute_per_sample_psd(cgan_gen, cgan_target, fs=160).tolist()
        print(f"  cGAN: topo={results['cGAN']['topology_similarity']:.4f}, "
              f"nmse={results['cGAN']['nmse']:.4f}, psd={results['cGAN']['psd_correlation']:.4f}, "
              f"freq={results['cGAN']['frequency_similarity']:.4f}", flush=True)
    else:
        print(f"  [SKIP] cGAN checkpoint not found: {cgan_path}", flush=True)

    # EEGDiff
    eegdiff_path = 'checkpoints/physionet/baseline_eegdiff/best_model.pth'
    if os.path.exists(eegdiff_path):
        print("[5/8] 评估 EEGDiff...", flush=True)
        from models.baselines import EEGDiff
        eegdiff_ckpt = torch.load(eegdiff_path, map_location=device, weights_only=False)
        eegdiff = EEGDiff(
            input_channels=eegdiff_ckpt['n_input_ch'],
            output_channels=eegdiff_ckpt['n_output_ch'],
            time_steps=eegdiff_ckpt['time_steps'],
            num_classes=eegdiff_ckpt['num_classes'],
            num_timesteps=1000, embed_dim=128
        ).to(device)
        eegdiff.load_state_dict(eegdiff_ckpt['model_state_dict'], strict=False)
        eegdiff.eval()

        eegdiff_mean = eegdiff_ckpt.get('data_mean', data_mean)
        eegdiff_std = eegdiff_ckpt.get('data_std', data_std)
        eegdiff_input = ((input_data - eegdiff_mean[:, input_ch_indices, :]) / eegdiff_std[:, input_ch_indices, :]).astype(np.float32)

        gen_list = []
        with torch.no_grad():
            for i in range(0, len(target_data), 16):
                batch_input = torch.FloatTensor(eegdiff_input[i:i+16]).to(device)
                batch_labels = torch.LongTensor(all_labels[i:i+16]).to(device)
                generated = eegdiff.sample(batch_input, task_label=batch_labels, noise_scale=0.02)
                gen_np = generated.cpu().numpy() * eegdiff_std + eegdiff_mean
                gen_list.append(gen_np)
        eegdiff_gen = np.concatenate(gen_list, axis=0)
        eegdiff_gen[:, input_ch_indices, :] = target_data[:, input_ch_indices, :]
        results['EEGDiff'] = compute_metrics(eegdiff_gen, target_data, fs=160)
        results['EEGDiff']['per_sample_topo'] = compute_per_sample_topo(eegdiff_gen, target_data).tolist()
        results['EEGDiff']['per_sample_psd'] = compute_per_sample_psd(eegdiff_gen, target_data, fs=160).tolist()
        print(f"  EEGDiff: topo={results['EEGDiff']['topology_similarity']:.4f}, "
              f"nmse={results['EEGDiff']['nmse']:.4f}, psd={results['EEGDiff']['psd_correlation']:.4f}, "
              f"freq={results['EEGDiff']['frequency_similarity']:.4f}", flush=True)
    else:
        print(f"  [SKIP] EEGDiff checkpoint not found: {eegdiff_path}", flush=True)

    # BrainDiff
    braindiff_path = 'checkpoints/physionet/baseline_braindiff/best_model.pth'
    if os.path.exists(braindiff_path):
        print("[6/8] 评估 BrainDiff...", flush=True)
        from models.baselines import BrainDiff
        bdiff_ckpt = torch.load(braindiff_path, map_location=device, weights_only=False)
        braindiff = BrainDiff(
            input_channels=bdiff_ckpt['n_input_ch'],
            output_channels=bdiff_ckpt['n_output_ch'],
            time_steps=bdiff_ckpt['time_steps'],
            num_classes=bdiff_ckpt['num_classes'],
            num_timesteps=1000, embed_dim=128
        ).to(device)
        braindiff.load_state_dict(bdiff_ckpt['model_state_dict'], strict=False)
        braindiff.eval()

        bdiff_mean = bdiff_ckpt.get('data_mean', data_mean)
        bdiff_std = bdiff_ckpt.get('data_std', data_std)
        bdiff_input = ((input_data - bdiff_mean[:, input_ch_indices, :]) / bdiff_std[:, input_ch_indices, :]).astype(np.float32)

        gen_list = []
        with torch.no_grad():
            for i in range(0, len(target_data), 16):
                batch_input = torch.FloatTensor(bdiff_input[i:i+16]).to(device)
                generated = braindiff.sample(batch_input, noise_scale=0.02)
                gen_np = generated.cpu().numpy() * bdiff_std + bdiff_mean
                gen_list.append(gen_np)
        braindiff_gen = np.concatenate(gen_list, axis=0)
        braindiff_gen[:, input_ch_indices, :] = target_data[:, input_ch_indices, :]
        results['BrainDiff'] = compute_metrics(braindiff_gen, target_data, fs=160)
        results['BrainDiff']['per_sample_topo'] = compute_per_sample_topo(braindiff_gen, target_data).tolist()
        results['BrainDiff']['per_sample_psd'] = compute_per_sample_psd(braindiff_gen, target_data, fs=160).tolist()
        print(f"  BrainDiff: topo={results['BrainDiff']['topology_similarity']:.4f}, "
              f"nmse={results['BrainDiff']['nmse']:.4f}, psd={results['BrainDiff']['psd_correlation']:.4f}, "
              f"freq={results['BrainDiff']['frequency_similarity']:.4f}", flush=True)
    else:
        print(f"  [SKIP] BrainDiff checkpoint not found: {braindiff_path}", flush=True)

    # Spline
    print("[7/8] 评估 Spline...", flush=True)
    spline_gen = spline_interpolation(input_data, input_ch_indices, n_output_ch,
                                       channel_names=CHANNEL_NAMES_64)
    results['Spline'] = compute_metrics(spline_gen, target_data, fs=160)
    results['Spline']['per_sample_topo'] = compute_per_sample_topo(spline_gen, target_data).tolist()
    results['Spline']['per_sample_psd'] = compute_per_sample_psd(spline_gen, target_data, fs=160).tolist()
    print(f"  Spline: topo={results['Spline']['topology_similarity']:.4f}, "
          f"nmse={results['Spline']['nmse']:.4f}, psd={results['Spline']['psd_correlation']:.4f}, "
          f"freq={results['Spline']['frequency_similarity']:.4f}", flush=True)

    # Kriging
    print("[8/8] 评估 Kriging...", flush=True)
    kriging_gen = kriging_interpolation(input_data, input_ch_indices, n_output_ch,
                                         channel_names=CHANNEL_NAMES_64)
    results['Kriging'] = compute_metrics(kriging_gen, target_data, fs=160)
    results['Kriging']['per_sample_topo'] = compute_per_sample_topo(kriging_gen, target_data).tolist()
    results['Kriging']['per_sample_psd'] = compute_per_sample_psd(kriging_gen, target_data, fs=160).tolist()
    print(f"  Kriging: topo={results['Kriging']['topology_similarity']:.4f}, "
          f"nmse={results['Kriging']['nmse']:.4f}, psd={results['Kriging']['psd_correlation']:.4f}, "
          f"freq={results['Kriging']['frequency_similarity']:.4f}", flush=True)

    return results


def statistical_comparison(results, dttd_key='DTTD'):
    stats = {}
    if dttd_key not in results:
        return stats
    dttd_topo = np.array(results[dttd_key]['per_sample_topo'])
    dttd_psd = np.array(results[dttd_key]['per_sample_psd'])

    for method, metrics in results.items():
        if method == dttd_key:
            continue
        if 'per_sample_topo' not in metrics:
            continue
        other_topo = np.array(metrics['per_sample_topo'])
        other_psd = np.array(metrics['per_sample_psd'])

        # 处理样本数不一致的情况（如CVAE/cGAN只用了部分样本）
        min_len = min(len(dttd_topo), len(other_topo))
        dttd_topo_cmp = dttd_topo[:min_len]
        other_topo_cmp = other_topo[:min_len]
        dttd_psd_cmp = dttd_psd[:min_len]
        other_psd_cmp = other_psd[:min_len]

        valid_topo = ~(np.isnan(dttd_topo_cmp) | np.isnan(other_topo_cmp))
        valid_psd = ~(np.isnan(dttd_psd_cmp) | np.isnan(other_psd_cmp))

        topo_p = None
        if valid_topo.sum() > 10:
            try:
                _, topo_p = wilcoxon(dttd_topo_cmp[valid_topo], other_topo_cmp[valid_topo])
            except Exception:
                topo_p = 1.0

        psd_p = None
        if valid_psd.sum() > 10:
            try:
                _, psd_p = wilcoxon(dttd_psd_cmp[valid_psd], other_psd_cmp[valid_psd])
            except Exception:
                psd_p = 1.0

        stats[method] = {
            'topo_p_value': float(topo_p) if topo_p is not None else None,
            'psd_p_value': float(psd_p) if psd_p is not None else None,
            'topo_diff_mean': float(np.nanmean(dttd_topo_cmp - other_topo_cmp)),
            'psd_diff_mean': float(np.nanmean(dttd_psd_cmp - other_psd_cmp))
        }

    return stats


def main():
    parser = argparse.ArgumentParser(description='三数据集重建质量评估')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['bci2a', 'hgd', 'physionet', 'all'],
                        help='评估的数据集')
    parser.add_argument('--bci2a-path', default='E:/data/BCI2a')
    parser.add_argument('--hgd-path', default='E:/data/HGD')
    parser.add_argument('--physionet-path', default='E:/data/PhysioNetMI')
    parser.add_argument('--n-samples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='paper_results/reconstruction_quality')
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    # BCI2a
    if args.dataset in ('bci2a', 'all') and os.path.exists(args.bci2a_path):
        bci2a_results = eval_bci2a(args.bci2a_path, args.n_samples, device, args.seed)
        bci2a_stats = statistical_comparison(bci2a_results)
        all_results['BCI2a'] = {'metrics': {}, 'statistics': bci2a_stats}
        for method, metrics in bci2a_results.items():
            all_results['BCI2a']['metrics'][method] = {
                k: v for k, v in metrics.items() if not k.startswith('per_sample')
            }
    else:
        if args.dataset in ('bci2a', 'all'):
            print(f"[SKIP] BCI2a data not found: {args.bci2a_path}", flush=True)

    # HGD
    if args.dataset in ('hgd', 'all') and os.path.exists(args.hgd_path):
        hgd_results = eval_hgd(args.hgd_path, args.n_samples, device, args.seed)
        hgd_stats = statistical_comparison(hgd_results)
        all_results['HGD'] = {'metrics': {}, 'statistics': hgd_stats}
        for method, metrics in hgd_results.items():
            all_results['HGD']['metrics'][method] = {
                k: v for k, v in metrics.items() if not k.startswith('per_sample')
            }
    else:
        if args.dataset in ('hgd', 'all'):
            print(f"[SKIP] HGD data not found: {args.hgd_path}", flush=True)

    # PhysioNet MI
    if args.dataset in ('physionet', 'all') and os.path.exists(args.physionet_path):
        physionet_results = eval_physionet(args.physionet_path, args.n_samples, device, args.seed)
        physionet_stats = statistical_comparison(physionet_results)
        all_results['PhysioNet_MI'] = {'metrics': {}, 'statistics': physionet_stats}
        for method, metrics in physionet_results.items():
            all_results['PhysioNet_MI']['metrics'][method] = {
                k: v for k, v in metrics.items() if not k.startswith('per_sample')
            }
    else:
        if args.dataset in ('physionet', 'all'):
            print(f"[SKIP] PhysioNet MI data not found: {args.physionet_path}", flush=True)

    # Save
    output_path = os.path.join(args.output_dir, 'three_dataset_reconstruction.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 90, flush=True)
    print("三数据集重建质量指标汇总 (各200样本)", flush=True)
    print("=" * 90, flush=True)
    header = f"{'数据集':<14} {'方法':<14} {'拓扑相似性↑':>12} {'NMSE↓':>12} {'PSD相关↑':>12} {'频带相似性↑':>12}"
    print(header, flush=True)
    print("-" * 90, flush=True)

    for dataset_name, dataset_results in all_results.items():
        for method, metrics in dataset_results['metrics'].items():
            print(f"{dataset_name:<14} {method:<14} "
                  f"{metrics['topology_similarity']:>12.4f} {metrics['nmse']:>12.4f} "
                  f"{metrics['psd_correlation']:>12.4f} {metrics['frequency_similarity']:>12.4f}", flush=True)
        print(flush=True)

    # Print statistics
    print("\n" + "=" * 70, flush=True)
    print("统计比较 (Wilcoxon signed-rank test: DTTD vs 其他方法)", flush=True)
    print("=" * 70, flush=True)
    for dataset_name, dataset_results in all_results.items():
        print(f"\n{dataset_name}:", flush=True)
        stats = dataset_results.get('statistics', {})
        for method, s in stats.items():
            topo_p = s.get('topo_p_value', 'N/A')
            psd_p = s.get('psd_p_value', 'N/A')
            topo_p_str = f"{topo_p:.4f}" if isinstance(topo_p, float) else topo_p
            psd_p_str = f"{psd_p:.4f}" if isinstance(psd_p, float) else psd_p
            print(f"  vs {method}: topo_diff={s['topo_diff_mean']:+.4f} (p={topo_p_str}), "
                  f"psd_diff={s['psd_diff_mean']:+.4f} (p={psd_p_str})", flush=True)

    print(f"\n结果已保存到: {output_path}", flush=True)


if __name__ == '__main__':
    main()
