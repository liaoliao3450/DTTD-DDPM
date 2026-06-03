import json
import os
import sys
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.data_augmentation_eval import load_dttd_model, generate_22ch_data, get_device, CH_IDX_9

ALL_CH = list(range(22))
MISSING_CH = [ch for ch in ALL_CH if ch not in CH_IDX_9]


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


def channel_metrics(gen, target, ch):
    x = gen[:, ch, :]
    y = target[:, ch, :]
    nmse = float(np.mean((x - y) ** 2) / (np.var(y) + 1e-10))
    corrs = []
    psd_corrs = []
    for i in range(len(x)):
        corr = np.corrcoef(x[i], y[i])[0, 1]
        if not np.isnan(corr):
            corrs.append(corr)
        _, px = welch(x[i], fs=250, nperseg=min(256, x.shape[1]))
        _, py = welch(y[i], fs=250, nperseg=min(256, y.shape[1]))
        pcorr = np.corrcoef(px, py)[0, 1]
        if not np.isnan(pcorr):
            psd_corrs.append(pcorr)
    return {
        'nmse': nmse,
        'waveform_corr': float(np.mean(corrs)) if corrs else 0.0,
        'psd_corr': float(np.mean(psd_corrs)) if psd_corrs else 0.0,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Channel-wise consistency evaluation for missing electrodes')
    parser.add_argument('--data-path', default='E:/data/BCI2a')
    parser.add_argument('--config', default='configs/bci2a_enhanced_config.yaml')
    parser.add_argument('--checkpoint', default='checkpoints/bci2a_enhanced/best_model.pth')
    parser.add_argument('--output', default='paper_results/reviewer_revision/channelwise_consistency.json')
    parser.add_argument('--guidance-scale', type=float, default=3.0)
    args = parser.parse_args()

    device = get_device()
    model = load_dttd_model(args.config, args.checkpoint, device)
    per_subject = {}

    for sid in range(1, 10):
        data_t, labels_t = load_raw_bci2a(args.data_path, sid, 'T')
        data_e, labels_e = load_raw_bci2a(args.data_path, sid, 'E')
        data_22 = np.concatenate([data_t, data_e], axis=0)
        labels = np.concatenate([labels_t, labels_e], axis=0)
        data_9 = data_22[:, CH_IDX_9, :]
        gen_22 = generate_22ch_data(model, data_9, labels, device, data_22, guidance_scale=args.guidance_scale)
        per_subject[f'S{sid}'] = {str(ch): channel_metrics(gen_22, data_22, ch) for ch in MISSING_CH}

    aggregate = {}
    for ch in MISSING_CH:
        nmse = [per_subject[s][str(ch)]['nmse'] for s in per_subject]
        wc = [per_subject[s][str(ch)]['waveform_corr'] for s in per_subject]
        pc = [per_subject[s][str(ch)]['psd_corr'] for s in per_subject]
        aggregate[str(ch)] = {
            'nmse_mean': float(np.mean(nmse)), 'nmse_std': float(np.std(nmse)),
            'waveform_corr_mean': float(np.mean(wc)), 'waveform_corr_std': float(np.std(wc)),
            'psd_corr_mean': float(np.mean(pc)), 'psd_corr_std': float(np.std(pc)),
        }

    output = {
        'missing_channels': MISSING_CH,
        'per_subject': per_subject,
        'aggregate': aggregate,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
