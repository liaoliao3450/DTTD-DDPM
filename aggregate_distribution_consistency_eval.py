import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.data_augmentation_eval import load_raw_bci2a, load_dttd_model, generate_22ch_data, get_device
from experiments.unified_reconstruction_eval import compute_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Aggregate distribution consistency across all subjects')
    parser.add_argument('--data-path', default='E:/data/BCI2a')
    parser.add_argument('--config', default='configs/bci2a_enhanced_config.yaml')
    parser.add_argument('--checkpoint', default='checkpoints/bci2a_enhanced/best_model.pth')
    parser.add_argument('--output', default='paper_results/reviewer_revision/aggregate_distribution_consistency.json')
    parser.add_argument('--guidance-scale', type=float, default=3.0)
    args = parser.parse_args()

    device = get_device()
    model = load_dttd_model(args.config, args.checkpoint, device)
    subject_metrics = {}

    for sid in range(1, 10):
        data_t, labels_t = load_raw_bci2a(args.data_path, sid, 'T')
        data_e, labels_e = load_raw_bci2a(args.data_path, sid, 'E')
        data_22 = np.concatenate([data_t, data_e], axis=0)
        labels = np.concatenate([labels_t, labels_e], axis=0)
        data_9 = data_22[:, [7, 9, 11, 1, 3, 5, 13, 15, 17], :]
        gen_22 = generate_22ch_data(model, data_9, labels, device, data_22, guidance_scale=args.guidance_scale)
        subject_metrics[f'S{sid}'] = compute_metrics(gen_22, data_22)

    aggregate = {}
    keys = ['nmse', 'topology_similarity', 'psd_correlation', 'frequency_similarity']
    for key in keys:
        vals = [subject_metrics[s][key] for s in subject_metrics]
        aggregate[key] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'median': float(np.median(vals)),
        }

    output = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'subject_metrics': subject_metrics,
        'aggregate': aggregate,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
