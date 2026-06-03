import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.data_augmentation_eval import (
    load_raw_bci2a,
    train_classifier,
    evaluate_classifier,
    load_dttd_model,
    generate_22ch_data,
    get_device,
    CH_IDX_9,
)


def generate_without_labels(model, data_9ch, labels, device, guidance_scale=3.0):
    dummy_labels = np.zeros_like(labels)
    return generate_22ch_data(model, data_9ch, dummy_labels, device, data_9ch, guidance_scale=guidance_scale)


def evaluate_augmented(train_22ch, train_9ch, train_labels, test_22ch, test_labels, generated_22ch, device):
    aug_data = np.concatenate([train_22ch, generated_22ch], axis=0)
    aug_labels = np.concatenate([train_labels, train_labels], axis=0)
    clf = train_classifier(aug_data, aug_labels, device, 22)
    metrics = evaluate_classifier(clf, test_22ch, test_labels, device)
    del clf
    torch.cuda.empty_cache()
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Condition ablation for reviewer revision')
    parser.add_argument('--data-path', default='E:/data/BCI2a')
    parser.add_argument('--config', default='configs/bci2a_enhanced_config.yaml')
    parser.add_argument('--checkpoint', default='checkpoints/bci2a_enhanced/best_model.pth')
    parser.add_argument('--output', default='paper_results/reviewer_revision/condition_ablation.json')
    parser.add_argument('--guidance-scale', type=float, default=3.0)
    args = parser.parse_args()

    device = get_device()
    model = load_dttd_model(args.config, args.checkpoint, device)
    output = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    cross_session = {}
    for sid in range(1, 10):
        train_22ch, train_labels = load_raw_bci2a(args.data_path, sid, 'T')
        test_22ch, test_labels = load_raw_bci2a(args.data_path, sid, 'E')
        train_9ch = train_22ch[:, CH_IDX_9, :]

        gen_full = generate_22ch_data(model, train_9ch, train_labels, device, train_22ch, guidance_scale=args.guidance_scale)
        gen_no_task = generate_without_labels(model, train_9ch, train_labels, device, guidance_scale=args.guidance_scale)

        cross_session[f'S{sid}'] = {
            'full_condition': evaluate_augmented(train_22ch, train_9ch, train_labels, test_22ch, test_labels, gen_full, device),
            'no_task_condition': evaluate_augmented(train_22ch, train_9ch, train_labels, test_22ch, test_labels, gen_no_task, device),
        }

    cross_subject = {}
    for test_sid in range(1, 10):
        train_22ch_list, train_labels_list = [], []
        for sid in range(1, 10):
            if sid == test_sid:
                continue
            for sess in ['T', 'E']:
                d, l = load_raw_bci2a(args.data_path, sid, sess)
                train_22ch_list.append(d)
                train_labels_list.append(l)
        train_22ch = np.concatenate(train_22ch_list, axis=0)
        train_labels = np.concatenate(train_labels_list, axis=0)
        train_9ch = train_22ch[:, CH_IDX_9, :]
        test_t, labels_t = load_raw_bci2a(args.data_path, test_sid, 'T')
        test_e, labels_e = load_raw_bci2a(args.data_path, test_sid, 'E')
        test_22ch = np.concatenate([test_t, test_e], axis=0)
        test_labels = np.concatenate([labels_t, labels_e], axis=0)

        gen_full = generate_22ch_data(model, train_9ch, train_labels, device, train_22ch, guidance_scale=args.guidance_scale)
        gen_no_task = generate_without_labels(model, train_9ch, train_labels, device, guidance_scale=args.guidance_scale)

        cross_subject[f'S{test_sid}'] = {
            'full_condition': evaluate_augmented(train_22ch, train_9ch, train_labels, test_22ch, test_labels, gen_full, device),
            'no_task_condition': evaluate_augmented(train_22ch, train_9ch, train_labels, test_22ch, test_labels, gen_no_task, device),
        }

    output['cross_session'] = cross_session
    output['cross_subject'] = cross_subject
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
