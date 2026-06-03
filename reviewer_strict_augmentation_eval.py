import os
import sys
import json
from datetime import datetime

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.data_augmentation_eval import (
    load_raw_bci2a,
    load_dttd_model,
    train_classifier,
    evaluate_classifier,
    generate_22ch_data,
    get_device,
    CH_IDX_9,
)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_four_arm_evaluation(train_22ch, train_9ch, train_labels, test_22ch, test_9ch, test_labels,
                            model, device, use_ddim=False, num_steps=50, guidance_scale=3.0):
    results = {}

    clf = train_classifier(train_22ch, train_labels, device, 22)
    acc_22, f1_22, kappa_22 = evaluate_classifier(clf, test_22ch, test_labels, device)
    results['baseline_22ch'] = {'accuracy': acc_22, 'f1': f1_22, 'kappa': kappa_22}
    del clf

    clf = train_classifier(train_9ch, train_labels, device, 9)
    acc_9, f1_9, kappa_9 = evaluate_classifier(clf, test_9ch, test_labels, device)
    results['baseline_9ch'] = {'accuracy': acc_9, 'f1': f1_9, 'kappa': kappa_9}
    del clf

    gen_22ch = generate_22ch_data(
        model, train_9ch, train_labels, device, train_22ch,
        use_ddim=use_ddim, num_steps=num_steps, guidance_scale=guidance_scale
    )

    clf = train_classifier(gen_22ch, train_labels, device, 22)
    acc_gen, f1_gen, kappa_gen = evaluate_classifier(clf, test_22ch, test_labels, device)
    results['generated_only_22ch'] = {'accuracy': acc_gen, 'f1': f1_gen, 'kappa': kappa_gen}
    del clf

    aug_data = np.concatenate([train_22ch, gen_22ch], axis=0)
    aug_labels = np.concatenate([train_labels, train_labels], axis=0)
    clf = train_classifier(aug_data, aug_labels, device, 22)
    acc_aug, f1_aug, kappa_aug = evaluate_classifier(clf, test_22ch, test_labels, device)
    results['dttd_augmented'] = {'accuracy': acc_aug, 'f1': f1_aug, 'kappa': kappa_aug}
    del clf

    torch.cuda.empty_cache()
    return results


def summarize_subject_results(subject_results):
    arms = ['baseline_22ch', 'baseline_9ch', 'generated_only_22ch', 'dttd_augmented']
    summary = {}
    for arm in arms:
        accs = [subject_results[s][arm]['accuracy'] for s in subject_results if s.startswith('S')]
        f1s = [subject_results[s][arm]['f1'] for s in subject_results if s.startswith('S') and 'f1' in subject_results[s][arm]]
        kappas = [subject_results[s][arm]['kappa'] for s in subject_results if s.startswith('S') and 'kappa' in subject_results[s][arm]]
        summary[arm] = {
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
        }
        if f1s:
            summary[arm]['f1_mean'] = float(np.mean(f1s))
            summary[arm]['f1_std'] = float(np.std(f1s))
        if kappas:
            summary[arm]['kappa_mean'] = float(np.mean(kappas))
            summary[arm]['kappa_std'] = float(np.std(kappas))
    return summary


def within_subject_eval(data_path, model, device, use_ddim=False, num_steps=50, guidance_scale=3.0, seed=42):
    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for sid in range(1, 10):
        data_t, labels_t = load_raw_bci2a(data_path, sid, 'T')
        data_e, labels_e = load_raw_bci2a(data_path, sid, 'E')
        all_22ch = np.concatenate([data_t, data_e], axis=0)
        all_labels = np.concatenate([labels_t, labels_e], axis=0)
        all_9ch = all_22ch[:, CH_IDX_9, :]
        fold_results = []
        for train_idx, test_idx in skf.split(all_22ch, all_labels):
            fold_results.append(run_four_arm_evaluation(
                all_22ch[train_idx], all_9ch[train_idx], all_labels[train_idx],
                all_22ch[test_idx], all_9ch[test_idx], all_labels[test_idx],
                model, device, use_ddim=use_ddim, num_steps=num_steps, guidance_scale=guidance_scale,
            ))
        results[f'S{sid}'] = {
            arm: {
                'accuracy': float(np.mean([fr[arm]['accuracy'] for fr in fold_results])),
                'std': float(np.std([fr[arm]['accuracy'] for fr in fold_results])),
            }
            for arm in fold_results[0]
        }
    results['average'] = summarize_subject_results(results)
    return results


def cross_session_eval(data_path, model, device, use_ddim=False, num_steps=50, guidance_scale=3.0):
    results = {}
    for sid in range(1, 10):
        train_22ch, train_labels = load_raw_bci2a(data_path, sid, 'T')
        test_22ch, test_labels = load_raw_bci2a(data_path, sid, 'E')
        results[f'S{sid}'] = run_four_arm_evaluation(
            train_22ch, train_22ch[:, CH_IDX_9, :], train_labels,
            test_22ch, test_22ch[:, CH_IDX_9, :], test_labels,
            model, device, use_ddim=use_ddim, num_steps=num_steps, guidance_scale=guidance_scale,
        )
    results['average'] = summarize_subject_results(results)
    return results


def cross_subject_loso_eval(data_path, model, device, use_ddim=False, num_steps=50, guidance_scale=3.0):
    results = {}
    for test_sid in range(1, 10):
        train_22ch_list, train_labels_list = [], []
        for sid in range(1, 10):
            if sid == test_sid:
                continue
            for sess in ['T', 'E']:
                data, labels = load_raw_bci2a(data_path, sid, sess)
                train_22ch_list.append(data)
                train_labels_list.append(labels)
        train_22ch = np.concatenate(train_22ch_list, axis=0)
        train_labels = np.concatenate(train_labels_list, axis=0)
        test_t, labels_t = load_raw_bci2a(data_path, test_sid, 'T')
        test_e, labels_e = load_raw_bci2a(data_path, test_sid, 'E')
        test_22ch = np.concatenate([test_t, test_e], axis=0)
        test_labels = np.concatenate([labels_t, labels_e], axis=0)
        results[f'S{test_sid}'] = run_four_arm_evaluation(
            train_22ch, train_22ch[:, CH_IDX_9, :], train_labels,
            test_22ch, test_22ch[:, CH_IDX_9, :], test_labels,
            model, device, use_ddim=use_ddim, num_steps=num_steps, guidance_scale=guidance_scale,
        )
    results['average'] = summarize_subject_results(results)
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Strict reviewer-aligned data augmentation evaluation')
    parser.add_argument('--data-path', default='E:/data/BCI2a')
    parser.add_argument('--config', default='configs/bci2a_enhanced_config.yaml')
    parser.add_argument('--checkpoint', default='checkpoints/bci2a_enhanced/best_model.pth')
    parser.add_argument('--output-dir', default='paper_results/reviewer_revision')
    parser.add_argument('--mode', default='all', choices=['within_subject', 'cross_session', 'cross_subject', 'all'])
    parser.add_argument('--use-ddim', action='store_true')
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--guidance-scale', type=float, default=3.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    model = load_dttd_model(args.config, args.checkpoint, device)

    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'seed': args.seed,
            'use_ddim': args.use_ddim,
            'num_steps': args.num_steps,
            'guidance_scale': args.guidance_scale,
            'evaluation_type': 'strict_train_only_augmentation',
        },
        'protocol_audit': {
            'generated_samples_from_train_split_only': True,
            'generated_samples_used_in_testing': False,
            'test_evaluation_on_real_samples_only': True,
            'loso_heldout_subject_identity_used': False,
            'main_comparison_arms': [
                'baseline_9ch', 'baseline_22ch', 'generated_only_22ch', 'dttd_augmented'
            ],
        }
    }

    if args.mode in ['within_subject', 'all']:
        results['within_subject'] = within_subject_eval(args.data_path, model, device, args.use_ddim, args.num_steps, args.guidance_scale, args.seed)
    if args.mode in ['cross_session', 'all']:
        results['cross_session'] = cross_session_eval(args.data_path, model, device, args.use_ddim, args.num_steps, args.guidance_scale)
    if args.mode in ['cross_subject', 'all']:
        results['cross_subject'] = cross_subject_loso_eval(args.data_path, model, device, args.use_ddim, args.num_steps, args.guidance_scale)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'strict_augmentation_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to: {output_path}')


if __name__ == '__main__':
    main()
