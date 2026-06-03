import json
import os
import numpy as np
from scipy.stats import wilcoxon


def cliffs_delta(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    gt = sum(ix > iy for ix in x for iy in y)
    lt = sum(ix < iy for ix in x for iy in y)
    return (gt - lt) / (len(x) * len(y))


def holm_bonferroni(pairs):
    indexed = sorted(enumerate(pairs), key=lambda t: t[1][1])
    m = len(pairs)
    adjusted = [None] * m
    for rank, (idx, (name, p)) in enumerate(indexed):
        adjusted[idx] = (name, min(1.0, (m - rank) * p))
    return adjusted


def load_subject_values(section, arm, metric='accuracy'):
    values = []
    for key, item in section.items():
        if key.startswith('S'):
            values.append(item[arm][metric])
    return np.array(values, dtype=float)


def compare(a_name, a_vals, b_name, b_vals):
    diff = a_vals - b_vals
    stat, p = wilcoxon(a_vals, b_vals, zero_method='wilcox', alternative='two-sided')
    return {
        'comparison': f'{a_name} vs {b_name}',
        'n': int(len(a_vals)),
        'mean_gain': float(np.mean(diff)),
        'median_gain': float(np.median(diff)),
        'wilcoxon_stat': float(stat),
        'p_value': float(p),
        'cliffs_delta': float(cliffs_delta(a_vals, b_vals)),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Statistical significance for reviewer revision experiments')
    parser.add_argument('--input', default='paper_results/reviewer_revision/strict_augmentation_results.json')
    parser.add_argument('--output', default='paper_results/reviewer_revision/statistical_significance.json')
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        results = json.load(f)

    output = {}
    for protocol in ['within_subject', 'cross_session', 'cross_subject']:
        if protocol not in results:
            continue
        section = results[protocol]
        aug = load_subject_values(section, 'dttd_augmented')
        base9 = load_subject_values(section, 'baseline_9ch')
        base22 = load_subject_values(section, 'baseline_22ch')
        gen_only = load_subject_values(section, 'generated_only_22ch')

        comparisons = [
            compare('dttd_augmented', aug, 'baseline_9ch', base9),
            compare('dttd_augmented', aug, 'baseline_22ch', base22),
            compare('dttd_augmented', aug, 'generated_only_22ch', gen_only),
        ]
        adjusted = holm_bonferroni([(c['comparison'], c['p_value']) for c in comparisons])
        for comp, (_, adj_p) in zip(comparisons, adjusted):
            comp['holm_bonferroni_p'] = adj_p
        output[protocol] = comparisons

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
