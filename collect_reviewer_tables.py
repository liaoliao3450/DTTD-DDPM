import json
import os
from typing import Any, Dict, List


def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def fmt_pct(x):
    return 'NA' if x is None else f'{x * 100:.2f}'


def fmt_num(x, digits=3):
    return 'NA' if x is None else f'{x:.{digits}f}'


def get_acc_mean(section: Dict[str, Any], arm: str):
    if not section or 'average' not in section:
        return None
    block = section['average'].get(arm)
    if isinstance(block, dict):
        return block.get('accuracy_mean', block.get('accuracy'))
    return block


def extract_condition_acc(entry):
    if isinstance(entry, dict):
        return entry.get('accuracy')
    if isinstance(entry, list) and entry:
        return entry[0]
    return None


def avg_subject_metric(section: Dict[str, Any], key: str):
    vals = []
    for sid, item in section.items():
        if not sid.startswith('S'):
            continue
        vals.append(extract_condition_acc(item[key]))
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def build_main_results(strict_results):
    rows = []
    for protocol_key, label in [
        ('within_subject', 'Within-subject'),
        ('cross_session', 'Cross-session'),
        ('cross_subject', 'Cross-subject (LOSO)'),
    ]:
        sec = strict_results.get(protocol_key)
        if not sec:
            continue
        rows.append([
            label,
            fmt_pct(get_acc_mean(sec, 'baseline_9ch')),
            fmt_pct(get_acc_mean(sec, 'baseline_22ch')),
            fmt_pct(get_acc_mean(sec, 'generated_only_22ch')),
            fmt_pct(get_acc_mean(sec, 'dttd_augmented')),
        ])
    return rows


def build_stats_rows(stats):
    rows = []
    for protocol, comps in stats.items():
        for item in comps:
            rows.append([
                protocol,
                item['comparison'],
                fmt_pct(item.get('mean_gain')),
                fmt_num(item.get('p_value'), 4),
                fmt_num(item.get('holm_bonferroni_p'), 4),
                fmt_num(item.get('cliffs_delta'), 3),
            ])
    return rows


def build_distribution_rows(dist):
    agg = dist.get('aggregate', {}) if dist else {}
    rows = []
    for key in ['nmse', 'topology_similarity', 'psd_correlation', 'frequency_similarity']:
        item = agg.get(key, {})
        rows.append([key, fmt_num(item.get('mean')), fmt_num(item.get('std')), fmt_num(item.get('median'))])
    return rows


def build_condition_rows(cond):
    rows = []
    for protocol in ['cross_session', 'cross_subject']:
        sec = cond.get(protocol, {}) if cond else {}
        rows.append([
            protocol,
            fmt_pct(avg_subject_metric(sec, 'full_condition')),
            fmt_pct(avg_subject_metric(sec, 'no_task_condition')),
        ])
    return rows


def build_erd_rows(erd):
    agg = erd.get('aggregate', {}) if erd else {}
    rows = []
    for key in ['mu_corr', 'beta_corr', 'mu_mae', 'beta_mae']:
        item = agg.get(key, {})
        rows.append([key, fmt_num(item.get('mean')), fmt_num(item.get('std'))])
    return rows


def build_channel_rows(ch):
    agg = ch.get('aggregate', {}) if ch else {}
    rows = []
    for channel, item in agg.items():
        rows.append([
            channel,
            fmt_num(item.get('nmse_mean')),
            fmt_num(item.get('waveform_corr_mean')),
            fmt_num(item.get('psd_corr_mean')),
        ])
    rows.sort(key=lambda r: int(r[0]))
    return rows


def build_direct_rows(direct):
    rows = []
    if not direct:
        return rows
    for protocol in ['cross_session', 'cross_subject']:
        sec = direct.get(protocol, {})
        gen_vals, aug_vals = [], []
        for sid, item in sec.items():
            if not sid.startswith('S'):
                continue
            gen = item.get('generated_only_22ch')
            aug = item.get('augmented')
            gen_vals.append(gen['accuracy'] if isinstance(gen, dict) else gen[0])
            aug_vals.append(aug['accuracy'] if isinstance(aug, dict) else aug[0])
        if gen_vals:
            rows.append([protocol, fmt_pct(sum(gen_vals)/len(gen_vals)), fmt_pct(sum(aug_vals)/len(aug_vals))])
    return rows


def md_table(headers: List[str], rows: List[List[str]]):
    out = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    out += ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]
    return '\n'.join(out)


def latex_table(headers: List[str], rows: List[List[str]], caption: str, label: str):
    cols = 'l' * len(headers)
    body = ['\\begin{table}[t]', '\\centering', f'\\caption{{{caption}}}', f'\\label{{{label}}}', f'\\begin{{tabular}}{{{cols}}}', '\\hline']
    body.append(' & '.join(headers) + ' \\\\')
    body.append('\\hline')
    body.extend([' & '.join(map(str, r)) + ' \\\\' for r in rows])
    body += ['\\hline', '\\end{tabular}', '\\end{table}']
    return '\n'.join(body)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Collect reviewer revision results into tables')
    parser.add_argument('--input-dir', default='paper_results/reviewer_revision')
    parser.add_argument('--output-dir', default='paper_results/reviewer_revision/tables')
    args = parser.parse_args()

    strict_results = load_json(os.path.join(args.input_dir, 'strict_augmentation_results.json')) or {}
    stats = load_json(os.path.join(args.input_dir, 'statistical_significance.json')) or {}
    dist = load_json(os.path.join(args.input_dir, 'aggregate_distribution_consistency.json')) or {}
    cond = load_json(os.path.join(args.input_dir, 'condition_ablation.json')) or {}
    erd = load_json(os.path.join(args.input_dir, 'erd_ers_quantification.json')) or {}
    ch = load_json(os.path.join(args.input_dir, 'channelwise_consistency.json')) or {}
    direct = load_json(os.path.join(args.input_dir, 'direct_regression_baseline.json')) or {}

    tables = {
        'main_results': {
            'headers': ['Protocol', '9ch', '22ch', 'Generated-only', 'Real+Generated'],
            'rows': build_main_results(strict_results),
            'caption': 'Classification results under the strict augmentation protocol.',
            'label': 'tab:strict_main_results',
        },
        'statistics': {
            'headers': ['Protocol', 'Comparison', 'Mean gain (%)', 'p', 'Holm p', 'Effect size'],
            'rows': build_stats_rows(stats),
            'caption': 'Paired statistical comparisons across subjects.',
            'label': 'tab:statistics',
        },
        'distribution_consistency': {
            'headers': ['Metric', 'Mean', 'Std', 'Median'],
            'rows': build_distribution_rows(dist),
            'caption': 'Aggregate distribution consistency over all subjects.',
            'label': 'tab:distribution_consistency',
        },
        'condition_ablation': {
            'headers': ['Protocol', 'Full condition', 'No task condition'],
            'rows': build_condition_rows(cond),
            'caption': 'Effect of task conditioning on augmentation performance.',
            'label': 'tab:condition_ablation',
        },
        'erd_ers': {
            'headers': ['Metric', 'Mean', 'Std'],
            'rows': build_erd_rows(erd),
            'caption': 'Group-level ERD/ERS quantification.',
            'label': 'tab:erd_ers',
        },
        'channelwise_consistency': {
            'headers': ['Missing ch', 'NMSE', 'Waveform corr', 'PSD corr'],
            'rows': build_channel_rows(ch),
            'caption': 'Channel-wise consistency on missing electrodes.',
            'label': 'tab:channelwise',
        },
        'direct_regression': {
            'headers': ['Protocol', 'Generated-only', 'Real+Generated'],
            'rows': build_direct_rows(direct),
            'caption': 'Direct regression baseline under the same augmentation protocol.',
            'label': 'tab:direct_regression',
        },
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'reviewer_tables.json'), 'w', encoding='utf-8') as f:
        json.dump(tables, f, indent=2)

    md_parts, tex_parts = [], []
    for name, table in tables.items():
        md_parts.append(f'## {name}\n\n' + md_table(table['headers'], table['rows']))
        tex_parts.append(latex_table(table['headers'], table['rows'], table['caption'], table['label']))

    with open(os.path.join(args.output_dir, 'reviewer_tables.md'), 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(md_parts))
    with open(os.path.join(args.output_dir, 'reviewer_tables.tex'), 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(tex_parts))

    print(f'Saved tables to: {args.output_dir}')


if __name__ == '__main__':
    main()
