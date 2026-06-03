import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.data_augmentation_eval import load_raw_bci2a, train_classifier, evaluate_classifier, get_device, CH_IDX_9


class DirectRegressionNet(nn.Module):
    def __init__(self, in_ch=9, out_ch=22):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Conv1d(128, out_ch, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


def train_regressor(train_9ch, train_22ch, device, epochs=80):
    model = DirectRegressionNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(train_9ch), torch.FloatTensor(train_22ch)),
        batch_size=32, shuffle=True, drop_last=True
    )
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
    return model


def generate_with_regressor(model, data_9ch, device):
    loader = DataLoader(TensorDataset(torch.FloatTensor(data_9ch)), batch_size=32, shuffle=False)
    out = []
    model.eval()
    with torch.no_grad():
        for (x,) in loader:
            out.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(out, axis=0)


def evaluate_protocol(train_22ch, train_9ch, train_labels, test_22ch, test_9ch, test_labels, device):
    reg = train_regressor(train_9ch, train_22ch, device)
    gen_22 = generate_with_regressor(reg, train_9ch, device)

    clf = train_classifier(gen_22, train_labels, device, 22)
    gen_only = evaluate_classifier(clf, test_22ch, test_labels, device)
    del clf

    aug_data = np.concatenate([train_22ch, gen_22], axis=0)
    aug_labels = np.concatenate([train_labels, train_labels], axis=0)
    clf = train_classifier(aug_data, aug_labels, device, 22)
    aug = evaluate_classifier(clf, test_22ch, test_labels, device)
    del clf
    torch.cuda.empty_cache()
    return {'generated_only_22ch': gen_only, 'augmented': aug}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Direct regression baseline for reviewer revision')
    parser.add_argument('--data-path', default='E:/data/BCI2a')
    parser.add_argument('--output', default='paper_results/reviewer_revision/direct_regression_baseline.json')
    args = parser.parse_args()

    device = get_device()
    results = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    cross_session = {}
    for sid in range(1, 10):
        train_22ch, train_labels = load_raw_bci2a(args.data_path, sid, 'T')
        test_22ch, test_labels = load_raw_bci2a(args.data_path, sid, 'E')
        cross_session[f'S{sid}'] = evaluate_protocol(
            train_22ch, train_22ch[:, CH_IDX_9, :], train_labels,
            test_22ch, test_22ch[:, CH_IDX_9, :], test_labels,
            device,
        )

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
        test_t, labels_t = load_raw_bci2a(args.data_path, test_sid, 'T')
        test_e, labels_e = load_raw_bci2a(args.data_path, test_sid, 'E')
        test_22ch = np.concatenate([test_t, test_e], axis=0)
        test_labels = np.concatenate([labels_t, labels_e], axis=0)
        cross_subject[f'S{test_sid}'] = evaluate_protocol(
            train_22ch, train_22ch[:, CH_IDX_9, :], train_labels,
            test_22ch, test_22ch[:, CH_IDX_9, :], test_labels,
            device,
        )

    results['cross_session'] = cross_session
    results['cross_subject'] = cross_subject
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
