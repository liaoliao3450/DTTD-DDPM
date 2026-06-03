"""Cross-dataset evaluation on the SEED dataset (Table 5)."""
from __future__ import annotations

import json
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from data import get_seed_dataloaders
from utils import load_config, get_device
from models import get_classifier
from experiments.real_eval_utils import (
    ensure_output_dir,
    load_reconstruction_model,
)


def collect_seed_dataset(loader):
    data_list, label_list = [], []
    for batch_data in tqdm(loader, desc="Collecting SEED data"):
        signals, labels = batch_data
        data_list.append(signals)
        label_list.append(labels)
    data = torch.cat(data_list, dim=0)
    labels = torch.cat(label_list, dim=0)
    return data, labels


def evaluate(classifier, data, labels, device):
    classifier.eval()
    dataset = TensorDataset(data, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_data, batch_labels in loader:
            outputs = classifier(batch_data.to(device))
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, f1


def run_seed_transfer(config_path: str = "configs/seed_config.yaml") -> None:
    config = load_config(config_path)
    device = get_device()
    output_dir = ensure_output_dir()

    train_loader, test_loader = get_seed_dataloaders(
        data_path=config['data']['data_path'],
        subject_ids=config['data'].get('subjects', list(range(1, 16))),
        batch_size=32,
        num_workers=0,
    )

    train_data, train_labels = collect_seed_dataset(train_loader)
    test_data, test_labels = collect_seed_dataset(test_loader)

    results = {}

    def train_classifier(num_channels):
        clf = get_classifier('eegnet', num_channels=num_channels, num_classes=3, time_steps=train_data.size(-1)).to(device)
        optimizer = torch.optim.Adam(clf.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        dataset = TensorDataset(train_data[:, :num_channels, :], train_labels)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        clf.train()
        for _ in range(20):
            for batch_data, batch_labels in loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad()
                loss = criterion(clf(batch_data), batch_labels)
                loss.backward()
                optimizer.step()
        return clf

    # Baseline: 9-channel subset
    print("Training 9-channel baseline classifier ...")
    clf_9 = train_classifier(num_channels=9)
    acc_9, f1_9 = evaluate(clf_9, test_data[:, :9, :], test_labels, device)
    results['9ch_baseline'] = {'accuracy': float(acc_9), 'f1': float(f1_9)}

    # Reconstruction from DTTD-DDPM (no fine-tune)
    print("Loading DTTD-DDPM for reconstruction ...")
    config_bci = load_config('configs/bci2a_enhanced_config.yaml')
    dttd_model = load_reconstruction_model('DTTD-DDPM', config_bci, device)
    
    # 检查时间步长是否匹配
    expected_time_steps = config_bci['model'].get('time_steps', 1000)
    actual_time_steps = test_data.size(-1)
    
    if actual_time_steps != expected_time_steps:
        print(f"⚠️ 时间步长不匹配: SEED={actual_time_steps}, 模型期望={expected_time_steps}")
        print("对SEED数据进行插值以匹配模型...")
        # 使用插值将SEED数据调整到模型期望的时间步长
        test_data_interp = torch.nn.functional.interpolate(
            test_data, size=expected_time_steps, mode='linear', align_corners=False
        )
        train_data_interp = torch.nn.functional.interpolate(
            train_data, size=expected_time_steps, mode='linear', align_corners=False
        )
    else:
        test_data_interp = test_data
        train_data_interp = train_data
    
    recon_dataset = TensorDataset(test_data_interp[:, :9, :], torch.zeros(len(test_data_interp), dtype=torch.long))
    recon_loader = DataLoader(recon_dataset, batch_size=32, shuffle=False)
    recon_list = []
    with torch.no_grad():
        for batch_data, _ in tqdm(recon_loader, desc="Reconstructing SEED"):
            noise_level = 0.02
            noisy_input = batch_data.to(device) + noise_level * torch.randn_like(batch_data).to(device)
            t = torch.zeros(noisy_input.size(0), dtype=torch.long, device=device)
            recon = dttd_model(noisy_input, t, None)
            recon_list.append(recon.cpu())
    recon_data = torch.cat(recon_list, dim=0)

    # Evaluate without fine-tuning
    # 将重建数据插值回原始时间步长以匹配分类器
    if actual_time_steps != expected_time_steps:
        recon_data_orig_time = torch.nn.functional.interpolate(
            recon_data, size=actual_time_steps, mode='linear', align_corners=False
        )
    else:
        recon_data_orig_time = recon_data
    
    acc_noft, f1_noft = evaluate(clf_9, recon_data_orig_time[:, :9, :], test_labels, device)
    results['dttd_no_finetune'] = {'accuracy': float(acc_noft), 'f1': float(f1_noft)}

    # Fine-tuned classifier on reconstructed data
    # 需要重建训练数据用于fine-tuning
    print("Reconstructing training data for fine-tuning ...")
    train_recon_dataset = TensorDataset(train_data_interp[:, :9, :], torch.zeros(len(train_data_interp), dtype=torch.long))
    train_recon_loader = DataLoader(train_recon_dataset, batch_size=32, shuffle=False)
    train_recon_list = []
    with torch.no_grad():
        for batch_data, _ in tqdm(train_recon_loader, desc="Reconstructing train SEED"):
            noise_level = 0.02
            noisy_input = batch_data.to(device) + noise_level * torch.randn_like(batch_data).to(device)
            t = torch.zeros(noisy_input.size(0), dtype=torch.long, device=device)
            recon = dttd_model(noisy_input, t, None)
            train_recon_list.append(recon.cpu())
    train_recon_data = torch.cat(train_recon_list, dim=0)
    
    # 插值回原始时间步长
    if actual_time_steps != expected_time_steps:
        train_recon_data_orig = torch.nn.functional.interpolate(
            train_recon_data, size=actual_time_steps, mode='linear', align_corners=False
        )
    else:
        train_recon_data_orig = train_recon_data
    
    print("Fine-tuning classifier on reconstructed SEED data ...")
    clf_ft = get_classifier('eegnet', num_channels=train_recon_data_orig.size(1), num_classes=3, time_steps=train_recon_data_orig.size(-1)).to(device)
    optimizer = torch.optim.Adam(clf_ft.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    dataset = TensorDataset(train_recon_data_orig, train_labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    clf_ft.train()
    for _ in range(20):
        for batch_data, batch_labels in loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            loss = criterion(clf_ft(batch_data), batch_labels)
            loss.backward()
            optimizer.step()
    acc_ft, f1_ft = evaluate(clf_ft, recon_data_orig_time, test_labels, device)
    results['dttd_finetuned'] = {'accuracy': float(acc_ft), 'f1': float(f1_ft)}

    output_json = Path(output_dir) / "seed_transfer.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Saved SEED transfer results to {output_json}")


if __name__ == '__main__':
    run_seed_transfer()
