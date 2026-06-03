"""Standalone script to reproduce Table 1 (reconstruction quality metrics)."""
from __future__ import annotations

import json
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from data import get_bci2a_dataloaders
from utils import load_config, get_device
from experiments.real_eval_utils import (
    ensure_output_dir,
    generate_samples_real,
    get_checkpoint_mapping,
    load_reconstruction_model,
    compute_reconstruction_metrics,
)


def run_table1(config_path: str = "configs/bci2a_enhanced_config.yaml") -> None:
    config = load_config(config_path)
    device = get_device()
    output_dir = ensure_output_dir()

    _, _, test_loader = get_bci2a_dataloaders(
        data_path=config["data"]["data_path"],
        batch_size=32,
        subject_ids=config["data"].get("subjects", list(range(1, 10))),
        num_workers=0,
        reconstruction_mode=True,
    )

    table_results = {}
    for model_name in get_checkpoint_mapping().keys():
        print(f"\n[Table1] Evaluating {model_name} ...")
        model = load_reconstruction_model(model_name, config, device)
        if model is None:
            continue

        generated, target, _ = generate_samples_real(model, model_name, test_loader, device, num_samples=200)
        metrics = compute_reconstruction_metrics(generated, target)
        table_results[model_name] = metrics

    traditional_path = Path(output_dir) / "traditional_baselines_results.json"
    if traditional_path.exists():
        print(f"\n[Table1] Merging traditional baselines from {traditional_path} ...")
        with open(traditional_path, "r", encoding="utf-8") as f:
            traditional_results = json.load(f)
        table_results.update(traditional_results)
    else:
        print(f"\n[WARN] Traditional baseline file missing: {traditional_path}")

    table1_json = Path(output_dir) / "table1_reconstruction.json"
    with open(table1_json, "w", encoding="utf-8") as f:
        json.dump(table_results, f, indent=2)
    print(f"\n[OK] Saved Table 1 results to {table1_json}")


if __name__ == "__main__":
    run_table1()
