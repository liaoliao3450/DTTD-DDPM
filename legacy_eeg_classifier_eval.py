"""
使用 legacy EEGClassifier 复现三种场景的 22/9 通道基线和 DTTD 增强：
1) 被试内 (Within-Subject, 5 折)
2) 跨会话 (Cross-Session, T→E)
3) 跨被试 (Cross-Subject, LOSO)

包含 DTTD 数据增强评估，与 classification_eval.py 保持一致。
"""
import os
import sys
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.dttd_enhanced_v2 import DTTDEnhanced
from utils import load_config, get_device

# 9 通道索引（与现有工程保持一致）
CH_IDX_9 = [7, 9, 11, 1, 3, 5, 13, 15, 17]


def load_raw_bci2a(data_path, subject_id, session="T"):
    """加载原始 BCI2a 数据 (与 classification_eval 保持一致)."""
    subject_str = f"0{subject_id}" if subject_id < 10 else str(subject_id)
    file_path = os.path.join(data_path, f"A{subject_str}{session}.mat")

    mat_data = loadmat(file_path)

    if "data" in mat_data:
        data = mat_data["data"]
    elif "X" in mat_data:
        data = mat_data["X"]
    else:
        max_key = max(
            mat_data.keys(),
            key=lambda k: mat_data[k].size if isinstance(mat_data[k], np.ndarray) else 0,
        )
        data = mat_data[max_key]

    if "label" in mat_data:
        labels = mat_data["label"].flatten()
    elif "labels" in mat_data:
        labels = mat_data["labels"].flatten()
    elif "y" in mat_data:
        labels = mat_data["y"].flatten()
    else:
        labels = mat_data["Y"].flatten()

    labels = labels.astype(np.int64)
    if labels.min() > 0:
        labels = labels - labels.min()

    return data.astype(np.float32), labels


class EEGClassifier(nn.Module):
    """
    legacy EEGClassifier (基于 EEGNet 架构)
    从 DS-DDPM 工程中简化移植，仅保留分类功能。
    """

    def __init__(
        self,
        channels: int = 22,
        n_samples: int = 1000,
        num_classes: int = 4,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.channels = channels
        self.n_samples = n_samples
        self.num_classes = num_classes

        # 时间卷积块
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)

        # 深度卷积块
        self.depthwiseConv = nn.Conv2d(
            F1, D * F1, (channels, 1), groups=F1, bias=False
        )
        self.batchnorm2 = nn.BatchNorm2d(D * F1)
        self.activation = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # 可分离卷积块
        self.separableConv = nn.Conv2d(
            D * F1, F2, (1, 16), padding=(0, 8), bias=False
        )
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # 计算展平后特征维度
        with torch.no_grad():
            x = torch.zeros(1, 1, self.channels, self.n_samples)
            x = self.conv1(x)
            x = self.batchnorm1(x)
            x = self.depthwiseConv(x)
            x = self.batchnorm2(x)
            x = self.activation(x)
            x = self.avgpool1(x)
            x = self.separableConv(x)
            x = self.batchnorm3(x)
            x = self.activation(x)
            x = self.avgpool2(x)
            feature_size = x.numel()

        self.classifier = nn.Linear(feature_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B, 1, C, T]
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwiseConv(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        x = self.separableConv(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits


def train_legacy_classifier(train_data, train_labels, device, num_channels=22, epochs=100):
    """使用 legacy EEGClassifier 训练分类器（与 classification_eval.py 保持一致）。"""
    clf = EEGClassifier(channels=num_channels, n_samples=train_data.shape[2]).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)  # 添加权重衰减
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)  # 添加学习率调度器

    # 使用固定随机种子生成器，确保可复现
    generator = torch.Generator()
    generator.manual_seed(42)
    
    loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(train_data), torch.LongTensor(train_labels)
        ),
        batch_size=32,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )

    clf.train()
    for _ in range(epochs):
        total_loss = 0.0
        for data, labels in loader:
            optimizer.zero_grad()
            logits = clf(data.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step(total_loss / len(loader))  # 根据平均loss调整学习率
    return clf


def eval_classifier(clf, test_data, test_labels, device):
    """评估分类器，返回 acc / f1 / kappa。"""
    clf.eval()
    loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(test_data), torch.LongTensor(test_labels)
        ),
        batch_size=64,
        shuffle=False,
    )
    preds, labels_all = [], []
    with torch.no_grad():
        for data, labels in loader:
            logits = clf(data.to(device))
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels_all.extend(labels.numpy())

    preds = np.asarray(preds)
    labels_all = np.asarray(labels_all)
    return {
        "accuracy": float(accuracy_score(labels_all, preds)),
        "f1": float(f1_score(labels_all, preds, average="macro")),
        "kappa": float(cohen_kappa_score(labels_all, preds)),
    }


def load_dttd_model(config_path, checkpoint_path, device):
    """加载DTTD模型"""
    config = load_config(config_path)
    model = DTTDEnhanced(config['model']).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    model.eval()
    data_scale_factor = ckpt.get('data_scale_factor', 1e5)
    print(f"[OK] 加载DTTD模型: {checkpoint_path}")
    return model, data_scale_factor


def generate_22ch_data(model, data_9ch, labels, device, data_scale_factor=1e5, guidance_scale=3.0):
    """用DTTD模型从9通道生成22通道数据"""
    model.eval()
    data_9ch_scaled = data_9ch * data_scale_factor
    
    loader = DataLoader(TensorDataset(torch.FloatTensor(data_9ch_scaled), torch.LongTensor(labels)),
                        batch_size=32, shuffle=False)
    
    generated_list = []
    with torch.no_grad():
        for batch_data, batch_labels in loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            if hasattr(model, 'sample'):
                gen = model.sample(batch_data, task_label=batch_labels, num_steps=10, 
                                   guidance_scale=guidance_scale).cpu().numpy()
            else:
                # 兼容旧版本
                gen = model.generate(batch_data, batch_labels).cpu().numpy()
            
            generated_list.append(gen / data_scale_factor)
    
    return np.concatenate(generated_list, axis=0)


# ==================== 三种评估场景 ====================

def within_subject_eval_legacy(data_path, model, device, data_scale_factor, subject_ids=range(1, 10), guidance_scale=3.0):
    """被试内 - 5 折 (与现有实现保持划分一致，仅换分类器)."""
    print("\n" + "=" * 60)
    print("被试内测试 (Within-Subject) - 5折交叉验证")
    print("=" * 60)

    results = {}

    for sid in subject_ids:
        print(f"\n被试 S{sid}:")
        data_t, labels_t = load_raw_bci2a(data_path, sid, "T")
        data_e, labels_e = load_raw_bci2a(data_path, sid, "E")
        all_data_22 = np.concatenate([data_t, data_e], axis=0)
        all_labels = np.concatenate([labels_t, labels_e], axis=0)
        all_data_9 = all_data_22[:, CH_IDX_9, :]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        accs_22, accs_9, accs_aug = [], [], []
        for fold, (train_idx, test_idx) in enumerate(skf.split(all_data_22, all_labels)):
            X_train_22, X_test_22 = all_data_22[train_idx], all_data_22[test_idx]
            y_train, y_test = all_labels[train_idx], all_labels[test_idx]
            X_train_9, X_test_9 = all_data_9[train_idx], all_data_9[test_idx]

            # 基线1：22通道
            clf22 = train_legacy_classifier(X_train_22, y_train, device, num_channels=22)
            metrics_22 = eval_classifier(clf22, X_test_22, y_test, device)
            accs_22.append(metrics_22["accuracy"])
            del clf22

            # 基线2：9通道
            clf9 = train_legacy_classifier(X_train_9, y_train, device, num_channels=9)
            metrics_9 = eval_classifier(clf9, X_test_9, y_test, device)
            accs_9.append(metrics_9["accuracy"])
            del clf9

            # DTTD增强
            gen_22ch = generate_22ch_data(model, X_train_9, y_train, device, 
                                          data_scale_factor, guidance_scale)
            aug_data = np.concatenate([X_train_22, gen_22ch], axis=0)
            aug_labels = np.concatenate([y_train, y_train], axis=0)
            
            clf_aug = train_legacy_classifier(aug_data, aug_labels, device, num_channels=22)
            metrics_aug = eval_classifier(clf_aug, X_test_22, y_test, device)
            accs_aug.append(metrics_aug["accuracy"])
            del clf_aug

            torch.cuda.empty_cache()

        mean_22, mean_9, mean_aug = np.mean(accs_22), np.mean(accs_9), np.mean(accs_aug)
        print(f"  22ch: {mean_22*100:.2f}%, 9ch: {mean_9*100:.2f}%, DTTD: {mean_aug*100:.2f}%")

        results[f"S{sid}"] = {
            "baseline_22ch": {"accuracy": float(mean_22), "std": float(np.std(accs_22))},
            "baseline_9ch": {"accuracy": float(mean_9), "std": float(np.std(accs_9))},
            "dttd_augmented": {"accuracy": float(mean_aug), "std": float(np.std(accs_aug))},
        }

    avg_22 = float(np.mean([r["baseline_22ch"]["accuracy"] for r in results.values()]))
    avg_9 = float(np.mean([r["baseline_9ch"]["accuracy"] for r in results.values()]))
    avg_aug = float(np.mean([r["dttd_augmented"]["accuracy"] for r in results.values()]))
    results["average"] = {"baseline_22ch": avg_22, "baseline_9ch": avg_9, "dttd_augmented": avg_aug}
    print(f"\n平均: 22ch={avg_22*100:.2f}%, 9ch={avg_9*100:.2f}%, DTTD={avg_aug*100:.2f}%")
    return results


def cross_session_eval_legacy(data_path, model, device, data_scale_factor, subject_ids=range(1, 10), guidance_scale=3.0):
    """跨会话 - T 训练, E 测试。"""
    print("\n" + "=" * 60)
    print("跨会话测试 (Cross-Session)")
    print("=" * 60)

    results = {}

    for sid in subject_ids:
        print(f"\n被试 S{sid}:")
        train_22, train_labels = load_raw_bci2a(data_path, sid, "T")
        test_22, test_labels = load_raw_bci2a(data_path, sid, "E")
        train_9 = train_22[:, CH_IDX_9, :]
        test_9 = test_22[:, CH_IDX_9, :]

        # 基线1：22通道
        clf22 = train_legacy_classifier(train_22, train_labels, device, num_channels=22)
        metrics_22 = eval_classifier(clf22, test_22, test_labels, device)
        print(f"  22ch: {metrics_22['accuracy']*100:.2f}%")
        del clf22

        # 基线2：9通道
        clf9 = train_legacy_classifier(train_9, train_labels, device, num_channels=9)
        metrics_9 = eval_classifier(clf9, test_9, test_labels, device)
        print(f"  9ch: {metrics_9['accuracy']*100:.2f}%")
        del clf9

        # DTTD增强
        gen_22ch = generate_22ch_data(model, train_9, train_labels, device, 
                                      data_scale_factor, guidance_scale)
        aug_data = np.concatenate([train_22, gen_22ch], axis=0)
        aug_labels = np.concatenate([train_labels, train_labels], axis=0)
        
        clf_aug = train_legacy_classifier(aug_data, aug_labels, device, num_channels=22)
        metrics_aug = eval_classifier(clf_aug, test_22, test_labels, device)
        print(f"  DTTD: {metrics_aug['accuracy']*100:.2f}%")
        del clf_aug

        results[f"S{sid}"] = {
            "baseline_22ch": metrics_22,
            "baseline_9ch": metrics_9,
            "dttd_augmented": metrics_aug,
        }
        torch.cuda.empty_cache()

    avg_22 = float(np.mean([r["baseline_22ch"]["accuracy"] for r in results.values()]))
    avg_9 = float(np.mean([r["baseline_9ch"]["accuracy"] for r in results.values()]))
    avg_aug = float(np.mean([r["dttd_augmented"]["accuracy"] for r in results.values()]))
    results["average"] = {"baseline_22ch": avg_22, "baseline_9ch": avg_9, "dttd_augmented": avg_aug}
    print(f"\n平均: 22ch={avg_22*100:.2f}%, 9ch={avg_9*100:.2f}%, DTTD={avg_aug*100:.2f}%")
    return results


def cross_subject_eval_legacy(data_path, model, device, data_scale_factor, subject_ids=range(1, 10), guidance_scale=3.0):
    """跨被试 - LOSO (与现有 cross_subject_loso_eval 一致，仅换分类器)."""
    print("\n" + "=" * 60)
    print("跨被试测试 (LOSO)")
    print("=" * 60)

    results = {}

    for test_sid in subject_ids:
        print(f"\n测试被试 S{test_sid}:")

        train_22_list, train_labels_list = [], []
        for sid in subject_ids:
            if sid == test_sid:
                continue
            # 只用第一个会话（Session T）作为训练集
            data, labels = load_raw_bci2a(data_path, sid, "T")
            train_22_list.append(data)
            train_labels_list.append(labels)

        train_22 = np.concatenate(train_22_list, axis=0)
        train_labels = np.concatenate(train_labels_list, axis=0)
        train_9 = train_22[:, CH_IDX_9, :]

        test_t, labels_t = load_raw_bci2a(data_path, test_sid, "T")
        test_e, labels_e = load_raw_bci2a(data_path, test_sid, "E")
        test_22 = np.concatenate([test_t, test_e], axis=0)
        test_labels = np.concatenate([labels_t, labels_e], axis=0)
        test_9 = test_22[:, CH_IDX_9, :]

        # 基线1：22通道
        clf22 = train_legacy_classifier(train_22, train_labels, device, num_channels=22)
        metrics_22 = eval_classifier(clf22, test_22, test_labels, device)
        print(f"  22ch: {metrics_22['accuracy']*100:.2f}%")
        del clf22

        # 基线2：9通道
        clf9 = train_legacy_classifier(train_9, train_labels, device, num_channels=9)
        metrics_9 = eval_classifier(clf9, test_9, test_labels, device)
        print(f"  9ch: {metrics_9['accuracy']*100:.2f}%")
        del clf9

        # DTTD增强
        gen_22ch = generate_22ch_data(model, train_9, train_labels, device, 
                                      data_scale_factor, guidance_scale)
        aug_data = np.concatenate([train_22, gen_22ch], axis=0)
        aug_labels = np.concatenate([train_labels, train_labels], axis=0)
        
        clf_aug = train_legacy_classifier(aug_data, aug_labels, device, num_channels=22)
        metrics_aug = eval_classifier(clf_aug, test_22, test_labels, device)
        print(f"  DTTD: {metrics_aug['accuracy']*100:.2f}%")
        del clf_aug

        results[f"S{test_sid}"] = {
            "baseline_22ch": metrics_22,
            "baseline_9ch": metrics_9,
            "dttd_augmented": metrics_aug,
        }
        torch.cuda.empty_cache()

    avg_22 = float(np.mean([r["baseline_22ch"]["accuracy"] for r in results.values()]))
    avg_9 = float(np.mean([r["baseline_9ch"]["accuracy"] for r in results.values()]))
    avg_aug = float(np.mean([r["dttd_augmented"]["accuracy"] for r in results.values()]))
    results["average"] = {"baseline_22ch": avg_22, "baseline_9ch": avg_9, "dttd_augmented": avg_aug}
    print(f"\n平均: 22ch={avg_22*100:.2f}%, 9ch={avg_9*100:.2f}%, DTTD={avg_aug*100:.2f}%")
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Legacy EEGClassifier baseline (3 场景 + DTTD)")
    parser.add_argument("--data-path", default="E:/data/BCI2a")
    parser.add_argument("--output-dir", default="paper_results/classification_legacy")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["within_subject", "cross_session", "cross_subject", "all"],
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dttd-config", default="configs/bci2a_enhanced_config.yaml", help="DTTD模型配置文件")
    parser.add_argument("--dttd-checkpoint", default="checkpoints/dttd/best_model.pth", help="DTTD模型检查点")
    parser.add_argument("--guidance-scale", type=float, default=3.0, help="DTTD生成时的guidance scale")
    args = parser.parse_args()

    # 设置固定随机种子，确保可复现
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = get_device()
    print(f"使用设备: {device}")
    print(f"随机种子: {args.seed}")

    # 加载DTTD模型
    dttd_config_path = os.path.join(project_root, args.dttd_config)
    dttd_checkpoint_path = os.path.join(project_root, args.dttd_checkpoint)
    if os.path.exists(dttd_config_path) and os.path.exists(dttd_checkpoint_path):
        model, data_scale_factor = load_dttd_model(dttd_config_path, dttd_checkpoint_path, device)
    else:
        print(f"[WARNING] DTTD模型文件不存在，跳过DTTD评估")
        print(f"  配置文件: {dttd_config_path}")
        print(f"  检查点: {dttd_checkpoint_path}")
        model, data_scale_factor = None, None

    results = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "seed": args.seed}

    if args.mode in ["within_subject", "all"]:
        if model is not None:
            results["within_subject"] = within_subject_eval_legacy(
                args.data_path, model, device, data_scale_factor, guidance_scale=args.guidance_scale
            )
        else:
            print("[ERROR] DTTD模型未加载，无法进行被试内评估")

    if args.mode in ["cross_session", "all"]:
        if model is not None:
            results["cross_session"] = cross_session_eval_legacy(
                args.data_path, model, device, data_scale_factor, guidance_scale=args.guidance_scale
            )
        else:
            print("[ERROR] DTTD模型未加载，无法进行跨会话评估")

    if args.mode in ["cross_subject", "all"]:
        if model is not None:
            results["cross_subject"] = cross_subject_eval_legacy(
                args.data_path, model, device, data_scale_factor, guidance_scale=args.guidance_scale
            )
        else:
            print("[ERROR] DTTD模型未加载，无法进行跨被试评估")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "classification_legacy_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {out_path}")


if __name__ == "__main__":
    main()


