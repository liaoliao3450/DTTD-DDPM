"""基于缓存数据快速训练DTTD PhysioNet模型"""
import os
import sys
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.dttd_physionet import DTTDPhysioNet

CHANNEL_NAMES_64 = [
    'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'T9', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'O1', 'Oz', 'O2', 'Iz'
]
INPUT_CHANNELS_16 = [
    'FC3', 'FC1', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4'
]
INPUT_CHANNEL_INDICES_16 = [CHANNEL_NAMES_64.index(ch) for ch in INPUT_CHANNELS_16]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载缓存数据
    cache_path = 'paper_results/physionet_mi/physionet_mi_cache.npz'
    cache = np.load(cache_path)
    data = cache['data'].astype(np.float32)
    labels = cache['labels'].astype(np.int64)

    print(f"数据形状: {data.shape}, 标签分布: {np.bincount(labels)}")

    # 标准化数据（不用1e5缩放，避免loss爆炸）
    data_mean = data.mean(axis=0, keepdims=True)
    data_std = data.std(axis=0, keepdims=True) + 1e-8
    scaled_data = (data - data_mean) / data_std

    # 划分训练/验证
    n_total = len(scaled_data)
    n_train = int(0.8 * n_total)
    indices = np.random.permutation(n_total)

    train_data = torch.FloatTensor(scaled_data[indices[:n_train]])
    train_labels = torch.LongTensor(labels[indices[:n_train]])
    val_data = torch.FloatTensor(scaled_data[indices[n_train:]])
    val_labels = torch.LongTensor(labels[indices[n_train:]])

    print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}")

    # 创建模型
    model_config = {
        'input_channels': 16,
        'output_channels': 64,
        'time_steps': 640,
        'embed_dim': 256,
        'task_dim': 64,
        'num_classes': 4,
        'num_heads': 8,
        'dropout': 0.1,
        'num_timesteps': 1000,
        'beta_start': 1e-4,
        'beta_end': 0.02,
        'schedule_type': 'linear',
        'fs': 160,
        'loss_type': 'l2',
        'use_classifier_guidance': True,
        'lambda_cls': 0.1,
        'warmup_epochs': 30,
    }

    model = DTTDPhysioNet(model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params/1e6:.2f}M")

    ch_idx = torch.tensor(INPUT_CHANNEL_INDICES_16, device=device)

    # 训练
    epochs = 200
    batch_size = 32
    lr = 1e-4

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_dataset = TensorDataset(val_data, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    save_path = 'paper_results/physionet_mi/dttd_physionet_best.pth'
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            loss = model.compute_loss(
                x_target=batch_data,
                channel_indices=ch_idx,
                task_label=batch_labels,
                loss_type='l2',
                current_epoch=epoch,
                noise_scale=0.1
            )

            if isinstance(loss, dict):
                loss_value = loss['total_loss']
            else:
                loss_value = loss

            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss_value.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches

        # 验证
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                loss = model.compute_loss(
                    x_target=batch_data,
                    channel_indices=ch_idx,
                    task_label=batch_labels,
                    loss_type='l2',
                    current_epoch=epoch,
                    noise_scale=0.1
                )
                if isinstance(loss, dict):
                    val_loss += loss['total_loss'].item()
                else:
                    val_loss += loss.item()
                n_val += 1

        avg_val_loss = val_loss / n_val

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'val_loss': avg_val_loss,
                'data_mean': data_mean,
                'data_std': data_std,
                'input_ch_indices': INPUT_CHANNEL_INDICES_16,
                'n_input_ch': 16,
                'n_output_ch': 64,
                'time_steps': 640,
                'num_classes': 4,
            }, save_path)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Train Loss: {avg_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}, Best: {best_loss:.6f}")

    print(f"\n训练完成! Best Val Loss: {best_loss:.6f}, 保存至: {save_path}")


if __name__ == '__main__':
    main()
