"""在三个数据集上训练EEGDiff和BrainDiff基线模型"""
import os
import sys
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.baselines import EEGDiff, BrainDiff
from utils import get_device


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 数据加载
# ============================================================

def load_bci2a_data(data_path='E:/data/BCI2a'):
    """加载BCI2a数据"""
    from data.bci2a import BCI2aDataset
    dataset = BCI2aDataset(data_path, subject_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                           reconstruction_mode=True)
    data = dataset.data_full
    labels = np.array(dataset.labels)
    input_ch_indices = dataset.input_indices
    n_output_ch = data.shape[1]
    time_steps = data.shape[2]

    data_mean = data.mean(axis=0, keepdims=True)
    data_std = data.std(axis=0, keepdims=True) + 1e-8
    data = (data - data_mean) / data_std

    n_total = len(data)
    n_train = int(0.8 * n_total)
    indices = np.random.permutation(n_total)

    return {
        'train_data': torch.FloatTensor(data[indices[:n_train]]),
        'train_labels': torch.LongTensor(labels[indices[:n_train]]),
        'val_data': torch.FloatTensor(data[indices[n_train:]]),
        'val_labels': torch.LongTensor(labels[indices[n_train:]]),
        'input_ch_indices': input_ch_indices,
        'n_input_ch': len(input_ch_indices),
        'n_output_ch': n_output_ch,
        'time_steps': time_steps,
        'num_classes': int(labels.max()) + 1,
        'data_mean': data_mean,
        'data_std': data_std,
    }


def load_hgd_data(data_path='E:/data/HGD'):
    """加载HGD数据"""
    from data.high_gamma_dataset import HighGammaDataset
    dataset = HighGammaDataset(data_path, subject_ids=list(range(1, 15)),
                               sessions='both', fs_target=250)
    data = dataset.data
    labels = dataset.labels
    input_ch_indices = dataset.input_ch_indices
    n_output_ch = dataset.num_output_channels
    time_steps = data.shape[2]

    data_mean = data.mean(axis=0, keepdims=True)
    data_std = data.std(axis=0, keepdims=True) + 1e-8
    data = (data - data_mean) / data_std

    n_total = len(data)
    n_train = int(0.8 * n_total)
    indices = np.random.permutation(n_total)

    return {
        'train_data': torch.FloatTensor(data[indices[:n_train]]),
        'train_labels': torch.LongTensor(labels[indices[:n_train]]),
        'val_data': torch.FloatTensor(data[indices[n_train:]]),
        'val_labels': torch.LongTensor(labels[indices[n_train:]]),
        'input_ch_indices': input_ch_indices,
        'n_input_ch': len(input_ch_indices),
        'n_output_ch': n_output_ch,
        'time_steps': time_steps,
        'num_classes': int(labels.max()) + 1,
        'data_mean': data_mean,
        'data_std': data_std,
    }


def load_physionet_data():
    """加载PhysioNet MI数据（使用与分类评估一致的all_subjects缓存）"""
    cache_path = 'paper_results/physionet_mi/physionet_mi_all_subjects.npz'
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"缓存不存在: {cache_path}")

    cache = np.load(cache_path, allow_pickle=True)
    subj_map = cache['subject_data_map'].item()
    data = np.concatenate([v[0] for v in subj_map.values()]).astype(np.float32)
    labels = np.concatenate([v[1] for v in subj_map.values()]).astype(np.int64)

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
    input_ch_indices = [CHANNEL_NAMES_64.index(ch) for ch in INPUT_CHANNELS_16]

    data_mean = data.mean(axis=0, keepdims=True)
    data_std = data.std(axis=0, keepdims=True) + 1e-8
    data = (data - data_mean) / data_std

    n_total = len(data)
    n_train = int(0.8 * n_total)
    indices = np.random.permutation(n_total)

    return {
        'train_data': torch.FloatTensor(data[indices[:n_train]]),
        'train_labels': torch.LongTensor(labels[indices[:n_train]]),
        'val_data': torch.FloatTensor(data[indices[n_train:]]),
        'val_labels': torch.LongTensor(labels[indices[n_train:]]),
        'input_ch_indices': input_ch_indices,
        'n_input_ch': len(input_ch_indices),
        'n_output_ch': 64,
        'time_steps': data.shape[2],
        'num_classes': 4,
        'data_mean': data_mean,
        'data_std': data_std,
    }


# ============================================================
# 训练函数
# ============================================================

def train_diffusion_model(model_class, model_name, dataset_info, output_dir, device,
                          epochs=200, batch_size=32, lr=1e-4):
    """训练扩散模型（EEGDiff或BrainDiff）"""
    os.makedirs(output_dir, exist_ok=True)

    n_input = dataset_info['n_input_ch']
    n_output = dataset_info['n_output_ch']
    time_steps = dataset_info['time_steps']
    num_classes = dataset_info['num_classes']
    input_ch_indices = dataset_info['input_ch_indices']

    model = model_class(
        input_channels=n_input, output_channels=n_output,
        time_steps=time_steps, num_classes=num_classes,
        num_timesteps=1000, embed_dim=128
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {n_params/1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_dataset = TensorDataset(dataset_info['train_data'], dataset_info['train_labels'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    val_dataset = TensorDataset(dataset_info['val_data'], dataset_info['val_labels'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    ch_idx = torch.tensor(input_ch_indices, device=device)
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
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
                noise_scale=0.1
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / n_batches

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
                    noise_scale=0.1
                )
                val_loss += loss.item()
                n_val += 1

        avg_val_loss = val_loss / n_val

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': model_name,
                'input_ch_indices': input_ch_indices,
                'n_input_ch': n_input,
                'n_output_ch': n_output,
                'time_steps': time_steps,
                'num_classes': num_classes,
                'data_mean': dataset_info['data_mean'],
                'data_std': dataset_info['data_std'],
            }, os.path.join(output_dir, 'best_model.pth'))

        if epoch % 10 == 0:
            print(f'  Epoch {epoch}/{epochs}, Train Loss: {avg_train_loss:.6f}, '
                  f'Val Loss: {avg_val_loss:.6f}, Best: {best_loss:.6f}')

    print(f'  {model_name}训练完成, Best Val Loss: {best_loss:.6f}')
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['bci2a', 'hgd', 'physionet', 'all'])
    parser.add_argument('--model', type=str, default='all',
                        choices=['eegdiff', 'braindiff', 'all'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    device = get_device()
    set_seed(42)

    datasets = {}
    if args.dataset in ('bci2a', 'all'):
        print("加载BCI2a数据...")
        datasets['bci2a'] = load_bci2a_data()
    if args.dataset in ('hgd', 'all'):
        print("加载HGD数据...")
        datasets['hgd'] = load_hgd_data()
    if args.dataset in ('physionet', 'all'):
        print("加载PhysioNet MI数据...")
        datasets['physionet'] = load_physionet_data()

    models_to_train = []
    if args.model in ('eegdiff', 'all'):
        models_to_train.append(('EEGDiff', EEGDiff))
    if args.model in ('braindiff', 'all'):
        models_to_train.append(('BrainDiff', BrainDiff))

    output_dirs = {
        'bci2a': {
            'EEGDiff': 'checkpoints/bci2a/baseline_eegdiff',
            'BrainDiff': 'checkpoints/bci2a/baseline_braindiff',
        },
        'hgd': {
            'EEGDiff': 'checkpoints/hgd/baseline_eegdiff',
            'BrainDiff': 'checkpoints/hgd/baseline_braindiff',
        },
        'physionet': {
            'EEGDiff': 'checkpoints/physionet/baseline_eegdiff',
            'BrainDiff': 'checkpoints/physionet/baseline_braindiff',
        },
    }

    for ds_name, ds_info in datasets.items():
        for model_name, model_class in models_to_train:
            print(f"\n{'='*70}")
            print(f"训练 {model_name} on {ds_name.upper()}")
            print(f"  输入通道: {ds_info['n_input_ch']}, 输出通道: {ds_info['n_output_ch']}, "
                  f"时间步: {ds_info['time_steps']}, 类别数: {ds_info['num_classes']}")
            print(f"{'='*70}")

            out_dir = output_dirs[ds_name][model_name]
            train_diffusion_model(
                model_class, model_name, ds_info, out_dir, device,
                epochs=args.epochs, batch_size=args.batch_size
            )

    print(f"\n{'='*70}")
    print("所有训练完成!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
