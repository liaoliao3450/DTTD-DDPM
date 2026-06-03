"""
在HGD和PhysioNet MI数据集上训练CVAE和cGAN基线模型

大数据集(128ch/64ch)使用转置卷积解码器，避免全连接层参数爆炸

Usage:
    python experiments/train_baselines_multidataset.py --dataset hgd --model cvae
    python experiments/train_baselines_multidataset.py --dataset hgd --model cgan
    python experiments/train_baselines_multidataset.py --dataset physionet --model cvae
    python experiments/train_baselines_multidataset.py --dataset physionet --model cgan
    python experiments/train_baselines_multidataset.py --dataset all --model all
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import get_device


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 适配大数据集的模型（使用转置卷积，避免全连接层参数爆炸）
# ============================================================

class CVAE_Large(nn.Module):
    """CVAE for large-channel datasets (HGD 128ch, PhysioNet 64ch)"""
    def __init__(self, input_channels, output_channels, time_steps, num_classes, latent_dim=256):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.time_steps = time_steps
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)
        )

        self.condition_embed = nn.Embedding(num_classes, 64)

        enc_out_dim = 256 * 32
        self.fc_mu = nn.Linear(enc_out_dim + 64, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim + 64, latent_dim)

        # 解码器 - 使用转置卷积
        self.decoder_fc = nn.Linear(latent_dim + 64, 256 * 64)

        self.decoder_conv = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.ConvTranspose1d(256, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, output_channels, kernel_size=3, padding=1),
        )

    def encode(self, x, condition):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        c = self.condition_embed(condition)
        h = torch.cat([h, c], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, condition):
        c = self.condition_embed(condition)
        z_c = torch.cat([z, c], dim=1)
        h = self.decoder_fc(z_c)
        h = h.view(h.size(0), 256, -1)
        h = self.decoder_conv(h)
        # 确保输出时间维度正确
        if h.size(2) > self.time_steps:
            h = h[:, :, :self.time_steps]
        elif h.size(2) < self.time_steps:
            h = F.pad(h, (0, self.time_steps - h.size(2)))
        return h

    def forward(self, x, condition):
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, condition)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + 0.01 * kl_loss


class Generator_Large(nn.Module):
    """cGAN Generator for large-channel datasets"""
    def __init__(self, input_channels, output_channels, time_steps, num_classes, latent_dim=128):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.time_steps = time_steps
        self.latent_dim = latent_dim

        # 输入编码器
        self.input_encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)
        )

        self.condition_embed = nn.Embedding(num_classes, 64)

        input_feature_dim = 256 * 16
        self.fc = nn.Linear(latent_dim + 64 + input_feature_dim, 256 * 64)

        self.deconv = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.ConvTranspose1d(256, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, output_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z, condition, input_data=None):
        batch_size = z.size(0)
        c = self.condition_embed(condition)

        if input_data is not None:
            input_features = self.input_encoder(input_data)
            input_features = input_features.view(batch_size, -1)
        else:
            input_features = torch.zeros(batch_size, 256 * 16).to(z.device)

        x = torch.cat([z, c, input_features], dim=1)
        x = self.fc(x)
        x = x.view(batch_size, 256, -1)
        x = self.deconv(x)
        if x.size(2) > self.time_steps:
            x = x[:, :, :self.time_steps]
        elif x.size(2) < self.time_steps:
            x = F.pad(x, (0, self.time_steps - x.size(2)))
        return x


class Discriminator_Large(nn.Module):
    """cGAN Discriminator for large-channel datasets"""
    def __init__(self, num_channels, time_steps, num_classes):
        super().__init__()
        self.condition_embed = nn.Embedding(num_classes, 1)

        self.main = nn.Sequential(
            nn.Conv1d(num_channels + 1, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(512, 1)

    def forward(self, x, condition):
        batch_size, _, time_dim = x.size()
        c = self.condition_embed(condition).unsqueeze(2).expand(batch_size, 1, time_dim)
        x = torch.cat([x, c], dim=1)
        x = self.main(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============================================================
# 数据加载
# ============================================================

def load_hgd_data(data_path):
    """加载HGD数据"""
    from data.high_gamma_dataset import HighGammaDataset
    dataset = HighGammaDataset(data_path, subject_ids=list(range(1, 15)),
                               sessions='both', fs_target=250)
    data = dataset.data
    labels = dataset.labels
    input_ch_indices = dataset.input_ch_indices
    n_output_ch = dataset.num_output_channels
    time_steps = data.shape[2]

    # 标准化
    data_mean = data.mean(axis=0, keepdims=True)
    data_std = data.std(axis=0, keepdims=True) + 1e-8
    data = (data - data_mean) / data_std

    # 划分训练/验证
    n_total = len(data)
    n_train = int(0.8 * n_total)
    indices = np.random.permutation(n_total)

    train_data = torch.FloatTensor(data[indices[:n_train]])
    train_labels = torch.LongTensor(labels[indices[:n_train]])
    val_data = torch.FloatTensor(data[indices[n_train:]])
    val_labels = torch.LongTensor(labels[indices[n_train:]])

    return {
        'train_data': train_data, 'train_labels': train_labels,
        'val_data': val_data, 'val_labels': val_labels,
        'input_ch_indices': input_ch_indices,
        'n_input_ch': len(input_ch_indices),
        'n_output_ch': n_output_ch,
        'time_steps': time_steps,
        'num_classes': int(labels.max()) + 1,
        'data_mean': data_mean, 'data_std': data_std,
    }


def load_physionet_data():
    """加载PhysioNet MI数据（使用与分类评估一致的all_subjects缓存）"""
    cache_path = 'paper_results/physionet_mi/physionet_mi_all_subjects.npz'
    if not os.path.exists(cache_path):
        print("缓存不存在，请先运行physionet_mi_classification.py生成all_subjects缓存")
        raise FileNotFoundError(cache_path)

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

    # 标准化
    data_mean = data.mean(axis=0, keepdims=True)
    data_std = data.std(axis=0, keepdims=True) + 1e-8
    data = (data - data_mean) / data_std

    # 四分类: T1(左拳), T2(右拳), T3(双脚), T4(休息)
    # labels已经是0,1,2,3，无需过滤

    # 划分训练/验证
    n_total = len(data)
    n_train = int(0.8 * n_total)
    indices = np.random.permutation(n_total)

    train_data = torch.FloatTensor(data[indices[:n_train]])
    train_labels = torch.LongTensor(labels[indices[:n_train]])
    val_data = torch.FloatTensor(data[indices[n_train:]])
    val_labels = torch.LongTensor(labels[indices[n_train:]])

    return {
        'train_data': train_data, 'train_labels': train_labels,
        'val_data': val_data, 'val_labels': val_labels,
        'input_ch_indices': input_ch_indices,
        'n_input_ch': len(input_ch_indices),
        'n_output_ch': 64,
        'time_steps': data.shape[2],
        'num_classes': 4,
        'data_mean': data_mean, 'data_std': data_std,
    }


# ============================================================
# 训练函数
# ============================================================

def train_cvae(dataset_info, output_dir, device, epochs=200, batch_size=32, lr=1e-3):
    """训练CVAE模型"""
    os.makedirs(output_dir, exist_ok=True)

    n_input = dataset_info['n_input_ch']
    n_output = dataset_info['n_output_ch']
    time_steps = dataset_info['time_steps']
    num_classes = dataset_info['num_classes']
    input_ch_indices = dataset_info['input_ch_indices']

    model = CVAE_Large(input_channels=n_input, output_channels=n_output,
                       time_steps=time_steps, num_classes=num_classes, latent_dim=256).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {n_params/1e6:.2f}M")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_dataset = TensorDataset(dataset_info['train_data'], dataset_info['train_labels'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    val_dataset = TensorDataset(dataset_info['val_data'], dataset_info['val_labels'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        n_batches = 0
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            input_data = batch_data[:, input_ch_indices, :]

            recon_x, mu, logvar = model(input_data, batch_labels)
            loss = model.loss_function(recon_x, batch_data, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / n_batches

        # Validate
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                input_data = batch_data[:, input_ch_indices, :]

                recon_x, mu, logvar = model(input_data, batch_labels)
                loss = model.loss_function(recon_x, batch_data, mu, logvar)
                val_loss += loss.item()
                n_val += 1

        avg_val_loss = val_loss / n_val

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': 'CVAE_Large',
                'input_ch_indices': input_ch_indices,
                'n_input_ch': n_input,
                'n_output_ch': n_output,
                'time_steps': time_steps,
                'num_classes': num_classes,
                'latent_dim': 256,
                'data_mean': dataset_info['data_mean'],
                'data_std': dataset_info['data_std'],
            }, os.path.join(output_dir, 'best_model.pth'))

        if epoch % 10 == 0:
            print(f'  Epoch {epoch}/{epochs}, Train Loss: {avg_train_loss:.6f}, '
                  f'Val Loss: {avg_val_loss:.6f}, Best: {best_val_loss:.6f}')

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': 'CVAE_Large',
        'input_ch_indices': input_ch_indices,
        'n_input_ch': n_input,
        'n_output_ch': n_output,
        'time_steps': time_steps,
        'num_classes': num_classes,
        'latent_dim': 256,
        'data_mean': dataset_info['data_mean'],
        'data_std': dataset_info['data_std'],
    }, os.path.join(output_dir, 'final_model.pth'))

    print(f'  CVAE训练完成, Best Val Loss: {best_val_loss:.6f}')
    return model


def train_cgan(dataset_info, output_dir, device, epochs=200, batch_size=32, lr=2e-4):
    """训练cGAN模型"""
    os.makedirs(output_dir, exist_ok=True)

    n_input = dataset_info['n_input_ch']
    n_output = dataset_info['n_output_ch']
    time_steps = dataset_info['time_steps']
    num_classes = dataset_info['num_classes']
    input_ch_indices = dataset_info['input_ch_indices']
    latent_dim = 128

    generator = Generator_Large(input_channels=n_input, output_channels=n_output,
                                time_steps=time_steps, num_classes=num_classes,
                                latent_dim=latent_dim).to(device)
    discriminator = Discriminator_Large(n_output, time_steps, num_classes).to(device)

    n_g_params = sum(p.numel() for p in generator.parameters())
    n_d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"  Generator参数量: {n_g_params/1e6:.2f}M, Discriminator参数量: {n_d_params/1e6:.2f}M")

    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    g_scheduler = optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=epochs)
    d_scheduler = optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max=epochs)

    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    train_dataset = TensorDataset(dataset_info['train_data'], dataset_info['train_labels'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    val_dataset = TensorDataset(dataset_info['val_data'], dataset_info['val_labels'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        g_loss_sum = 0
        d_loss_sum = 0
        n_batches = 0

        for batch_data, batch_labels in train_loader:
            batch_size_actual = batch_data.size(0)
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            input_data = batch_data[:, input_ch_indices, :]

            real_label = torch.ones(batch_size_actual, 1).to(device)
            fake_label = torch.zeros(batch_size_actual, 1).to(device)

            # Train Discriminator
            z = torch.randn(batch_size_actual, latent_dim).to(device)
            fake_data = generator(z, batch_labels, input_data)

            d_real = discriminator(batch_data, batch_labels)
            d_fake = discriminator(fake_data.detach(), batch_labels)

            d_loss = bce_loss(d_real, real_label) + bce_loss(d_fake, fake_label)

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            z = torch.randn(batch_size_actual, latent_dim).to(device)
            fake_data = generator(z, batch_labels, input_data)
            d_fake = discriminator(fake_data, batch_labels)

            g_adv_loss = bce_loss(d_fake, real_label)
            g_recon_loss = mse_loss(fake_data, batch_data)
            g_loss = g_adv_loss + 10.0 * g_recon_loss

            g_optimizer.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            g_optimizer.step()

            g_loss_sum += g_loss.item()
            d_loss_sum += d_loss.item()
            n_batches += 1

        g_scheduler.step()
        d_scheduler.step()

        # Validate
        generator.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                input_data = batch_data[:, input_ch_indices, :]

                z = torch.randn(batch_data.size(0), latent_dim).to(device)
                fake_data = generator(z, batch_labels, input_data)
                loss = mse_loss(fake_data, batch_data)
                val_loss += loss.item()
                n_val += 1

        avg_val_loss = val_loss / n_val

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': generator.state_dict(),
                'model_type': 'Generator_Large',
                'input_ch_indices': input_ch_indices,
                'n_input_ch': n_input,
                'n_output_ch': n_output,
                'time_steps': time_steps,
                'num_classes': num_classes,
                'latent_dim': latent_dim,
                'data_mean': dataset_info['data_mean'],
                'data_std': dataset_info['data_std'],
            }, os.path.join(output_dir, 'best_model.pth'))

        if epoch % 10 == 0:
            print(f'  Epoch {epoch}/{epochs}, G Loss: {g_loss_sum/n_batches:.6f}, '
                  f'D Loss: {d_loss_sum/n_batches:.6f}, Val MSE: {avg_val_loss:.6f}, '
                  f'Best: {best_val_loss:.6f}')

    # Save final model
    torch.save({
        'model_state_dict': generator.state_dict(),
        'model_type': 'Generator_Large',
        'input_ch_indices': input_ch_indices,
        'n_input_ch': n_input,
        'n_output_ch': n_output,
        'time_steps': time_steps,
        'num_classes': num_classes,
        'latent_dim': latent_dim,
        'data_mean': dataset_info['data_mean'],
        'data_std': dataset_info['data_std'],
    }, os.path.join(output_dir, 'final_model.pth'))

    print(f'  cGAN训练完成, Best Val MSE: {best_val_loss:.6f}')
    return generator


def main():
    parser = argparse.ArgumentParser(description='Train CVAE/cGAN baselines on HGD and PhysioNet MI')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['hgd', 'physionet', 'all'],
                        help='Dataset to train on')
    parser.add_argument('--model', type=str, default='all',
                        choices=['cvae', 'cgan', 'all'],
                        help='Model to train')
    parser.add_argument('--hgd-path', type=str, default='E:/data/HGD',
                        help='Path to HGD dataset')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    datasets = ['hgd', 'physionet'] if args.dataset == 'all' else [args.dataset]
    models = ['cvae', 'cgan'] if args.model == 'all' else [args.model]

    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"加载数据集: {dataset_name}")
        print(f"{'='*70}")

        if dataset_name == 'hgd':
            dataset_info = load_hgd_data(args.hgd_path)
        else:
            dataset_info = load_physionet_data()

        print(f"  输入通道: {dataset_info['n_input_ch']}, "
              f"输出通道: {dataset_info['n_output_ch']}, "
              f"时间步: {dataset_info['time_steps']}, "
              f"类别数: {dataset_info['num_classes']}")
        print(f"  训练样本: {len(dataset_info['train_data'])}, "
              f"验证样本: {len(dataset_info['val_data'])}")

        for model_name in models:
            print(f"\n{'='*70}")
            print(f"训练 {model_name.upper()} on {dataset_name.upper()}")
            print(f"{'='*70}")

            output_dir = f'checkpoints/{dataset_name}/baseline_{model_name}'

            if model_name == 'cvae':
                train_cvae(dataset_info, output_dir, device,
                          epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
            else:
                train_cgan(dataset_info, output_dir, device,
                          epochs=args.epochs, batch_size=args.batch_size, lr=2e-4)

    print(f"\n{'='*70}")
    print("所有训练完成!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
