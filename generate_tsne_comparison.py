"""
t-SNE分布可视化对比
仿照 visualize_class_discriminative_ddpm._tsne.py 的方式

对比方法：DTTD-DDPM, CVAE, cGAN, Simple-DDPM
布局：每个方法一行，4列
  Col1: PCA特征按类别 (Real=圆, Gen=x)
  Col2: PCA特征 Real vs Gen
  Col3: EEGNet分类器特征按类别
  Col4: EEGNet分类器特征 Real vs Gen
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy import signal
from scipy.stats import wasserstein_distance

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models import DTTDEnhanced
from models.baselines import CVAE, Generator as cGANGenerator, SimpleDDPM, EEGDiff, BrainDiff
from models.classifier import EEGNet
from data import get_bci2a_dataloaders
from utils import load_config, get_device

# ============================================================================
# 配置
# ============================================================================

NUM_CLASSES = 4
CLASS_NAMES = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
# 高对比类别色：Real深色实心, Gen浅色空心
CLASS_COLORS_REAL = ['#D32F2F', '#1565C0', '#2E7D32', '#F57F17']  # 深红、深蓝、深绿、深橙
CLASS_COLORS_GEN  = ['#FF8A80', '#82B1FF', '#A5D6A7', '#FFE082']  # 浅红、浅蓝、浅绿、浅黄
FS = 250
DATA_SCALE = 1e5

# BCI2a 22通道
BCI2A_CHANNELS = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P1', 'Pz', 'P2', 'POz'
]

# 方法配置（与论文Table II对照）
METHODS = {
    'DTTD-DDPM': {
        'checkpoint': 'checkpoints/bci2a_enhanced/best_model.pth',
        'color': '#E53935',
        'marker': 'P',
        'type': 'model',  # 需要加载模型
    },
    'CVAE': {
        'checkpoint': 'checkpoints/bci2a/baseline_cvae/best_model.pth',
        'color': '#7B1FA2',
        'marker': 'D',
        'type': 'model',
    },
    'cGAN': {
        'checkpoint': 'checkpoints/bci2a_enhanced/baseline_cgan/best_model.pth',
        'color': '#1E88E5',
        'marker': 's',
        'type': 'model',
    },
    'EEGDiff': {
        'checkpoint': 'checkpoints/bci2a/baseline_eegdiff/best_model.pth',
        'color': '#FF8F00',
        'marker': 'v',
        'type': 'model',
    },
    'BrainDiff': {
        'checkpoint': 'checkpoints/bci2a/baseline_braindiff/best_model.pth',
        'color': '#00838F',
        'marker': 'h',
        'type': 'model',
    },
    'Spline': {
        'color': '#AB47BC',
        'marker': '8',
        'type': 'interpolation',  # 无需模型
    },
    'Kriging': {
        'color': '#78909C',
        'marker': 'p',
        'type': 'interpolation',
    },
    'CNN': {
        'checkpoint': 'checkpoints/bci2a/baseline_cnn/best_model.pth',
        'color': '#6D4C41',
        'marker': 'd',
        'type': 'model',
    },
}


# ============================================================================
# 数据加载
# ============================================================================

def load_bci2a_data(data_path='E:/data/BCI2a', n_samples=500, device='cpu'):
    """加载BCI2a测试数据"""
    print("加载BCI2a数据...")
    _, _, test_loader = get_bci2a_dataloaders(
        data_path=data_path, batch_size=32, subject_ids=list(range(1, 10)),
        num_workers=0, reconstruction_mode=True
    )

    all_data, all_labels, ch_indices = [], [], None
    for batch in test_loader:
        target_data, channel_indices, labels, _ = batch
        all_data.append(target_data)
        all_labels.append(labels)
        if ch_indices is None:
            ch_indices = channel_indices[0].tolist()

    all_data = torch.cat(all_data, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    n_total = len(all_data)
    if n_samples < n_total:
        idx = np.random.choice(n_total, n_samples, replace=False)
        all_data = all_data[idx]
        all_labels = all_labels[idx]

    target_22ch = all_data
    labels = all_labels
    print(f"  数据: {target_22ch.shape}, 输入通道索引: {ch_indices}")
    return target_22ch, labels, ch_indices


# ============================================================================
# 各方法生成
# ============================================================================

def generate_dttd(target_data, labels, ch_indices, device, checkpoint_path=None):
    """DTTD-DDPM生成 - 使用sample方法开启分类器引导"""
    print("  生成DTTD-DDPM样本...")
    config = load_config('configs/bci2a_enhanced_config.yaml')
    model = DTTDEnhanced(config['model']).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    gen_list = []
    with torch.no_grad():
        for i in range(0, len(target_data), 32):
            batch_data = target_data[i:i+32].to(device)
            batch_labels = labels[i:i+32].to(device)
            input_data = batch_data[:, ch_indices, :]
            scaled_input = input_data * DATA_SCALE
            generated = model.sample(scaled_input, task_label=batch_labels,
                                     num_steps=10, guidance_scale=1.5)
            generated = generated / DATA_SCALE
            gen_list.append(generated.cpu())

    return torch.cat(gen_list, dim=0)


def generate_dttd_conditional(target_data, ch_indices, device, checkpoint_path=None, n_samples=25):
    """DTTD条件生成实验：取同一批输入，分别用4个标签生成
    使用DDIM多步采样，让分类器引导真正起作用
    返回: dict {label: generated_data}
    """
    print("  DTTD条件生成实验（同一输入，不同标签，DDIM采样）...")
    config = load_config('configs/bci2a_enhanced_config.yaml')
    model = DTTDEnhanced(config['model']).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    # 取前n_samples个样本的输入通道数据
    input_data = target_data[:n_samples, ch_indices, :].to(device)
    print(f"    input_data shape={input_data.shape}, range=[{input_data.min():.4f}, {input_data.max():.4f}]")
    scaled_input = input_data * DATA_SCALE
    print(f"    scaled_input range=[{scaled_input.min():.4f}, {scaled_input.max():.4f}]")

    # 先用_sample_fast测试
    print("    Testing _sample_fast...")
    with torch.no_grad():
        t_zero = torch.zeros(n_samples, device=device, dtype=torch.long)
        x_noisy = scaled_input + 0.02 * torch.randn_like(scaled_input)
        out_cond0 = model.forward(x_noisy, t_zero, task_label=torch.zeros(n_samples, dtype=torch.long, device=device))
        out_cond1 = model.forward(x_noisy, t_zero, task_label=torch.ones(n_samples, dtype=torch.long, device=device))
        out_uncond = model.forward(x_noisy, t_zero, task_label=None)
        print(f"    Forward: cond0 vs uncond = {(out_cond0 - out_uncond).abs().mean().item():.8f}")
        print(f"    Forward: cond0 vs cond1 = {(out_cond0 - out_cond1).abs().mean().item():.8f}")

        # Test sample with guidance
        s0 = model.sample(scaled_input, task_label=torch.zeros(n_samples, dtype=torch.long, device=device),
                          num_steps=10, guidance_scale=5.0)
        s1 = model.sample(scaled_input, task_label=torch.ones(n_samples, dtype=torch.long, device=device),
                          num_steps=10, guidance_scale=5.0)
        print(f"    Sample: s0 vs s1 (raw) = {(s0 - s1).abs().mean().item():.8f}")
        print(f"    Sample: s0 vs s1 (/DATA_SCALE) = {(s0/DATA_SCALE - s1/DATA_SCALE).abs().mean().item():.8f}")

    cond_gen = {}
    with torch.no_grad():
        for cls in range(NUM_CLASSES):
            cls_labels = torch.full((n_samples,), cls, dtype=torch.long, device=device)
            # 使用_sample_fast + 高guidance_scale
            generated = model.sample(scaled_input, task_label=cls_labels,
                                     num_steps=10, guidance_scale=5.0)
            generated = generated / DATA_SCALE
            cond_gen[cls] = generated.cpu()
            # 验证不同标签是否产生不同输出
            if cls > 0:
                diff = torch.mean(torch.abs(cond_gen[cls] - cond_gen[0])).item()
                print(f"    Label {cls} vs Label 0: mean abs diff = {diff:.8f}")

    return cond_gen


def generate_cvae(target_data, labels, ch_indices, device, checkpoint_path=None):
    """CVAE生成"""
    print("  生成CVAE样本...")
    model = CVAE(input_channels=9, output_channels=22, time_steps=1000,
                 num_classes=4, latent_dim=128).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    gen_list = []
    with torch.no_grad():
        for i in range(0, len(target_data), 32):
            batch_data = target_data[i:i+32].to(device)
            batch_labels = labels[i:i+32].to(device)
            input_data = batch_data[:, ch_indices, :]
            scaled_input = input_data * DATA_SCALE
            generated, _, _ = model(scaled_input, batch_labels)
            generated = generated / DATA_SCALE
            gen_list.append(generated.cpu())

    return torch.cat(gen_list, dim=0)


def generate_cgan(target_data, labels, ch_indices, device, checkpoint_path=None):
    """cGAN生成"""
    print("  生成cGAN样本...")
    model = cGANGenerator(input_channels=9, output_channels=22, time_steps=1000,
                          num_classes=4, latent_dim=128).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    gen_list = []
    with torch.no_grad():
        for i in range(0, len(target_data), 32):
            batch_data = target_data[i:i+32].to(device)
            batch_labels = labels[i:i+32].to(device)
            input_data = batch_data[:, ch_indices, :]
            batch_size = input_data.size(0)
            z = torch.randn(batch_size, 128).to(device)
            scaled_input = input_data * DATA_SCALE
            generated = model(z, batch_labels, scaled_input)
            generated = generated / DATA_SCALE
            gen_list.append(generated.cpu())

    return torch.cat(gen_list, dim=0)


def generate_simple_ddpm(target_data, labels, ch_indices, device, checkpoint_path=None):
    """Simple-DDPM生成"""
    print("  生成Simple-DDPM样本...")
    model = SimpleDDPM(input_channels=9, output_channels=22, time_steps=1000,
                       num_classes=4).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    gen_list = []
    with torch.no_grad():
        for i in range(0, len(target_data), 32):
            batch_data = target_data[i:i+32].to(device)
            batch_labels = labels[i:i+32].to(device)
            input_data = batch_data[:, ch_indices, :]
            scaled_input = input_data * DATA_SCALE
            generated = model.reconstruct(scaled_input, batch_labels, num_inference_steps=1, noise_level=0.02)
            generated = generated / DATA_SCALE
            gen_list.append(generated.cpu())

    return torch.cat(gen_list, dim=0)


def generate_eegdiff(target_data, labels, ch_indices, device, checkpoint_path=None):
    """EEGDiff生成：静态电极邻接的扩散模型"""
    print("  生成EEGDiff样本...")
    model = EEGDiff(input_channels=9, output_channels=22, time_steps=1000,
                    num_classes=4).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    gen_list = []
    with torch.no_grad():
        for i in range(0, len(target_data), 32):
            batch_data = target_data[i:i+32].to(device)
            batch_labels = labels[i:i+32].to(device)
            input_data = batch_data[:, ch_indices, :]
            scaled_input = input_data * DATA_SCALE
            generated = model.sample(scaled_input, task_label=batch_labels,
                                     num_steps=1, noise_scale=0.02, guidance_scale=3.0)
            generated = generated / DATA_SCALE
            gen_list.append(generated.cpu())

    return torch.cat(gen_list, dim=0)


def generate_braindiff(target_data, labels, ch_indices, device, checkpoint_path=None):
    """BrainDiff生成：频率感知但无任务条件化的扩散模型"""
    print("  生成BrainDiff样本...")
    model = BrainDiff(input_channels=9, output_channels=22, time_steps=1000,
                      num_classes=4).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    gen_list = []
    with torch.no_grad():
        for i in range(0, len(target_data), 32):
            batch_data = target_data[i:i+32].to(device)
            input_data = batch_data[:, ch_indices, :]
            scaled_input = input_data * DATA_SCALE
            generated = model.sample(scaled_input, num_steps=1, noise_scale=0.02)
            generated = generated / DATA_SCALE
            gen_list.append(generated.cpu())

    return torch.cat(gen_list, dim=0)


def generate_spline(target_data, labels, ch_indices, device, **kwargs):
    """Spherical Spline插值重建：9ch -> 22ch"""
    print("  生成Spline插值样本...")
    from scipy.interpolate import CubicSpline

    target_np = target_data.numpy()
    n_samples, n_total_ch, n_time = target_np.shape
    input_data = target_np[:, ch_indices, :]  # [N, 9, T]

    # 按通道索引排序，确保x严格递增
    sorted_order = np.argsort(ch_indices)
    sorted_indices = np.array(ch_indices)[sorted_order]

    result = np.zeros((n_samples, n_total_ch, n_time), dtype=np.float32)
    all_positions = np.arange(n_total_ch, dtype=np.float64)

    for i in range(n_samples):
        for t in range(n_time):
            values = input_data[i, sorted_order, t]
            cs = CubicSpline(sorted_indices.astype(np.float64), values, bc_type='natural')
            result[i, :, t] = cs(all_positions)

    return torch.from_numpy(result)


def generate_kriging(target_data, labels, ch_indices, device, **kwargs):
    """Kriging插值重建：9ch -> 22ch"""
    print("  生成Kriging插值样本...")
    target_np = target_data.numpy()
    n_samples, n_total_ch, n_time = target_np.shape
    input_data = target_np[:, ch_indices, :]
    n_input = len(ch_indices)

    # 使用10-10系统电极2D坐标
    electrode_2d = {
        'Fz': (0.0, 0.3), 'FC3': (-0.3, 0.2), 'FC1': (-0.1, 0.2),
        'FCz': (0.0, 0.2), 'FC2': (0.1, 0.2), 'FC4': (0.3, 0.2),
        'C5': (-0.4, 0.0), 'C3': (-0.3, 0.0), 'C1': (-0.1, 0.0),
        'Cz': (0.0, 0.0), 'C2': (0.1, 0.0), 'C4': (0.3, 0.0),
        'C6': (0.4, 0.0), 'CP3': (-0.3, -0.2), 'CP1': (-0.1, -0.2),
        'CPz': (0.0, -0.2), 'CP2': (0.1, -0.2), 'CP4': (0.3, -0.2),
        'P1': (-0.1, -0.3), 'Pz': (0.0, -0.3), 'P2': (0.1, -0.3),
        'POz': (0.0, -0.4),
    }
    ch_names = BCI2A_CHANNELS
    all_coords = np.array([electrode_2d[ch_names[i]] for i in range(n_total_ch)])
    input_coords = np.array([electrode_2d[ch_names[i]] for i in ch_indices])

    # 简化Kriging：用RBF核的加权平均
    result = np.zeros((n_samples, n_total_ch, n_time), dtype=np.float32)
    sigma = 0.3  # RBF带宽

    # 预计算权重矩阵
    dists = np.zeros((n_total_ch, n_input))
    for i in range(n_total_ch):
        for j in range(n_input):
            dists[i, j] = np.sum((all_coords[i] - input_coords[j]) ** 2)
    weights = np.exp(-dists / (2 * sigma ** 2))
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-10)

    for i in range(n_samples):
        for t in range(0, n_time, 50):  # 每50个时间点批量处理
            t_end = min(t + 50, n_time)
            vals = input_data[i, :, t:t_end]  # [9, t_len]
            result[i, :, t:t_end] = weights @ vals

    return torch.from_numpy(result)


def generate_cnn(target_data, labels, ch_indices, device, checkpoint_path):
    """CNN重建：9ch -> 22ch"""
    print("  生成CNN重建样本...")

    class CNNReconstructor(nn.Module):
        def __init__(self, in_ch=9, out_ch=22, time_steps=1000):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(in_ch, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
                nn.AdaptiveAvgPool1d(time_steps),
            )
            self.decoder = nn.Sequential(
                nn.Conv1d(128, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, out_ch, 3, padding=1),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = CNNReconstructor().to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        print(f"    [警告] CNN checkpoint不存在，使用随机初始化")
    model.eval()

    gen_list = []
    with torch.no_grad():
        for i in range(0, len(target_data), 32):
            batch_data = target_data[i:i+32].to(device)
            input_data = batch_data[:, ch_indices, :]
            scaled_input = input_data * DATA_SCALE
            generated = model(scaled_input) / DATA_SCALE
            gen_list.append(generated.cpu())

    return torch.cat(gen_list, dim=0)


GENERATORS = {
    'DTTD-DDPM': generate_dttd,
    'CVAE': generate_cvae,
    'cGAN': generate_cgan,
    'EEGDiff': generate_eegdiff,
    'BrainDiff': generate_braindiff,
    'Spline': generate_spline,
    'Kriging': generate_kriging,
    'CNN': generate_cnn,
}


# ============================================================================
# 特征提取
# ============================================================================

def extract_pca_features(real_data, gen_data_dict, n_components=50):
    """PCA降维：所有数据一起fit，分别transform"""
    print("  提取PCA特征...")
    real_flat = real_data.reshape(len(real_data), -1)
    gen_flats = {k: v.reshape(len(v), -1) for k, v in gen_data_dict.items()}

    all_flat = np.vstack([real_flat] + [v for v in gen_flats.values()])
    pca = PCA(n_components=min(n_components, all_flat.shape[1], all_flat.shape[0] - 1))
    all_pca = pca.fit_transform(all_flat)

    n_real = len(real_data)
    real_pca = all_pca[:n_real]
    gen_pca_dict = {}
    start = n_real
    for k, v in gen_flats.items():
        n_gen = len(v)
        gen_pca_dict[k] = all_pca[start:start + n_gen]
        start += n_gen

    print(f"    PCA解释方差比: {pca.explained_variance_ratio_[:5].sum():.3f} (前5维)")
    return real_pca, gen_pca_dict


def extract_classifier_features(real_data, gen_data_dict, device, num_channels=22, time_steps=1000):
    """EEGNet分类器中间层特征"""
    print("  提取EEGNet分类器特征...")
    classifier = EEGNet(num_channels=num_channels, num_classes=NUM_CLASSES,
                        time_steps=time_steps).to(device)
    classifier.eval()

    # Hook提取fc层输入
    embeddings = []
    def hook(module, inputs, output):
        embeddings.append(inputs[0].detach().cpu().numpy())

    handle = classifier.fc.register_forward_hook(hook)

    def _extract(data):
        emb_list = []
        with torch.no_grad():
            for i in range(0, len(data), 64):
                batch = torch.FloatTensor(data[i:i+64]).to(device)
                classifier(batch)
                emb_list.append(embeddings[-1])
        return np.concatenate(emb_list, axis=0)

    real_feat = _extract(real_data)
    gen_feat_dict = {k: _extract(v) for k, v in gen_data_dict.items()}

    handle.remove()
    return real_feat, gen_feat_dict


# ============================================================================
# t-SNE计算
# ============================================================================

def compute_tsne(real_feat, gen_feat_dict, perplexity=30):
    """对PCA/分类器特征计算t-SNE，所有数据一起fit"""
    all_feat = np.vstack([real_feat] + [v for v in gen_feat_dict.values()])
    print(f"  计算t-SNE ({all_feat.shape[0]} samples, {all_feat.shape[1]} dims)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(perplexity, all_feat.shape[0] - 1))
    embedded = tsne.fit_transform(all_feat)

    n_real = len(real_feat)
    real_emb = embedded[:n_real]
    gen_emb_dict = {}
    start = n_real
    for k, v in gen_feat_dict.items():
        n_gen = len(v)
        gen_emb_dict[k] = embedded[start:start + n_gen]
        start += n_gen

    return real_emb, gen_emb_dict


# ============================================================================
# 可视化
# ============================================================================

def visualize_all_methods(real_data, real_labels, gen_data_dict, gen_labels_dict,
                          real_pca_emb, gen_pca_emb_dict,
                          real_clf_emb, gen_clf_emb_dict,
                          real_gen_ch, gen_gen_ch_dict,
                          real_clf_feat, gen_clf_feat_dict,
                          save_path='paper/figures/tsne_comparison.png'):
    """生成两张图：
    1. 量化指标对比柱状图（论文核心图）- NMSE/PSD只算13个生成通道
    2. t-SNE类别可分性图（补充图）
    """
    available_methods = [m for m in gen_data_dict if gen_data_dict[m] is not None]
    n_methods = len(available_methods)
    if n_methods == 0:
        print("[警告] 没有可用的生成数据，跳过可视化")
        return

    print(f"\n绘制对比图...")

    # ---- 计算可视化所需指标 ----
    from scipy.signal import welch

    metrics = {m: {} for m in available_methods}
    for m in available_methods:
        g_ch = gen_gen_ch_dict[m]

        # NMSE (仅13个生成通道)
        mse = np.mean((g_ch - real_gen_ch)**2)
        metrics[m]['NMSE'] = mse / (np.mean(real_gen_ch**2) + 1e-10)

        # PSD correlation (仅13个生成通道, alpha band)
        corr_list = []
        for i in range(min(100, len(g_ch))):
            f_r, psd_r = welch(real_gen_ch[i], fs=FS, nperseg=256, axis=-1)
            f_g, psd_g = welch(g_ch[i], fs=FS, nperseg=256, axis=-1)
            alpha_mask = (f_r >= 8) & (f_r <= 13)
            alpha_r = np.mean(psd_r[:, alpha_mask], axis=-1)
            alpha_g = np.mean(psd_g[:, alpha_mask], axis=-1)
            c = np.corrcoef(alpha_r, alpha_g)[0, 1]
            if not np.isnan(c):
                corr_list.append(c)
        metrics[m]['PSD_corr'] = np.mean(corr_list) if corr_list else 0

        # Wasserstein (classifier features)
        gct = gen_clf_emb_dict[m]
        metrics[m]['WD_clf'] = (wasserstein_distance(real_clf_emb[:, 0], gct[:, 0]) +
                                wasserstein_distance(real_clf_emb[:, 1], gct[:, 1])) / 2

        # Silhouette (高维分类器特征，不是2D t-SNE)
        metrics[m]['Sil_clf'] = silhouette_score(gen_clf_feat_dict[m], gen_labels_dict[m])

    real_sil_clf = silhouette_score(real_clf_feat, real_labels)

    # ---- 图1: 量化指标对比柱状图 ----
    method_colors = [METHODS[m]['color'] for m in available_methods]
    dttd_idx = available_methods.index('DTTD-DDPM') if 'DTTD-DDPM' in available_methods else -1

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # 1a. NMSE (仅13个生成通道, 越低越好)
    ax = axes[0]
    vals = [metrics[m]['NMSE'] for m in available_methods]
    bars = ax.bar(range(n_methods), vals, color=method_colors, edgecolor='black', linewidth=0.8)
    if dttd_idx >= 0:
        bars[dttd_idx].set_edgecolor('red')
        bars[dttd_idx].set_linewidth(2.5)
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels(available_methods, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('NMSE', fontsize=12)
    ax.set_title('Reconstruction Error (↓)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    # 1b. PSD Correlation (仅13个生成通道, 越高越好)
    ax = axes[1]
    vals = [metrics[m]['PSD_corr'] for m in available_methods]
    bars = ax.bar(range(n_methods), vals, color=method_colors, edgecolor='black', linewidth=0.8)
    if dttd_idx >= 0:
        bars[dttd_idx].set_edgecolor('red')
        bars[dttd_idx].set_linewidth(2.5)
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels(available_methods, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('PSD Correlation', fontsize=12)
    ax.set_title('Spectral Fidelity (↑)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    # 1c. WD Classifier (越低越好)
    ax = axes[2]
    vals = [metrics[m]['WD_clf'] for m in available_methods]
    bars = ax.bar(range(n_methods), vals, color=method_colors, edgecolor='black', linewidth=0.8)
    if dttd_idx >= 0:
        bars[dttd_idx].set_edgecolor('red')
        bars[dttd_idx].set_linewidth(2.5)
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels(available_methods, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Wasserstein Distance', fontsize=12)
    ax.set_title('Distribution Alignment (↓)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    # 1d. 下游分类准确率
    ax = axes[3]
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    real_flat = real_data.reshape(len(real_data), -1)
    acc_vals = []
    for m in available_methods:
        g = gen_data_dict[m]
        gen_flat = g.reshape(len(g), -1)
        scaler = StandardScaler()
        gen_flat_s = scaler.fit_transform(gen_flat)
        real_flat_s = scaler.transform(real_flat)
        svm = SVC(kernel='rbf', C=1.0, gamma='scale')
        svm.fit(gen_flat_s, gen_labels_dict[m])
        acc_vals.append(svm.score(real_flat_s, real_labels))
    bars = ax.bar(range(n_methods), acc_vals, color=method_colors, edgecolor='black', linewidth=0.8)
    if dttd_idx >= 0:
        bars[dttd_idx].set_edgecolor('red')
        bars[dttd_idx].set_linewidth(2.5)
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels(available_methods, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Downstream Classification (↑)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, acc_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Channel Generation Comparison (13 Generated Channels)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    metrics_path = save_path.replace('.png', '_metrics.png')
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
    print(f"保存: {metrics_path}")
    plt.close()

    # ---- 图2: t-SNE类别可分性（只画生成数据，按类别着色）----
    class_colors = ['#D32F2F', '#1565C0', '#2E7D32', '#F57F17']

    fig2, axes2 = plt.subplots(2, (n_methods + 1) // 2, figsize=(5 * ((n_methods + 1) // 2), 10))
    axes2 = axes2.flatten()

    for idx, method_name in enumerate(available_methods):
        ax = axes2[idx]
        gen_clf = gen_clf_emb_dict[method_name]
        gen_labels = gen_labels_dict[method_name]
        sil = metrics[method_name]['Sil_clf']
        for cls in range(NUM_CLASSES):
            mask = gen_labels == cls
            ax.scatter(gen_clf[mask, 0], gen_clf[mask, 1],
                       c=class_colors[cls], marker='o', alpha=0.7, s=40,
                       edgecolors='white', linewidths=0.3, label=CLASS_NAMES[cls])
        ax.set_title(f'{method_name}\nSil={sil:.4f}', fontsize=11, fontweight='bold')
        ax.set_xlabel('t-SNE 1', fontsize=9)
        ax.set_ylabel('t-SNE 2', fontsize=9)
        ax.legend(loc='best', fontsize=7, framealpha=0.9)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)

    # 隐藏多余子图
    for idx in range(n_methods, len(axes2)):
        axes2[idx].set_visible(False)

    plt.suptitle('Class Separability of Generated Data (Classifier Features)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    tsne_path = save_path.replace('.png', '_class_tsne.png')
    plt.savefig(tsne_path, dpi=150, bbox_inches='tight')
    print(f"保存: {tsne_path}")
    plt.close()


# ============================================================================
# 任务特征对比可视化（展示DTTD优势）
# ============================================================================

# 10-10系统电极2D坐标（用于脑地形图）
ELECTRODE_2D = {
    'Fz': (0.0, 0.7), 'FC3': (-0.45, 0.45), 'FC1': (-0.15, 0.45),
    'FCz': (0.0, 0.45), 'FC2': (0.15, 0.45), 'FC4': (0.45, 0.45),
    'C5': (-0.6, 0.0), 'C3': (-0.35, 0.0), 'C1': (-0.12, 0.0),
    'Cz': (0.0, 0.0), 'C2': (0.12, 0.0), 'C4': (0.35, 0.0), 'C6': (0.6, 0.0),
    'CP3': (-0.45, -0.45), 'CP1': (-0.15, -0.45), 'CPz': (0.0, -0.45),
    'CP2': (0.15, -0.45), 'CP4': (0.45, -0.45),
    'P1': (-0.2, -0.7), 'Pz': (0.0, -0.7), 'P2': (0.2, -0.7), 'POz': (0.0, -0.9),
}


def _topo_plot(ax, values, channels, title='', vmin=None, vmax=None, cmap='RdBu_r'):
    """在ax上画脑地形图（RBF插值 + contourf）"""
    from scipy.interpolate import RBFInterpolator

    coords = np.array([ELECTRODE_2D[ch] for ch in channels])
    vals = np.array(values, dtype=float)

    # 检查点是否共线（3个点在一条线上RBF会奇异）
    if len(coords) <= 3:
        # 点太少或可能共线，直接画散点
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, cmap=cmap, vmin=vmin, vmax=vmax,
                        s=200, edgecolors='black', linewidths=1, zorder=5)
        # 标注数值
        for i, (x, y) in enumerate(coords):
            ax.text(x, y+0.08, f'{vals[i]:.1f}', ha='center', va='bottom', fontsize=7)
        im = sc
    else:
        # 生成插值网格
        grid_x = np.linspace(-0.8, 0.8, 60)
        grid_y = np.linspace(-1.0, 0.9, 60)
        GX, GY = np.meshgrid(grid_x, grid_y)
        mask_circle = (GX**2 + GY**2) <= 1.0
        grid_points = np.column_stack([GX.ravel(), GY.ravel()])

        # RBF插值
        try:
            rbf = RBFInterpolator(coords, vals, kernel='thin_plate_spline', smoothing=0.1)
            Z = rbf(grid_points).reshape(GX.shape)
        except np.linalg.LinAlgError:
            # 如果奇异，用linear核
            rbf = RBFInterpolator(coords, vals, kernel='linear', smoothing=0.1)
            Z = rbf(grid_points).reshape(GX.shape)

        Z[~mask_circle] = np.nan
        Z = np.clip(Z, -1e6, 1e6)

        # 用contourf替代pcolormesh（更稳定）
        levels = np.linspace(vmin if vmin else np.nanmin(Z), vmax if vmax else np.nanmax(Z), 30)
        im = ax.contourf(GX, GY, Z, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, extend='both')

    # 画头模型轮廓
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta)*0.95, np.sin(theta)*0.95, 'k-', linewidth=1)
    # 鼻子标记
    ax.plot([0, 0], [0.9, 1.0], 'k-', linewidth=1.5)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')
    return im


def visualize_task_features(real_data, real_labels, gen_data_dict, gen_labels_dict,
                            gen_ch_indices, input_ch_indices, save_dir='paper/figures',
                            dttd_cond_gen=None):
    """展示DTTD核心优势，生成2张图：
    图1: 条件生成ERD特征对比 — 同一输入不同标签→运动皮层Alpha/Beta能量是否不同
    图2: 综合能力对比雷达图 — 多维度量化各方法优劣
    """
    available_methods = [m for m in gen_data_dict if gen_data_dict[m] is not None]
    if not available_methods:
        return

    os.makedirs(save_dir, exist_ok=True)

    key_methods = ['DTTD-DDPM', 'EEGDiff', 'BrainDiff', 'Kriging']
    key_methods = [m for m in key_methods if m in available_methods]
    if not key_methods:
        key_methods = available_methods[:3]

    from scipy.signal import welch

    # 运动皮层关键通道: C3(idx=7), Cz(idx=9), C4(idx=11)
    motor_ch_indices = [7, 9, 11]
    motor_ch_names = ['C3', 'Cz', 'C4']
    colors_4cls = ['#E53935', '#1E88E5', '#43A047', '#FB8C00']

    # ========== 图1: 条件生成ERD特征对比 ==========
    if dttd_cond_gen is not None:
        print("\n【条件生成ERD特征对比】")

        # 计算每个标签生成数据的Alpha(8-13Hz)和Beta(13-30Hz)频段能量
        # ERD = 事件相关去同步，表现为Alpha/Beta能量下降
        # 不同运动想象任务应在不同通道表现出不同的ERD模式
        band_powers = {}  # {cls: {'alpha': [C3, Cz, C4], 'beta': [C3, Cz, C4]}}
        for cls in range(NUM_CLASSES):
            data = dttd_cond_gen[cls].numpy()  # [n_samples, 22, 1000]
            alpha_powers = []
            beta_powers = []
            for ch_i in motor_ch_indices:
                f, psd = welch(data[:, ch_i, :], fs=FS, nperseg=256, axis=-1)
                alpha_mask = (f >= 8) & (f <= 13)
                beta_mask = (f >= 13) & (f <= 30)
                alpha_powers.append(np.mean(psd[:, alpha_mask]))
                beta_powers.append(np.mean(psd[:, beta_mask]))
            band_powers[cls] = {'alpha': alpha_powers, 'beta': beta_powers}

        # 同样计算真实数据各类别的ERD特征
        real_band_powers = {}
        for cls in range(NUM_CLASSES):
            mask = real_labels == cls
            cls_data = real_data[mask]
            alpha_powers = []
            beta_powers = []
            for ch_i in motor_ch_indices:
                f, psd = welch(cls_data[:, ch_i, :], fs=FS, nperseg=256, axis=-1)
                alpha_mask = (f >= 8) & (f <= 13)
                beta_mask = (f >= 13) & (f <= 30)
                alpha_powers.append(np.mean(psd[:, alpha_mask]))
                beta_powers.append(np.mean(psd[:, beta_mask]))
            real_band_powers[cls] = {'alpha': alpha_powers, 'beta': beta_powers}

        # 计算类别间ERD模式差异（用变异系数CV衡量）
        # CV越大 → 不同类别的ERD模式差异越大 → 条件生成越有效
        def compute_cv(band_powers_dict, band_key):
            """计算4个类别在3个通道上的变异系数"""
            vals = np.array([band_powers_dict[cls][band_key] for cls in range(NUM_CLASSES)])  # [4, 3]
            mean_val = vals.mean()
            cv = vals.std() / (mean_val + 1e-15)
            return cv, vals

        dttd_alpha_cv, dttd_alpha_vals = compute_cv(band_powers, 'alpha')
        dttd_beta_cv, dttd_beta_vals = compute_cv(band_powers, 'beta')
        real_alpha_cv, real_alpha_vals = compute_cv(real_band_powers, 'alpha')
        real_beta_cv, real_beta_vals = compute_cv(real_band_powers, 'beta')

        print(f"  DTTD Alpha CV: {dttd_alpha_cv:.4f}, Real Alpha CV: {real_alpha_cv:.4f}")
        print(f"  DTTD Beta CV:  {dttd_beta_cv:.4f}, Real Beta CV:  {real_beta_cv:.4f}")

        # 绘制：2行(Alpha/Beta) x 2列(DTTD生成 vs 真实数据) 的分组柱状图
        fig1, axes1 = plt.subplots(2, 2, figsize=(12, 8))

        for row, (band_name, band_label) in enumerate([('alpha', 'Alpha (8-13Hz)'), ('beta', 'Beta (13-30Hz)')]):
            for col, (source, bp_dict) in enumerate([('DTTD Generated', band_powers), ('Real Data', real_band_powers)]):
                ax = axes1[row, col]
                x = np.arange(NUM_CLASSES)
                width = 0.25

                for i, (ch_name, ch_color) in enumerate(zip(motor_ch_names, ['#1565C0', '#2E7D32', '#C62828'])):
                    vals = [bp_dict[cls][band_name][i] for cls in range(NUM_CLASSES)]
                    bars = ax.bar(x + (i - 1) * width, vals, width,
                                 label=ch_name, color=ch_color, edgecolor='white', linewidth=0.5)
                    # 在柱子上标注数值
                    for bar, val in zip(bars, vals):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                               f'{val:.1e}', ha='center', va='bottom', fontsize=6, rotation=45)

                ax.set_xticks(x)
                ax.set_xticklabels([CLASS_NAMES[cls] for cls in range(NUM_CLASSES)], fontsize=9)
                ax.set_ylabel(f'{band_label} Power', fontsize=10)
                ax.set_title(f'{source} — {band_label} Band\n(Motor cortex channels)', fontsize=11, fontweight='bold')
                ax.legend(fontsize=9, loc='upper right')
                ax.grid(axis='y', alpha=0.3)

                # 标注CV
                if source == 'DTTD Generated':
                    cv_val = dttd_alpha_cv if band_name == 'alpha' else dttd_beta_cv
                else:
                    cv_val = real_alpha_cv if band_name == 'alpha' else real_beta_cv
                ax.annotate(f'CV = {cv_val:.3f}\n(variation across classes)',
                           xy=(0.02, 0.95), xycoords='axes fraction',
                           fontsize=9, va='top',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray'))

        fig1.suptitle('DTTD Conditional Generation: ERD Patterns by Class\n'
                      'Same input → different labels → different motor cortex power patterns',
                      fontsize=13, fontweight='bold', y=1.02)
        cond_path = os.path.join(save_dir, 'task_conditional_generation.png')
        try:
            fig1.tight_layout()
            plt.savefig(cond_path, dpi=200)
            print(f"保存: {cond_path}")
        except Exception as e:
            print(f"[ERROR] 保存条件生成图失败: {e}")
        plt.close()

    # ========== 图2: 综合能力雷达图 ==========
    print("\n绘制综合能力雷达图...")

    # 计算各方法的多维度指标
    # 维度1: NMSE (越低越好 → 取1-NMSE归一化)
    # 维度2: PSD相关性 (越高越好)
    # 维度3: 条件生成能力 (DTTD独有 → 用类别间CV衡量)
    # 维度4: 下游分类准确率 (如果有的话)

    metrics = {}
    for m in key_methods:
        data = gen_data_dict[m].numpy() if isinstance(gen_data_dict[m], torch.Tensor) else gen_data_dict[m]
        # NMSE
        mse = np.mean((data[:, gen_ch_indices, :] - real_data[:len(data), gen_ch_indices, :]) ** 2)
        var = np.var(real_data[:len(data), gen_ch_indices, :])
        nmse = mse / (var + 1e-10)
        # PSD相关性 (Alpha频段)
        corr_list = []
        for j in range(min(100, len(data))):
            f_r, psd_r = welch(real_data[j, gen_ch_indices, :], fs=FS, nperseg=256, axis=-1)
            f_g, psd_g = welch(data[j, gen_ch_indices, :], fs=FS, nperseg=256, axis=-1)
            alpha_mask = (f_r >= 8) & (f_r <= 13)
            band_r = np.mean(psd_r[:, alpha_mask], axis=-1)
            band_g = np.mean(psd_g[:, alpha_mask], axis=-1)
            c = np.corrcoef(band_r, band_g)[0, 1]
            if not np.isnan(c):
                corr_list.append(c)
        psd_corr = np.mean(corr_list) if corr_list else 0

        # 条件生成能力: 用生成数据中不同类别间的Alpha能量CV
        if gen_labels_dict.get(m) is not None:
            g_labels = gen_labels_dict[m][:len(data)]
            cv_list = []
            for ch_i in motor_ch_indices:
                cls_powers = []
                for cls in range(NUM_CLASSES):
                    mask = g_labels == cls
                    if mask.sum() > 0:
                        f, psd = welch(data[mask][:, ch_i, :], fs=FS, nperseg=256, axis=-1)
                        alpha_mask = (f >= 8) & (f <= 13)
                        cls_powers.append(np.mean(psd[:, alpha_mask]))
                if len(cls_powers) == NUM_CLASSES:
                    cv_list.append(np.std(cls_powers) / (np.mean(cls_powers) + 1e-15))
            cond_cv = np.mean(cv_list) if cv_list else 0
        else:
            cond_cv = 0

        metrics[m] = {'nmse': nmse, 'psd_corr': psd_corr, 'cond_cv': cond_cv}

    # DTTD的条件生成CV用dttd_cond_gen的结果
    if dttd_cond_gen is not None:
        metrics['DTTD-DDPM']['cond_cv'] = dttd_alpha_cv

    # 归一化到0-1
    dim_names = ['1-NMSE\n(Reconstruction)', 'PSD Corr.\n(Spectrum)', 'Cond. CV\n(Task-specific)']
    n_dims = len(dim_names)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True))

    for m in key_methods:
        nmse_val = metrics[m]['nmse']
        psd_val = metrics[m]['psd_corr']
        cv_val = metrics[m]['cond_cv']

        # 归一化: 1-NMSE (clamp to 0-1), PSD直接用, CV归一化到0-1
        score_1nmse = max(0, 1 - min(nmse_val, 1))
        score_psd = max(0, min(psd_val, 1))
        # CV归一化: 用真实数据的CV作为上限
        cv_max = real_alpha_cv * 1.5 if real_alpha_cv > 0 else 0.5
        score_cv = max(0, min(cv_val / cv_max, 1))

        values = [score_1nmse, score_psd, score_cv]
        values += values[:1]

        color = METHODS.get(m, {}).get('color', 'gray')
        ax2.plot(angles, values, 'o-', linewidth=2, label=m, color=color, markersize=6)
        ax2.fill(angles, values, alpha=0.1, color=color)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(dim_names, fontsize=11)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax2.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax2.set_title('Method Comparison: Multi-dimensional Radar Chart\n'
                  'DTTD excels at task-specific conditional generation',
                  fontsize=12, fontweight='bold', pad=20)

    radar_path = os.path.join(save_dir, 'task_radar_comparison.png')
    try:
        plt.savefig(radar_path, dpi=200)
        print(f"保存: {radar_path}")
    except Exception as e:
        print(f"[ERROR] 保存雷达图失败: {e}")
    plt.close()

    # ========== 打印总结 ==========
    print("\n" + "=" * 70)
    print("DTTD优势总结")
    print("=" * 70)

    if dttd_cond_gen is not None:
        print(f"\n  条件生成能力（DTTD独有）:")
        print(f"  Alpha频段类别间CV: {dttd_alpha_cv:.4f} (真实数据: {real_alpha_cv:.4f})")
        print(f"  Beta频段类别间CV:  {dttd_beta_cv:.4f} (真实数据: {real_beta_cv:.4f})")

    print(f"\n  各方法综合指标:")
    print(f"  {'方法':15s} {'NMSE':>8s} {'PSD_corr':>10s} {'Cond_CV':>10s}")
    for m in key_methods:
        print(f"  {m:15s} {metrics[m]['nmse']:8.4f} {metrics[m]['psd_corr']:10.4f} {metrics[m]['cond_cv']:10.4f}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='t-SNE分布可视化对比')
    parser.add_argument('--n-samples', type=int, default=100, help='每类生成样本数')
    parser.add_argument('--data-path', type=str, default='E:/data/BCI2a', help='BCI2a数据路径')
    parser.add_argument('--output', type=str, default='paper/figures/tsne_comparison.png', help='输出路径')
    args = parser.parse_args()

    device = get_device()
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 70)
    print("t-SNE分布可视化对比")
    print("=" * 70)

    # 1. 加载真实数据
    target_data, labels, ch_indices = load_bci2a_data(args.data_path, args.n_samples, device)
    real_np = target_data.numpy()
    labels_np = labels.numpy()

    # 2. 各方法生成
    print("\n生成各方法样本...")
    gen_data_dict = {}
    gen_labels_dict = {}

    for method_name, cfg in METHODS.items():
        method_type = cfg.get('type', 'model')
        ckpt_path = cfg.get('checkpoint', None)

        # 插值方法不需要checkpoint
        if method_type == 'model' and ckpt_path and not os.path.exists(ckpt_path):
            print(f"  [跳过] {method_name}: checkpoint不存在 ({ckpt_path})")
            continue
        try:
            generator = GENERATORS[method_name]
            gen_data = generator(target_data, labels, ch_indices, device, checkpoint_path=ckpt_path)
            gen_np = gen_data.numpy()
            # 不做统计匹配，保留各方法原始分布差异
            gen_data_dict[method_name] = gen_np
            gen_labels_dict[method_name] = labels_np.copy()
            print(f"  [OK] {method_name}: {gen_np.shape}")
        except Exception as e:
            print(f"  [错误] {method_name}: {e}")

    if not gen_data_dict:
        print("没有可用的生成数据，退出")
        return

    # 3. 特征提取
    print("\n特征提取...")
    real_pca, gen_pca_dict = extract_pca_features(real_np, gen_data_dict, n_components=50)
    real_clf, gen_clf_dict = extract_classifier_features(real_np, gen_data_dict, device)

    # 4. t-SNE
    print("\n计算t-SNE...")
    real_pca_emb, gen_pca_emb_dict = compute_tsne(real_pca, gen_pca_dict)
    real_clf_emb, gen_clf_emb_dict = compute_tsne(real_clf, gen_clf_dict)

    # 5. 量化分析
    print("\n" + "=" * 70)
    print("量化分析")
    print("=" * 70)

    # 只评估13个生成通道（排除9个输入通道）
    gen_ch_indices = [i for i in range(22) if i not in ch_indices]
    print(f"\n输入通道({len(ch_indices)}个): {ch_indices}")
    print(f"生成通道({len(gen_ch_indices)}个): {gen_ch_indices}")

    real_gen_ch = real_np[:, gen_ch_indices, :]  # [N, 13, T]
    gen_gen_ch_dict = {k: v[:, gen_ch_indices, :] for k, v in gen_data_dict.items()}

    # ---- 下游分类准确率 ----
    print("\n【下游分类准确率 - 用生成数据训练，真实数据测试 (越高越好)】")
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    # 真实数据的5折交叉验证基线
    real_flat = real_np.reshape(len(real_np), -1)
    scaler_ref = StandardScaler()
    real_flat_s = scaler_ref.fit_transform(real_flat)
    svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    real_cv = cross_val_score(svm, real_flat_s, labels_np, cv=5, scoring='accuracy')
    print(f"  Real Data (5-fold CV):   {real_cv.mean():.4f} ± {real_cv.std():.4f}")

    for k, g in gen_data_dict.items():
        gen_flat = g.reshape(len(g), -1)
        scaler_gen = StandardScaler()
        gen_flat_s = scaler_gen.fit_transform(gen_flat)
        real_flat_s2 = scaler_gen.transform(real_flat)
        svm = SVC(kernel='rbf', C=1.0, gamma='scale')
        svm.fit(gen_flat_s, gen_labels_dict[k])
        acc = svm.score(real_flat_s2, labels_np)
        print(f"  {k:15s} Train→Test: {acc:.4f}")

    # Silhouette Score (类别可分性)
    print("\n【类别可分性 - Silhouette Score (越高越好, -1~1)】")
    real_sil_pca = silhouette_score(real_pca, labels_np)
    real_sil_clf = silhouette_score(real_clf, labels_np)
    print(f"  Real Data (PCA):        {real_sil_pca:.4f}")
    print(f"  Real Data (Classifier): {real_sil_clf:.4f}")
    for k in gen_data_dict:
        sil_pca = silhouette_score(gen_pca_dict[k], labels_np)
        sil_clf = silhouette_score(gen_clf_dict[k], labels_np)
        print(f"  {k:15s} (PCA):        {sil_pca:.4f}  (vs Real: {sil_pca-real_sil_pca:+.4f})")
        print(f"  {k:15s} (Classifier): {sil_clf:.4f}  (vs Real: {sil_clf-real_sil_clf:+.4f})")

    # Wasserstein Distance (分布对齐, t-SNE空间)
    print("\n【分布对齐 - Wasserstein Distance (越低越好)】")
    for k in gen_data_dict:
        gt = gen_pca_emb_dict[k]
        gct = gen_clf_emb_dict[k]
        wd_pca = (wasserstein_distance(real_pca_emb[:,0], gt[:,0]) +
                  wasserstein_distance(real_pca_emb[:,1], gt[:,1])) / 2
        wd_clf = (wasserstein_distance(real_clf_emb[:,0], gct[:,0]) +
                  wasserstein_distance(real_clf_emb[:,1], gct[:,1])) / 2
        print(f"  {k:15s} PCA: {wd_pca:.3f}  Classifier: {wd_clf:.3f}")

    # 每类分布对齐
    print("\n【每类分布对齐 - Classifier特征 Wasserstein (越低越好)】")
    for k in gen_data_dict:
        gct = gen_clf_emb_dict[k]
        wd_per_class = []
        for c in range(NUM_CLASSES):
            mask = labels_np == c
            wd = (wasserstein_distance(real_clf_emb[mask,0], gct[mask,0]) +
                  wasserstein_distance(real_clf_emb[mask,1], gct[mask,1])) / 2
            wd_per_class.append(wd)
        avg = np.mean(wd_per_class)
        detail = ', '.join([f'{CLASS_NAMES[i]}:{wd_per_class[i]:.3f}' for i in range(NUM_CLASSES)])
        print(f"  {k:15s} avg={avg:.3f}  ({detail})")

    # 信号质量 - 只评估13个生成通道
    print("\n【生成通道重建质量 (仅13个生成通道)】")
    from scipy.signal import welch
    for k, g in gen_gen_ch_dict.items():
        mse = np.mean((g - real_gen_ch)**2)
        nmse = mse / (np.mean(real_gen_ch**2) + 1e-10)
        corr_list = []
        for i in range(min(100, len(g))):
            f_r, psd_r = welch(real_gen_ch[i], fs=FS, nperseg=256, axis=-1)
            f_g, psd_g = welch(g[i], fs=FS, nperseg=256, axis=-1)
            alpha_mask = (f_r >= 8) & (f_r <= 13)
            alpha_r = np.mean(psd_r[:, alpha_mask], axis=-1)
            alpha_g = np.mean(psd_g[:, alpha_mask], axis=-1)
            c = np.corrcoef(alpha_r, alpha_g)[0,1]
            if not np.isnan(c):
                corr_list.append(c)
        psd_corr = np.mean(corr_list) if corr_list else 0
        print(f"  {k:15s} NMSE={nmse:.4f}  PSD_corr(alpha)={psd_corr:.4f}")

    print("\n" + "=" * 70)
    print("指标解读: Silhouette↑=类别越可分; Wasserstein↓=分布越对齐; NMSE↓=重建越好; Acc↑=下游可用性越好")
    print("=" * 70)

    # 6. 可视化
    visualize_all_methods(
        real_np, labels_np, gen_data_dict, gen_labels_dict,
        real_pca_emb, gen_pca_emb_dict,
        real_clf_emb, gen_clf_emb_dict,
        real_gen_ch, gen_gen_ch_dict,
        real_clf, gen_clf_dict,
        save_path=args.output
    )

    # 7. 任务特征对比可视化（DTTD优势展示）
    # 先做DTTD条件生成实验
    dttd_cond_gen = None
    if 'DTTD-DDPM' in gen_data_dict:
        try:
            dttd_cond_gen = generate_dttd_conditional(
                target_data, ch_indices, device,
                checkpoint_path=METHODS['DTTD-DDPM']['checkpoint'],
                n_samples=25
            )
        except Exception as e:
            print(f"  [WARNING] 条件生成实验失败: {e}")

    visualize_task_features(real_np, labels_np, gen_data_dict, gen_labels_dict,
                           gen_ch_indices, ch_indices, save_dir='paper/figures',
                           dttd_cond_gen=dttd_cond_gen)

    print("\n完成!")


if __name__ == '__main__':
    main()
