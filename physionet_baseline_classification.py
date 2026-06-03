"""
PhysioNet MI 基线分类评估

对基线方法在PhysioNet MI数据集上进行跨被试(10折跨被试交叉验证)重建分类和增强分类评估。

使用方法:
    python experiments/physionet_baseline_classification.py
    python experiments/physionet_baseline_classification.py --methods CVAE,EEGDiff
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.baselines import CVAE, ConditionalGAN, EEGDiff, BrainDiff
from models.traditional_baselines import SplineInterpolation, KrigingInterpolation
from utils import get_device

CHECKPOINT_MAP = {
    'CVAE': 'checkpoints/physionet/baseline_cvae/best_model.pth',
    'cGAN': 'checkpoints/physionet/baseline_cgan/best_model.pth',
    'EEGDiff': 'checkpoints/physionet/baseline_eegdiff/best_model.pth',
    'BrainDiff': 'checkpoints/physionet/baseline_braindiff/best_model.pth',
}


# ==================== EEGNet分类器 ====================

class EEGNetClassifier(nn.Module):
    def __init__(self, num_channels, num_classes=4, time_steps=640):
        super().__init__()
        F1, D, F2 = 8, 2, 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (num_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(0.5)
        )
        flat_size = F2 * (time_steps // 32)
        self.classifier = nn.Linear(flat_size, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv3(self.conv2(self.conv1(x)))
        return self.classifier(x.flatten(1))


def train_classifier(train_data, train_labels, device, num_channels, epochs=200,
                     extra_data=None, extra_labels=None):
    """训练EEGNet分类器。extra_data/extra_labels用于增强，避免numpy concatenate的内存开销。"""
    num_classes = len(np.unique(train_labels))
    time_steps = train_data.shape[2]

    # z-score归一化（与DTTD评估脚本一致）
    channel_mean = train_data.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    channel_std = train_data.std(axis=(0, 2), keepdims=True).astype(np.float32) + 1e-8
    train_data_norm = ((train_data - channel_mean) / channel_std).astype(np.float32)
    if extra_data is not None and extra_labels is not None:
        extra_data_norm = ((extra_data - channel_mean) / channel_std).astype(np.float32)

    # 固定随机种子确保可复现
    torch.manual_seed(42)
    np.random.seed(42)

    clf = EEGNetClassifier(num_channels, num_classes, time_steps).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)

    generator = torch.Generator()
    generator.manual_seed(42)

    ds1 = TensorDataset(torch.FloatTensor(train_data_norm), torch.LongTensor(train_labels))
    if extra_data is not None and extra_labels is not None:
        ds2 = TensorDataset(torch.FloatTensor(extra_data_norm), torch.LongTensor(extra_labels))
        dataset = torch.utils.data.ConcatDataset([ds1, ds2])
    else:
        dataset = ds1

    loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False, generator=generator)

    clf.train()
    best_loss = float('inf')
    best_state = None
    for epoch in range(epochs):
        total_loss = 0.0
        for data, labels in loader:
            opt.zero_grad()
            loss = criterion(clf(data.to(device)), labels.to(device))
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in clf.state_dict().items()}
    
    # 恢复最佳模型
    if best_state is not None:
        clf.load_state_dict(best_state)
    return clf, channel_mean, channel_std


def evaluate_classifier(clf, test_data, test_labels, device, channel_mean=None, channel_std=None):
    clf.eval()
    if channel_mean is not None and channel_std is not None:
        test_data = ((test_data - channel_mean) / channel_std).astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(test_data), torch.LongTensor(test_labels)),
        batch_size=32, shuffle=False)
    preds, labels_all = [], []
    with torch.no_grad():
        for data, labels in loader:
            preds.extend(torch.argmax(clf(data.to(device)), dim=1).cpu().numpy())
            labels_all.extend(labels.numpy())
    return {
        'accuracy': float(accuracy_score(labels_all, preds)),
        'f1': float(f1_score(labels_all, preds, average='macro')),
        'kappa': float(cohen_kappa_score(labels_all, preds))
    }


# ==================== 模型加载 ====================

def load_baseline_model(model_name, device, in_ch, out_ch, t_steps, n_cls):
    """加载基线模型，返回 (model, scale_info, ckpt_indices)"""
    ckpt_path = CHECKPOINT_MAP.get(model_name)
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print(f"[SKIP] {model_name} checkpoint not found: {ckpt_path}")
        return None, None, None

    try:
        # 先加载checkpoint获取模型配置
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        # 从checkpoint读取模型参数
        ckpt_latent_dim = ckpt.get('latent_dim', 128)
        ckpt_in_ch = ckpt.get('n_input_ch', in_ch)
        ckpt_out_ch = ckpt.get('n_output_ch', out_ch)
        ckpt_t_steps = ckpt.get('time_steps', t_steps)
        ckpt_n_cls = ckpt.get('num_classes', n_cls)

        if model_name == 'CVAE':
            # 从state_dict推断完整模型架构
            sd = ckpt.get('model_state_dict', ckpt.get('state_dict', {}))
            # 推断encoder结构
            enc_layers = []
            i = 0
            while f'encoder.{i}.weight' in sd:
                w = sd[f'encoder.{i}.weight']
                in_ch = w.shape[1]
                out_ch = w.shape[0]
                ksize = w.shape[2]
                enc_layers.extend([
                    nn.Conv1d(in_ch, out_ch, kernel_size=ksize, padding=ksize//2),
                    nn.BatchNorm1d(out_ch), nn.ReLU(),
                ])
                i += 3  # Conv, BN, ReLU
            # 去掉最后一个ReLU，换成AdaptiveAvgPool
            enc_layers = enc_layers[:-1]
            # 推断pool_size: fc_mu.weight shape[1] = enc_out_ch * pool_size + 64
            fc_mu_in = sd['fc_mu.weight'].shape[1]
            enc_out_ch = sd[f'encoder.{i-3}.weight'].shape[0]
            pool_size = (fc_mu_in - 64) // enc_out_ch
            enc_layers.append(nn.AdaptiveAvgPool1d(pool_size))

            # 推断decoder_conv结构
            decoder_conv_spec = None
            if any('decoder_conv' in k for k in sd.keys()):
                decoder_conv_spec = []
                j = 0
                while f'decoder_conv.{j}.weight' in sd:
                    w = sd[f'decoder_conv.{j}.weight']
                    b = sd[f'decoder_conv.{j}.bias']
                    # 区分Conv1d和ConvTranspose1d: Conv1d的bias.shape==w.shape[0]
                    if w.shape[0] == b.shape[0]:
                        decoder_conv_spec.append(
                            nn.Conv1d(w.shape[1], w.shape[0], kernel_size=w.shape[2], padding=w.shape[2]//2))
                    else:
                        decoder_conv_spec.append(
                            nn.ConvTranspose1d(w.shape[0], w.shape[1], kernel_size=w.shape[2], padding=w.shape[2]//2))
                    # 检查下一层是否是BN
                    if f'decoder_conv.{j+1}.weight' in sd and sd[f'decoder_conv.{j+1}.weight'].dim() == 1:
                        bn_ch = sd[f'decoder_conv.{j+1}.weight'].shape[0]
                        decoder_conv_spec.append(nn.BatchNorm1d(bn_ch))
                        decoder_conv_spec.append(nn.ReLU())
                        j += 3
                    else:
                        j += 1  # 最后一层Conv没有BN/ReLU

            model = CVAE(ckpt_in_ch, ckpt_out_ch, ckpt_t_steps, ckpt_n_cls,
                         latent_dim=ckpt_latent_dim, encoder_layers=enc_layers,
                         pool_size=pool_size, decoder_conv_spec=decoder_conv_spec).to(device)
            # 根据checkpoint调整decoder_fc的输出维度
            if 'decoder_fc.weight' in sd:
                dec_fc_out = sd['decoder_fc.weight'].shape[0]
                dec_fc_in = sd['decoder_fc.weight'].shape[1]
                if dec_fc_out != model.decoder_fc.out_features:
                    model.decoder_fc = nn.Linear(dec_fc_in, dec_fc_out).to(device)
        elif model_name == 'cGAN':
            model = ConditionalGAN(ckpt_in_ch, ckpt_out_ch, ckpt_t_steps, ckpt_n_cls,
                                   latent_dim=ckpt_latent_dim).to(device)
        elif model_name == 'EEGDiff':
            model = EEGDiff(ckpt_in_ch, ckpt_out_ch, ckpt_t_steps, ckpt_n_cls).to(device)
        elif model_name == 'BrainDiff':
            model = BrainDiff(ckpt_in_ch, ckpt_out_ch, ckpt_t_steps, ckpt_n_cls).to(device)
        else:
            return None, None, None

        state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # 从checkpoint获取input_ch_indices
        ckpt_input_indices = ckpt.get('input_ch_indices', None)

        # PhysioNet所有模型使用z-score
        scale_info = {'method': 'none'}
        if 'data_mean' in ckpt and 'data_std' in ckpt:
            scale_info = {
                'method': 'zscore',
                'data_mean': ckpt['data_mean'],
                'data_std': ckpt['data_std']
            }

        print(f"[OK] Loaded {model_name} from {ckpt_path}, scale={scale_info['method']}, "
              f"ckpt_indices={ckpt_input_indices}, latent_dim={ckpt_latent_dim}")
        return model, scale_info, ckpt_input_indices
    except Exception as e:
        print(f"[ERROR] Failed to load {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


# ==================== 重建生成 ====================

@torch.no_grad()
def generate_reconstruction(model, model_name, data_input, labels, device,
                            scale_info=None, input_ch_indices=None, batch_size=32):
    """用基线模型从低通道输入生成高通道重建"""
    model.eval()

    if scale_info is None:
        scale_info = {'method': 'none'}

    # 预处理输入
    if scale_info['method'] == 'zscore':
        mean = scale_info['data_mean']
        std = scale_info['data_std']
        if isinstance(mean, torch.Tensor):
            mean = mean.cpu().numpy()
            std = std.cpu().numpy()
        # mean/std是全通道的，输入只需取对应通道
        if input_ch_indices is not None and mean.shape[-2] != data_input.shape[1]:
            input_mean = mean[:, input_ch_indices, :]
            input_std = std[:, input_ch_indices, :]
        else:
            input_mean = mean
            input_std = std
        data_scaled = (data_input - input_mean) / input_std
    else:
        data_scaled = data_input

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(data_scaled), torch.LongTensor(labels)),
        batch_size=batch_size, shuffle=False)

    # 预分配输出数组，避免concatenate的内存峰值
    n_samples = data_input.shape[0]
    n_t = data_input.shape[2]
    generated = None
    offset = 0

    for batch_data, batch_labels in loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        bs = batch_data.size(0)

        try:
            if model_name == 'CVAE':
                recon, _, _ = model(batch_data, batch_labels)
            elif model_name == 'cGAN':
                z = torch.randn(batch_data.size(0), model.latent_dim, device=device)
                recon = model(z, batch_labels, input_data=batch_data)
            elif model_name in ('EEGDiff', 'BrainDiff'):
                recon = model.sample(batch_data, task_label=batch_labels, num_steps=1,
                                     noise_scale=0.02, guidance_scale=1.0)
            else:
                recon = batch_data
        except Exception as e:
            print(f"[WARN] Generation failed for {model_name}: {e}")
            recon = torch.zeros(batch_data.size(0), data_input.shape[1], data_input.shape[2],
                                device=device)

        recon_np = recon.cpu().numpy()
        if generated is None:
            out_ch = recon_np.shape[1]
            generated = np.empty((n_samples, out_ch, n_t), dtype=np.float32)
        generated[offset:offset+bs] = recon_np
        offset += bs

    # 裁剪极端值：z-score空间中限制在[-5,5]，避免反缩放后产生极端值
    if scale_info['method'] == 'zscore':
        clip_count = np.sum(np.abs(generated) > 5)
        if clip_count > 0:
            print(f"    [CLIP] 裁剪 {clip_count} 个极端值 (|x|>5)")
        generated = np.clip(generated, -5.0, 5.0)

    # 反缩放输出
    if scale_info['method'] == 'zscore':
        mean = scale_info['data_mean']
        std = scale_info['data_std']
        if isinstance(mean, torch.Tensor):
            mean = mean.cpu().numpy()
            std = std.cpu().numpy()
        generated = generated * std + mean

    return generated.astype(np.float32)


# ==================== 传统插值方法 ====================

def generate_traditional_reconstruction(method_name, data_input, n_input, n_output):
    """使用传统插值方法从低通道输入生成高通道重建"""
    if method_name == 'Spline':
        interp = SplineInterpolation(input_channels=n_input, output_channels=n_output)
    elif method_name == 'Kriging':
        interp = KrigingInterpolation(input_channels=n_input, output_channels=n_output)
    else:
        raise ValueError(f"Unknown traditional method: {method_name}")

    # 分批处理避免内存溢出
    batch_size = 32
    generated_list = []
    for i in range(0, len(data_input), batch_size):
        batch = data_input[i:i+batch_size]
        recon = interp.reconstruct(batch)
        if torch.is_tensor(recon):
            recon = recon.cpu().numpy()
        generated_list.append(recon.astype(np.float32))

    return np.concatenate(generated_list, axis=0)


# ==================== PhysioNet跨被试 ====================

def physionet_cross_subject(models, device, traditional_methods=None, start_fold=1):
    """PhysioNet MI跨被试评估 (10折跨被试交叉验证)"""
    if traditional_methods is None:
        traditional_methods = []
    print("\n" + "=" * 60)
    print("PhysioNet MI 跨被试 (10-fold cross-subject CV)")
    print("=" * 60)

    from data.physionet_mi import PhysioNetMIDataset, INPUT_CHANNEL_INDICES_16
    input_indices = INPUT_CHANNEL_INDICES_16

    # 尝试从缓存加载
    cache_path = 'paper_results/physionet_mi/physionet_mi_all_subjects.npz'
    if os.path.exists(cache_path):
        print(f"从缓存加载: {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        subj_map = cache['subject_data_map'].item()
        data = np.concatenate([v[0] for v in subj_map.values()]).astype(np.float32)
        labels = np.concatenate([v[1] for v in subj_map.values()]).astype(np.int64)
        subject_ids = np.concatenate([
            np.full(len(v[0]), sid, dtype=np.int64) for sid, v in subj_map.items()
        ])
    else:
        print("加载PhysioNet数据集...")
        dataset = PhysioNetMIDataset()
        data = dataset.data.astype(np.float32)
        labels = dataset.labels.astype(np.int64)
        subject_ids = getattr(dataset, 'subject_ids', None)

    n_output = data.shape[1]
    n_input = len(input_indices)
    print(f"数据: {n_output}ch输出, {n_input}ch输入, 样本={len(data)}")

    unique_subjects = np.unique(subject_ids) if subject_ids is not None else [0]
    n_subjects = len(unique_subjects)
    print(f"被试数: {n_subjects}")

    # 10折跨被试划分：将被试随机分成10组，每组约n_subjects/10个被试
    rng = np.random.RandomState(42)
    shuffled_subjects = unique_subjects.copy()
    rng.shuffle(shuffled_subjects)
    folds = np.array_split(shuffled_subjects, 10)

    results = {}

    for fold_idx, test_subjs in enumerate(folds):
        if fold_idx + 1 < start_fold:
            print(f"\n--- Fold {fold_idx+1}/10: 跳过（断点续传）---")
            continue
        test_subj_set = set(test_subjs.tolist())
        train_subjs = [s for s in unique_subjects if s not in test_subj_set]

        print(f"\n--- Fold {fold_idx+1}/10: test subjects {sorted(test_subjs.tolist())} ---")

        # 按被试划分训练/测试
        train_mask = np.isin(subject_ids, list(train_subjs))
        test_mask = np.isin(subject_ids, list(test_subj_set))

        train_full = data[train_mask].astype(np.float32)
        train_labels = labels[train_mask].astype(np.int64)
        test_full = data[test_mask].astype(np.float32)
        test_labels = labels[test_mask].astype(np.int64)

        train_input = train_full[:, input_indices, :]
        test_input = test_full[:, input_indices, :]

        print(f"  train={len(train_full)}, test={len(test_full)}")

        # 基线
        clf, cm_full, cs_full = train_classifier(train_full, train_labels, device, n_output)
        m_full = evaluate_classifier(clf, test_full, test_labels, device, cm_full, cs_full)
        del clf

        clf, cm_input, cs_input = train_classifier(train_input, train_labels, device, n_input)
        m_input = evaluate_classifier(clf, test_input, test_labels, device, cm_input, cs_input)
        del clf

        fold_result = {
            'baseline_full': {
                'accuracy': m_full['accuracy'], 'kappa': m_full['kappa'],
            },
            'baseline_input': {
                'accuracy': m_input['accuracy'], 'kappa': m_input['kappa'],
            },
        }
        print(f"  {n_output}ch: acc={m_full['accuracy']*100:.2f}%, kappa={m_full['kappa']:.4f} | "
              f"{n_input}ch: acc={m_input['accuracy']*100:.2f}%, kappa={m_input['kappa']:.4f}")

        for model_name, (model, scale_info, ckpt_indices) in models.items():
            if model is None:
                continue
            model_input_indices = ckpt_indices if ckpt_indices is not None else input_indices
            train_input_model = train_full[:, model_input_indices, :]

            # 1.0倍增强：对所有训练数据生成重建（训练集翻倍）
            aug_ratio = 1.0
            n_aug = max(1, int(len(train_labels) * aug_ratio))
            rng = np.random.RandomState(42)
            aug_idx = rng.choice(len(train_labels), n_aug, replace=False)
            gen_full = generate_reconstruction(model, model_name, train_input_model[aug_idx], train_labels[aug_idx],
                                               device, scale_info=scale_info,
                                               input_ch_indices=model_input_indices)

            # 增强（用extra_data避免numpy concatenate内存开销）
            clf, cm, cs = train_classifier(train_full, train_labels, device, n_output,
                                   extra_data=gen_full, extra_labels=train_labels[aug_idx])
            m_aug = evaluate_classifier(clf, test_full, test_labels, device, cm, cs)
            del clf

            fold_result[f'{model_name}_aug'] = {
                'accuracy': m_aug['accuracy'], 'kappa': m_aug['kappa'],
            }
            print(f"  {model_name}: aug acc={m_aug['accuracy']*100:.2f}%, kappa={m_aug['kappa']:.4f}")
            torch.cuda.empty_cache()

        # 传统插值方法评估
        for trad_name in traditional_methods:
            # 1.0倍增强：对所有训练数据生成重建（训练集翻倍）
            aug_ratio = 1.0
            n_aug = max(1, int(len(train_labels) * aug_ratio))
            rng = np.random.RandomState(42)
            aug_idx = rng.choice(len(train_labels), n_aug, replace=False)
            gen_full = generate_traditional_reconstruction(
                trad_name, train_input[aug_idx], n_input, n_output)

            # 增强
            clf, cm, cs = train_classifier(train_full, train_labels, device, n_output,
                                   extra_data=gen_full, extra_labels=train_labels[aug_idx])
            m_aug = evaluate_classifier(clf, test_full, test_labels, device, cm, cs)
            del clf

            fold_result[f'{trad_name}_aug'] = {
                'accuracy': m_aug['accuracy'], 'kappa': m_aug['kappa'],
            }
            print(f"  {trad_name}: aug acc={m_aug['accuracy']*100:.2f}%, kappa={m_aug['kappa']:.4f}")

        results[f'fold{fold_idx+1}'] = fold_result

    # 平均
    avg_result = {}
    first_key = list(results.keys())[0]
    for key in results[first_key].keys():
        acc_vals = [r[key]['accuracy'] for r in results.values() if key in r]
        kappa_vals = [r[key]['kappa'] for r in results.values() if key in r and 'kappa' in r[key]]
        avg_result[key] = {
            'accuracy': float(np.mean(acc_vals)),
            'std': float(np.std(acc_vals)),
            'kappa': float(np.mean(kappa_vals)) if kappa_vals else None,
            'kappa_std': float(np.std(kappa_vals)) if kappa_vals else None,
        }
    results['average'] = avg_result
    print(f"\n平均结果:")
    for key, val in avg_result.items():
        k_str = f", kappa={val['kappa']:.4f}" if val['kappa'] is not None else ""
        print(f"  {key}: acc={val['accuracy']*100:.2f}±{val.get('std', 0)*100:.2f}%{k_str}")

    return results


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='PhysioNet MI基线分类评估')
    parser.add_argument('--output-dir', default='paper_results/baseline_classification')
    parser.add_argument('--methods', type=str, default='all',
                        help='Comma-separated: CVAE,cGAN,EEGDiff,BrainDiff')
    parser.add_argument('--start-fold', type=int, default=1,
                        help='从第几折开始（1-based），用于断点续传')
    args = parser.parse_args()

    # 重置CUDA错误状态（避免之前运行残留的device-side assert）
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    device = get_device()
    np.random.seed(42)
    torch.manual_seed(42)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.methods == 'all':
        method_names = ['Spline', 'Kriging', 'CVAE', 'cGAN', 'EEGDiff', 'BrainDiff']
    else:
        method_names = [m.strip() for m in args.methods.split(',')]

    # 分离传统方法和深度学习方法
    TRADITIONAL_METHODS = {'Spline', 'Kriging'}
    traditional_methods = [n for n in method_names if n in TRADITIONAL_METHODS]
    deep_method_names = [n for n in method_names if n not in TRADITIONAL_METHODS]

    # 先获取数据集配置
    from data.physionet_mi import INPUT_CHANNEL_INDICES_16
    in_ch = len(INPUT_CHANNEL_INDICES_16)

    # 尝试从缓存获取数据维度
    cache_path = 'paper_results/physionet_mi/physionet_mi_all_subjects.npz'
    if os.path.exists(cache_path):
        cache = np.load(cache_path, allow_pickle=True)
        subj_map = cache['subject_data_map'].item()
        first_subj = list(subj_map.values())[0]
        out_ch = first_subj[0].shape[1]
        t_steps = first_subj[0].shape[2]
        all_labels = np.concatenate([v[1] for v in subj_map.values()])
        n_cls = len(np.unique(all_labels))
    else:
        from data.physionet_mi import PhysioNetMIDataset
        ds = PhysioNetMIDataset()
        out_ch = ds.data.shape[1]
        t_steps = ds.data.shape[2]
        n_cls = len(np.unique(ds.labels))
        del ds

    # 加载模型
    print("加载模型...")
    models = {}
    for name in deep_method_names:
        model, scale_info, ckpt_indices = load_baseline_model(name, device, in_ch, out_ch, t_steps, n_cls)
        if model is not None:
            # 使用checkpoint中的input_ch_indices（可能与脚本定义不同）
            if ckpt_indices is not None:
                print(f"  {name}: 使用checkpoint的input_ch_indices={ckpt_indices}")
            models[name] = (model, scale_info, ckpt_indices)

    if not models and not traditional_methods:
        print("[ERROR] 没有可用的模型或传统方法，退出")
        return

    if traditional_methods:
        print(f"传统插值方法: {traditional_methods}")

    all_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'methods': list(models.keys()) + traditional_methods,
    }

    all_results['cross_subject'] = physionet_cross_subject(models, device, traditional_methods,
                                                             start_fold=args.start_fold)

    # 保存结果
    out_path = os.path.join(args.output_dir, 'physionet_baseline_classification.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存到 {out_path}")


if __name__ == '__main__':
    main()
