"""
HGD 基线分类评估

对基线方法在HGD数据集上进行跨会话和跨被试(LOSO)重建分类和增强分类评估。

使用方法:
    python experiments/hgd_baseline_classification.py
    python experiments/hgd_baseline_classification.py --methods CVAE,EEGDiff
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
    'CVAE': 'checkpoints/hgd/baseline_cvae/best_model.pth',
    'cGAN': 'checkpoints/hgd/baseline_cgan/best_model.pth',
    'EEGDiff': 'checkpoints/hgd/baseline_eegdiff/best_model.pth',
    'BrainDiff': 'checkpoints/hgd/baseline_braindiff/best_model.pth',
}


# ==================== EEGNet分类器 ====================

class EEGNetClassifier(nn.Module):
    def __init__(self, num_channels, num_classes=4, time_steps=1000):
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


def train_classifier(train_data, train_labels, device, num_channels, num_classes=4,
                     time_steps=None, epochs=60,
                     extra_data=None, extra_labels=None):
    """训练EEGNet分类器。extra_data/extra_labels用于增强，避免numpy concatenate的内存开销。"""
    if time_steps is None:
        time_steps = train_data.shape[2]
    torch.manual_seed(42)
    np.random.seed(42)

    clf = EEGNetClassifier(num_channels, num_classes, time_steps).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)

    generator = torch.Generator()
    generator.manual_seed(42)

    ds1 = TensorDataset(torch.FloatTensor(train_data), torch.LongTensor(train_labels))
    if extra_data is not None and extra_labels is not None:
        ds2 = TensorDataset(torch.FloatTensor(extra_data), torch.LongTensor(extra_labels))
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
    return clf


def evaluate_classifier(clf, test_data, test_labels, device):
    clf.eval()
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

        # HGD所有模型使用z-score
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
    generated_list = []

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

    # 预分配输出数组，避免concatenate的内存峰值
    n_samples = data_input.shape[0]
    n_t = data_input.shape[2]
    # 先用一个batch推断输出形状
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(data_scaled), torch.LongTensor(labels)),
        batch_size=batch_size, shuffle=False)

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

    batch_size = 32
    generated_list = []
    for i in range(0, len(data_input), batch_size):
        batch = data_input[i:i+batch_size]
        recon = interp.reconstruct(batch)
        if torch.is_tensor(recon):
            recon = recon.cpu().numpy()
        generated_list.append(recon.astype(np.float32))

    return np.concatenate(generated_list, axis=0)


# ==================== HGD跨会话 ====================

def hgd_cross_session(data_path, models, device, traditional_methods=None):
    """HGD跨会话评估: 每个被试在train session上训练，test session上测试"""
    if traditional_methods is None:
        traditional_methods = []
    print("\n" + "=" * 60)
    print("HGD 跨会话 (train→test)")
    print("=" * 60)

    from data.high_gamma_dataset import HighGammaDataset
    input_indices = None
    n_output = None
    n_input = None
    DS = 4  # 降采样倍数，与high_gamma_classification一致

    results = {}

    for sid in range(1, 15):
        print(f"\n--- Subject S{sid} ---")
        try:
            train_ds = HighGammaDataset(data_path, subject_ids=[sid], sessions='train', fs_target=250)
            test_ds = HighGammaDataset(data_path, subject_ids=[sid], sessions='test', fs_target=250)
        except Exception as e:
            print(f"  被试{sid}加载失败: {e}")
            continue

        if input_indices is None:
            input_indices = train_ds.input_ch_indices
            n_output = train_ds.num_output_channels
            n_input = len(input_indices)

        train_full_hr = train_ds.data.astype(np.float32)
        train_labels = train_ds.labels.astype(np.int64)
        test_full_hr = test_ds.data.astype(np.float32)
        test_labels = test_ds.labels.astype(np.int64)

        # 降采样用于分类，原始分辨率用于模型推理
        train_full = train_full_hr[:, :, ::DS]
        test_full = test_full_hr[:, :, ::DS]
        train_input = train_full[:, input_indices, :]
        test_input = test_full[:, input_indices, :]
        train_input_hr = train_full_hr[:, input_indices, :]
        test_input_hr = test_full_hr[:, input_indices, :]

        print(f"  train={len(train_full)}, test={len(test_full)}, time_steps={train_full.shape[2]}")

        # 基线
        clf = train_classifier(train_full, train_labels, device, n_output)
        m_full = evaluate_classifier(clf, test_full, test_labels, device)
        del clf

        clf = train_classifier(train_input, train_labels, device, n_input)
        m_input = evaluate_classifier(clf, test_input, test_labels, device)
        del clf

        subj_result = {
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
            # 模型推理使用原始分辨率（1000时间步），输出再降采样用于分类
            train_input_model_hr = train_full_hr[:, model_input_indices, :]
            gen_full_hr = generate_reconstruction(model, model_name, train_input_model_hr, train_labels,
                                                  device, scale_info=scale_info,
                                                  input_ch_indices=model_input_indices)
            gen_full = gen_full_hr[:, :, ::DS]  # 降采样用于分类
            del gen_full_hr

            aug_data = np.concatenate([train_full, gen_full], axis=0)
            aug_labels = np.concatenate([train_labels, train_labels], axis=0)
            clf = train_classifier(aug_data, aug_labels, device, n_output)
            m_aug = evaluate_classifier(clf, test_full, test_labels, device)
            del clf

            subj_result[f'{model_name}_aug'] = {
                'accuracy': m_aug['accuracy'], 'kappa': m_aug['kappa'],
            }
            print(f"  {model_name}: aug acc={m_aug['accuracy']*100:.2f}%, kappa={m_aug['kappa']:.4f}")
            torch.cuda.empty_cache()

        for trad_name in traditional_methods:
            gen_full = generate_traditional_reconstruction(
                trad_name, train_input, n_input, n_output)

            aug_data = np.concatenate([train_full, gen_full], axis=0)
            aug_labels = np.concatenate([train_labels, train_labels], axis=0)
            clf = train_classifier(aug_data, aug_labels, device, n_output)
            m_aug = evaluate_classifier(clf, test_full, test_labels, device)
            del clf

            subj_result[f'{trad_name}_aug'] = {
                'accuracy': m_aug['accuracy'], 'kappa': m_aug['kappa'],
            }
            print(f"  {trad_name}: aug acc={m_aug['accuracy']*100:.2f}%, kappa={m_aug['kappa']:.4f}")

        results[f'S{sid}'] = subj_result

    # 平均
    if results:
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
        print(f"\n跨会话平均结果:")
        for key, val in avg_result.items():
            k_str = f", kappa={val['kappa']:.4f}" if val['kappa'] is not None else ""
            print(f"  {key}: acc={val['accuracy']*100:.2f}±{val.get('std', 0)*100:.2f}%{k_str}")

    return results


# ==================== HGD跨被试 ====================

def hgd_cross_subject(data_path, models, device, traditional_methods=None, start_subject=1):
    """HGD跨被试 LOSO"""
    if traditional_methods is None:
        traditional_methods = []
    print("\n" + "=" * 60)
    print("HGD 跨被试 (LOSO)")
    print("=" * 60)

    from data.high_gamma_dataset import HighGammaDataset
    # 先加载一个被试获取通道信息
    tmp_ds = HighGammaDataset(data_path, subject_ids=[1], sessions='both', fs_target=250)
    input_indices = tmp_ds.input_ch_indices
    n_output = tmp_ds.num_output_channels
    n_input = len(input_indices)
    del tmp_ds

    # 逐被试加载数据（同时保留原始分辨率和降采样版本）
    DS = 4  # 降采样倍数，与high_gamma_classification一致
    subject_data_hr = {}  # 原始分辨率，用于模型推理
    subject_data_ds = {}  # 降采样版本，用于分类
    subject_labels = {}
    for sid in range(1, 15):
        try:
            ds = HighGammaDataset(data_path, subject_ids=[sid], sessions='both', fs_target=250)
            subject_data_hr[sid] = ds.data.astype(np.float32)
            subject_data_ds[sid] = ds.data.astype(np.float32)[:, :, ::DS]
            subject_labels[sid] = ds.labels.astype(np.int64)
            del ds
        except Exception as e:
            print(f"  被试{sid}加载失败: {e}")
            continue

    if not subject_data_ds:
        print("[ERROR] 没有可用的被试数据")
        return {}

    n_cls = 4  # HGD有4个类别
    print(f"数据: {n_output}ch输出, {n_input}ch输入, 类别={n_cls}")
    print(f"输入通道索引: {input_indices}")

    results = {}

    for test_sid in range(start_subject, 15):
        print(f"\n--- Test Subject S{test_sid} ---")
        if test_sid not in subject_data_ds:
            continue

        train_data_list, train_labels_list = [], []
        train_global_to_local = []  # [(sid, local_idx), ...] 映射全局索引到被试本地索引
        test_data_list, test_labels_list = [], []

        for sid in range(1, 15):
            if sid not in subject_data_ds:
                continue
            n_samples = len(subject_labels[sid])
            if sid == test_sid:
                test_data_list.append(subject_data_ds[sid])
                test_labels_list.append(subject_labels[sid])
            else:
                train_data_list.append(subject_data_ds[sid])
                train_labels_list.append(subject_labels[sid])
                for i in range(n_samples):
                    train_global_to_local.append((sid, i))

        if not train_data_list or not test_data_list:
            continue

        train_full = np.concatenate(train_data_list, axis=0).astype(np.float32)
        train_labels = np.concatenate(train_labels_list, axis=0).astype(np.int64)
        test_full = np.concatenate(test_data_list, axis=0).astype(np.float32)
        test_labels = np.concatenate(test_labels_list, axis=0).astype(np.int64)

        train_input = train_full[:, input_indices, :]
        test_input = test_full[:, input_indices, :]

        # 基线
        clf = train_classifier(train_full, train_labels, device, n_output)
        m_full = evaluate_classifier(clf, test_full, test_labels, device)
        del clf

        clf = train_classifier(train_input, train_labels, device, n_input)
        m_input = evaluate_classifier(clf, test_input, test_labels, device)
        del clf

        subj_result = {
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

            # 0.2倍增强：只对20%训练数据生成重建
            aug_ratio = 0.2
            n_aug = max(1, int(len(train_labels) * aug_ratio))
            rng = np.random.RandomState(42)
            aug_idx = rng.choice(len(train_labels), n_aug, replace=False)

            # 按需从各被试HR数据中提取增强子集的输入通道，避免拼接全部HR数据
            aug_hr_list = []
            for gi in aug_idx:
                sid, local_idx = train_global_to_local[gi]
                aug_hr_list.append(subject_data_hr[sid][local_idx:local_idx+1, model_input_indices, :])
            train_input_model_hr = np.concatenate(aug_hr_list, axis=0).astype(np.float32)
            del aug_hr_list

            gen_full_hr = generate_reconstruction(model, model_name, train_input_model_hr, train_labels[aug_idx],
                                                  device, scale_info=scale_info,
                                                  input_ch_indices=model_input_indices)
            gen_full = gen_full_hr[:, :, ::DS]  # 降采样用于分类
            del gen_full_hr, train_input_model_hr

            # 增强（用extra_data避免numpy concatenate内存开销）
            clf = train_classifier(train_full, train_labels, device, n_output,
                                   extra_data=gen_full, extra_labels=train_labels[aug_idx])
            m_aug = evaluate_classifier(clf, test_full, test_labels, device)
            del clf, gen_full

            subj_result[f'{model_name}_aug'] = {
                'accuracy': m_aug['accuracy'], 'kappa': m_aug['kappa'],
            }
            print(f"  {model_name}: aug acc={m_aug['accuracy']*100:.2f}%, kappa={m_aug['kappa']:.4f}")
            torch.cuda.empty_cache()

        # 传统插值方法评估
        for trad_name in traditional_methods:
            # 0.2倍增强：只对20%训练数据生成重建
            aug_ratio = 0.2
            n_aug = max(1, int(len(train_labels) * aug_ratio))
            rng = np.random.RandomState(42)
            aug_idx = rng.choice(len(train_labels), n_aug, replace=False)
            gen_full = generate_traditional_reconstruction(
                trad_name, train_input[aug_idx], n_input, n_output)

            # 增强（用extra_data避免numpy concatenate内存开销）
            clf = train_classifier(train_full, train_labels, device, n_output,
                                   extra_data=gen_full, extra_labels=train_labels[aug_idx])
            m_aug = evaluate_classifier(clf, test_full, test_labels, device)
            del clf, gen_full

            subj_result[f'{trad_name}_aug'] = {
                'accuracy': m_aug['accuracy'], 'kappa': m_aug['kappa'],
            }
            print(f"  {trad_name}: aug acc={m_aug['accuracy']*100:.2f}%, kappa={m_aug['kappa']:.4f}")

        results[f'S{test_sid}'] = subj_result

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
    parser = argparse.ArgumentParser(description='HGD基线分类评估')
    parser.add_argument('--data-path', default='E:/data/HGD')
    parser.add_argument('--output-dir', default='paper_results/baseline_classification')
    parser.add_argument('--methods', type=str, default='all',
                        help='Comma-separated: CVAE,cGAN,EEGDiff,BrainDiff')
    parser.add_argument('--eval-mode', type=str, default='cross_subject',
                        choices=['all', 'cross_session', 'cross_subject'],
                        help='Which evaluation to run: all, cross_session, or cross_subject')
    parser.add_argument('--start-subject', type=int, default=2,
                        help='Start from this subject ID (for resuming interrupted runs)')
    args = parser.parse_args()

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

    # 先加载数据集获取通道信息
    from data.high_gamma_dataset import HighGammaDataset
    dataset = HighGammaDataset(args.data_path, subject_ids=[1], sessions='both', fs_target=250)
    in_ch = len(dataset.input_ch_indices)
    out_ch = dataset.num_output_channels
    t_steps = dataset.data.shape[2]
    n_cls = len(np.unique(dataset.labels))
    del dataset

    # 加载模型
    print("加载模型...")
    models = {}
    for name in deep_method_names:
        model, scale_info, ckpt_indices = load_baseline_model(name, device, in_ch, out_ch, t_steps, n_cls)
        if model is not None:
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

    eval_mode = getattr(args, 'eval_mode', 'all')
    print(f"[INFO] eval_mode = {eval_mode}")
    if eval_mode in ('all', 'cross_session'):
        all_results['cross_session'] = hgd_cross_session(args.data_path, models, device, traditional_methods)
    if eval_mode in ('all', 'cross_subject'):
        all_results['cross_subject'] = hgd_cross_subject(args.data_path, models, device, traditional_methods,
                                                          start_subject=args.start_subject)

    # 保存结果（合并已有的跨被试结果）
    out_path = os.path.join(args.output_dir, 'hgd_baseline_classification.json')
    existing = {}
    if os.path.exists(out_path):
        with open(out_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)

    # 合并跨被试结果
    if 'cross_subject' in all_results and 'cross_subject' in existing:
        for key, val in existing['cross_subject'].items():
            if key not in all_results['cross_subject']:
                all_results['cross_subject'][key] = val
    # 合并跨会话结果（保留已有）
    if 'cross_session' in existing and 'cross_session' not in all_results:
        all_results['cross_session'] = existing['cross_session']

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存到 {out_path}")


if __name__ == '__main__':
    main()
