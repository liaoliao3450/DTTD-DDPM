"""
数据生成脚本 (整合版)

使用训练好的DTTD模型从9通道生成22通道EEG数据

使用方法:
    python experiments/generate_data.py --checkpoint checkpoints/dttd/best_model.pth --output generated_data/
"""
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat, savemat
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.dttd_enhanced_v2 import DTTDEnhanced
from utils import load_config, get_device

# 9通道索引
CH_IDX_9 = [7, 9, 11, 1, 3, 5, 13, 15, 17]


def load_raw_bci2a(data_path, subject_id, session='T'):
    """加载原始BCI2a数据"""
    subject_str = f'0{subject_id}' if subject_id < 10 else str(subject_id)
    file_path = os.path.join(data_path, f'A{subject_str}{session}.mat')
    
    mat_data = loadmat(file_path)
    
    if 'data' in mat_data:
        data = mat_data['data']
    elif 'X' in mat_data:
        data = mat_data['X']
    else:
        max_key = max(mat_data.keys(), key=lambda k: mat_data[k].size if isinstance(mat_data[k], np.ndarray) else 0)
        data = mat_data[max_key]
    
    if 'label' in mat_data:
        labels = mat_data['label'].flatten()
    elif 'labels' in mat_data:
        labels = mat_data['labels'].flatten()
    elif 'y' in mat_data:
        labels = mat_data['y'].flatten()
    else:
        labels = mat_data['Y'].flatten()
    
    labels = labels.astype(np.int64)
    if labels.min() > 0:
        labels = labels - labels.min()
    
    return data.astype(np.float32), labels


def load_dttd_model(config_path, checkpoint_path, device):
    """加载DTTD模型"""
    config = load_config(config_path)
    model = DTTDEnhanced(config['model']).to(device)
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    model.eval()
    
    # 获取数据缩放因子
    data_scale_factor = ckpt.get('data_scale_factor', 1e5)
    
    print(f"[OK] 加载DTTD模型: {checkpoint_path}")
    print(f"     数据缩放因子: {data_scale_factor}")
    
    return model, data_scale_factor


def generate_22ch_data(model, data_9ch, labels, device, data_scale_factor=1e5,
                       use_ddim=False, num_steps=50, guidance_scale=3.0):
    """
    用DTTD模型从9通道生成22通道数据
    
    Args:
        model: DTTD模型
        data_9ch: 9通道输入数据 [N, 9, T]
        labels: 类别标签 [N]
        device: 设备
        data_scale_factor: 数据缩放因子
        use_ddim: 是否使用DDIM采样
        num_steps: DDIM采样步数
        guidance_scale: 分类器引导强度
    
    Returns:
        generated_22ch: 生成的22通道数据 [N, 22, T]
    """
    model.eval()
    
    # 放大输入数据
    data_9ch_scaled = data_9ch * data_scale_factor
    
    loader = DataLoader(TensorDataset(torch.FloatTensor(data_9ch_scaled), torch.LongTensor(labels)),
                        batch_size=32, shuffle=False)
    
    generated_list = []
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(loader, desc="生成22通道"):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            if use_ddim and hasattr(model, 'sample_ddim'):
                gen = model.sample_ddim(
                    batch_data, 
                    task_label=batch_labels,
                    num_inference_steps=num_steps,
                    eta=0.0,
                    guidance_scale=guidance_scale
                ).cpu().numpy()
            elif hasattr(model, 'sample'):
                gen = model.sample(
                    batch_data,
                    task_label=batch_labels,
                    num_steps=10,
                    guidance_scale=guidance_scale,
                    use_full_denoising=False
                ).cpu().numpy()
            else:
                # 直接前向传播
                t = torch.zeros(batch_data.size(0), dtype=torch.long, device=device)
                gen = model(batch_data, t, batch_labels).cpu().numpy()
            
            # 缩小回原始尺度
            gen = gen / data_scale_factor
            generated_list.append(gen)
    
    return np.concatenate(generated_list, axis=0)


def generate_for_subject(model, data_path, subject_id, session, device, 
                         data_scale_factor, output_dir, **kwargs):
    """为单个被试生成数据"""
    print(f"\n处理被试 S{subject_id} Session {session}...")
    
    # 加载原始数据
    data_22ch, labels = load_raw_bci2a(data_path, subject_id, session)
    data_9ch = data_22ch[:, CH_IDX_9, :]
    
    print(f"  原始数据: {data_22ch.shape}")
    print(f"  9通道输入: {data_9ch.shape}")
    
    # 生成22通道
    generated_22ch = generate_22ch_data(model, data_9ch, labels, device, 
                                        data_scale_factor, **kwargs)
    
    print(f"  生成数据: {generated_22ch.shape}")
    
    # 保存
    subject_str = f'0{subject_id}' if subject_id < 10 else str(subject_id)
    output_file = os.path.join(output_dir, f'A{subject_str}{session}_generated.mat')
    
    savemat(output_file, {
        'data_original': data_22ch,
        'data_generated': generated_22ch,
        'data_9ch': data_9ch,
        'labels': labels,
        'ch_idx_9': CH_IDX_9
    })
    
    print(f"  保存到: {output_file}")
    
    return {
        'subject_id': subject_id,
        'session': session,
        'num_samples': len(labels),
        'output_file': output_file
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='使用DTTD生成22通道EEG数据')
    parser.add_argument('--config', default='configs/bci2a_enhanced_config.yaml')
    parser.add_argument('--checkpoint', default='checkpoints/dttd/best_model.pth')
    parser.add_argument('--data-path', default='E:/data/BCI2a')
    parser.add_argument('--output', default='generated_data/')
    parser.add_argument('--subjects', type=int, nargs='+', default=list(range(1, 10)))
    parser.add_argument('--sessions', type=str, nargs='+', default=['T', 'E'])
    parser.add_argument('--use-ddim', action='store_true', help='使用DDIM采样')
    parser.add_argument('--num-steps', type=int, default=50, help='DDIM采样步数')
    parser.add_argument('--guidance-scale', type=float, default=3.0, help='引导强度')
    args = parser.parse_args()
    
    device = get_device()
    print(f"使用设备: {device}")
    
    # 加载模型
    model, data_scale_factor = load_dttd_model(args.config, args.checkpoint, device)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 生成数据
    print("\n" + "="*60)
    print("开始生成数据")
    print("="*60)
    
    results = []
    for subject_id in args.subjects:
        for session in args.sessions:
            result = generate_for_subject(
                model, args.data_path, subject_id, session, device,
                data_scale_factor, args.output,
                use_ddim=args.use_ddim,
                num_steps=args.num_steps,
                guidance_scale=args.guidance_scale
            )
            results.append(result)
    
    # 保存生成记录
    with open(os.path.join(args.output, 'generation_log.json'), 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'data_scale_factor': data_scale_factor,
            'use_ddim': args.use_ddim,
            'num_steps': args.num_steps,
            'guidance_scale': args.guidance_scale,
            'results': results
        }, f, indent=2)
    
    print("\n" + "="*60)
    print("数据生成完成")
    print("="*60)
    print(f"共生成 {len(results)} 个文件")
    print(f"保存到: {args.output}")


if __name__ == '__main__':
    main()
