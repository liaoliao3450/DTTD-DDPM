"""
DTTD模型训练脚本 (整合版)

训练DTTD-DDPM模型用于EEG通道重建
- 支持BCI2a数据集
- 使用无预处理模式（原始数据已预处理）
- 数据缩放因子: 1e5

使用方法:
    python experiments/train_model.py --epochs 200 --batch-size 32
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.bci2a import BCI2aDataset
from models.dttd_enhanced_v2 import DTTDEnhanced
from utils import load_config, set_seed


def train_epoch(model, train_loader, optimizer, device, epoch, config, data_scale_factor=1e5):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_cls_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch in pbar:
        if len(batch) == 4:
            target_data, ch_indices, labels, _ = batch
        else:
            target_data, ch_indices, labels = batch
        
        target_data = target_data.to(device)
        labels = labels.to(device)
        ch_idx = ch_indices[0].tolist() if ch_indices.dim() > 1 else ch_indices[0].tolist()
        
        # 缩放数据
        target_data = target_data * data_scale_factor
        
        # 提取输入通道
        input_data = target_data[:, ch_idx, :]
        
        # 添加轻量噪声
        noise_scale = 0.02
        noisy_input = input_data + torch.randn_like(input_data) * noise_scale
        
        # 随机时间步
        t = torch.randint(0, config['model']['num_timesteps'], (target_data.size(0),), device=device)
        
        optimizer.zero_grad()
        
        # 前向传播
        output = model(noisy_input, t, labels)
        
        # 重建损失
        recon_loss = nn.functional.mse_loss(output, target_data)
        
        # 分类器引导损失
        cls_loss = torch.tensor(0.0, device=device)
        if hasattr(model, 'classifier') and model.classifier is not None:
            with torch.no_grad():
                cls_logits = model.classifier(output)
            cls_loss = nn.functional.cross_entropy(cls_logits, labels)
            
            lambda_cls = config['model'].get('lambda_cls', 0.1)
            warmup_epochs = config['model'].get('warmup_epochs', 10)
            if epoch < warmup_epochs:
                lambda_cls = lambda_cls * (epoch / warmup_epochs)
            
            total_loss_batch = recon_loss + lambda_cls * cls_loss
        else:
            total_loss_batch = recon_loss
        
        total_loss_batch.backward()
        
        # 梯度裁剪
        if config['training'].get('gradient_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        total_recon_loss += recon_loss.item()
        total_cls_loss += cls_loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{total_loss/num_batches:.4f}',
            'recon': f'{total_recon_loss/num_batches:.4f}',
            'cls': f'{total_cls_loss/num_batches:.4f}'
        })
    
    return {
        'total_loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'cls_loss': total_cls_loss / num_batches
    }


def validate(model, val_loader, device, config, data_scale_factor=1e5):
    """验证"""
    model.eval()
    total_loss = 0.0
    total_corr = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 4:
                target_data, ch_indices, labels, _ = batch
            else:
                target_data, ch_indices, labels = batch
            
            target_data = target_data.to(device)
            labels = labels.to(device)
            ch_idx = ch_indices[0].tolist() if ch_indices.dim() > 1 else ch_indices[0].tolist()
            
            target_data = target_data * data_scale_factor
            input_data = target_data[:, ch_idx, :]
            t = torch.zeros(target_data.size(0), dtype=torch.long, device=device)
            
            output = model(input_data, t, labels)
            
            mse = nn.functional.mse_loss(output, target_data)
            
            # 计算相关系数
            output_flat = output.view(output.size(0), -1)
            target_flat = target_data.view(target_data.size(0), -1)
            corr = torch.mean(torch.sum(output_flat * target_flat, dim=1) / 
                            (torch.norm(output_flat, dim=1) * torch.norm(target_flat, dim=1) + 1e-8))
            
            total_loss += mse.item()
            total_corr += corr.item()
            num_batches += 1
    
    return {
        'val_loss': total_loss / num_batches,
        'val_corr': total_corr / num_batches
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='训练DTTD模型')
    parser.add_argument('--config', type=str, default='configs/bci2a_enhanced_config.yaml')
    parser.add_argument('--output-dir', type=str, default='checkpoints/dttd')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    config = load_config(args.config)
    
    # 使用无预处理模式
    config['data']['preprocess_mode'] = 'none'
    config['data']['normalization_mode'] = 'none'
    
    # 数据缩放因子
    data_scale_factor = 1e5
    
    print("\n" + "="*60)
    print("DTTD模型训练")
    print("="*60)
    print(f"预处理模式: {config['data']['preprocess_mode']}")
    print(f"数据缩放因子: {data_scale_factor}")
    print(f"训练轮数: {args.epochs}")
    print(f"输出目录: {args.output_dir}")
    
    # 加载数据
    print("\n加载数据...")
    train_dataset = BCI2aDataset(
        data_path=config['data']['data_path'],
        subject_ids=config['data']['subject_ids'],
        train=True,
        reconstruction_mode=True,
        preprocess_mode='none',
        normalization_mode='none',
        normalize=False
    )
    
    val_dataset = BCI2aDataset(
        data_path=config['data']['data_path'],
        subject_ids=config['data']['subject_ids'],
        train=False,
        reconstruction_mode=True,
        preprocess_mode='none',
        normalization_mode='none',
        normalize=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    
    # 创建模型
    print("\n创建模型...")
    model = DTTDEnhanced(config['model']).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 训练循环
    print("\n" + "="*60)
    print("开始训练")
    print("="*60)
    
    best_val_loss = float('inf')
    best_epoch = 0
    history = {'train': [], 'val': []}
    
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, config, data_scale_factor)
        val_metrics = validate(model, val_loader, device, config, data_scale_factor)
        scheduler.step()
        
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train - Loss: {train_metrics['total_loss']:.4f}, Recon: {train_metrics['recon_loss']:.4f}")
        print(f"  Val   - Loss: {val_metrics['val_loss']:.4f}, Corr: {val_metrics['val_corr']:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 保存最佳模型
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['val_loss'],
                'val_corr': val_metrics['val_corr'],
                'config': config,
                'data_scale_factor': data_scale_factor
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  ✓ 保存最佳模型 (val_loss: {best_val_loss:.4f})")
        
        # 定期保存
        if epoch % 50 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
                'data_scale_factor': data_scale_factor
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'config': config,
        'data_scale_factor': data_scale_factor
    }, os.path.join(args.output_dir, 'final_model.pth'))
    
    # 保存训练历史
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*60)
    print("训练完成")
    print("="*60)
    print(f"最佳模型: Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")
    print(f"模型保存到: {args.output_dir}")


if __name__ == '__main__':
    main()
