"""
快速训练DTTD Enhanced V2模型（使用新频率模块）
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.dttd_enhanced_v2 import DTTDEnhanced
from data.bci2a import BCI2aDataset
from utils import load_config, set_seed


def train_model(config_path='configs/bci2a_enhanced_config.yaml', 
                epochs=200, 
                batch_size=32,
                lr=1e-4,
                save_path='checkpoints/bci2a_enhanced_v2/best_model.pth'):
    """训练模型"""
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载配置
    config = load_config(config_path)
    
    # 加载数据
    print("\n加载BCI2a数据集...")
    data_path = config['data'].get('data_path', 'E:/data/BCI2a')
    
    # 加载Session 1数据用于训练 - 使用无预处理模式
    train_dataset = BCI2aDataset(
        data_path=data_path,
        subject_ids=list(range(1, 10)),
        normalize=False,  # ⭐ 关闭归一化
        train=True,
        normalization_mode='none',  # ⭐ 无归一化
        preprocess_mode='none'  # ⭐ 无预处理
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # 获取通道索引
    ch_idx = train_dataset.input_indices
    ch_idx = torch.tensor(ch_idx, device=device)
    print(f"输入通道索引: {ch_idx.tolist()}")
    
    # 创建模型
    print("\n创建模型...")
    model = DTTDEnhanced(config['model']).to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {total_params:,}")
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 训练
    print(f"\n开始训练 ({epochs} epochs)...")
    best_loss = float('inf')
    
    # ⭐ 数据缩放因子（EEG数据值很小，需要放大）
    data_scale_factor = 1e5
    print(f"数据缩放因子: {data_scale_factor}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            # BCI2a数据集返回: (target_data, channel_indices, label, subject_id)
            if isinstance(batch, (list, tuple)) and len(batch) == 4:
                data = batch[0].to(device) * data_scale_factor  # ⭐ 放大数据
                # channel_indices已经在外面定义了
                labels = batch[2].to(device)  # [B]
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")
            
            optimizer.zero_grad()
            
            # 计算损失
            loss = model.compute_loss(
                x_target=data,
                channel_indices=ch_idx,
                task_label=labels,
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
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss_value.item():.4f}'})
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  保存最佳模型 (loss={best_loss:.4f})")
    
    print(f"\n训练完成！最佳损失: {best_loss:.4f}")
    print(f"模型保存到: {save_path}")
    
    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)  # 默认200 epochs
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
