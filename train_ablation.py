"""
DTTD Ablation Models 训练脚本
支持训练不同的消融模型变体
"""
import sys
import os
import shutil
import yaml
from tqdm import tqdm
# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import time

from models.dttd_ablation import create_model
from data import get_bci2a_dataloaders, get_seed_dataloaders
from utils import (
    set_seed, load_config, save_checkpoint, get_device,
    AverageMeter, EarlyStopping, create_dirs, get_lr,
    compute_mse, compute_mae, MetricsTracker
)


def train_epoch(model, train_loader, optimizer, device, epoch, scaler=None):
    """训练一个epoch"""
    model.train()
    losses = AverageMeter()
    mse_metric = AverageMeter()
    mae_metric = AverageMeter()
    
    # 使用非阻塞内存传输
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 创建进度条
    pbar = tqdm(train_loader, desc=f'Train Epoch {epoch}')
    
    for batch_idx, batch_data in enumerate(pbar):
        # 准备数据
        # 重建模式：输入是目标数据、通道索引和标签
        target_data, channel_indices, labels = batch_data
        target_data = target_data.to(device, non_blocking=True)
        channel_indices = channel_indices[0].to(device, non_blocking=True)  # 所有样本共享相同索引
        labels = labels.to(device, non_blocking=True)
        
        # 从完整数据中提取输入通道
        input_data = target_data[:, channel_indices, :]  # [B, 9, T]
        
        # 确保目标数据是完整的22个通道
        if target_data.size(1) != 22:
            print(f"警告：目标数据通道数={target_data.size(1)}，预期为22。请检查数据加载器。")
   
        # 混合精度训练
        with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):
            # 前向传播 - 使用 input_data (9通道) 作为模型输入
            model_output = model(input_data, channel_indices, labels)
            # 计算损失时使用完整的目标数据 (22通道)
            loss = F.mse_loss(model_output, target_data)
            mse = compute_mse(model_output, target_data)
            mae = compute_mae(model_output, target_data)
        
        # 反向传播和优化
        optimizer.zero_grad(set_to_none=True)  # 更快的归零
        # 计算梯度
        if scaler is not None:
            scaler.scale(loss).backward()
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # 更新参数
            optimizer.step()
        
        # 更新统计信息
        losses.update(loss.item() if hasattr(loss, 'item') else loss, target_data.size(0))
        mse_metric.update(mse.item() if hasattr(mse, 'item') else mse, target_data.size(0))
        mae_metric.update(mae.item() if hasattr(mae, 'item') else mae, target_data.size(0))
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{losses.avg:.4f}",
            'mse': f"{mse_metric.avg:.4f}",
            'mae': f"{mae_metric.avg:.4f}"
        })
    
    return {
        'loss': losses.avg,
        'mse': mse_metric.avg,
        'mae': mae_metric.avg
    }


def validate(model, val_loader, device, epoch=0):
    """验证模型"""
    model.eval()
    losses = AverageMeter()
    mse_metric = AverageMeter()
    mae_metric = AverageMeter()
    
    # 使用非阻塞内存传输
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Val Epoch {epoch}' if epoch else 'Validating')
        
        for batch_idx, batch_data in enumerate(pbar):
            # 准备数据
            # 重建模式：输入是目标数据、通道索引和标签
            target_data, channel_indices, labels = batch_data
            target_data = target_data.to(device, non_blocking=True)
            channel_indices = channel_indices[0].to(device, non_blocking=True)  # 所有样本共享相同索引
            labels = labels.to(device, non_blocking=True)
            
            # 从完整数据中提取输入通道（9个选定的通道）
            input_data = target_data[:, channel_indices, :]  # [B, 9, T]
            
            # 确保目标数据是完整的22个通道
            if target_data.size(1) != 22:
                print(f"警告：目标数据通道数={target_data.size(1)}，预期为22。请检查数据加载器。")
            
            # 使用混合精度进行前向传播
            with torch.amp.autocast(device_type='cuda'):
                # 前向传播 - 使用 input_data (9通道) 作为模型输入
                model_output = model(input_data, channel_indices, labels)
                # 计算损失时使用完整的目标数据 (22通道)
                loss = F.mse_loss(model_output, target_data)
                mse = F.mse_loss(model_output, target_data)
                
                # 更新指标 - 确保mse和mae是标量
                mse_scalar = mse.item() if hasattr(mse, 'item') else mse
                mse_metric.update(mse_scalar, target_data.size(0))
                mae = compute_mae(model_output, target_data)
                mae_scalar = mae.item() if hasattr(mae, 'item') else mae
                mae_metric.update(mae_scalar, target_data.size(0))
            
            # 更新统计信息
            losses.update(loss.item(), target_data.size(0))
            
            # 更新进度条
            pbar.set_postfix({
                'val_loss': f"{losses.avg:.4f}",
                'val_mse': f"{mse_metric.avg:.4f}",
                'val_mae': f"{mae_metric.avg:.4f}"
            })
    
    return {
        'val_loss': losses.avg,
        'val_mse': mse_metric.avg,
        'val_mae': mae_metric.avg
    }


def main():
    parser = argparse.ArgumentParser(description='Train DTTD Ablation Models')
    parser.add_argument('--config', type=str, default='configs/bci2a_enhanced_config.yaml',
                        help='path to config file')
    parser.add_argument('--model', type=str, default='dttd_no_freq',
                        choices=['dttd_no_topo', 'dttd_no_freq', 'dttd_no_task', 'dttd_no_topo_freq'],
                        help='ablation model to train (default: dttd_no_topo)')
    parser.add_argument('--experiment', type=str, default= 'dttd_no_topo',
                        help='experiment name (default: model_name)')
    parser.add_argument('--resume', type=str, default= 'bci2a_enhanced',
                        help='path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--device', type=str, default= 'cuda',
                        help='device to use (default: cuda if available, else cpu)')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖配置中的设备设置（如果指定）
    if args.device is not None:
        config['training']['device'] = args.device
    
    # 设置随机种子
    set_seed(args.seed if 'seed' not in config['training'] else config['training']['seed'])
    
    # 设备设置和CUDA优化
    use_cuda = torch.cuda.is_available() and (config['training'].get('device', 'cuda') == 'cuda')
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    # 打印CUDA信息
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        # 启用cuDNN自动调优
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        # 清空GPU缓存
        torch.cuda.empty_cache()
    
    print(f"使用设备: {device}")
    
    # 设置默认张量类型为FloatTensor
    torch.set_default_tensor_type(torch.FloatTensor)
    
    # 实验名称
    experiment_name = args.experiment if args.experiment else f"{args.model}_{args.seed}"
    
    # 创建目录
    save_dir = os.path.join(config['training']['save_dir'], 'ablation', experiment_name)
    log_dir = os.path.join(config['training']['log_dir'], 'ablation', experiment_name)
    
    create_dirs([save_dir, log_dir])
    
    # 保存配置
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # 加载数据 - 只使用第一个会话（训练集）
    if config['data']['dataset'] == 'bci2a':
        # 获取所有9个被试的第一个会话（训练集）
        subject_ids = list(range(1, 10))  # 9个被试
        print(f'使用9个被试的第一个会话（训练集）进行训练和验证，被试ID: {subject_ids}')
        
        # 确保使用正确的输入通道（9个运动想象相关通道）
        input_channels = config['data'].get('selected_channels', [
            'C3', 'Cz', 'C4', 'FC3', 'FCz', 'FC4', 'CP3', 'CPz', 'CP4'
        ])
        print(f'使用输入通道: {input_channels}')
        
        # 获取数据加载器，设置val_split=0.2表示从训练集中划分20%作为验证集
        # 使用重建模式（9通道重建22通道）
        reconstruction_mode = config['model'].get('reconstruction', True)
        print(f'使用重建模式: {reconstruction_mode} (9通道 -> 22通道)' if reconstruction_mode else '使用普通模式 (9通道 -> 9通道)')
        
        train_loader, val_loader, _ = get_bci2a_dataloaders(
            data_path=config['data']['data_path'],
            subject_ids=subject_ids,
            batch_size=config['training']['batch_size'],
            selected_channels=input_channels,
            num_workers=config['training'].get('num_workers', 4),
            reconstruction_mode=reconstruction_mode,  # 使用重建模式
            val_split=0.2,  # 20%的训练数据作为验证集
            random_seed=config['training'].get('seed', 42),
            normalization_mode='channel_global'  # 使用通道级全局归一化
        )
    elif config['data']['dataset'] == 'seed':
        train_loader, val_loader = get_seed_dataloaders(
            data_path=config['data']['data_path'],
            subject_ids=config['data']['subject_ids'],
            batch_size=config['training']['batch_size'],
            selected_channels=config['data'].get('selected_channels'),
            num_workers=config['training']['num_workers'],
            reconstruction_mode=True
        )
    else:
        raise ValueError(f"Unsupported dataset: {config['data']['dataset']}")
    
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    
    # 创建模型
    print(f"创建模型: {args.model}")
    model = create_model(args.model, config['model']).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    
    # 将模型设置为训练模式
    model.train()
    
    # 打印模型信息
    print(f"模型已移至 {next(model.parameters()).device}")
    
    # 优化器
    if config['training']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0.0)
        )
    elif config['training']['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0.01)
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['training']['optimizer']}")
    
    # 学习率调度器
    scheduler_type = config['training'].get('scheduler')
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs']
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True,
            min_lr=1e-6
        )
    else:
        scheduler = None
    
    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> 加载检查点 '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if scheduler and 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"=> 加载检查点 '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> 未找到检查点 '{args.resume}'")
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    # 早停
    early_stopping = EarlyStopping(
        patience=config['training'].get('patience', 30),  # 增加耐心值
        delta=0.001  # 设置一个最小变化阈值
    )
    
    # 学习率预热函数
    def adjust_learning_rate(optimizer, epoch, warmup_epochs, base_lr):
        if epoch < warmup_epochs:
            # 线性预热
            lr = base_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # 训练循环
    warmup_epochs = config['training'].get('warmup_epochs', 10)
    base_lr = config['training']['learning_rate']
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        # 学习率预热
        if epoch < warmup_epochs:
            adjust_learning_rate(optimizer, epoch, warmup_epochs, base_lr)
        
        # 训练一个epoch
        train_metrics = train_epoch(
            model, 
            train_loader, 
            optimizer, 
            device, 
            epoch, 
            scaler=scaler
        )
        
        # 验证
        val_metrics = validate(
            model, 
            val_loader, 
            device, 
            epoch
        )
        
        # 更新学习率
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['val_loss'])
            else:
                scheduler.step()
        
        # 记录到TensorBoard
        writer.add_scalar('train/loss', train_metrics['loss'], epoch)
        writer.add_scalar('train/mse', train_metrics['mse'], epoch)
        writer.add_scalar('train/mae', train_metrics['mae'], epoch)
        writer.add_scalar('val/loss', val_metrics['val_loss'], epoch)
        writer.add_scalar('val/mse', val_metrics['val_mse'], epoch)
        writer.add_scalar('val/mae', val_metrics['val_mae'], epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        # 清空GPU缓存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # 保存检查点
        is_best = val_metrics['val_loss'] < best_val_loss
        best_val_loss = min(val_metrics['val_loss'], best_val_loss)
        
        # 保存检查点（包含模型名称）
        checkpoint_path = os.path.join(save_dir, f'checkpoint_{args.model}.pth')
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            loss=val_metrics['val_loss'],
            save_path=checkpoint_path
        )
        
        # 如果是当前最佳模型，则保存为最佳模型
        if is_best:
            best_path = os.path.join(save_dir, f'model_best_{args.model}.pth')
            shutil.copyfile(checkpoint_path, best_path)
            print(f'新的最佳模型已保存到: {best_path}')
        
        # 早停检查
        early_stopping(val_metrics['val_loss'])
        if early_stopping.early_stop:
            print(f"早停于第 {epoch} 轮")
            break
        
        print(f'Epoch: {epoch+1:03d} | Train Loss: {train_metrics["loss"]:.6f} | Val Loss: {val_metrics["val_loss"]:.6f} | Best Val: {best_val_loss:.6f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    print(f"训练完成，最佳验证损失: {best_val_loss:.6f}")
    
    # 保存最终模型（包含模型名称）
    final_model_path = os.path.join(save_dir, f'final_model_{args.model}.pth')
    torch.save({
        'model': args.model,
        'state_dict': model.state_dict(),
        'config': config
    }, final_model_path)
    print(f'最终模型已保存到: {final_model_path}')
    
    # 保存训练结果
    results = {
        'best_val_loss': best_val_loss,
        'num_params': sum(p.numel() for p in model.parameters()),
        'config': config
    }
    
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    writer.close()


if __name__ == '__main__':
    main()
