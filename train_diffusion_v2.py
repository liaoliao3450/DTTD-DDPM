"""
DTTD模型训练脚本
"""
import sys
import os
# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from tqdm import tqdm

from models import DTTD, DTTDSimple, DTTDX0, DTTDEnhanced
from data import get_bci2a_dataloaders, get_seed_dataloaders
from utils import (
    set_seed, load_config, save_checkpoint, get_device,
    AverageMeter, EarlyStopping, create_dirs, get_lr,
    compute_mse, compute_mae, MetricsTracker
)


def train_epoch(model, train_loader, optimizer, device, epoch, reconstruction_mode=True):
    """训练一个epoch"""
    model.train()
    losses = AverageMeter()
    recon_losses = AverageMeter()
    cls_losses = AverageMeter()
    
    # 梯度监控
    grad_norms = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch_data in enumerate(pbar):
        # 准备数据
        if reconstruction_mode:
            # 重建模式：(target_data, channel_indices, labels, subject_ids)
            if len(batch_data) == 4:
                target_data, channel_indices, labels, _ = batch_data
                target_data = target_data.to(device)
                # 使用第一个样本的channel_indices作为整个batch的channel_indices
                channel_indices = channel_indices[0].to(device)  # [num_input_channels]
                labels = labels.to(device)
            else:
                # 兼容旧版本
                target_data, channel_indices, labels = batch_data
                target_data = target_data.to(device)
                # 使用第一个样本的channel_indices作为整个batch的channel_indices
                channel_indices = channel_indices[0].to(device)  # [num_input_channels]
                labels = labels.to(device)
            
            # 前向传播（通道重建）
            loss_output = model.compute_loss(
                x_target=target_data,
                channel_indices=channel_indices,
                task_label=labels,
                current_epoch=epoch
            )
        else:
            # 普通模式：(input_data, labels, subject_ids)
            if len(batch_data) == 3:
                input_data, labels, _ = batch_data
                input_data = input_data.to(device)
                labels = labels.to(device)
            else:
                # 兼容旧版本
                input_data, labels = batch_data
                input_data = input_data.to(device)
                labels = labels.to(device)
            
            # 前向传播（普通训练）
            loss_output = model.compute_loss(
                x_target=input_data,
                channel_indices=None,
                task_label=labels,
                current_epoch=epoch
            )
        
        # 处理损失输出
        if isinstance(loss_output, dict):
            loss = loss_output['total_loss']
            recon_loss = loss_output.get('reconstruction_loss', None)
            cls_loss = loss_output.get('classification_loss', None)
        else:
            loss = loss_output
            recon_loss = None
            cls_loss = None
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        grad_norms.append(grad_norm.item())
        
        optimizer.step()
        
        # 更新统计
        batch_size = target_data.size(0) if reconstruction_mode else input_data.size(0)
        losses.update(loss.item(), batch_size)
        if recon_loss is not None:
            recon_losses.update(recon_loss.item() if torch.is_tensor(recon_loss) else recon_loss, batch_size)
        if cls_loss is not None:
            cls_losses.update(cls_loss.item() if torch.is_tensor(cls_loss) else cls_loss, batch_size)
        
        # 更新进度条
        postfix = {
            'loss': f'{losses.avg:.4f}',
            'grad': f'{grad_norm:.3f}'
        }
        if recon_loss is not None:
            postfix['recon'] = f'{recon_losses.avg:.4f}'
        if cls_loss is not None:
            postfix['cls'] = f'{cls_losses.avg:.4f}'
        pbar.set_postfix(postfix)
    
    # 返回统计信息
    metrics = {
        'total_loss': losses.avg,
        'grad_norm': sum(grad_norms) / len(grad_norms) if grad_norms else 0
    }
    if recon_losses.count > 0:
        metrics['recon_loss'] = recon_losses.avg
    if cls_losses.count > 0:
        metrics['cls_loss'] = cls_losses.avg
        
    return metrics


@torch.no_grad()
def validate(model, val_loader, device, epoch=None, reconstruction_mode=True):
    """验证"""
    model.eval()
    losses = AverageMeter()
    recon_losses = AverageMeter()
    cls_losses = AverageMeter()
    
    for batch_data in val_loader:
        # 准备数据
        if reconstruction_mode:
            # 重建模式：(target_data, channel_indices, labels, subject_ids)
            if len(batch_data) == 4:
                target_data, channel_indices, labels, _ = batch_data
                target_data = target_data.to(device)
                # 使用第一个样本的channel_indices作为整个batch的channel_indices
                channel_indices = channel_indices[0].to(device)  # [num_input_channels]
                labels = labels.to(device)
            else:
                # 兼容旧版本
                target_data, channel_indices, labels = batch_data
                target_data = target_data.to(device)
                # 使用第一个样本的channel_indices作为整个batch的channel_indices
                channel_indices = channel_indices[0].to(device)  # [num_input_channels]
                labels = labels.to(device)
            
            # 前向传播（通道重建）
            loss_output = model.compute_loss(
                x_target=target_data,
                channel_indices=channel_indices,
                task_label=labels,
                current_epoch=epoch
            )
        else:
            # 普通模式：(input_data, labels, subject_ids)
            if len(batch_data) == 3:
                input_data, labels, _ = batch_data
                input_data = input_data.to(device)
                labels = labels.to(device)
            else:
                # 兼容旧版本
                input_data, labels = batch_data
                input_data = input_data.to(device)
                labels = labels.to(device)
            
            # 前向传播（普通训练）
            loss_output = model.compute_loss(
                x_target=input_data,
                channel_indices=None,
                task_label=labels,
                current_epoch=epoch
            )
        
        # 处理损失输出
        if isinstance(loss_output, dict):
            loss = loss_output['total_loss']
            recon_loss = loss_output.get('reconstruction_loss', None)
            cls_loss = loss_output.get('classification_loss', None)
        else:
            loss = loss_output
            recon_loss = None
            cls_loss = None
        
        # 更新统计
        batch_size = target_data.size(0) if reconstruction_mode else input_data.size(0)
        losses.update(loss.item(), batch_size)
        if recon_loss is not None:
            recon_losses.update(recon_loss.item() if torch.is_tensor(recon_loss) else recon_loss, batch_size)
        if cls_loss is not None:
            cls_losses.update(cls_loss.item() if torch.is_tensor(cls_loss) else cls_loss, batch_size)
    
    # 返回统计信息
    metrics = {
        'total_loss': losses.avg,
    }
    if recon_losses.count > 0:
        metrics['recon_loss'] = recon_losses.avg
    if cls_losses.count > 0:
        metrics['cls_loss'] = cls_losses.avg
        
    return metrics


def main(args):
    # 加载配置
    config = load_config(args.config)
    
    # 设置随机种子
    set_seed(config['training']['seed'])
    
    # 设备
    device = get_device() if config['training']['device'] == 'cuda' else torch.device('cpu')
    
    # 创建目录
    create_dirs([
        config['training']['save_dir'],
        config['training']['log_dir']
    ])
    
    # 加载数据
    if config['data']['dataset'] == 'bci2a':
        train_loader, val_loader, test_loader = get_bci2a_dataloaders(
            data_path=config['data']['data_path'],
            subject_ids=config['data']['subject_ids'],
            batch_size=config['training']['batch_size'],
            selected_channels=config['data']['selected_channels'],
            num_workers=config['training']['num_workers'],
            reconstruction_mode=config['data'].get('reconstruction_mode', True),
            normalization_mode=config['data'].get('normalization_mode', 'channel_global'),
            val_split=config['data'].get('val_split', 0.2),
            random_seed=config['training'].get('seed', 42)
        )
    elif config['data']['dataset'] == 'seed':
        # 注意：确保get_seed_dataloaders也返回三个loader，或者根据实际情况调整
        train_loader, val_loader, test_loader = get_seed_dataloaders(
            data_path=config['data']['data_path'],
            subject_ids=config['data']['subject_ids'],
            batch_size=config['training']['batch_size'],
            selected_channels=config['data']['selected_channels'],
            num_workers=config['training']['num_workers']
        )
    else:
        raise ValueError(f"Unknown dataset: {config['data']['dataset']}")
    
    # 将test_loader保存到config中，供后续使用
    config['test_loader'] = test_loader
    
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    
    # 创建模型
    model_type = config['model'].get('model_type', 'x0pred')  # 默认使用x0预测模型
    if model_type == 'enhanced':
        print("使用增强版模型 (DTTDEnhanced) - 拓扑+频率+X0预测")
        model = DTTDEnhanced(config['model']).to(device)
    elif model_type == 'x0pred':
        print("使用X0预测模型 (DTTDX0) - 直接预测原始信号")
        model = DTTDX0(config['model']).to(device)
    elif model_type == 'simple':
        print("使用简化模型 (DTTDSimple) - 预测噪声")
        model = DTTDSimple(config['model']).to(device)
    else:
        print("使用完整模型 (DTTD) - 预测噪声")
        model = DTTD(config['model']).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器
    if config['training']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif config['training']['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
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
        scheduler_step_on_epoch = True
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
        scheduler_step_on_epoch = True
    elif scheduler_type == 'plateau':
        # 智能调度器：只在loss不降时才降低学习率
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True,
            min_lr=1e-6
        )
        scheduler_step_on_epoch = False  # 需要传入loss
    else:
        scheduler = None
        scheduler_step_on_epoch = False
    
    # TensorBoard
    writer = SummaryWriter(config['training']['log_dir'])
    
    # 早停 - 增加patience避免过早停止
    early_stopping = EarlyStopping(patience=50, mode='min')
    
    # 训练循环
    best_loss = float('inf')
    
    # 检查是否为通道重建模式
    reconstruction_mode = config['model'].get('input_channels', 9) != config['model'].get('output_channels', 9)
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        # 训练
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, reconstruction_mode)
        train_loss = train_metrics['total_loss']
        
        # 验证（传递epoch以保持lambda一致）
        val_metrics = validate(model, val_loader, device, epoch, reconstruction_mode)
        val_loss = val_metrics['total_loss']
        
        # 更新学习率
        if scheduler is not None:
            if scheduler_step_on_epoch:
                # 对于 ReduceLROnPlateau 调度器，需要传入验证损失
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    # 其他调度器直接调用 step()
                    scheduler.step()
            else:
                # 如果不是在epoch级别更新，则使用训练损失
                scheduler.step(train_loss)
                
            # 更新当前学习率
            if hasattr(scheduler, 'get_last_lr'):
                # 对于支持get_last_lr()的调度器
                lr_list = scheduler.get_last_lr()
                current_lr = lr_list[0] if isinstance(lr_list, (list, tuple)) else lr_list
            else:
                # 对于不支持的调度器，直接从optimizer获取
                current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        # 记录梯度信息
        if 'grad_norm' in train_metrics:  # 确保grad_norm存在
            writer.add_scalar('Gradient/norm', train_metrics['grad_norm'], epoch)
        
        # ⭐ 记录分解的损失（重建损失和分类损失）
        if 'recon_loss' in train_metrics and train_metrics['recon_loss'] is not None:
            writer.add_scalar('Loss/train_reconstruction', train_metrics['recon_loss'], epoch)
        if 'cls_loss' in train_metrics and train_metrics['cls_loss'] is not None:
            writer.add_scalar('Loss/train_classification', train_metrics['cls_loss'], epoch)
        
        # 打印信息
        log_msg = (f'Epoch {epoch}/{config["training"]["num_epochs"]} - '
                   f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                   f'LR: {current_lr:.6f}, Grad: {train_metrics["grad_norm"]:.3f}')
        if train_metrics['recon_loss'] is not None and train_metrics['cls_loss'] is not None:
            log_msg += f', Recon: {train_metrics["recon_loss"]:.4f}, Cls: {train_metrics["cls_loss"]:.4f}'
        print(log_msg)
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(config['training']['save_dir'], 'best_model.pth')
            )
        
        # 定期保存
        if epoch % config['training']['save_interval'] == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(config['training']['save_dir'], f'checkpoint_epoch_{epoch}.pth')
            )
        
        # 早停检查
        if early_stopping(val_loss, model):
            print(f'早停于 epoch {epoch}')
            # 加载最佳模型
            checkpoint = torch.load(early_stopping.path)
            model.load_state_dict(checkpoint['state_dict'])
            break
    
    # 保存最终模型
    save_checkpoint(
        model, optimizer, epoch, val_loss,
        os.path.join(config['training']['save_dir'], 'final_model.pth')
    )
    
    writer.close()
    print('训练完成!')


if __name__ == '__main__':
    # ========== 在这里设置参数 ==========
    CONFIG_FILE = 'configs/bci2a_enhanced_config.yaml'  # 配置文件路径
    # CONFIG_FILE = 'configs/seed_config.yaml'  # 或使用SEED数据集
    # ====================================
    
    # 创建参数对象
    class Args:
        def __init__(self):
            self.config = CONFIG_FILE
    
    args = Args()
    main(args)

