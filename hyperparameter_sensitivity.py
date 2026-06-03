"""
DTTD Enhanced 超参数敏感度分析

针对模型的核心创新超参数进行敏感度分析：
1. lambda_cls - 分类器引导损失权重
2. noise_scale - 训练时输入噪声比例
3. guidance_scale - 采样时分类器引导强度
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.dttd_enhanced_v2 import DTTDEnhanced
from data import get_bci2a_dataloaders
from utils import load_config, set_seed

# 设置matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def compute_metrics(pred, target):
    """计算重建指标"""
    mse = F.mse_loss(pred, target).item()
    mae = F.l1_loss(pred, target).item()
    
    pred_np = pred.detach().cpu().numpy().flatten()
    target_np = target.detach().cpu().numpy().flatten()
    
    if np.std(pred_np) < 1e-8 or np.std(target_np) < 1e-8:
        corr = 0.0
    else:
        corr = np.corrcoef(pred_np, target_np)[0, 1]
        if np.isnan(corr):
            corr = 0.0
    
    return {'mse': mse, 'mae': mae, 'correlation': corr}


def compute_psd_correlation(pred, target, fs=250):
    """计算功率谱密度相关性"""
    from scipy import signal
    
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    correlations = []
    for i in range(min(pred_np.shape[0], 10)):  # 限制计算量
        for j in range(pred_np.shape[1]):
            f_pred, psd_pred = signal.welch(pred_np[i, j], fs=fs, nperseg=min(256, pred_np.shape[2]))
            f_target, psd_target = signal.welch(target_np[i, j], fs=fs, nperseg=min(256, target_np.shape[2]))
            
            if np.std(psd_pred) > 1e-8 and np.std(psd_target) > 1e-8:
                corr = np.corrcoef(psd_pred, psd_target)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
    
    return np.mean(correlations) if correlations else 0.0


def load_model(checkpoint_path, config, device):
    """加载模型"""
    model = DTTDEnhanced(config['model']).to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"模型加载成功")
    
    model.eval()
    return model


def guidance_scale_sensitivity(model, test_loader, device, num_samples=300):
    """
    分类器引导强度 (guidance_scale) 敏感度分析
    测试不同guidance_scale对重建质量的影响
    """
    print("\n" + "="*70)
    print("Guidance Scale 敏感度分析")
    print("="*70)
    
    guidance_scales = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    results = {}
    
    # 输入通道索引
    input_indices = [7, 9, 11, 1, 3, 5, 13, 15, 17]
    
    for gs in guidance_scales:
        print(f"\nGuidance Scale = {gs}")
        
        all_mse = []
        all_corr = []
        all_psd_corr = []
        sample_count = 0
        
        model.eval()
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc=f"GS={gs}"):
                if sample_count >= num_samples:
                    break
                
                if len(batch_data) == 4:
                    target_data, _, labels, _ = batch_data
                else:
                    target_data, _, labels = batch_data
                
                target_data = target_data.to(device)
                labels = labels.to(device)
                
                # 提取输入通道
                input_data = target_data[:, input_indices, :]
                
                try:
                    # 使用指定的guidance_scale进行采样
                    output = model.sample(input_data, task_label=labels, guidance_scale=gs)
                    
                    metrics = compute_metrics(output, target_data)
                    all_mse.append(metrics['mse'])
                    all_corr.append(metrics['correlation'])
                    
                    psd_corr = compute_psd_correlation(output, target_data)
                    all_psd_corr.append(psd_corr)
                except Exception as e:
                    continue
                
                sample_count += target_data.size(0)
        
        results[str(gs)] = {
            'mse': np.mean(all_mse) if all_mse else float('inf'),
            'correlation': np.mean(all_corr) if all_corr else 0.0,
            'psd_correlation': np.mean(all_psd_corr) if all_psd_corr else 0.0
        }
        
        print(f"  MSE: {results[str(gs)]['mse']:.6f}, "
              f"Corr: {results[str(gs)]['correlation']:.4f}, "
              f"PSD Corr: {results[str(gs)]['psd_correlation']:.4f}")
    
    return results


def noise_level_sensitivity(model, test_loader, device, num_samples=300):
    """
    输入噪声水平敏感度分析
    测试不同输入噪声水平对重建质量的影响
    """
    print("\n" + "="*70)
    print("输入噪声水平敏感度分析")
    print("="*70)
    
    noise_levels = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    results = {}
    
    input_indices = [7, 9, 11, 1, 3, 5, 13, 15, 17]
    
    for noise_level in noise_levels:
        print(f"\n噪声水平 = {noise_level*100:.0f}%")
        
        all_mse = []
        all_corr = []
        all_psd_corr = []
        sample_count = 0
        
        model.eval()
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc=f"Noise={noise_level}"):
                if sample_count >= num_samples:
                    break
                
                if len(batch_data) == 4:
                    target_data, _, labels, _ = batch_data
                else:
                    target_data, _, labels = batch_data
                
                target_data = target_data.to(device)
                labels = labels.to(device)
                
                # 提取输入通道并添加噪声
                input_data = target_data[:, input_indices, :]
                if noise_level > 0:
                    noise = torch.randn_like(input_data) * noise_level
                    input_data = input_data + noise
                
                try:
                    output = model.sample(input_data, task_label=labels)
                    
                    metrics = compute_metrics(output, target_data)
                    all_mse.append(metrics['mse'])
                    all_corr.append(metrics['correlation'])
                    
                    psd_corr = compute_psd_correlation(output, target_data)
                    all_psd_corr.append(psd_corr)
                except:
                    continue
                
                sample_count += target_data.size(0)
        
        results[str(noise_level)] = {
            'mse': np.mean(all_mse) if all_mse else float('inf'),
            'correlation': np.mean(all_corr) if all_corr else 0.0,
            'psd_correlation': np.mean(all_psd_corr) if all_psd_corr else 0.0
        }
        
        print(f"  MSE: {results[str(noise_level)]['mse']:.6f}, "
              f"Corr: {results[str(noise_level)]['correlation']:.4f}, "
              f"PSD Corr: {results[str(noise_level)]['psd_correlation']:.4f}")
    
    return results


def num_steps_sensitivity(model, test_loader, device, num_samples=300):
    """
    采样步数敏感度分析
    测试不同采样步数对重建质量和速度的影响
    """
    print("\n" + "="*70)
    print("采样步数敏感度分析")
    print("="*70)
    
    num_steps_list = [1, 5, 10, 20, 50]
    results = {}
    
    input_indices = [7, 9, 11, 1, 3, 5, 13, 15, 17]
    
    for num_steps in num_steps_list:
        print(f"\n采样步数 = {num_steps}")
        
        all_mse = []
        all_corr = []
        sample_count = 0
        
        import time
        start_time = time.time()
        
        model.eval()
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc=f"Steps={num_steps}"):
                if sample_count >= num_samples:
                    break
                
                if len(batch_data) == 4:
                    target_data, _, labels, _ = batch_data
                else:
                    target_data, _, labels = batch_data
                
                target_data = target_data.to(device)
                labels = labels.to(device)
                
                input_data = target_data[:, input_indices, :]
                
                try:
                    output = model.sample(input_data, task_label=labels, num_steps=num_steps)
                    
                    metrics = compute_metrics(output, target_data)
                    all_mse.append(metrics['mse'])
                    all_corr.append(metrics['correlation'])
                except:
                    continue
                
                sample_count += target_data.size(0)
        
        elapsed_time = time.time() - start_time
        
        results[str(num_steps)] = {
            'mse': np.mean(all_mse) if all_mse else float('inf'),
            'correlation': np.mean(all_corr) if all_corr else 0.0,
            'time_per_sample': elapsed_time / sample_count if sample_count > 0 else 0
        }
        
        print(f"  MSE: {results[str(num_steps)]['mse']:.6f}, "
              f"Corr: {results[str(num_steps)]['correlation']:.4f}, "
              f"Time: {results[str(num_steps)]['time_per_sample']*1000:.2f}ms/sample")
    
    return results


def plot_sensitivity_results(results, output_dir):
    """绘制敏感度分析结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Guidance Scale - MSE
    if 'guidance_scale' in results:
        gs_data = results['guidance_scale']
        gs_values = [float(k) for k in gs_data.keys()]
        mse_values = [gs_data[k]['mse'] for k in gs_data.keys()]
        corr_values = [gs_data[k]['correlation'] for k in gs_data.keys()]
        
        axes[0, 0].plot(gs_values, mse_values, 'o-', color='blue', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Guidance Scale')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].set_title('Guidance Scale vs MSE')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(gs_values, corr_values, 'o-', color='green', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Guidance Scale')
        axes[0, 1].set_ylabel('Correlation')
        axes[0, 1].set_title('Guidance Scale vs Correlation')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 2. Noise Level
    if 'noise_level' in results:
        noise_data = results['noise_level']
        noise_values = [float(k) for k in noise_data.keys()]
        mse_values = [noise_data[k]['mse'] for k in noise_data.keys()]
        corr_values = [noise_data[k]['correlation'] for k in noise_data.keys()]
        
        axes[0, 2].plot(noise_values, mse_values, 'o-', color='red', linewidth=2, markersize=8)
        axes[0, 2].set_xlabel('Noise Level')
        axes[0, 2].set_ylabel('MSE')
        axes[0, 2].set_title('Noise Level vs MSE')
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 0].plot(noise_values, corr_values, 'o-', color='purple', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Noise Level')
        axes[1, 0].set_ylabel('Correlation')
        axes[1, 0].set_title('Noise Level vs Correlation')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 3. Num Steps
    if 'num_steps' in results:
        steps_data = results['num_steps']
        steps_values = [int(k) for k in steps_data.keys()]
        mse_values = [steps_data[k]['mse'] for k in steps_data.keys()]
        time_values = [steps_data[k]['time_per_sample']*1000 for k in steps_data.keys()]
        
        axes[1, 1].plot(steps_values, mse_values, 'o-', color='orange', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('Num Steps')
        axes[1, 1].set_ylabel('MSE')
        axes[1, 1].set_title('Sampling Steps vs MSE')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].plot(steps_values, time_values, 'o-', color='brown', linewidth=2, markersize=8)
        axes[1, 2].set_xlabel('Num Steps')
        axes[1, 2].set_ylabel('Time (ms/sample)')
        axes[1, 2].set_title('Sampling Steps vs Time')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hyperparameter_sensitivity.png'), dpi=150)
    plt.close()
    print(f"\n敏感度分析图已保存到: {output_dir}/hyperparameter_sensitivity.png")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='DTTD Enhanced 超参数敏感度分析')
    parser.add_argument('--config', type=str, default='configs/bci2a_enhanced_config.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/bci2a_enhanced/best_model.pth')
    parser.add_argument('--output-dir', type=str, default='results/sensitivity')
    parser.add_argument('--num-samples', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载配置
    config = load_config(args.config)
    
    # 加载测试数据
    _, _, test_loader = get_bci2a_dataloaders(
        data_path=config['data']['data_path'],
        subject_ids=config['data']['subject_ids'],
        batch_size=32,
        selected_channels=config['data'].get('selected_channels'),
        num_workers=0,
        reconstruction_mode=True
    )
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 加载模型
    print("加载模型...")
    model = load_model(args.checkpoint, config, device)
    
    all_results = {}
    
    # 1. Guidance Scale 敏感度分析
    gs_results = guidance_scale_sensitivity(model, test_loader, device, args.num_samples)
    all_results['guidance_scale'] = gs_results
    
    # 2. 噪声水平敏感度分析
    noise_results = noise_level_sensitivity(model, test_loader, device, args.num_samples)
    all_results['noise_level'] = noise_results
    
    # 3. 采样步数敏感度分析
    steps_results = num_steps_sensitivity(model, test_loader, device, args.num_samples)
    all_results['num_steps'] = steps_results
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'hyperparameter_sensitivity.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_path}")
    
    # 绘制图表
    plot_sensitivity_results(all_results, args.output_dir)
    
    # 打印汇总
    print("\n" + "="*80)
    print("超参数敏感度分析结果汇总")
    print("="*80)
    
    print("\n1. Guidance Scale 敏感度:")
    print(f"{'GS':<10} {'MSE':<12} {'Corr':<12} {'PSD Corr':<12}")
    print("-"*50)
    for gs, metrics in all_results['guidance_scale'].items():
        print(f"{gs:<10} {metrics['mse']:<12.6f} {metrics['correlation']:<12.4f} {metrics['psd_correlation']:<12.4f}")
    
    print("\n2. 噪声水平敏感度:")
    print(f"{'Noise':<10} {'MSE':<12} {'Corr':<12} {'PSD Corr':<12}")
    print("-"*50)
    for noise, metrics in all_results['noise_level'].items():
        print(f"{float(noise)*100:.0f}%{'':<7} {metrics['mse']:<12.6f} {metrics['correlation']:<12.4f} {metrics['psd_correlation']:<12.4f}")
    
    print("\n3. 采样步数敏感度:")
    print(f"{'Steps':<10} {'MSE':<12} {'Corr':<12} {'Time(ms)':<12}")
    print("-"*50)
    for steps, metrics in all_results['num_steps'].items():
        print(f"{steps:<10} {metrics['mse']:<12.6f} {metrics['correlation']:<12.4f} {metrics['time_per_sample']*1000:<12.2f}")
    
    # 找出最优参数
    print("\n" + "="*80)
    print("最优参数推荐")
    print("="*80)
    
    # 最优Guidance Scale（最低MSE）
    best_gs = min(all_results['guidance_scale'].items(), key=lambda x: x[1]['mse'])
    print(f"最优 Guidance Scale: {best_gs[0]} (MSE={best_gs[1]['mse']:.6f})")
    
    # 最优噪声水平
    best_noise = min(all_results['noise_level'].items(), key=lambda x: x[1]['mse'])
    print(f"最优噪声水平: {float(best_noise[0])*100:.0f}% (MSE={best_noise[1]['mse']:.6f})")
    
    # 最优采样步数（平衡质量和速度）
    best_steps = min(all_results['num_steps'].items(), key=lambda x: x[1]['mse'])
    print(f"最优采样步数: {best_steps[0]} (MSE={best_steps[1]['mse']:.6f})")


if __name__ == '__main__':
    main()
