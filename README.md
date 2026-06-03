# DTTD 实验代码（论文复现导览）

本目录包含 DTTD-DDPM 项目的实验与绘图脚本。下面只保留、标出**论文中真正用到的那一批脚本**，其它调试/历史代码不删，按需参考即可。

## 📁 论文主线脚本一览

### 1. 训练与综合实验入口

| 文件 | 作用 | 典型用法 |
|------|------|----------|
| `train_model.py` | 训练 DTTD 主模型 | `python experiments/train_model.py --config configs/bci2a_enhanced_config.yaml` |
| `run_paper_experiments.py` | 一键跑论文中的分类 + 消融等主要实验 | `python experiments/run_paper_experiments.py` |
| `run_paper_experiments_complete.py` | 更完整版本（含表格汇总） | `python experiments/run_paper_experiments_complete.py` |
| `classification_eval.py` | 三种场景分类评估（被试内 / 跨会话 / 跨被试） | `python experiments/classification_eval.py --mode all` |
| `ablation_study.py` / `ablation_cross_session.py` | 消融实验（对应论文中消融表格/图） | `python experiments/ablation_study.py` |
| `table1_reconstruction.py` | 生成重建质量指标（对应重建结果表） | `python experiments/table1_reconstruction.py` |
| `sensitivity_analysis.py` | 噪声 / guidance 等超参敏感度分析 | `python experiments/sensitivity_analysis.py --mode cross_session` |

> 一般复现实验建议优先用：`train_model.py` + `run_paper_experiments.py`。

### 2. 论文图表相关脚本

| 文件 | 对应内容 | 典型用法 |
|------|----------|----------|
| `generate_per_subject_barplot.py` | 三种场景下各被试分类柱状图（within / cross-session / cross-subject） | `python experiments/generate_per_subject_barplot.py` |
| `generate_waveform_comparison.py` | **波形对比图**（论文中的波形图） | `python experiments/generate_waveform_comparison.py` |
| `generate_psd_topomap.py` | PSD + 拓扑图（功率谱与头皮图） | `python experiments/generate_psd_topomap.py` |
| `generate_tsne_comparison.py` | t-SNE 分布比较图 | `python experiments/generate_tsne_comparison.py` |
| `generate_sensitivity_plot.py` | 敏感度曲线图 | `python experiments/generate_sensitivity_plot.py` |
| `generate_architecture_diagram.py` | 模型结构示意图 | `python experiments/generate_architecture_diagram.py` |

这些脚本输出的图片默认保存在 `paper_results/figures/` 下，可直接插入 LaTeX 论文。

### 3. 数据与真实场景实验

| 文件 | 作用 |
|------|------|
| `generate_data.py` | 用训练好的 DTTD 生成 22 通道重建数据 |
| `run_complete_real_experiments.py` | 真实场景/真实数据实验批量运行 |
| `real_eval_utils.py` | 真实实验评估的公用工具 |
| `unified_reconstruction_eval.py` | 统一的重建评估入口（快速算相关系数等） |

## 🚀 快速开始（论文主线）

### 1. 训练模型

```bash
# 使用默认配置训练
python experiments/train_model.py --epochs 200 --batch-size 32

# 指定配置文件
python experiments/train_model.py --config configs/bci2a_enhanced_config.yaml --output-dir checkpoints/dttd
```

### 2. 生成数据

```bash
# 为所有被试生成22通道数据
python experiments/generate_data.py --checkpoint checkpoints/dttd/best_model.pth --output generated_data/

# 使用DDIM采样 (更高质量)
python experiments/generate_data.py --use-ddim --num-steps 50 --guidance-scale 3.0
```

### 3. 分类评估（三种场景）

```bash
# 运行所有三种评估场景
python experiments/classification_eval.py --mode all

# 仅运行跨会话评估
python experiments/classification_eval.py --mode cross_session

# 仅运行跨被试评估 (LOSO)
python experiments/classification_eval.py --mode cross_subject
```

### 4. 消融实验

```bash
# 运行消融实验
python experiments/ablation_study.py --checkpoint checkpoints/dttd/best_model.pth
```

### 5. 敏感度分析

```bash
# 跨会话敏感度分析
python experiments/sensitivity_analysis.py --mode cross_session

# 分析结果保存到 paper_results/sensitivity/
```

### 6. 生成可视化与波形图

```bash
# 生成所有可视化图表
python experiments/visualization.py --type all

# 仅生成波形对比图
python experiments/visualization.py --type waveform

# 仅生成条形图
python experiments/visualization.py --type barplot
```

## 📊 评估场景说明（与论文对应）

### 被试内 (Within-Subject)
- 合并两个会话数据
- 5折交叉验证
- 评估同一被试内的分类性能

### 跨会话 (Cross-Session)
- Session T 训练 → Session E 测试
- 评估跨时间的泛化能力

### 跨被试 (Cross-Subject / LOSO)
- Leave-One-Subject-Out
- 8个被试训练 → 1个被试测试
- 评估跨个体的泛化能力

## 📈 输出目录

```
paper_results/
├── classification/          # 分类评估结果
│   └── classification_results.json
├── ablation/               # 消融实验结果
│   └── ablation_results.json
├── sensitivity/            # 敏感度分析结果
│   ├── sensitivity_results.json
│   └── sensitivity_analysis.png
└── figures/                # 可视化图表
    ├── waveform_comparison.png
    ├── tsne_comparison.png
    ├── spectrum_comparison.png
    └── *_per_subject.png
```

## ⚙️ 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data_scale_factor` | 1e5 | EEG数据缩放因子 |
| `guidance_scale` | 3.0 | 分类器引导强度 |
| `noise_scale` | 0.02 | 输入噪声比例 |
| `num_augment` | 1 | 数据增强倍数 |

## 📝 注意事项

## 🗂️ 其它/调试脚本说明

以下脚本主要用于早期实验或细节调试，对论文主线不必使用：

- **调试相关**：`debug_baselines.py`, `debug_detailed.py`, `debug_metrics.py`, `debug_spline.py`, `debug_topo_sim.py`, `quick_reconstruction_metrics.py`
- **旧版 / 备选实现**：`ablation_v2.py`, `data_augmentation_eval.py`, `baseline_classification_test.py`, `test_noise_level_ablation.py`, `train_ablation.py`, `train_diffusion_v2.py`, `train_dttd_no_preprocess.py`, `train_model_v2.py`, `train_strong_classifier.py`, `seed_transfer.py`, `sensitivity_cross_session.py`, `sensitivity_cross_subject.py`, `evaluate_comprehensive.py`, `run_all_paper_experiments.py`

它们目前**不会影响论文复现**，如需清理代码，可以将这一批脚本整体移动到单独的 `legacy/` 目录中备份。***

1. **数据缩放**: EEG数据通常是微伏级别(~1e-5)，训练时需要放大到合理范围
2. **无预处理**: BCI2a数据已预处理，不需要额外滤波
3. **模型保存**: checkpoint中包含`data_scale_factor`，加载时会自动使用
