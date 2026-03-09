# Quick Start

## 环境安装

```bash
pip install -r requirements.txt
```

如需可视化支持，还需安装 matplotlib：

```bash
pip install matplotlib
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | str | `nsl` | 数据集名称：`nsl` / `unsw` / `cic` |
| `--epochs` | int | `4` | Stage 1 离线训练轮数 |
| `--epoch_1` | int | `1` | Stage 2 每个在线步骤的训练轮数 |
| `--percent` | float | `0.8` | 在线测试集比例（用于模拟流式到达） |
| `--flip_percent` | float | `0.2` | 伪标签翻转比例（模拟标签噪声） |
| `--sample_interval` | int | `2000` | 每步在线学习的样本数 |
| `--cuda` | str | `0` | 使用的 GPU 编号，无 GPU 时自动使用 CPU |

## 训练指令

### 1. NSL-KDD 数据集

```bash
python online_training.py --dataset nsl --epochs 800 --epoch_1 1 --flip_percent 0.05 --sample_interval 2000
```

### 2. UNSW-NB15 数据集

```bash
python online_training.py --dataset unsw --epochs 800 --epoch_1 1 --flip_percent 0.05 --sample_interval 2784
```

### 3. CIC-IDS-2017 数据集

```bash
python online_training.py --dataset cic --epochs 800 --epoch_1 1 --flip_percent 0.05 --sample_interval 3000
```

### 4. 指定 GPU 训练

```bash
python online_training.py --dataset nsl --epochs 800 --epoch_1 1 --flip_percent 0.05 --sample_interval 2000 --cuda 1
```

### 5. 快速调试（少量 epoch）

```bash
python online_training.py --dataset nsl --epochs 4 --epoch_1 1 --flip_percent 0.2 --sample_interval 2000
```

## 输出结果

训练完成后，所有结果保存在 `result/` 目录下，每次运行生成一个子文件夹：

```
result/{dataset}_seed{seed}_{timestamp}/
├── metrics.json      # 训练配置 + 各阶段损失 + 在线指标 + 最终评估结果
├── model.pth         # 模型权重 & 优化器状态
├── predictions.npz   # 测试集真实标签(y_true) & 预测标签(y_pred)
└── *_summary.png     # 可视化训练报告（损失曲线、指标变化、混淆矩阵等）
```

`metrics.json` 示例结构：

```json
{
  "config": { "dataset": "nsl", "seed": 5009, "epochs": 800, ... },
  "stage1_losses": [0.0123, 0.0098, ...],
  "stage2_losses": [0.0087, 0.0076, ...],
  "online_metrics": {
    "1": { "accuracy": 0.92, "precision": 0.89, "recall": 0.95, "f1": 0.92 },
    ...
  },
  "final_results": {
    "encoder":  { "accuracy": 0.94, "precision": 0.91, "recall": 0.96, "f1": 0.93 },
    "decoder":  { "accuracy": 0.93, "precision": 0.90, "recall": 0.95, "f1": 0.92 },
    "combined": { "accuracy": 0.95, "precision": 0.92, "recall": 0.97, "f1": 0.94 }
  }
}
```
