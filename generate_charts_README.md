# generate_charts.py 使用示例

`generate_charts.py` 用于把训练产物（`results.csv`、`results.png` 等）转换成各类评估图表。下面是完整用法说明与依赖安装。

## 1. 基本用法：从 results.csv 生成所有图表

```bash
python generate_charts.py --csv results.csv
```

## 2. 指定输出目录和分辨率

```bash
python generate_charts.py --csv results.csv --outdir ./charts --dpi 200
```

## 3. 生成高质量打印级图表

```bash
python generate_charts.py --csv results.csv --outdir ./charts_hi --dpi 300
```

## 生成的文件列表

| 文件名 | 说明 |
| --- | --- |
| `results.png` | 6 合 1 训练曲线总览（loss + mAP + LR） |
| `BoxPR_curve.png` | Precision-Recall 曲线 |
| `BoxP_curve.png` | Precision 随置信度阈值变化 |
| `BoxR_curve.png` | Recall 随置信度阈值变化 |
| `BoxF1_curve.png` | F1 随置信度阈值变化（标注最优阈值） |
| `metrics_over_epochs.png` | P/R/mAP50/mAP50-95 随 epoch 变化 |
| `confusion_matrix.png` | 2×2 混淆矩阵热力图 |

## 依赖安装

```bash
pip install matplotlib pandas numpy
```

## 仅生成单个图表（import 使用）

```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

# 如果只想生成单个图，直接 import 对应函数
# from generate_charts import plot_results, plot_box_pr_curve, ...
```
