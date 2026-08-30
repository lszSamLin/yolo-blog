"""
generate_charts.py — 使用示例
================================

# 1. 基本用法：从 results.csv 生成所有图表

python generate_charts.py --csv results.csv

# 2. 指定输出目录和分辨率

python generate_charts.py --csv results.csv --outdir ./charts --dpi 200

# 3. 生成高质量打印级图表

python generate_charts.py --csv results.csv --outdir ./charts_hi --dpi 300

# 生成的文件列表：

#   results.png              — 6合1训练曲线总览（loss + mAP + LR）

#   BoxPR_curve.png          — Precision-Recall 曲线

#   BoxP_curve.png           — Precision 随置信度阈值变化

#   BoxR_curve.png           — Recall 随置信度阈值变化

#   BoxF1_curve.png          — F1 随置信度阈值变化（标注最优阈值）

#   metrics_over_epochs.png  — P/R/mAP50/mAP50-95 随 epoch 变化

#   confusion_matrix.png     — 2×2 混淆矩阵热力图

"""

# ── 依赖安装 ──

# pip install matplotlib pandas numpy

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

# 如果只想生成单个图，直接 import 对应函数

# from generate_charts import plot_results, plot_box_pr_curve, ...
