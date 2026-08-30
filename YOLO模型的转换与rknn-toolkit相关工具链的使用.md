# YOLO模型的转换与RKNN-Toolkit相关工具链的使用

## 引言

将训练好的 YOLO 模型部署到边缘设备（如 Rockchip NPU）需要经历模型转换流程。RKNN-Toolkit2 是瑞芯微官方提供的模型转换和部署工具链，支持将多种框架模型转换为 RKNN 格式，以便在 Rockchip SoC 上高效运行。本文将详细介绍 YOLO 模型转换的完整流程、常见问题排查以及性能优化技巧。

> **参考来源**：[RKNN-Toolkit2 GitHub](https://github.com/airockchip/rknn-toolkit2) | [RKNN-Toolkit2 Documentation](https://github.com/airockchip/rknn-toolkit2/blob/master/docs/en/02_rknn_user_guide/01_toolkit_user_guide.md)

---

## 一、RKNN-Toolkit2 概述

### 1.1 支持的平台

| 平台 | SoC 型号 | NPU TOPS | 内存 | 适用场景 |
|------|---------|----------|------|---------|
| **RK3588** | 旗舰级 | 6 TOPS | 支持 LPDDR4/5 | 高性能边缘计算、工业机器人 |
| **RK3576** | 高端级 | 2-4 TOPS | 支持 LPDDR4 | 智能安防、自动驾驶辅助 |
| **RK3568** | 中端级 | 0.8 TOPS | 支持 LPDDR4 | 智能摄像头、POS机 |
| **RK3566** | 中端级 | 0.8 TOPS | 支持 LPDDR4 | 消费类电子产品 |
| **RK3562** | 入门级 | 0.8 TOPS | 支持 LPDDR3 | 低成本 IoT 设备 |

### 1.2 支持的模型格式

```
支持导入的框架格式:
├── PyTorch (.pt)          ← 直接导入（部分算子限制）
├── ONNX (.onnx)           ← 推荐，最通用
├── Tensorflow SavedModel  ← 完整支持
├── Tensorflow GraphDef (.pb)
├── Tensorflow Lite (.tflite)
├── Caffe (.caffemodel)
└── Darknet (.weights)     ← YOLO 原生格式

支持导出的格式:
└── RKNN (.rknn)           ← 目标平台原生格式

```

### 1.3 工具链组成

```
RKNN-Toolkit2 包含:
├── rknn-toolkit2-python   # Python SDK（模型转换）
├── rknn-server            # 推理服务（可选）
├── rknn-track             # 目标跟踪服务（可选）
└── examples               # 示例代码

```

---

## 二、环境配置

### 2.1 PC 端安装（模型转换）

```bash
# 方式1: pip 安装（推荐）
pip install rknn-toolkit2==2.1.0

# 方式2: 从源码安装
git clone https://github.com/airockchip/rknn-toolkit2.git
cd rknn-toolkit2
pip install -e .

# 验证安装
python -c "from rknn.api import RKNN; print('RKNN imported successfully')"

```

**依赖要求**：

```bash
pip install numpy onnx protobuf pillow
# ONNX 模型导出时需要
pip install onnx onnxsim

```

### 2.2 开发板端安装（推理部署）

```bash
# RK3588 开发板
# 方法1: 从SDK包安装
curl -fsSL https://sdk.rock-chips.com/rknn-toolkit2/rknn-toolkit2-linux-aarch64.run | bash

# 方法2: 使用 apt
apt-get install rknn-toolkit2

# 设置环境变量
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
export PATH=/usr/local/bin:$PATH

# 验证 NPU 驱动
cat /proc/driver/rockchip/npu/info
lsmod | grep npu

```

### 2.3 开发板环境检查

```python
import subprocess

def check_npu_status():
    """检查 NPU 状态"""
    # 检查 NPU 驱动
    result = subprocess.run(
        ["cat", "/proc/driver/rockchip/npu/info"],
        capture_output=True, text=True
    )
    print("NPU 信息:")
    print(result.stdout)

    # 检查 NPU 驱动加载
    result = subprocess.run(
        ["lsmod"], capture_output=True, text=True
    )
    if "rockchip_npu" in result.stdout:
        print("NPU 驱动已加载 ✓")
    else:
        print("NPU 驱动未加载 ✗")

    # 检查 NPU 频率
    result = subprocess.run(
        ["cat", "/sys/class/devfreq/*npu*/cur_freq"],
        capture_output=True, text=True
    )
    print(f"NPU 当前频率: {result.stdout.strip()} Hz")

check_npu_status()

```

---

## 三、YOLO 模型导出为 ONNX

### 3.1 使用 Ultralytics 导出

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO("best.pt")

# 导出为 ONNX
model.export(
    format="onnx",
    imgsz=640,            # 输入尺寸
    batch=1,              # 批次大小
    dynamic=False,        # 固定尺寸导出
    simplify=True,        # 使用 onnx-simplifier 简化
    opset=11,             # ONNX opset 版本
    device=0,             # 导出时使用的设备
    half=False,           # 不导出 FP16（在 RKNN 中处理）
)

# 导出的文件: best.onnx

```

### 3.2 ONNX 导出参数详解

```python
model.export(
    format="onnx",
    imgsz=[640, 640],     # [height, width]，必须一致
    batch=1,              # 批次大小（固定导出时设为1）
    dynamic=False,        # False=固定尺寸, True=动态尺寸
    simplify=True,        # 使用 onnx-simplifier 优化图结构
    opset=11,             # ONNX opset 版本（11-17 支持）
    int8=False,           # 不导出 INT8（在 RKNN 中处理）
    half=False,           # 不导出 FP16
    reduce=False,         # 不减少精度
    stride=32,            # 模型 stride（YOLOv8 默认 32）
    inplace=True,         # 原地操作
    keras=False,          # 不使用 Keras
    optimize=False,       # 不使用 ONNX optimizer
    il2r=False,           # 不使用 I2R 转换
    imgnet=False,         # 不使用 ImageNet 预处理
)

```

### 3.3 ONNX 模型验证

```python
import onnx
import onnxruntime as ort
import numpy as np

# 加载 ONNX 模型
onnx_model = onnx.load("best.onnx")
onnx.checker.check_model(onnx_model)  # 检查模型合法性
print("ONNX 模型验证通过 ✓")

# 打印模型信息
print(f"输入: {onnx_model.graph.input[0].name} - {onnx_model.graph.input[0].type.tensor_type.elem_type}")
print(f"输出: {onnx_model.graph.output[0].name}")
print(f"节点数: {len(onnx_model.graph.node)}")

# 使用 ONNX Runtime 验证推理
session = ort.InferenceSession("best.onnx")
input_name = session.get_inputs()[0].name
print(f"输入名称: {input_name}, 形状: {session.get_inputs()[0].shape}")

# 模拟推理
dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
outputs = session.run(None, {input_name: dummy_input})
print(f"推理输出数量: {len(outputs)}")
for i, out in enumerate(outputs):
    print(f"  输出 {i}: 形状={out.shape}, dtype={out.dtype}")

# 对比 Ultralytics 推理结果
from ultralytics import YOLO
ultralytics_model = YOLO("best.pt")
ultralytics_result = ultralytics_model.predict("test_image.jpg", verbose=False)
print(f"Ultralytics 推理结果数量: {len(ultralytics_result[0].boxes)}")

```

---

## 四、ONNX 转换为 RKNN

### 4.1 基本转换流程

```python
from rknn.api import RKNN

# 创建 RKNN 实例
rknn = RKNN()

# ═══════════════════════════════════════════════════════════
# Step 1: 配置转换参数（Toolkit2 中 config 必须在加载模型之前调用）
# ═══════════════════════════════════════════════════════════
print("--> Configuring model")
ret = rknn.config(
    mean_values=[[0, 0, 0]],       # 归一化均值（与训练/推理预处理一致）
    std_values=[[255, 255, 255]],  # 归一化方差，即 /255 归一化到 [0, 1]
    target_platform=['rk3588'],    # 目标平台
)
if ret != 0:
    print(f"Config failed: {ret}")
    exit(ret)
print("Done")

# ═══════════════════════════════════════════════════════════
# Step 2: 加载 ONNX 模型
# ═══════════════════════════════════════════════════════════
print("--> Loading model")
ret = rknn.load_onnx(model="best.onnx")
if ret != 0:
    print(f"Load ONNX failed: {ret}")
    exit(ret)
print("Done")

# ═══════════════════════════════════════════════════════════
# Step 3: 构建模型
# ═══════════════════════════════════════════════════════════
print("--> Building model")
ret = rknn.build(do_quantization=False)
if ret != 0:
    print(f"Build failed: {ret}")
    exit(ret)
print("Done")

# ═══════════════════════════════════════════════════════════
# Step 4: 导出 RKNN 模型
# ═══════════════════════════════════════════════════════════
print("--> Exporting RKNN model")
ret = rknn.export_rknn("best.rknn")
if ret != 0:
    print(f"Export failed: {ret}")
    exit(ret)
print("Done")

# ═══════════════════════════════════════════════════════════
# Step 5: 释放资源
# ═══════════════════════════════════════════════════════════
rknn.release()
print("All done!")

```

### 4.2 关键参数详解

> **⚠️ Toolkit1 / Toolkit2 参数对照**：下面 4.2.1～4.2.3 介绍的 `channel_mean_value`、`reorder_channel`、`targets` 是 **RKNN-Toolkit1（已停止维护）** 的参数名。在 **RKNN-Toolkit2** 中应使用对应的新写法：
>
> | Toolkit1（旧） | Toolkit2（现行） | 说明 |
> |--------------|----------------|------|
> | `channel_mean_value='0 0 0 255'` | `mean_values=[[0, 0, 0]]` + `std_values=[[255, 255, 255]]` | 均值与方差分开传入，与 PyTorch 预处理一一对应 |
> | `reorder_channel='0 1 2'` | （已移除） | v2 不在转换期做通道重排，BGR→RGB 请在预处理代码中完成 |
> | `targets=['rk3588']` | `target_platform='rk3588'` | 支持字符串或列表 |
> | `batch_size=1` | （已移除） | v2 在推理阶段决定 batch |
>
> 新项目一律使用 Toolkit2 写法；旧写法仅供维护历史项目时参考。

#### 4.2.1 channel_mean_value（Toolkit1 参数）

```python
# 格式: 'mean_R mean_G mean_B std_R std_G std_B'

# 常见配置:
rknn.config(channel_mean_value='0 0 0 255')
# 含义: (pixel - 0) / 255 = pixel / 255
# 即归一化到 [0, 1]

rknn.config(channel_mean_value='123.675 116.28 103.53 58.395 57.12 57.375')
# ImageNet 归一化: (pixel - mean) / std

# 不需要归一化:
rknn.config(channel_mean_value='0 0 0 1 1 1')

```

#### 4.2.2 targets

```python
# 目标平台
rknn.config(targets=['rk3588'])   # RK3588
rknn.config(targets=['rk3576'])   # RK3576
rknn.config(targets=['rk3568'])   # RK3568
rknn.config(targets=['rk3566'])   # RK3566
rknn.config(targets=['rk3562'])   # RK3562

```

#### 4.2.3 reorder_channel（Toolkit1 参数，Toolkit2 已移除）

```python
# 通道重排
rknn.config(reorder_channel='0 1 2')  # BGR → RGB (不需要)
rknn.config(reorder_channel='2 1 0')  # RGB → BGR (不需要，因为已归一化)

```

### 4.3 动态量化（INT8）

```python
from rknn.api import RKNN
import glob

rknn = RKNN()

# 配置（Toolkit2 中 config 必须在加载模型之前调用）
print("--> Configuring model")
rknn.config(
    mean_values=[[0, 0, 0]],       # 归一化均值
    std_values=[[255, 255, 255]],  # 归一化方差（/255）
    target_platform='rk3588',
)

# 加载模型
print("--> Loading model")
rknn.load_onnx(model="best.onnx")

# 生成校准数据集文件（txt，每行一张图片路径，建议约 100 张有代表性的图片）
images = sorted(glob.glob("calibration_images/*.jpg"))
with open("calibration_dataset.txt", "w") as f:
    f.write("\n".join(images[:100]))

# INT8 量化构建
print("--> Building model (INT8 Quantization)")
ret = rknn.build(do_quantization=True,
                 dataset="calibration_dataset.txt",
                 quantized_algorithm="normal")   # normal / mmse / kl_divergence
if ret != 0:
    print(f"Build failed: {ret}")
    exit(ret)

# 导出
print("--> Exporting RKNN model")
ret = rknn.export_rknn("best_int8.rknn")
if ret != 0:
    print(f"Export failed: {ret}")
    exit(ret)

rknn.release()
print("INT8 模型导出成功: best_int8.rknn")

```

**量化效果对比**：

```
模型格式       文件大小      推理速度      精度损失
─────────────────────────────────────────────
FP32 (.rknn)   ~50MB        基准         0%
FP16 (.rknn)   ~25MB        ~1.5x        ~0.1%
INT8 (.rknn)   ~12MB        ~2-3x        ~0.5-1%

```

---

## 五、开发板端推理部署

### 5.1 Python 推理代码

```python
from rknn.api import RKNN
import cv2
import numpy as np

# 加载 RKNN 模型
rknn = RKNN()
print("--> Loading RKNN model")
ret = rknn.load_rknn("best.rknn")
if ret != 0:
    print("Load RKNN failed!")
    exit(ret)
print("Done")

# 初始化 NPU
print("--> Init runtime environment")
ret = rknn.init_runtime(target='rk3588')
if ret != 0:
    print("Init runtime failed!")
    exit(ret)
print("Done")

# 推理
print("--> Running model")
img = cv2.imread("test_image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640, 640))
img = np.expand_dims(img.astype(np.float32), axis=0) / 255.0

outputs = rknn.inference(inputs=[img])
print(f"推理输出: {len(outputs)} 个张量")

# 后处理（YOLOv8 输出格式）
# outputs[0]: [1, 4, 8400]  - 边界框 (x, y, w, h)
# outputs[1]: [1, 1, 8400]  - 置信度
# outputs[2]: [1, num_classes, 8400] - 类别概率

rknn.release()

```

### 5.2 性能监控

```python
import time

# 性能测试
times = []
for i in range(100):
    start = time.perf_counter()
    outputs = rknn.inference(inputs=[img])
    end = time.perf_counter()
    times.append(end - start)

avg_time = np.mean(times) * 1000  # ms
fps = 1000 / avg_time
print(f"平均推理时间: {avg_time:.2f} ms")
print(f"推理速度: {fps:.1f} FPS")

```

---

## 六、常见问题与解决

### 6.1 模型转换失败

```
问题1: Build failed (错误码非0)
原因: ONNX 模型中有不支持的算子

解决:
1. 使用 onnx-simplifier 简化模型
   python -m onnxsim best.onnx best_simplified.onnx

2. 检查不支持的算子
   python -c "
   import onnx
   m = onnx.load('best.onnx')
   ops = set([n.op_type for n in m.graph.node])
   print('算子类型:', ops)
   "

3. 升级 RKNN-Toolkit2 版本
   pip install --upgrade rknn-toolkit2

```

```
问题2: Unsupported operation: xxx
解决:
1. 更新 RKNN-Toolkit2 到最新版本
2. 在 PC 端使用 ONNX opset 11 导出
3. 检查是否使用了 RKNN 不支持的算子

```

### 6.2 推理结果不正确

```
问题: 推理结果与 ONNX 不一致
原因: 归一化参数不匹配

解决:
1. 确认 training 和 inference 使用相同的归一化
2. 检查 channel_mean_value 配置
3. 确认图像预处理流程一致

验证方法:
# 在 PC 端和开发板上分别推理同一张图片
# 对比输出结果

```

### 6.3 性能不达标

```
问题: 推理速度不理想
解决:
1. 尝试 INT8 量化（速度提升 2-3x）
2. 减小输入尺寸（如 320×320）
3. 使用 half 精度（FP16）
4. 检查 NPU 是否运行在高性能模式
   echo "performance" > /sys/class/devfreq/*npu*/governor

```

### 6.4 NPU 内存不足

```
问题: 推理时 NPU 内存溢出
解决:
1. 减小输入尺寸（imgsz=320）
2. 使用更小的模型（yolov8n → yolov8s）
3. 检查是否有其他应用占用 NPU
4. 重启开发板释放内存

```

---

## 总结

YOLO 模型转换到 RKNN 的关键步骤：

1. **导出 ONNX**：使用 `model.export(format="onnx")`，建议设置 `simplify=True`
2. **验证 ONNX**：在 PC 端用 ONNX Runtime 验证推理结果与原始模型一致
3. **配置转换参数**：`channel_mean_value` 和 `targets` 是关键参数
4. **选择量化策略**：FP16 精度损失小，INT8 速度提升大但需要校准数据
5. **部署推理**：在开发板上加载 `.rknn` 文件并进行后处理

**注意事项**：
- ONNX opset 建议使用 11，兼容性最好
- 校准数据应具有代表性，覆盖所有类别和光照条件
- 转换失败时优先检查算子支持情况
- 推理前务必对比 PC 端和开发板端的输出结果

---

*参考资料：*
- *[RKNN-Toolkit2 GitHub](https://github.com/airockchip/rknn-toolkit2)*
- *[RKNN-Toolkit2 User Guide](https://github.com/airockchip/rknn-toolkit2/blob/master/docs/en/02_rknn_user_guide/01_toolkit_user_guide.md)*

---

> **📌 系列导航**：[← 上一篇：模型量化深度解析](模型量化深度解析.md) · [📖 导读目录](README.md) · [下一篇：yolo模型在npu的python部署 →](YOLO模型在npu的python部署.md)
