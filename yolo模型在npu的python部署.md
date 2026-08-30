# YOLO模型在NPU的Python部署

## 引言

Rockchip NPU（Neural Processing Unit）是瑞芯微芯片内置的神经网络加速器，专为边缘 AI 推理设计。将 YOLO 模型部署到 NPU 上，可以在低功耗设备（如 RK3588、RK3568）上实现实时目标检测。本文介绍基于 RKNN-Toolkit2 的 Python 部署方案。

> **参考来源**：[RKNN-Toolkit2 Python API](https://github.com/airockchip/rknn-toolkit2) | [Rockchip NPU Documentation](https://github.com/airockchip/rknn-toolkit2/blob/master/docs/en/03_rknn_runtime/)

---

## 一、硬件平台选择

### 1.1 Rockchip NPU 系列

| 芯片 | NPU TOPS | 内存 | 适用场景 | 典型功耗 | 参考价格（人民币） |
|------|----------|------|---------|---------|-----------------|
| **RK3588** | 6 TOPS | LPDDR4/5 | 高端边缘设备 | ~15W | 500-800元 |
| **RK3576** | 6 TOPS | LPDDR4/5 | 高端边缘设备 | ~10W | 400-700元 |
| **RK3568** | 0.8 TOPS | LPDDR4 | 中端设备 | ~5W | 200-400元 |
| **RK3566** | 0.8 TOPS | LPDDR4 | 中端设备 | ~3W | 150-300元 |
| **RK3562** | 0.8 TOPS | LPDDR4 | 入门级设备 | ~2W | 80-150元 |

> **选择建议**：RK3588 适合需要高吞吐量的复杂场景（如多路视频分析）；RK3568 适合单路摄像头入门应用；RK3562 适合极简低成本场景。

### 1.2 开发板详细对比

#### RK3588 系列（旗舰级）

| 开发板 | 内存选项 | 扩展接口 | 适合场景 | 价格区间 |
|--------|---------|---------|---------|---------|
| **Radxa ROCK 5B** | 8GB/16GB | PCIe 3.0, M.2 NVMe | 通用计算、AI推理 | 600-900元 |
| **Orange Pi 5 Plus** | 8GB/16GB | 双M.2, HDMI双出 | 多屏显示、边缘服务器 | 700-1000元 |
| **Orange Pi 5** | 4GB/8GB | M.2 NVMe | 性价比选择 | 400-600元 |
| **Rock 5C** | 4GB/8GB | USB4, HDMI | 紧凑型项目 | 350-500元 |
| **NanoPC-T6** | 8GB/16GB | 工业级接口 | 工业应用 | 800-1200元 |
| **NanoPi R5S** | 4GB/8GB | 双网口 | 网关、路由器 | 500-700元 |

#### RK3568 系列（中端级）

| 开发板 | 内存选项 | 扩展接口 | 适合场景 | 价格区间 |
|--------|---------|---------|---------|---------|
| **Orange Pi 3B** | 4GB/8GB | M.2, CSI | 通用入门 | 250-400元 |
| **Rock 3A** | 4GB/8GB | M.2 Key E | WiFi/BLE扩展 | 300-450元 |
| **NanoPi R5S** | 4GB | 双网口 | 网络应用 | 400-550元 |
| **Orange Pi 3B LTS** | 4GB | 工业级稳定 | 长期运行项目 | 300-450元 |

#### RK3562 系列（入门级）

| 开发板 | 内存选项 | 适合场景 | 价格区间 |
|--------|---------|---------|---------|
| **Orange Pi 3A** | 2GB/4GB | 轻量级AI应用 | 150-250元 |
| **NanoPi NEO6** | 2GB | 极低成本项目 | 100-180元 |

### 1.3 选型决策指南

```
如何选择合适的开发板？

1. 确定性能需求
   ├── 单路 1080p 实时检测 (15+ FPS) → RK3568 即可
   ├── 多路 720p 检测 或 4K 单路   → RK3588 推荐
   └── 极低功耗持续运行           → RK3562 考虑

2. 确定接口需求
   ├── 需要 CSI 摄像头接口     → 选择带 MIPI CSI 的开发板
   ├── 需要以太网             → NanoPi R5S 或 Orange Pi 3B
   └── 需要 NVMe 存储         → Rock 5B 或 Orange Pi 5

3. 确定预算
   ├── 100-300元              → RK3562 系列
   ├── 300-600元              → RK3568 系列
   └── 600元以上              → RK3588 系列

```

### 1.4 外设兼容性注意事项

- **摄像头**：确认 MIPI CSI 通道数和分辨率支持（RK3588 支持双 13MP 或单 32MP）
- **显示屏**：部分开发板仅支持 HDMI，工业场景推荐支持 LVDS/eDP 的型号
- **散热**：RK3588 高性能模式下需要主动散热（散热风扇）
- **电源**：建议使用 5V/3A 以上的稳定电源，避免 NPU 高负载时电压不稳

---

## 二、环境配置

### 2.1 开发板系统准备

```bash
# Ubuntu/Debian 系统
# 安装依赖
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    libpython3-dev \
    libatlas-base-dev \
    liblapack-dev \
    gfortran \
    git \
    cmake \
    libopencv-dev \
    ca-certificates \
    wget \
    unzip

# 安装 RKNN Runtime
# 方法1: 从开发板镜像中获取（推荐，版本匹配最好）
# 在开发板官网下载对应系统的 SDK，其中已预装 RKNN Runtime

# 方法2: 从 SDK 包手动安装
# 下载对应芯片型号的 .deb 包（注意：不同芯片的 Runtime 不通用）
dpkg -i rknn-toolkit2-*-linux-aarch64.deb

# 验证 NPU 驱动
cat /proc/driver/rockchip/npu/info

# 预期输出示例:
# npu version: 3.3.2
# npu status: normal

```

### 2.2 Python 环境配置

```bash
# 创建虚拟环境（推荐使用 conda）
python3 -m venv ~/rknn_env
source ~/rknn_env/bin/activate

# 升级 pip 到最新版本
pip install --upgrade pip setuptools wheel

# 安装 rknn-toolkit2（注意：需下载与开发板芯片匹配的版本）
# RK3588 对应的版本
pip install rknn-toolkit2==2.1.0 --no-deps

# 安装运行时依赖（开发板上运行）
pip install numpy>=1.21 opencv-python>=4.5 pillow scipy

# 可选：安装性能分析工具
pip install py-spy psutil

# 验证安装
python3 -c "from rknn.api import RKNN; print('RKNN OK')"

```

### 2.3 版本兼容性说明

| 芯片型号 | 推荐 RKNN Runtime 版本 | 支持 Python 版本 | 注意事项 |
|---------|----------------------|----------------|---------|
| RK3588 | >= 2.0.0 | 3.8 - 3.11 | 建议使用 Python 3.8 或 3.10 |
| RK3576 | >= 2.1.0 | 3.8 - 3.11 | 需使用最新 SDK |
| RK3568 | >= 2.0.0 | 3.8 - 3.10 | 部分版本对 Python 3.11 支持不稳定 |
| RK3562 | >= 2.0.0 | 3.8 - 3.10 | 功能与 RK3568 基本一致 |

> **重要**：RKNN-Toolkit2 需要在 **x86 主机**上进行模型转换，在 **ARM 开发板**上运行推理。两者使用的库版本必须匹配。

### 2.4 验证 NPU 状态

```python
import subprocess
import sys

def check_npu_status():
    """检查 NPU 驱动状态"""
    print("=" * 50)
    print("NPU 环境诊断")
    print("=" * 50)
    
    # 1. 检查 NPU 驱动信息
    print("\n[1] NPU 驱动信息:")
    try:
        result = subprocess.run(
            ["cat", "/proc/driver/rockchip/npu/info"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"  错误: {result.stderr}")
    except Exception as e:
        print(f"  无法读取 NPU 信息: {e}")
    
    # 2. 检查 NPU 模块加载状态
    print("[2] NPU 内核模块:")
    try:
        result = subprocess.run(
            ["lsmod"], capture_output=True, text=True
        )
        npu_modules = ['rockchip_npu', 'rknpu', 'rockchip_rga']
        for mod in npu_modules:
            status = "已加载" if mod in result.stdout else "未加载"
            print(f"  {mod}: {status}")
    except Exception as e:
        print(f"  检查模块失败: {e}")
    
    # 3. 检查 NPU 频率
    print("[3] NPU 频率:")
    try:
        result = subprocess.run(
            ["cat", "/sys/class/devfreq/*npu*/cur_freq"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            freq_hz = int(result.stdout.strip())
            freq_mhz = freq_hz / 1000000
            print(f"  当前频率: {freq_mhz:.1f} MHz")
        else:
            print("  无法读取 NPU 频率")
    except Exception as e:
        print(f"  检查频率失败: {e}")
    
    # 4. 检查 NPU 温度
    print("[4] NPU 温度:")
    try:
        result = subprocess.run(
            ["cat", "/sys/class/thermal/thermal_zone*/temp"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            temps = [int(line.strip()) for line in result.stdout.strip().split('\n')]
            for i, t in enumerate(temps):
                print(f"  温度区 {i}: {t/1000.0:.1f} °C")
    except Exception as e:
        print(f"  无法读取温度: {e}")
    
    # 5. Python 环境检查
    print("[5] Python 环境:")
    print(f"  Python 版本: {sys.version}")
    try:
        from rknn.api import RKNN
        print(f"  RKNN 库: 已安装")
    except ImportError as e:
        print(f"  RKNN 库: 未安装 - {e}")
    
    try:
        import numpy
        print(f"  NumPy 版本: {numpy.__version__}")
    except ImportError:
        print(f"  NumPy: 未安装")
    
    try:
        import cv2
        print(f"  OpenCV 版本: {cv2.__version__}")
    except ImportError:
        print(f"  OpenCV: 未安装")
    
    print("=" * 50)

if __name__ == "__main__":
    check_npu_status()

```

### 2.5 常见安装问题排查

```bash
# === 问题1: pip 安装失败，网络超时 ===
# 解决方案：使用国内镜像源
pip install rknn-toolkit2==2.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# === 问题2: 导入 rknn 时报 "librknnrt.so not found" ===
# 解决方案：添加库路径
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
# 或者将路径写入 /etc/ld.so.conf.d/rknn.conf
echo "/usr/lib/aarch64-linux-gnu" | sudo tee /etc/ld.so.conf.d/rknn.conf
sudo ldconfig

# === 问题3: OpenCV 版本冲突 ===
# 解决方案：安装特定版本
pip uninstall opencv-python opencv-contrib-python -y
pip install opencv-python==4.8.1.78

# === 问题4: NPU 驱动未加载 ===
# 解决方案：手动加载驱动
sudo modprobe rockchip_npu
# 检查是否加载成功
lsmod | grep rockchip_npu

# === 问题5: 虚拟环境中找不到 rknn 模块 ===
# 解决方案：确认使用了正确的 Python
which python3
python3 -c "import rknn"

# === 问题6: 开发板镜像版本过旧 ===
# 解决方案：升级到最新 SDK
# 从开发板厂商官网下载最新镜像并重新烧录

```

---

## 三、Python 推理部署

### 3.1 完整推理类

```python
#!/usr/bin/env python3
"""
YOLO NPU 推理示例
目标平台: RK3588
支持: 图片推理、视频推理、批量处理
"""

import cv2
import numpy as np
import time
from pathlib import Path
from rknn.api import RKNN


class YOLONPUInference:
    """
    YOLO NPU 推理器
    
    支持 YOLOv8 / YOLOv5 / YOLOv10 等常见 YOLO 系列模型
    输出格式兼容 Ultralytics 标准格式
    """
    
    def __init__(self, model_path="best.rknn", target="rk3588", imgsz=640):
        """
        初始化推理器
        
        Args:
            model_path: RKNN 模型文件路径
            target: 目标芯片型号 (rk3588 / rk3568 / rk3566 / rk3562)
            imgsz: 模型输入尺寸 (默认 640)
        """
        self.imgsz = imgsz
        self.target = target
        self.rknn = RKNN()
        self.num_classes = 80  # COCO 默认
        self.stride = 32       # YOLO 默认步长
        
        # 加载模型
        print(f"[1/5] Loading RKNN model: {model_path}")
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {ret}")
        print("    Model loaded successfully")
        
        # 初始化运行时
        print(f"[2/5] Initializing runtime on {target}...")
        ret = self.rknn.init_runtime(target=target)
        if ret != 0:
            raise RuntimeError(f"Failed to init runtime: {ret}")
        print("    Runtime initialized")
        
        # 获取输入输出信息
        inputs = self.rknn.graph.get_inputs()
        outputs = self.rknn.graph.get_outputs()
        self.input_name = inputs[0].name
        self.output_names = [o.name for o in outputs]
        
        # 尝试从模型名称推断类别数
        self._infer_num_classes(outputs)
        
        print(f"[3/5] Input: {self.input_name}")
        print(f"      Outputs: {self.output_names}")
        print(f"      Num classes: {self.num_classes}")
        
        # 加载类别名称
        self.class_names = self._load_class_names()
        print(f"[4/5] Loaded {len(self.class_names)} class names")
        
        # 性能统计
        self.fps_history = []
        print("[5/5] Ready!")
    
    def _infer_num_classes(self, outputs):
        """根据输出张量推断类别数量"""
        try:
            # YOLOv8 输出格式: [1, 4+n, 8400]
            shape = outputs[0].shape
            if len(shape) >= 2 and shape[1] > 4:
                self.num_classes = shape[1] - 4
        except Exception:
            pass
    
    def _load_class_names(self):
        """加载类别名称文件"""
        # 常见名称文件路径
        name_files = [
            "coco.names",
            "classes.txt",
            "class_names.txt",
        ]
        for name_file in name_files:
            try:
                with open(name_file, 'r', encoding='utf-8') as f:
                    return [line.strip() for line in f.readlines() if line.strip()]
            except FileNotFoundError:
                continue
        # 默认 COCO 80 类
        return [f"class_{i}" for i in range(self.num_classes)]
    
    def _preprocess(self, img):
        """
        图像预处理
        
        将原始图像转换为模型可接受的输入格式：
        1. 保持宽高比缩放
        2. 填充到正方形
        3. BGR -> RGB 转换
        4. 归一化到 [0, 1]
        
        Args:
            img: 原始 BGR 图像 (numpy array)
            
        Returns:
            img_input: 预处理后的输入张量 [1, 3, imgsz, imgsz]
            scale: 缩放比例
            pad_w, pad_h: 填充像素数
        """
        h, w = img.shape[:2]
        
        # 计算保持宽高比的缩放比例
        scale = self.imgsz / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 保持宽高比缩放
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # 填充到正方形 (使用灰色填充)
        img_padded = np.ones((self.imgsz, self.imgsz, 3), dtype=np.uint8) * 128
        # 居中放置
        top = (self.imgsz - new_h) // 2
        left = (self.imgsz - new_w) // 2
        img_padded[top:top+new_h, left:left+new_w] = img_resized
        
        # RGB 转换
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        
        # 归一化并扩展维度
        img_input = np.expand_dims(img_rgb.astype(np.float32), axis=0) / 255.0
        
        # 记录填充参数
        pad_w = left
        pad_h = top
        
        return img_input, scale, pad_w, pad_h
    
    def _postprocess(self, outputs, orig_shape, scale, pad_w, pad_h):
        """
        后处理：解码 NPU 输出，过滤低置信度框，执行 NMS
        
        YOLOv8 输出格式: [1, 4+n, 8400]
        - 前 4 维: [x_center, y_center, width, height] (相对于 640x640 输入)
        - 后 n 维: 各类别的置信度分数
        
        Args:
            outputs: NPU 推理输出列表
            orig_shape: 原始图像形状 (h, w, c)
            scale: 缩放比例
            pad_w: 水平填充像素
            pad_h: 垂直填充像素
            
        Returns:
            results: 检测结果列表，每项包含 class_id, class_name, confidence, bbox
        """
        # 转置输出: [1, 4+n, 8400] -> [1, 8400, 4+n]
        output = outputs[0].transpose(0, 2, 1)
        
        # 解析边界框和分数
        # boxes: [x_center, y_center, width, height]
        xc = output[0, :, 0]
        yc = output[0, :, 1]
        ww = output[0, :, 2]
        hh = output[0, :, 3]
        scores = output[0, :, 4:]
        num_classes = scores.shape[1]
        
        # 解码边界框坐标
        # x1, y1, x2, y2 对应左上角和右下角
        x1 = xc - ww / 2.0
        y1 = yc - hh / 2.0
        x2 = xc + ww / 2.0
        y2 = yc + hh / 2.0
        
        # 将坐标从模型输入空间映射回原始图像空间
        # 公式: orig_coord = (model_coord - pad) / scale
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale
        
        # 裁剪到图像边界
        oh, ow = orig_shape[:2]
        x1 = np.clip(x1, 0, ow)
        y1 = np.clip(y1, 0, oh)
        x2 = np.clip(x2, 0, ow)
        y2 = np.clip(y2, 0, oh)
        
        # 获取类别和置信度
        class_ids = np.argmax(scores, axis=1)
        confs = np.max(scores, axis=1)
        
        # 过滤低置信度检测结果
        mask = confs > 0.25
        boxes = np.column_stack([x1[mask], y1[mask], x2[mask], y2[mask]])
        confs = confs[mask]
        class_ids = class_ids[mask]
        
        # 执行非极大值抑制 (NMS)
        if len(boxes) > 0:
            # cv2.dnn.NMSBoxes 需要 float32 格式的 box
            boxes_float = boxes.astype(np.float32)
            indices = cv2.dnn.NMSBoxes(
                boxes_float.tolist(), 
                confs.astype(np.float32).tolist(), 
                score_threshold=0.25, 
                nms_threshold=0.45
            )
            # NMSBoxes 返回的 indices 可能为空的列表
            if len(indices) > 0:
                indices = indices.flatten()
            else:
                indices = np.array([], dtype=np.int32)
        else:
            indices = np.array([], dtype=np.int32)
        
        # 构建结果列表
        results = []
        for idx in indices:
            result = {
                'class_id': int(class_ids[idx]),
                'class_name': self.class_names[class_ids[idx]] if class_ids[idx] < len(self.class_names) else f"class_{class_ids[idx]}",
                'confidence': float(confs[idx]),
                'bbox': [int(x) for x in boxes[idx]]
            }
            results.append(result)
        
        return results
    
    def infer(self, img):
        """
        执行单次推理
        
        Args:
            img: BGR 格式的输入图像
            
        Returns:
            results: 检测结果列表
        """
        # 预处理
        img_input, scale, pad_w, pad_h = self._preprocess(img)
        
        # NPU 推理
        outputs = self.rknn.inference(inputs=[img_input])
        
        # 后处理
        results = self._postprocess(outputs, img.shape, scale, pad_w, pad_h)
        
        return results
    
    def infer_batch(self, images):
        """
        批量推理
        
        Args:
            images: 图像列表 (每个为 BGR numpy array)
            
        Returns:
            all_results: 每张图片的检测结果列表
        """
        all_results = []
        for img in images:
            results = self.infer(img)
            all_results.append(results)
        return all_results
    
    def get_fps(self):
        """获取当前平均 FPS"""
        if self.fps_history:
            return 1.0 / np.mean(self.fps_history)
        return 0.0
    
    def release(self):
        """释放 NPU 资源"""
        if self.rknn:
            self.rknn.release()


def main():
    """主函数：演示图片推理"""
    # 初始化推理器
    detector = YOLONPUInference(
        model_path="best.rknn",
        target="rk3588",
        imgsz=640
    )
    
    # 测试推理
    img = cv2.imread("test_image.jpg")
    if img is None:
        print("Failed to load image")
        detector.release()
        return
    
    # 预热（NPU 首次推理较慢）
    for _ in range(5):
        detector.infer(img)
    
    # 计时推理
    n_runs = 20
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        results = detector.infer(img)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print(f"\n推理结果:")
    for r in results:
        x1, y1, x2, y2 = r['bbox']
        print(f"  {r['class_name']}: {r['confidence']:.2f} [{x1},{y1},{x2},{y2}]")
    
    print(f"\n性能: 平均 {avg_time*1000:.1f} ms, {fps:.1f} FPS")
    
    detector.release()


if __name__ == "__main__":
    main()

```

### 3.2 实时视频推理

```python
#!/usr/bin/env python3
"""
YOLO NPU 实时视频推理
支持摄像头输入、视频文件输入
"""

import cv2
import numpy as np
import time
import threading
from rknn.api import RKNN


class RealtimeDetector:
    """
    实时目标检测器
    
    支持摄像头实时推理和视频文件推理
    包含 FPS 显示和结果绘制功能
    """
    
    def __init__(self, model_path, target="rk3588", imgsz=640):
        self.imgsz = imgsz
        self.target = target
        
        # 加载模型
        print("Loading model...")
        self.rknn = RKNN()
        self.rknn.load_rknn(model_path)
        self.rknn.init_runtime(target=target)
        print("Model loaded.")
        
        # 类别名称
        self.class_names = self._load_names()
        
        # 性能统计
        self.fps_history = []
        self.frame_count = 0
        self.start_time = time.perf_counter()
    
    def _load_names(self):
        """加载类别名称"""
        for name_file in ["coco.names", "classes.txt"]:
            try:
                with open(name_file, "r", encoding="utf-8") as f:
                    return [line.strip() for line in f if line.strip()]
            except FileNotFoundError:
                continue
        return [f"class_{i}" for i in range(80)]
    
    def _preprocess(self, img):
        """
        图像预处理
        返回: (input_tensor, scale, pad_w, pad_h)
        """
        h, w = img.shape[:2]
        scale = self.imgsz / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 保持宽高比缩放
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # 居中填充到正方形
        img_padded = np.full((self.imgsz, self.imgsz, 3), 128, dtype=np.uint8)
        top = (self.imgsz - new_h) // 2
        left = (self.imgsz - new_w) // 2
        img_padded[top:top+new_h, left:left+new_w] = img_resized
        
        # BGR -> RGB + 归一化
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_input = np.expand_dims(img_rgb.astype(np.float32), 0) / 255.0
        
        return img_input, scale, left, top
    
    def _postprocess(self, outputs, orig_shape, scale, pad_w, pad_h):
        """
        后处理：解码输出，过滤和 NMS
        """
        output = outputs[0].transpose(0, 2, 1)
        xc = output[0, :, 0]
        yc = output[0, :, 1]
        ww = output[0, :, 2]
        hh = output[0, :, 3]
        scores = output[0, :, 4:]
        
        # 解码边界框
        x1 = (xc - ww / 2.0 - pad_w) / scale
        y1 = (yc - hh / 2.0 - pad_h) / scale
        x2 = (xc + ww / 2.0 - pad_w) / scale
        y2 = (yc + hh / 2.0 - pad_h) / scale
        
        # 裁剪到图像边界
        oh, ow = orig_shape[:2]
        x1 = np.clip(x1, 0, ow)
        y1 = np.clip(y1, 0, oh)
        x2 = np.clip(x2, 0, ow)
        y2 = np.clip(y2, 0, oh)
        
        class_ids = np.argmax(scores, axis=1)
        confs = np.max(scores, axis=1)
        
        # 过滤低置信度
        mask = confs > 0.25
        boxes = np.column_stack([x1[mask], y1[mask], x2[mask], y2[mask]])
        confs = confs[mask]
        class_ids = class_ids[mask]
        
        # NMS
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(
                boxes.astype(np.float32).tolist(),
                confs.astype(np.float32).tolist(),
                score_threshold=0.25,
                nms_threshold=0.45
            )
            indices = indices.flatten() if len(indices) > 0 else np.array([], dtype=np.int32)
        else:
            indices = np.array([], dtype=np.int32)
        
        results = []
        for idx in indices:
            results.append({
                'class': self.class_names[class_ids[idx]] if class_ids[idx] < len(self.class_names) else f"class_{class_ids[idx]}",
                'conf': float(confs[idx]),
                'box': [int(x) for x in boxes[idx]]
            })
        return results
    
    def detect(self, img):
        """执行单帧检测"""
        img_input, scale, pad_w, pad_h = self._preprocess(img)
        outputs = self.rknn.inference(inputs=[img_input])
        return self._postprocess(outputs, img.shape, scale, pad_w, pad_h)
    
    def draw_results(self, img, results):
        """
        在图像上绘制检测结果
        
        绘制内容包括：
        - 边界框（绿色）
        - 类别标签和置信度（绿色背景白字）
        - FPS 信息（左上角）
        """
        # 绘制检测结果
        for r in results:
            x1, y1, x2, y2 = r['box']
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 绘制标签背景
            label = f"{r['class']} {r['conf']:.2f}"
            t = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(img, (x1, y1 - t[1] - 8), (x1 + t[0], y1), (0, 255, 0), -1)
            # 绘制标签文字
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 绘制 FPS
        self.frame_count += 1
        elapsed = time.perf_counter() - self.start_time
        if elapsed > 0:
            fps = self.frame_count / elapsed
            cv2.putText(img, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return img
    
    def run_camera(self, camera_id=0, save_output=None, show_window=True):
        """
        运行摄像头实时检测
        
        Args:
            camera_id: 摄像头 ID（0 为默认摄像头）
            save_output: 输出视频文件路径（可选）
            show_window: 是否显示结果窗口
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("Failed to open camera")
            return
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 视频写入器
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps_cap = cap.get(cv2.CAP_PROP_FPS) or 30.0
            writer = cv2.VideoWriter(save_output, fourcc, fps_cap,
                                     (int(cap.get(3)), int(cap.get(4))))
        
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # 检测
            start = time.perf_counter()
            results = self.detect(frame)
            elapsed = time.perf_counter() - start
            self.fps_history.append(elapsed)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            
            # 绘制结果
            self.draw_results(frame, results)
            
            # 显示
            if show_window:
                cv2.imshow("YOLO NPU Inference", frame)
            
            # 保存
            if writer:
                writer.write(frame)
            
            # 退出条件
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 清理资源
        cap.release()
        if writer:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()
        self.rknn.release()
    
    def run_video(self, video_path, save_output=None, show_window=True):
        """
        运行视频文件检测
        
        Args:
            video_path: 视频文件路径
            save_output: 输出视频路径（可选）
            show_window: 是否显示结果窗口
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return
        
        fps_cap = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 视频写入器
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_output, fourcc, fps_cap,
                                     (int(cap.get(3)), int(cap.get(4))))
        
        print(f"Processing video: {video_path} ({total_frames} frames)")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测
            results = self.detect(frame)
            self.draw_results(frame, results)
            
            # 显示进度
            frame_idx += 1
            if show_window and frame_idx % 10 == 0:
                print(f"  Progress: {frame_idx}/{total_frames}")
            
            if show_window:
                cv2.imshow("YOLO NPU Video", frame)
            
            if writer:
                writer.write(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if writer:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()
        self.rknn.release()


if __name__ == "__main__":
    # 方式1: 摄像头实时检测
    detector = RealtimeDetector("best.rknn", target="rk3588")
    detector.run_camera(camera_id=0, save_output="output.mp4")
    
    # 方式2: 视频文件检测（取消注释使用）
    # detector.run_video("input_video.mp4", save_output="output.mp4")

```

### 3.3 批量处理支持

```python
#!/usr/bin/env python3
"""
YOLO NPU 批量图片处理
支持图片文件夹扫描和批量推理
"""

import cv2
import numpy as np
import time
import json
from pathlib import Path
from rknn.api import RKNN


class BatchDetector:
    """
    批量图片检测器
    
    支持：
    - 扫描文件夹中所有图片
    - 批量推理
    - 结果保存为 JSON 和可视化图片
    """
    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    def __init__(self, model_path, target="rk3588", imgsz=640):
        self.imgsz = imgsz
        self.rknn = RKNN()
        self.rknn.load_rknn(model_path)
        self.rknn.init_runtime(target=target)
        self.class_names = self._load_names()
    
    def _load_names(self):
        for name_file in ["coco.names", "classes.txt"]:
            try:
                with open(name_file, "r", encoding="utf-8") as f:
                    return [line.strip() for line in f if line.strip()]
            except FileNotFoundError:
                continue
        return [f"class_{i}" for i in range(80)]
    
    def _preprocess(self, img):
        h, w = img.shape[:2]
        scale = self.imgsz / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        img_padded = np.full((self.imgsz, self.imgsz, 3), 128, dtype=np.uint8)
        top = (self.imgsz - new_h) // 2
        left = (self.imgsz - new_w) // 2
        img_padded[top:top+new_h, left:left+new_w] = img_resized
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        return np.expand_dims(img_rgb.astype(np.float32), 0) / 255.0, scale, left, top
    
    def _postprocess(self, outputs, orig_shape, scale, pad_w, pad_h):
        output = outputs[0].transpose(0, 2, 1)
        xc = output[0, :, 0]
        yc = output[0, :, 1]
        ww = output[0, :, 2]
        hh = output[0, :, 3]
        scores = output[0, :, 4:]
        
        x1 = (xc - ww / 2.0 - pad_w) / scale
        y1 = (yc - hh / 2.0 - pad_h) / scale
        x2 = (xc + ww / 2.0 - pad_w) / scale
        y2 = (yc + hh / 2.0 - pad_h) / scale
        
        oh, ow = orig_shape[:2]
        x1 = np.clip(x1, 0, ow)
        y1 = np.clip(y1, 0, oh)
        x2 = np.clip(x2, 0, ow)
        y2 = np.clip(y2, 0, oh)
        
        class_ids = np.argmax(scores, axis=1)
        confs = np.max(scores, axis=1)
        mask = confs > 0.25
        
        if not np.any(mask):
            return []
        
        boxes = np.column_stack([x1[mask], y1[mask], x2[mask], y2[mask]])
        confs = confs[mask]
        class_ids = class_ids[mask]
        
        indices = cv2.dnn.NMSBoxes(
            boxes.astype(np.float32).tolist(),
            confs.astype(np.float32).tolist(),
            0.25, 0.45
        )
        indices = indices.flatten() if len(indices) > 0 else np.array([], dtype=np.int32)
        
        results = []
        for idx in indices:
            results.append({
                'class_id': int(class_ids[idx]),
                'class_name': self.class_names[class_ids[idx]] if class_ids[idx] < len(self.class_names) else f"class_{class_ids[idx]}",
                'confidence': float(confs[idx]),
                'bbox': [int(x) for x in boxes[idx]]
            })
        return results
    
    def detect(self, img):
        img_input, scale, pad_w, pad_h = self._preprocess(img)
        outputs = self.rknn.inference(inputs=[img_input])
        return self._postprocess(outputs, img.shape, scale, pad_w, pad_h)
    
    def draw_results(self, img, results):
        """绘制检测结果到图像"""
        for r in results:
            x1, y1, x2, y2 = r['bbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{r['class_name']} {r['confidence']:.2f}"
            t = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(img, (x1, y1 - t[1] - 8), (x1 + t[0], y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return img
    
    def process_folder(self, input_dir, output_dir=None, save_images=True):
        """
        批量处理文件夹中的所有图片
        
        Args:
            input_dir: 输入图片文件夹路径
            output_dir: 输出文件夹路径（可选，默认为 input_dir/results）
            save_images: 是否保存带标注的图片
            
        Returns:
            all_results: 所有图片的检测结果字典
        """
        input_path = Path(input_dir)
        if output_dir is None:
            output_path = input_path / "results"
        else:
            output_path = Path(output_dir)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 收集所有图片
        image_files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            image_files.extend(input_path.glob(f"**/*{ext}"))
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return {}
        
        print(f"Found {len(image_files)} images")
        
        all_results = {}
        times = []
        
        for img_path in image_files:
            # 读取图像
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Failed to read: {img_path}")
                continue
            
            # 推理
            start = time.perf_counter()
            results = self.detect(img)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            
            # 保存结果
            rel_path = img_path.relative_to(input_path)
            result_key = str(rel_path)
            all_results[result_key] = {
                'detections': results,
                'time_ms': elapsed * 1000
            }
            
            # 保存带标注的图片
            if save_images:
                img_copy = img.copy()
                self.draw_results(img_copy, results)
                out_img_path = output_path / rel_path
                out_img_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_img_path), img_copy)
            
            print(f"  {rel_path}: {len(results)} objects, {elapsed*1000:.1f} ms")
        
        # 打印统计
        if times:
            avg_time = np.mean(times)
            print(f"\nBatch processing complete:")
            print(f"  Total images: {len(image_files)}")
            print(f"  Average time: {avg_time*1000:.1f} ms")
            print(f"  Average FPS: {1.0/avg_time:.1f}")
            print(f"  Results saved to: {output_path}")
        
        # 保存 JSON 结果
        json_path = output_path / "results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"  JSON results saved to: {json_path}")
        
        return all_results
    
    def release(self):
        """释放资源"""
        self.rknn.release()


if __name__ == "__main__":
    # 批量处理示例
    detector = BatchDetector("best.rknn", target="rk3588")
    results = detector.process_folder(
        input_dir="./test_images",
        output_dir="./test_images/results"
    )
    detector.release()

```

### 3.4 性能基准测试

```python
"""
在 RK3588 上测试不同模型的性能
"""
import time
import numpy as np
from rknn.api import RKNN


def benchmark_model(model_path, target="rk3588", imgsz=640, n_runs=50):
    """基准测试单个模型"""
    rknn = RKNN()
    rknn.load_rknn(model_path)
    rknn.init_runtime(target=target)
    
    # 准备输入
    img = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
    
    # 预热
    for _ in range(5):
        rknn.inference(inputs=[img])
    
    # 计时
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        rknn.inference(inputs=[img])
        times.append(time.perf_counter() - start)
    
    rknn.release()
    
    return {
        "model": model_path,
        "avg_ms": np.mean(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
        "fps": 1.0 / np.mean(times),
    }

# 测试不同模型
models = [
    ("yolov8n.rknn", "YOLOv8 Nano"),
    ("yolov8s.rknn", "YOLOv8 Small"),
    ("yolov8m.rknn", "YOLOv8 Medium"),
    ("yolov8s_int8.rknn", "YOLOv8 Small INT8"),
]

print(f"{'模型':<20} {'平均(ms)':<12} {'FPS':<10} {'最小(ms)':<12}")
print("-" * 55)
for model_path, name in models:
    try:
        result = benchmark_model(model_path)
        print(f"{name:<20} {result['avg_ms']:<12.1f} {result['fps']:<10.1f} {result['min_ms']:<12.1f}")
    except Exception as e:
        print(f"{name:<20} ERROR: {e}")

```

---

## 四、性能优化技巧

### 4.1 输入尺寸优化

```python
# 根据需求选择最优输入尺寸
# RK3588 上建议:
# - 速度优先: imgsz=320 (~30+ FPS)
# - 平衡: imgsz=640 (~15-20 FPS)
# - 精度优先: imgsz=1280 (~5-8 FPS)

def choose_imgsz(task):
    if task == "speed":
        return 320
    elif task == "balanced":
        return 640
    elif task == "accuracy":
        return 1280
    else:
        return 640

```

### 4.2 模型量化优化

```python
"""
模型量化是提升 NPU 推理速度的关键手段

量化类型对比:
- FP32: 最高精度，最慢速度
- FP16: 精度略降，速度提升约 1.5 倍
- INT8: 精度可能略降，速度提升 2-3 倍

RKNN-Toolkit2 量化示例（在 x86 主机上执行）:
"""

from rknn.api import RKNN

rknn = RKNN()

# 1. 配置模型（注意：Toolkit2 中 config 必须在 load_onnx 之前调用）
print('--> Configuring model')
rknn.config(
    mean_values=[[0, 0, 0]],        # 归一化均值（与训练/推理预处理一致）
    std_values=[[255, 255, 255]],   # 归一化方差（即 /255）
    target_platform='rk3588',       # 目标芯片平台
)

# 2. 加载 ONNX 模型
print('--> Loading model')
rknn.load_onnx(model='yolov8s.onnx')

# 3. 构建 INT8 量化模型（需要校准数据集）
print('--> Building model (INT8)')
rknn.build(do_quantization=True,
           dataset='calibration_dataset.txt',  # 校准数据集文件（txt，每行一张图片路径）
           quantized_algorithm='normal')       # 量化算法：normal / mmse / kl_divergence

# 4. 导出 RKNN 模型
print('--> Export RKNN model')
rknn.export_rknn('yolov8s_int8.rknn')

# 5. 释放资源
rknn.release()

```

### 4.3 算子融合优化

```python
"""
算子融合（Operator Fusion）是将多个相邻算子合并为一个复合算子的技术。
在模型转换阶段，RKNN-Toolkit2 会自动进行 Conv+BN、Conv+Scale 等常见融合。

如果模型转换后性能不理想，可以：
1. 确保转换时开启优化级别 2 或 3
2. 手动在 ONNX 层面合并 Conv+BN（使用 onnxsim 等工具）
3. 检查模型中是否有无法融合的冗余算子
"""

# 使用 onnx-simplifier 预处理 ONNX 模型（在 x86 主机上）
import onnx
from onnxsim import simplify

# 加载并简化 ONNX 模型
model = onnx.load('yolov8s.onnx')
model_sim, check = simplify(model)
assert check, "Simplified ONNX model could not be validated"

# 保存简化后的模型
onnx.save(model_sim, 'yolov8s_simplified.onnx')
print("Model simplified and ready for RKNN conversion")

```

### 4.4 多线程并行处理

```python
"""
多线程方案：一个线程负责采集图像，另一个线程负责推理，
两者通过队列通信，实现流水线并行。
"""

import cv2
import numpy as np
import time
import queue
import threading
from rknn.api import RKNN


class PipelineDetector:
    """
    流水线检测器
    
    结构：
    [采集线程] --> [图像队列] --> [推理线程] --> [显示线程]
    
    优势：采集和推理并行，最大化利用 NPU 空闲时间
    """
    
    def __init__(self, model_path, target="rk3588", imgsz=640):
        self.imgsz = imgsz
        self.rknn = RKNN()
        self.rknn.load_rknn(model_path)
        self.rknn.init_runtime(target=target)
        self.class_names = self._load_names()
        
        # 队列
        self.img_queue = queue.Queue(maxsize=2)  # 最多缓存 2 帧
        self.result_queue = queue.Queue(maxsize=2)
        
        # 运行标志
        self.running = False
        
        # 性能统计
        self.fps_history = []
        self.frame_count = 0
        self.start_time = time.perf_counter()
    
    def _load_names(self):
        for name_file in ["coco.names", "classes.txt"]:
            try:
                with open(name_file, "r", encoding="utf-8") as f:
                    return [line.strip() for line in f if line.strip()]
            except FileNotFoundError:
                continue
        return [f"class_{i}" for i in range(80)]
    
    def _preprocess(self, img):
        h, w = img.shape[:2]
        scale = self.imgsz / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        img_padded = np.full((self.imgsz, self.imgsz, 3), 128, dtype=np.uint8)
        top = (self.imgsz - new_h) // 2
        left = (self.imgsz - new_w) // 2
        img_padded[top:top+new_h, left:left+new_w] = img_resized
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        return np.expand_dims(img_rgb.astype(np.float32), 0) / 255.0, scale, left, top
    
    def _postprocess(self, outputs, orig_shape, scale, pad_w, pad_h):
        output = outputs[0].transpose(0, 2, 1)
        xc = output[0, :, 0]
        yc = output[0, :, 1]
        ww = output[0, :, 2]
        hh = output[0, :, 3]
        scores = output[0, :, 4:]
        
        x1 = (xc - ww / 2.0 - pad_w) / scale
        y1 = (yc - hh / 2.0 - pad_h) / scale
        x2 = (xc + ww / 2.0 - pad_w) / scale
        y2 = (yc + hh / 2.0 - pad_h) / scale
        
        oh, ow = orig_shape[:2]
        x1 = np.clip(x1, 0, ow)
        y1 = np.clip(y1, 0, oh)
        x2 = np.clip(x2, 0, ow)
        y2 = np.clip(y2, 0, oh)
        
        class_ids = np.argmax(scores, axis=1)
        confs = np.max(scores, axis=1)
        mask = confs > 0.25
        
        if not np.any(mask):
            return []
        
        boxes = np.column_stack([x1[mask], y1[mask], x2[mask], y2[mask]])
        confs = confs[mask]
        class_ids = class_ids[mask]
        
        indices = cv2.dnn.NMSBoxes(
            boxes.astype(np.float32).tolist(),
            confs.astype(np.float32).tolist(),
            0.25, 0.45
        )
        indices = indices.flatten() if len(indices) > 0 else np.array([], dtype=np.int32)
        
        results = []
        for idx in indices:
            results.append({
                'class': self.class_names[class_ids[idx]] if class_ids[idx] < len(self.class_names) else f"class_{class_ids[idx]}",
                'conf': float(confs[idx]),
                'box': [int(x) for x in boxes[idx]]
            })
        return results
    
    def _inference_thread(self):
        """推理线程：从队列取图像，推理后将结果放入结果队列"""
        while self.running:
            try:
                # 非阻塞获取图像（超时 0.1 秒）
                img, frame_info = self.img_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # 预处理和推理
            img_input, scale, pad_w, pad_h = self._preprocess(img)
            outputs = self.rknn.inference(inputs=[img_input])
            results = self._postprocess(outputs, img.shape, scale, pad_w, pad_h)
            
            # 放入结果队列
            try:
                self.result_queue.put((results, frame_info), timeout=0.1)
            except queue.Full:
                pass  # 跳过，保持流水线畅通
    
    def _display_thread(self):
        """显示线程：从结果队列取结果并显示"""
        while self.running:
            try:
                results, frame_info = self.result_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            img = frame_info['image']
            
            # 绘制结果
            for r in results:
                x1, y1, x2, y2 = r['box']
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{r['class']} {r['conf']:.2f}"
                t = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(img, (x1, y1 - t[1] - 8), (x1 + t[0], y1), (0, 255, 0), -1)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # 绘制 FPS
            self.frame_count += 1
            elapsed = time.perf_counter() - self.start_time
            if elapsed > 0:
                fps = self.frame_count / elapsed
                cv2.putText(img, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("YOLO NPU Pipeline", img)
    
    def run(self, camera_id=0):
        """启动流水线检测"""
        self.running = True
        
        # 启动线程
        t_infer = threading.Thread(target=self._inference_thread, daemon=True)
        t_display = threading.Thread(target=self._display_thread, daemon=True)
        t_infer.start()
        t_display.start()
        
        # 采集线程（主线程）
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("Failed to open camera")
            self.running = False
            return
        
        print("Press 'q' to quit")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 将图像放入队列（阻塞如果队列满）
            try:
                self.img_queue.put((frame.copy(), {'image': frame}), timeout=0.05)
            except queue.Full:
                pass  # 队列满，跳过当前帧
        
        # 清理
        self.running = False
        cap.release()
        cv2.destroyAllWindows()
        self.rknn.release()


if __name__ == "__main__":
    detector = PipelineDetector("best.rknn", target="rk3588")
    detector.run(camera_id=0)

```

### 4.5 内存优化

```python
"""
内存优化策略：

1. 避免创建多余的内存拷贝
2. 复用图像缓冲区
3. 及时释放不再使用的资源
4. 控制批量大小避免 OOM
"""

import cv2
import numpy as np
from rknn.api import RKNN


class MemoryOptimizedDetector:
    """
    内存优化版检测器
    
    关键优化点：
    - 复用预处理缓冲区，避免每帧重新分配内存
    - 及时释放中间结果
    - 使用 np.empty 替代 np.ones 减少内存分配
    """
    
    def __init__(self, model_path, target="rk3588", imgsz=640):
        self.imgsz = imgsz
        self.rknn = RKNN()
        self.rknn.load_rknn(model_path)
        self.rknn.init_runtime(target=target)
        self.class_names = self._load_names()
        
        # 预分配缓冲区（避免频繁内存分配）
        self.input_buffer = np.empty((1, 3, imgsz, imgsz), dtype=np.float32)
        self.padded_buffer = np.empty((imgsz, imgsz, 3), dtype=np.uint8)
    
    def _load_names(self):
        try:
            with open("coco.names", "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            return [f"class_{i}" for i in range(80)]
    
    def detect(self, img):
        """
        内存优化的推理方法
        - 避免不必要的图像拷贝
        - 复用预分配缓冲区
        """
        h, w = img.shape[:2]
        scale = self.imgsz / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 缩放（原地操作，避免额外分配）
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # 填充到正方形
        self.padded_buffer[:] = 128  # 用 128 填充背景
        top = (self.imgsz - new_h) // 2
        left = (self.imgsz - new_w) // 2
        self.padded_buffer[top:top+new_h, left:left+new_w] = img_resized
        
        # BGR -> RGB
        img_rgb = cv2.cvtColor(self.padded_buffer, cv2.COLOR_BGR2RGB)
        
        # 归一化到预分配缓冲区
        self.input_buffer[0] = img_rgb.astype(np.float32) / 255.0
        
        # 推理
        outputs = self.rknn.inference(inputs=[self.input_buffer])
        
        # 后处理
        return self._postprocess(outputs, (h, w, 3), scale, left, top)
    
    def _postprocess(self, outputs, orig_shape, scale, pad_w, pad_h):
        output = outputs[0].transpose(0, 2, 1)
        xc = output[0, :, 0]
        yc = output[0, :, 1]
        ww = output[0, :, 2]
        hh = output[0, :, 3]
        scores = output[0, :, 4:]
        
        x1 = (xc - ww / 2.0 - pad_w) / scale
        y1 = (yc - hh / 2.0 - pad_h) / scale
        x2 = (xc + ww / 2.0 - pad_w) / scale
        y2 = (yc + hh / 2.0 - pad_h) / scale
        
        oh, ow = orig_shape[:2]
        x1 = np.clip(x1, 0, ow)
        y1 = np.clip(y1, 0, oh)
        x2 = np.clip(x2, 0, ow)
        y2 = np.clip(y2, 0, oh)
        
        class_ids = np.argmax(scores, axis=1)
        confs = np.max(scores, axis=1)
        mask = confs > 0.25
        
        if not np.any(mask):
            return []
        
        boxes = np.column_stack([x1[mask], y1[mask], x2[mask], y2[mask]])
        confs = confs[mask]
        class_ids = class_ids[mask]
        
        indices = cv2.dnn.NMSBoxes(
            boxes.astype(np.float32).tolist(),
            confs.astype(np.float32).tolist(),
            0.25, 0.45
        )
        indices = indices.flatten() if len(indices) > 0 else np.array([], dtype=np.int32)
        
        results = []
        for idx in indices:
            results.append({
                'class': self.class_names[class_ids[idx]] if class_ids[idx] < len(self.class_names) else f"class_{class_ids[idx]}",
                'conf': float(confs[idx]),
                'box': [int(x) for x in boxes[idx]]
            })
        return results
    
    def release(self):
        self.rknn.release()

```

---

## 五、常见问题

### 5.1 推理结果与 PC 端不一致

**问题现象**：NPU 端推理结果与 PC 端（Ultralytics）结果差异较大。

**排查步骤**：

```python
# 步骤1：确认预处理完全一致
# PC 端（Ultralytics YOLOv8 默认）:
# 1. letterbox 缩放（保持宽高比，灰色填充）
# 2. HWC -> CHW 转置
# 3. BGR -> RGB
# 4. 除以 255 归一化

# NPU 端：确保完全相同的流程
def verify_preprocess(img, pc_img):
    """验证两端预处理一致性"""
    # 读取 PC 端预处理后的输入
    # 与 NPU 端预处理结果对比
    diff = np.abs(img - pc_img).max()
    print(f"Max difference: {diff}")
    assert diff < 1e-5, "Preprocessing mismatch!"

# 步骤2：检查模型转换参数
# 确保转换时使用的归一化参数与训练时一致
rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588')

```

**常见原因及解决**：
- **归一化不一致**：确认 PC 端和 NPU 端都使用 `/255.0` 归一化
- **颜色空间不一致**：确认都使用 RGB（不是 BGR）
- **填充方式不一致**：确认都使用 letterbox 填充（灰色 114）
- **类别顺序不一致**：确认 classes.txt 与训练时一致
- **输出布局 / 置信度刻度**：若 NPU 端置信度整体偏低、顶部被压缩或呈台阶状，很可能是导出含 `score_sum` 的 9 头布局阻碍了 INT8 阶段的 sigmoid 融合——改用 `[box, Sigmoid(cls)]×3` 的 6 头布局可无损修复（mAP 逐位一致）。实测见 [6 头 vs 9 头评测报告](../coco_benchmark_结果_9头vs6头/评测报告.md)
- **校准集预处理**：INT8 校准图必须与部署预处理一致（预 letterbox 灰 114）；用拉伸图校准会把置信度磨平、mAP 显著劣化

### 5.2 NPU 内存不足

**问题现象**：推理时报 "NPU memory allocation failed" 或推理中断。

**排查和解决**：

```bash
# 1. 检查当前 NPU 内存使用情况
cat /proc/driver/rockchip/npu/info

# 2. 查看系统内存
free -h

# 3. 检查是否有其他进程占用 NPU
ps aux | grep rknn

# 4. 解决方案
# 方案 A：减小输入尺寸
#   imgsz=640 -> imgsz=320

# 方案 B：使用 INT8 量化（减少模型大小约 4 倍）
#   见第四节 4.2 模型量化优化

# 方案 C：关闭其他占用 NPU 的应用
sudo killall <other_rknn_process>

# 方案 D：重启开发板释放内存
sudo reboot

```

```python
# Python 端内存监控
import resource

def get_memory_usage():
    """获取当前进程内存使用"""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print(f"Max RSS: {usage.ru_maxrss / 1024:.1f} MB")
    
    # 检查 NPU 可用内存
    try:
        import subprocess
        result = subprocess.run(
            ["cat", "/proc/driver/rockchip/npu/info"],
            capture_output=True, text=True
        )
        print(f"NPU Info:\n{result.stdout}")
    except Exception as e:
        print(f"Cannot read NPU info: {e}")

```

### 5.3 推理速度慢

**问题现象**：FPS 远低于预期。

**排查和优化**：

```python
# 1. 检查 NPU 运行频率
import subprocess

def check_npu_frequency():
    result = subprocess.run(
        ["cat", "/sys/class/devfreq/*npu*/cur_freq"],
        capture_output=True, text=True
    )
    freq_mhz = int(result.stdout.strip()) / 1000000
    print(f"NPU 频率: {freq_mhz:.0f} MHz")
    # RK3588 最高频率应为 1000 MHz
    if freq_mhz < 500:
        print("警告: NPU 频率过低，请检查电源和散热")

# 2. 检查 NPU 功耗模式
def set_npu_performance():
    """设置为高性能模式"""
    try:
        subprocess.run(
            ["echo", "performance", "|", "sudo", "tee", "/sys/class/devfreq/*npu*/governor"],
            shell=True, check=True
        )
        print("NPU 已设置为 performance 模式")
    except Exception as e:
        print(f"设置失败: {e}")

# 3. 检查 CPU 频率（NPU 推理时 CPU 不应成为瓶颈）
def check_cpu_frequency():
    import glob
    freqs = glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq")
    for f in freqs:
        with open(f) as file:
            print(f"{f}: {file.read().strip()} Hz")

# 4. 可能的优化方向
"""
优化方向:
├── 1. INT8 量化（速度提升 2-3 倍，精度损失通常 <1% mAP）
├── 2. 减小输入尺寸（320x320 比 640x640 快约 2-3 倍）
├── 3. 使用 FP16 精度（比 FP32 快约 1.5 倍，精度无损）
├── 4. 调整 NPU 频率为高性能模式
├── 5. 算子融合（确保转换时 optimization_level=2）
├── 6. 多线程流水线（采集和推理并行）
└── 7. 减少后处理中的 Python 开销（使用 NumPy 向量化操作）
"""

```

### 5.4 模型转换失败

**问题现象**：在 x86 主机上转换 ONNX 到 RKNN 时报错。

```bash
# 常见错误及解决方案

# 错误1: "Unsupported operator: XXX"
# 解决：检查 ONNX 模型中是否包含 NPU 不支持的算子
# 使用 Netron 可视化检查模型结构
# 替换不支持的算子为等效支持算子

# 错误2: "Quantization failed"
# 解决：
# 1. 检查校准数据集是否足够（建议至少 100 张）
# 2. 尝试使用 FP16 代替 INT8
# 3. 调整校准阈值

# 错误3: "Shape inference failed"
# 解决：
# 1. 确保 ONNX 模型有正确的输入 shape
# 2. 使用 onnx-simplifier 简化模型
# 3. 检查动态轴是否正确设置

# 错误4: 转换后推理结果为全零
# 解决：
# 1. 检查输入预处理是否与训练时一致
# 2. 检查归一化参数是否正确
# 3. 对比转换前后的模型输出

```

### 5.5 摄像头采集问题

**问题现象**：OpenCV 无法打开摄像头或帧率不稳定。

```python
import cv2

def setup_camera(camera_id=0):
    """
    摄像头初始化，解决常见采集问题
    """
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        # 尝试其他方式打开
        print("尝试替代方式打开摄像头...")
        # 方法1: 使用 V4L2 后端
        cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        if not cap.isOpened():
            # 方法2: 使用 GStreamer
            cap = cv2.VideoCapture(
                f"v4l2src device=/dev/video{camera_id} ! "
                f"video/x-raw,width=1280,height=720,framerate=30/1 ! "
                f"videoconvert ! appsink",
                cv2.CAP_GSTREAMER
            )
        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {camera_id}")
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲延迟
    
    # 验证设置
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"摄像头设置: {width}x{height} @ {fps} FPS")
    
    return cap


def troubleshoot_camera():
    """
    摄像头故障排查
    """
    import subprocess
    
    # 检查可用的摄像头设备
    print("可用摄像头设备:")
    result = subprocess.run(
        ["ls", "/dev/video*"],
        capture_output=True, text=True
    )
    print(result.stdout)
    
    # 检查摄像头权限
    result = subprocess.run(
        ["ls", "-la", "/dev/video0"],
        capture_output=True, text=True
    )
    print(f"摄像头权限: {result.stdout.strip()}")
    
    # 检查相关服务
    result = subprocess.run(
        ["systemctl", "is-active", "uvcvideo"],
        capture_output=True, text=True
    )
    print(f"UVC 驱动状态: {result.stdout.strip()}")


if __name__ == "__main__":
    troubleshoot_camera()
    cap = setup_camera(0)
    # 使用 cap 进行推理...
    cap.release()

```

### 5.6 交叉编译和部署问题

```bash
# 问题1: 交叉编译工具链找不到
# 解决：安装交叉编译工具链
sudo apt-get install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# 问题2: 开发板上运行时缺少动态库
# 解决：确保 rknn-toolkit2 运行时已正确安装
# 检查库文件是否存在
ls -la /usr/lib/aarch64-linux-gnu/librknnrt.so*
ls -la /usr/lib/aarch64-linux-gnu/librknn_api.so*

# 问题3: 模型文件格式错误
# 解决：确保 .rknn 文件是在与开发板相同芯片架构下转换的
# RK3588 的 .rknn 不能直接在 RK3568 上使用

# 问题4: Python 路径问题
# 解决：在开发板上设置正确的 Python 路径
export PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH

```

---

## 总结

本文详细介绍了在 Rockchip NPU 上使用 Python 部署 YOLO 模型的全流程，从硬件选型到最终部署，覆盖了实际工程中常见的问题和优化方法。核心要点如下：

1. **硬件选型**：根据性能需求和预算选择合适的 Rockchip 芯片。RK3588 适合高性能场景，RK3568 适合中端应用，RK3562 适合低成本入门。注意不同芯片的 NPU Runtime 不通用，需匹配使用。

2. **环境配置**：模型转换在 x86 主机上完成，推理在 ARM 开发板上运行。两个环境的库版本必须匹配。注意 Python 版本兼容性（建议 3.8-3.10），并正确配置动态库路径。

3. **推理部署**：核心流程为预处理 → NPU 推理 → 后处理。预处理需与训练时保持一致（letterbox 缩放、RGB 转换、归一化）。后处理需正确解码边界框坐标并映射回原始图像空间，执行 NMS 去重。

4. **性能优化**：
   - **量化**：INT8 量化可将速度提升 2-3 倍，是 NPU 部署的首选
   - **输入尺寸**：根据实际需求选择 320/640/1280
   - **算子融合**：转换时开启 optimization_level=2，自动融合 Conv+BN 等算子
   - **多线程流水线**：采集和推理并行，最大化吞吐量
   - **内存优化**：预分配缓冲区，复用内存，减少动态分配

5. **调试方法**：当推理结果与 PC 端不一致时，逐一排查预处理、归一化、颜色空间、类别顺序等各个环节。使用 NPU 状态检查工具监控驱动、频率、温度等关键指标。

6. **常见问题**：本文涵盖了模型转换失败、NPU 内存不足、推理速度慢、摄像头采集异常等典型问题的排查和解决方案。实际部署中如遇新问题，建议参考 [RKNN-Toolkit2 官方文档](https://github.com/airockchip/rknn-toolkit2) 和开发者社区。

通过合理选型、规范配置和针对性优化，YOLO 模型在 Rockchip NPU 上可以达到实时推理性能，满足边缘 AI 应用的实际需求。

---

*参考资料：*
- *[RKNN-Toolkit2 Python API](https://github.com/airockchip/rknn-toolkit2)*
- *[RKNN Runtime Documentation](https://github.com/airockchip/rknn-toolkit2/blob/master/docs/en/03_rknn_runtime/)*
- *[Rockchip NPU Driver Documentation](https://github.com/airockchip/rknn-toolkit2/blob/master/docs/en/02_rknn_toolkit2_installation/)*
- *[Ultralytics YOLO Documentation](https://docs.ultralytics.com/)*

---

> **📌 系列导航**：[← 上一篇：Yolo模型的转换与rknn-toolkit相关工具链的使用](Yolo模型的转换与rknn-toolkit相关工具链的使用.md) · [📖 导读目录](README.md) · [下一篇：模型在npu的cpp部署 →](模型在npu的cpp部署.md)
