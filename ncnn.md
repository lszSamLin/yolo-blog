# NCNN 部署指南（Ultralytics 官方文档归档）

> **系列定位**：本文是「部署格式专题」之一 —— 移动端 / 嵌入式通用推理框架（腾讯开源）。内容为 Ultralytics 官方英文文档完整归档，与中文实操文章互补。总目录见 [README 导读](README.md)。
>
> - 中文实操路线：[yolo模型推理与部署详解](YOLO模型推理与部署详解.md)
> - 关联阅读：[模型量化深度解析](模型量化深度解析.md) · [onnx.md](onnx.md)（转换链路起点）

---

（以下为官方文档原文归档）

## Ultralytics YOLO NCNN Export

Deploying [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models on devices with limited computational power, such as mobile or embedded systems, requires careful format selection. Using an optimized format ensures that even resource-constrained devices can handle advanced computer vision tasks efficiently.

Exporting to NCNN format allows you to optimize your [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) models for lightweight device-based applications. This guide covers how to convert your models to NCNN format for improved performance on mobile and embedded devices.

## Why Export to NCNN?

![NCNN high-performance neural network inference framework](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ncnn-overview.avif)

The [NCNN](https://github.com/Tencent/ncnn) framework, developed by Tencent, is a high-performance [neural network](https://www.ultralytics.com/glossary/neural-network-nn) inference computing framework optimized specifically for mobile platforms, including mobile phones, embedded devices, and IoT devices. NCNN is compatible with a wide range of platforms, including Linux, Android, iOS, and macOS.

NCNN is known for its fast processing speed on mobile CPUs and enables rapid deployment of [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models to mobile platforms, making it an excellent choice for building AI-powered applications.

## Key Features of NCNN Models

NCNN models provide several key features that enable on-device [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml), helping developers deploy models on mobile, embedded, and edge devices:

- **Efficient and High-Performance**: NCNN models are lightweight and optimized for mobile and embedded devices like Raspberry Pi with limited resources, while maintaining high [accuracy](https://www.ultralytics.com/glossary/accuracy) on computer vision tasks.
- **Quantization**: NCNN supports quantization, a technique that reduces the [precision](https://www.ultralytics.com/glossary/precision) of model weights and activations to improve performance and reduce memory footprint.
- **Compatibility**: NCNN models are compatible with popular deep learning frameworks including [TensorFlow](https://www.tensorflow.org/), [Caffe](https://caffe.berkeleyvision.org/), and [ONNX](https://onnx.ai/), allowing developers to leverage existing models and workflows.
- **Ease of Use**: NCNN provides user-friendly tools for converting models between formats, ensuring smooth interoperability across different development environments.
- **Vulkan GPU Acceleration**: NCNN supports Vulkan for GPU-accelerated inference across multiple vendors including AMD, Intel, and other non-NVIDIA GPUs, enabling high-performance deployment on a wider range of hardware.

## Deployment Options with NCNN

NCNN models are compatible with a variety of deployment platforms:

- **Mobile Deployment**: Optimized for Android and iOS, enabling seamless integration into mobile applications for efficient on-device inference.
- **Embedded Systems and IoT Devices**: Ideal for resource-constrained devices like Raspberry Pi and NVIDIA Jetson. If standard inference on a Raspberry Pi with the [Ultralytics Guide](https://docs.ultralytics.com/guides/raspberry-pi) is insufficient, NCNN can provide significant performance improvements.
- **Desktop and Server Deployment**: Supports deployment across Linux, Windows, and macOS for development, training, and evaluation workflows.

## Vulkan GPU Acceleration

NCNN supports GPU acceleration through Vulkan, enabling high-performance inference on a wide range of GPUs including AMD, Intel, and other non-NVIDIA graphics cards. This is particularly useful for:

- **Cross-Vendor GPU Support**: Unlike CUDA, which is limited to NVIDIA GPUs, Vulkan works across multiple GPU vendors.
- **Multi-GPU Systems**: Select a specific Vulkan device in systems with multiple GPUs using `device="vulkan:0"`, `device="vulkan:1"`, etc.
- **Edge and Desktop Deployments**: Leverage GPU acceleration on devices where CUDA is not available.

To use Vulkan acceleration, specify the Vulkan device when running inference:

Vulkan Inference

```python
from ultralytics import YOLO

# Load the exported NCNN model

ncnn_model = YOLO("./yolo26n_ncnn_model")

# Run inference with Vulkan GPU acceleration (first Vulkan device)

results = ncnn_model("https://ultralytics.com/images/bus.jpg", device="vulkan:0")

# Use second Vulkan device in multi-GPU systems

results = ncnn_model("https://ultralytics.com/images/bus.jpg", device="vulkan:1")
```

Vulkan Requirements

Ensure you have Vulkan drivers installed for your GPU. Most modern GPU drivers include Vulkan support by default. You can verify Vulkan availability using tools like `vulkaninfo` on Linux or the Vulkan SDK on Windows.

## Supported Tasks

NCNN export supports all seven Ultralytics tasks. Semantic segmentation and depth estimation are available only with YOLO26, the only family that ships those heads.

| Task | [YOLOv8](https://docs.ultralytics.com/models/yolov8) | [YOLO11](https://docs.ultralytics.com/models/yolo11) | [YOLO26](https://docs.ultralytics.com/models/yolo26) |
| --- | --- | --- | --- |
| [Detect](https://docs.ultralytics.com/tasks/detect) | ✅ | ✅ | ✅ |
| [Segment](https://docs.ultralytics.com/tasks/segment) | ✅ | ✅ | ✅ |
| [Semantic](https://docs.ultralytics.com/tasks/semantic) | ❌ | ❌ | ✅ |
| [Depth](https://docs.ultralytics.com/tasks/depth) | ❌ | ❌ | ✅ |
| [Classify](https://docs.ultralytics.com/tasks/classify) | ✅ | ✅ | ✅ |
| [Pose](https://docs.ultralytics.com/tasks/pose) | ✅ | ✅ | ✅ |
| [OBB](https://docs.ultralytics.com/tasks/obb) | ✅ | ✅ | ✅ |

## Export to NCNN: Converting Your YOLO26 Model

You can expand model compatibility and deployment flexibility by converting YOLO26 models to NCNN format.

### Installation

To install the required packages, run:

Installation

```bash
# Install the required package for YOLO26
pip install ultralytics
```

For detailed instructions and best practices, see the [Ultralytics Installation guide](https://docs.ultralytics.com/quickstart). If you encounter any difficulties, consult our [Common Issues guide](https://docs.ultralytics.com/guides/yolo-common-issues) for solutions.

### Usage

All [Ultralytics YOLO26 models](https://docs.ultralytics.com/models) are designed to support export out of the box, making it easy to integrate them into your preferred deployment workflow. You can [view the full list of supported export formats and configuration options](https://docs.ultralytics.com/modes/export) to choose the best setup for your application.

The NCNN format supports the [Export](https://docs.ultralytics.com/modes/export), [Predict](https://docs.ultralytics.com/modes/predict), and [Validate](https://docs.ultralytics.com/modes/val) modes. Export your model, then load the exported model to run inference or validate its accuracy.

Export

```python
from ultralytics import YOLO

# Load a YOLO26 model

model = YOLO("yolo26n.pt")

# Export the model to NCNN format

model.export(format="ncnn")  # creates '/yolo26n_ncnn_model'
```

Predict

```python
from ultralytics import YOLO

# Load the exported NCNN model

model = YOLO("./yolo26n_ncnn_model")

# Run inference

results = model("https://ultralytics.com/images/bus.jpg")
```

Validate

```python
from ultralytics import YOLO

# Load the exported NCNN model

model = YOLO("./yolo26n_ncnn_model")

# Validate accuracy on the COCO8 dataset

metrics = model.val(data="coco8.yaml")
```

### Export Arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `format` | `str` | `'ncnn'` | Target format for the exported model, defining compatibility with various deployment environments. |
| `imgsz` | `int` or `tuple` | `640` | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions. |
| `quantize` | `int` or `str` | `None` | Quantization precision: `16` (FP16) reduces model size and can speed up inference; `32`/unset is FP32. Replaces the deprecated `half` flag. |
| `batch` | `int` | `1` | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode. |
| `device` | `str` | `None` | Specifies the device for exporting: GPU (`device=0`), CPU (`device=cpu`), MPS for Apple silicon (`device=mps`). |

For more details about the export process, visit the [Ultralytics documentation page on exporting](https://docs.ultralytics.com/modes/export).

## Deploying Exported YOLO26 NCNN Models

After exporting your Ultralytics YOLO26 models to NCNN format, you can deploy them using the `YOLO("yolo26n_ncnn_model/")` method as shown in the usage example above. For platform-specific deployment instructions, see the following resources:

- **[Android](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android)**: Build and integrate NCNN models for [object detection](https://www.ultralytics.com/glossary/object-detection) in Android applications.
- **[macOS](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-macos)**: Deploy NCNN models on macOS systems.
- **[Linux](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux)**: Deploy NCNN models on Linux devices including Raspberry Pi and similar embedded systems.
- **[Windows x64](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-windows-x64-using-visual-studio-community-2017)**: Deploy NCNN models on Windows x64 using Visual Studio.

## Summary

This guide covered exporting Ultralytics YOLO26 models to NCNN format for improved efficiency and speed on resource-constrained devices.

For additional details, refer to the [official NCNN documentation](https://ncnn.readthedocs.io/en/latest/index.html). For other export options, visit our [integration guide page](https://docs.ultralytics.com/integrations).

---

> **📌 系列导航**：[📖 返回导读目录](README.md) · [中文实操：yolo模型推理与部署详解 →](YOLO模型推理与部署详解.md)
