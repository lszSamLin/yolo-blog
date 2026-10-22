# ONNX 部署指南（Ultralytics 官方文档归档）

> **系列定位**：本文是「部署格式专题」之一 —— 跨框架交换格式，几乎所有转换链路（RKNN / TensorRT / OpenVINO）的起点。内容为 Ultralytics 官方英文文档完整归档，与中文实操文章互补。总目录见 [README 导读](README.md)。
>
> - 中文实操路线：[Yolo模型的转换与rknn-toolkit相关工具链的使用](YOLO模型的转换与rknn-toolkit相关工具链的使用.md)
> - 关联阅读：[yolo模型推理与部署详解](YOLO模型推理与部署详解.md) · [模型量化深度解析](模型量化深度解析.md)

---

（以下为官方文档原文归档）

## ONNX Export for YOLO26 Models

~43% faster inference.

- Exporting the Ultralytics YOLO26 model to ONNX can deliver up to a 43% boost in inference speed, enabling faster and more efficient deployment.

Often, when deploying [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models, you'll need a model format that's both flexible and compatible with multiple platforms.

Exporting [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) models to ONNX format streamlines deployment and ensures optimal performance across various environments. This guide will show you how to easily convert your YOLO26 models to ONNX and enhance their scalability and effectiveness in real-world applications.

**Watch:** Ultralytics YOLO26 vs Ultralytics YOLO11 ONNX Inference Test | ~43% Faster Inference with YOLO26 🚀

## ONNX and ONNX Runtime

[ONNX](https://onnx.ai/), which stands for Open [Neural Network](https://www.ultralytics.com/glossary/neural-network-nn) Exchange, is a community project that Facebook and Microsoft initially developed. The ongoing development of ONNX is a collaborative effort supported by various organizations like IBM, Amazon (through AWS), and Google. The project aims to create an open file format designed to represent [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) models in a way that allows them to be used across different AI frameworks and hardware.

ONNX models can be used to transition between different frameworks seamlessly. For instance, a [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) model trained in PyTorch can be exported to ONNX format and then easily imported into TensorFlow.

![ONNX model portability across deep learning frameworks](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/onnx-model-portability.avif)

Alternatively, ONNX models can be used with ONNX Runtime. [ONNX Runtime](https://onnxruntime.ai/) is a versatile cross-platform accelerator for machine learning models that is compatible with frameworks like PyTorch, [TensorFlow](https://www.ultralytics.com/glossary/tensorflow), scikit-learn, etc.

ONNX Runtime optimizes the execution of ONNX models by leveraging hardware-specific capabilities. This optimization allows the models to run efficiently and with high performance on various hardware platforms, including CPUs, GPUs, and specialized accelerators.

![ONNX Runtime cross-platform inference acceleration](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/onnx-and-onnx-runtime.avif)

Whether used independently or in tandem with ONNX Runtime, ONNX provides a flexible solution for machine learning [model deployment](https://www.ultralytics.com/glossary/model-deployment) and compatibility.

## Key Features of ONNX Models

The ability of ONNX to handle various formats can be attributed to the following key features:

- **Common Model Representation**: ONNX defines a common set of operators (like convolutions, layers, etc.) and a standard data format. When a model is converted to ONNX format, its architecture and weights are translated into this common representation. This uniformity ensures that the model can be understood by any framework that supports ONNX.
- **Versioning and Backward Compatibility**: ONNX maintains a versioning system for its operators. This ensures that even as the standard evolves, models created in older versions remain usable. Backward compatibility is a crucial feature that prevents models from becoming obsolete quickly.
- **Graph-based Model Representation**: ONNX represents models as computational graphs. This graph-based structure is a universal way of representing machine learning models, where nodes represent operations or computations, and edges represent the tensors flowing between them. This format is easily adaptable to various frameworks which also represent models as graphs.
- **Tools and Ecosystem**: There is a rich ecosystem of tools around ONNX that assist in model conversion, visualization, and optimization. These tools make it easier for developers to work with ONNX models and to convert models between different frameworks seamlessly.

## Common Usage of ONNX

Before we jump into how to export YOLO26 models to the ONNX format, let's take a look at where ONNX models are usually used.

### CPU Deployment

ONNX models are often deployed on CPUs due to their compatibility with ONNX Runtime. This runtime is optimized for CPU execution. It significantly improves inference speed and makes real-time CPU deployments feasible.

### Supported Deployment Options

While ONNX models are commonly used on CPUs, they can also be deployed on the following platforms:

- **GPU Acceleration**: ONNX fully supports GPU acceleration, particularly NVIDIA CUDA. For native training and inference on AMD GPUs, use [PyTorch ROCm](https://docs.ultralytics.com/integrations/amd); an ONNX file still requires an AMD-compatible runtime.
- **Edge and Mobile Devices**: ONNX extends to edge and mobile devices, perfect for on-device and real-time inference scenarios. It's lightweight and compatible with edge hardware, and serves as the basis for vendor NPU formats such as [Huawei Ascend](https://docs.ultralytics.com/integrations/ascend), [Qualcomm QNN](https://docs.ultralytics.com/integrations/qnn) for Snapdragon devices, and [RKNN](https://docs.ultralytics.com/integrations/rockchip-rknn) for Rockchip NPUs.
- **Web Browsers**: ONNX can run directly in web browsers, powering interactive and dynamic web-based AI applications.

## Supported Tasks

ONNX export supports all seven Ultralytics tasks. Semantic segmentation and depth estimation are available only with YOLO26, the only family that ships those heads.

| Task | [YOLOv8](https://docs.ultralytics.com/models/yolov8) | [YOLO11](https://docs.ultralytics.com/models/yolo11) | [YOLO26](https://docs.ultralytics.com/models/yolo26) |
| --- | --- | --- | --- |
| [Detect](https://docs.ultralytics.com/tasks/detect) | ✅ | ✅ | ✅ |
| [Segment](https://docs.ultralytics.com/tasks/segment) | ✅ | ✅ | ✅ |
| [Semantic](https://docs.ultralytics.com/tasks/semantic) | ❌ | ❌ | ✅ |
| [Depth](https://docs.ultralytics.com/tasks/depth) | ❌ | ❌ | ✅ |
| [Classify](https://docs.ultralytics.com/tasks/classify) | ✅ | ✅ | ✅ |
| [Pose](https://docs.ultralytics.com/tasks/pose) | ✅ | ✅ | ✅ |
| [OBB](https://docs.ultralytics.com/tasks/obb) | ✅ | ✅ | ✅ |

## Exporting YOLO26 Models to ONNX

You can expand model compatibility and deployment flexibility by converting YOLO26 models to ONNX format. [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26) provides a straightforward export process that can significantly enhance your model's performance across different platforms.

### Installation

To install the required package, run:

Installation

```bash
# Install the required package for YOLO26
pip install ultralytics
```

For detailed instructions and best practices related to the installation process, check our [YOLO26 Installation guide](https://docs.ultralytics.com/quickstart). While installing the required packages for YOLO26, if you encounter any difficulties, consult our [Common Issues guide](https://docs.ultralytics.com/guides/yolo-common-issues) for solutions and tips.

### Usage

Before diving into the usage instructions, be sure to check out the range of [YOLO26 models offered by Ultralytics](https://docs.ultralytics.com/models). This will help you choose the most appropriate model for your project requirements.

The ONNX format supports the [Export](https://docs.ultralytics.com/modes/export), [Predict](https://docs.ultralytics.com/modes/predict), and [Validate](https://docs.ultralytics.com/modes/val) modes. Export your model, then load the exported model to run inference or validate its accuracy.

Export

```python
from ultralytics import YOLO

# Load a YOLO26 model

model = YOLO("yolo26n.pt")

# Export the model to ONNX format

model.export(format="onnx")  # creates 'yolo26n.onnx'

# Export an INT8-quantized ONNX model with calibration data

model.export(format="onnx", quantize=8, data="coco8.yaml")  # creates 'yolo26n_int8.onnx'
```

Predict

```python
from ultralytics import YOLO

# Load the exported ONNX model

model = YOLO("yolo26n.onnx")

# Run inference

results = model("https://ultralytics.com/images/bus.jpg")
```

Validate

```python
from ultralytics import YOLO

# Load the exported ONNX model

model = YOLO("yolo26n.onnx")

# Validate accuracy on the COCO8 dataset

metrics = model.val(data="coco8.yaml")
```

### Export Arguments

When exporting your YOLO26 model to ONNX format, you can customize the process using various arguments to optimize for your specific deployment needs:

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `format` | `str` | `'onnx'` | Target format for the exported model, defining compatibility with various deployment environments. |
| `imgsz` | `int` or `tuple` | `640` | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions. |
| `quantize` | `int` or `str` | `None` | Quantization precision: `16` (FP16) or `8` (INT8 static quantization with ONNX Runtime using calibration images from `data`, producing an `_int8.onnx` model); `32`/unset is FP32. Replaces the deprecated `half`/`int8` flags. |
| `data` | `str` | `None` | Dataset YAML used for INT8 calibration; classification instead takes a dataset directory or a built-in dataset name. If omitted with `quantize=8`, Ultralytics selects the default calibration dataset for the model task. |
| `fraction` | `float` | `1.0` | Fraction of calibration images to use for INT8 quantization. |
| `dynamic` | `bool` | `False` | Allows dynamic input sizes, enhancing flexibility in handling varying image dimensions. |
| `simplify` | `bool` | `True` | Simplifies the model graph with `onnxslim`, potentially improving performance and compatibility. |
| `opset` | `int` | `None` | Specifies the ONNX opset version for compatibility with different ONNX parsers and runtimes. If not set, uses the latest supported version. |
| `nms` | `bool` | `False` | Adds Non-Maximum Suppression (NMS), essential for accurate and efficient detection post-processing. |
| `batch` | `int` | `1` | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode. |
| `device` | `str` | `None` | Specifies the device for exporting: GPU (`device=0`), CPU (`device=cpu`), MPS for Apple silicon (`device=mps`). |

For more details about the export process, visit the [Ultralytics documentation page on exporting](https://docs.ultralytics.com/modes/export).

## Deploying Exported YOLO26 ONNX Models

Once you've successfully exported your Ultralytics YOLO26 models to ONNX format, the next step is deploying these models in various environments. For detailed instructions on deploying your ONNX models, take a look at the following resources:

- **[ONNX Runtime Python API Documentation](https://onnxruntime.ai/docs/api/python/api_summary.html)**: This guide provides essential information for loading and running ONNX models using ONNX Runtime.
- **[Deploying on Edge Devices](https://onnxruntime.ai/docs/tutorials/iot-edge/)**: Check out this docs page for different examples of deploying ONNX models on edge.
- **[ONNX Tutorials on GitHub](https://github.com/onnx/tutorials)**: A collection of comprehensive tutorials that cover various aspects of using and implementing ONNX models in different scenarios.
- **[Triton Inference Server](https://docs.ultralytics.com/guides/triton-inference-server)**: Learn how to deploy your ONNX models with NVIDIA's Triton Inference Server for high-performance, scalable deployments.

## Summary

In this guide, you've learned how to export Ultralytics YOLO26 models to ONNX format to increase their interoperability and performance across various platforms. You were also introduced to the ONNX Runtime and ONNX deployment options.

ONNX export is just one of many [export formats](https://docs.ultralytics.com/modes/export) supported by Ultralytics YOLO26, allowing you to deploy your models in virtually any environment. Depending on your specific needs, you might also want to explore other export options like [TensorRT](https://docs.ultralytics.com/integrations/tensorrt) for maximum GPU performance or [CoreML](https://docs.ultralytics.com/integrations/coreml) for Apple devices.

For further details on usage, visit the [ONNX official documentation](https://onnx.ai/onnx/intro/).

Also, if you'd like to know more about other Ultralytics YOLO26 integrations, visit our [integration guide page](https://docs.ultralytics.com/integrations). You'll find plenty of useful resources and insights there.

---

> **📌 系列导航**：[📖 返回导读目录](README.md) · [中文实操：Yolo模型的转换与rknn-toolkit相关工具链的使用 →](YOLO模型的转换与rknn-toolkit相关工具链的使用.md)
