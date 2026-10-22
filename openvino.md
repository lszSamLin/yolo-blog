# OpenVINO 部署指南（Ultralytics 官方文档归档）

> **系列定位**：本文是「部署格式专题」之一 —— Intel 平台（x86 CPU / 核显 / NPU）。内容为 Ultralytics 官方英文文档完整归档，与中文实操文章互补。总目录见 [README 导读](README.md)。
>
> - 中文实操路线：[yolo模型推理与部署详解](YOLO模型推理与部署详解.md)
> - 关联阅读：[模型量化深度解析](模型量化深度解析.md) · [onnx.md](onnx.md)（转换链路起点）

---

（以下为官方文档原文归档）

## Intel OpenVINO Export

![OpenVINO Intel AI inference toolkit](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/openvino-ecosystem.avif)

In this guide, we cover exporting YOLO26 models to the [OpenVINO](https://docs.openvino.ai/) format, which can provide up to 3x [CPU](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html) speedup, as well as accelerating YOLO inference on Intel [GPU](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html) and [NPU](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html) hardware.

OpenVINO, short for Open Visual Inference & [Neural Network](https://www.ultralytics.com/glossary/neural-network-nn) Optimization toolkit, is a comprehensive toolkit for optimizing and deploying AI inference models. Even though the name contains Visual, OpenVINO also supports various additional tasks including language, audio, time series, etc.

**Watch:** How to Export Ultralytics YOLO26 to Intel OpenVINO Format for Faster Inference 🚀

## Supported Tasks

OpenVINO export supports all seven Ultralytics tasks. Semantic segmentation and depth estimation are available only with YOLO26, the only family that ships those heads.

| Task | [YOLOv8](https://docs.ultralytics.com/models/yolov8) | [YOLO11](https://docs.ultralytics.com/models/yolo11) | [YOLO26](https://docs.ultralytics.com/models/yolo26) |
| --- | --- | --- | --- |
| [Detect](https://docs.ultralytics.com/tasks/detect) | ✅ | ✅ | ✅ |
| [Segment](https://docs.ultralytics.com/tasks/segment) | ✅ | ✅ | ✅ |
| [Semantic](https://docs.ultralytics.com/tasks/semantic) | ❌ | ❌ | ✅ |
| [Depth](https://docs.ultralytics.com/tasks/depth) | ❌ | ❌ | ✅ |
| [Classify](https://docs.ultralytics.com/tasks/classify) | ✅ | ✅ | ✅ |
| [Pose](https://docs.ultralytics.com/tasks/pose) | ✅ | ✅ | ✅ |
| [OBB](https://docs.ultralytics.com/tasks/obb) | ✅ | ✅ | ✅ |

## Usage Examples

The OpenVINO format supports the [Export](https://docs.ultralytics.com/modes/export), [Predict](https://docs.ultralytics.com/modes/predict), and [Validate](https://docs.ultralytics.com/modes/val) modes. Export your model, then load the exported model to run inference or validate its accuracy on Intel CPU, integrated/discrete GPU, or NPU.

Export

```python
from ultralytics import YOLO

# Load a YOLO26 model

model = YOLO("yolo26n.pt")

# Export the model to OpenVINO format

model.export(format="openvino")  # creates 'yolo26n_openvino_model/'
```

Predict

```python
from ultralytics import YOLO

# Load the exported OpenVINO model

model = YOLO("yolo26n_openvino_model/")

# Run inference

results = model("https://ultralytics.com/images/bus.jpg")

# Run inference on a specific device, available devices: ["intel:gpu", "intel:npu", "intel:cpu"]

results = model("https://ultralytics.com/images/bus.jpg", device="intel:gpu")
```

Validate

```python
from ultralytics import YOLO

# Load the exported OpenVINO model

model = YOLO("yolo26n_openvino_model/")

# Validate accuracy on the COCO8 dataset

metrics = model.val(data="coco8.yaml")
```

## Export Arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `format` | `str` | `'openvino'` | Target format for the exported model, defining compatibility with various deployment environments. |
| `imgsz` | `int` or `tuple` | `640` | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions. |
| `quantize` | `int` or `str` | `None` | Quantization precision: `16` (FP16) or `8` (INT8/PTQ; needs calibration `data`/`fraction`); `32`/unset is FP32. Replaces the deprecated `half`/`int8` flags. |
| `dynamic` | `bool` | `False` | Allows dynamic input sizes, enhancing flexibility in handling varying image dimensions. |
| `nms` | `bool` | `False` | Adds Non-Maximum Suppression (NMS), essential for accurate and efficient detection post-processing. |
| `batch` | `int` | `1` | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode. |
| `data` | `str` | `None` | Path to the [dataset](https://docs.ultralytics.com/datasets) YAML, essential for quantization; classification instead takes a dataset directory or a built-in dataset name. If omitted with `quantize=8`, Ultralytics selects the default calibration dataset for the model task. |
| `fraction` | `float` | `1.0` | Specifies the fraction of the dataset to use for INT8 quantization calibration. Allows for calibrating on a subset of the full dataset, useful for experiments or when resources are limited. If not specified with INT8 enabled, the full dataset will be used. |

For more details about the export process, visit the [Ultralytics documentation page on exporting](https://docs.ultralytics.com/modes/export).

Warning

OpenVINO™ is compatible with most Intel® processors but to ensure optimal performance:

1. Verify OpenVINO™ support
   Check whether your Intel® chip is officially supported by OpenVINO™ using [Intel's compatibility list](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html).
2. Identify your accelerator
   Determine if your processor includes an integrated NPU (Neural Processing Unit) or GPU (integrated GPU) by consulting [Intel's hardware guide](https://www.intel.com/content/www/us/en/support/articles/000097597/processors.html).
3. Install the latest drivers
   If your chip supports an NPU or GPU but OpenVINO™ isn't detecting it, you may need to install or update the associated drivers. Follow the [driver‑installation instructions](https://medium.com/openvino-toolkit/how-to-run-openvino-on-a-linux-ai-pc-52083ce14a98) to enable full acceleration.

By following these three steps, you can ensure OpenVINO™ runs optimally on your Intel® hardware.

## Benefits of OpenVINO

1. **Performance**: OpenVINO delivers high-performance inference by utilizing the power of Intel CPUs, integrated and discrete GPUs, and FPGAs.
2. **Support for Heterogeneous Execution**: OpenVINO provides an API to write once and deploy on any supported Intel hardware (CPU, GPU, FPGA, VPU, etc.).
3. **Model Optimizer**: OpenVINO provides a Model Optimizer that imports, converts, and optimizes models from popular [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) frameworks such as PyTorch, [TensorFlow](https://www.ultralytics.com/glossary/tensorflow), TensorFlow Lite, Keras, ONNX, PaddlePaddle, and Caffe.
4. **Ease of Use**: The toolkit comes with a large collection of [tutorial notebooks](https://github.com/openvinotoolkit/openvino_notebooks) (including [YOLO26 optimization](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov26-optimization)) teaching different aspects of the toolkit.

## OpenVINO Export Structure

When you export a model to OpenVINO format, it results in a directory containing the following:

1. **XML file**: Describes the network topology.
2. **BIN file**: Contains the weights and biases binary data.
3. **Mapping file**: Holds mapping of original model output tensors to OpenVINO tensor names.

You can use these files to run inference with the OpenVINO Inference Engine.

## Using OpenVINO Export in Deployment

Once your model is successfully exported to the OpenVINO format, you have two primary options for running inference:

1. Use the `ultralytics` package, which provides a high-level API and wraps the OpenVINO Runtime.
2. Use the native `openvino` package for more advanced or customized control over inference behavior.

### Inference with Ultralytics

The ultralytics package allows you to easily run inference using the exported OpenVINO model via the predict method. You can also specify the target device (e.g., `intel:gpu`, `intel:npu`, `intel:cpu`) using the device argument.

```python
from ultralytics import YOLO

# Load the exported OpenVINO model

ov_model = YOLO("yolo26n_openvino_model/")  # the path of your exported OpenVINO model

# Run inference with the exported model

ov_model.predict(device="intel:gpu")  # specify the device you want to run inference on
```

This approach is ideal for fast prototyping or deployment when you don't need full control over the inference pipeline.

### Inference with OpenVINO Runtime

The OpenVINO Runtime provides a unified API for inference across all supported Intel hardware. It also provides advanced capabilities like load balancing across Intel hardware and asynchronous execution. For more information on running inference, refer to the [YOLO26 notebooks](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov26-optimization).

Remember, you'll need the XML and BIN files as well as any application-specific settings like input size, scale factor for normalization, etc., to correctly set up and use the model with the Runtime.

In your deployment application, you would typically do the following steps:

1. Initialize OpenVINO by creating `core = Core()`.
2. Load the model using the `core.read_model()` method.
3. Compile the model using the `core.compile_model()` function.
4. Prepare the input (image, text, audio, etc.).
5. Run inference using `compiled_model(input_data)`.

For more detailed steps and code snippets, refer to the [OpenVINO documentation](https://docs.openvino.ai/) or [API tutorial](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/openvino-api/openvino-api.ipynb).

## OpenVINO YOLO26 Benchmarks

The Ultralytics team benchmarked YOLO26 across various model formats and [precision](https://www.ultralytics.com/glossary/precision), evaluating speed and accuracy on different Intel devices compatible with OpenVINO.

Note

- The benchmarking results below are for reference and might vary based on the exact hardware and software configuration of a system, as well as the current workload of the system at the time the benchmarks are run.
- All benchmarks were run with `openvino` Python package version [2026.2.1](https://pypi.org/project/openvino/2026.2.1).
- YOLO26 models on NPU are only supported on Intel® Core™ Ultra™ systems with 2xxV series and 3xx series and above.

### Intel® Core™ Ultra

The Intel® Core™ Ultra™ series represents a new benchmark in high-performance computing, engineered to meet the evolving demands of modern users—from gamers and creators to professionals leveraging AI. This next-generation lineup is more than a traditional CPU series; it combines powerful CPU cores, integrated high-performance GPU capabilities, and a dedicated Neural Processing Unit (NPU) within a single chip, offering a unified solution for diverse and intensive computing workloads.

At the heart of the Intel® Core Ultra™ architecture is a hybrid design that enables exceptional performance across traditional processing tasks, GPU-accelerated workloads, and AI-driven operations. The inclusion of the NPU enhances on-device AI inference, enabling faster, more efficient machine learning and data processing across a wide range of applications.

The Core Ultra™ family includes various models tailored for different performance needs, with options ranging from energy-efficient designs to high-power variants marked by the "H" designation—ideal for laptops and compact form factors that demand serious computing power. Across the lineup, users benefit from the synergy of CPU, GPU, and NPU integration, delivering remarkable efficiency, responsiveness, and multitasking capabilities.

As part of Intel's ongoing innovation, the Core Ultra™ series sets a new standard for future-ready computing. With multiple models available and more on the horizon, this series underscores Intel's commitment to delivering cutting-edge solutions for the next generation of intelligent, AI-enhanced devices.

Benchmarks below run on Intel® Core™ Ultra™ X7 358H, Intel® Core™ Ultra™ 7 258V and Intel® Core™ Ultra™ 7 155H at FP32, FP16 and INT8 precision.

#### Intel® Core™ Ultra™ X7 358H

Benchmarks

![Intel Core Ultra GPU benchmarks](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/openvino-ultraX7-358H-gpu.avif)

Detailed Benchmark Results

| Model | Format | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO26n | PyTorch (CPU) | FP32 | ✅ | 5.3 | 0.4765 | 29.28 |
| YOLO26n | OpenVINO | FP32 | ✅ | 9.7 | 0.4762 | 4.09 |
| YOLO26n | OpenVINO | FP16 | ✅ | 5.1 | 0.4762 | 4.03 |
| YOLO26n | OpenVINO | INT8 | ✅ | 3.2 | 0.4634 | 4.24 |
| YOLO26s | PyTorch (CPU) | FP32 | ✅ | 19.5 | 0.5703 | 59.32 |
| YOLO26s | OpenVINO | FP32 | ✅ | 36.7 | 0.5616 | 5.08 |
| YOLO26s | OpenVINO | FP16 | ✅ | 18.7 | 0.5616 | 5.14 |
| YOLO26s | OpenVINO | INT8 | ✅ | 10.0 | 0.5462 | 4.47 |
| YOLO26m | PyTorch (CPU) | FP32 | ✅ | 42.2 | 0.6196 | 143.21 |
| YOLO26m | OpenVINO | FP32 | ✅ | 78.4 | 0.6166 | 7.1 |
| YOLO26m | OpenVINO | FP16 | ✅ | 39.5 | 0.6166 | 7.06 |
| YOLO26m | OpenVINO | INT8 | ✅ | 20.6 | 0.6055 | 5.61 |
| YOLO26l | PyTorch (CPU) | FP32 | ✅ | 50.7 | 0.6215 | 181.16 |
| YOLO26l | OpenVINO | FP32 | ✅ | 95.3 | 0.6205 | 8.45 |
| YOLO26l | OpenVINO | FP16 | ✅ | 48.1 | 0.6205 | 8.49 |
| YOLO26l | OpenVINO | INT8 | ✅ | 25.2 | 0.5938 | 6.25 |
| YOLO26x | PyTorch (CPU) | FP32 | ✅ | 113.2 | 0.6512 | 379.75 |
| YOLO26x | OpenVINO | FP32 | ✅ | 213.3 | 0.6568 | 13.03 |
| YOLO26x | OpenVINO | FP16 | ✅ | 107.1 | 0.6568 | 12.98 |
| YOLO26x | OpenVINO | INT8 | ✅ | 54.8 | 0.6385 | 8.74 |

#### Intel® Core™ Ultra™ 7 258V

Benchmarks

![Intel Core Ultra GPU benchmarks](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/openvino-ultra7-258V-gpu.avif)

Detailed Benchmark Results

| Model | Format | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO26n | PyTorch (CPU) | FP32 | ✅ | 5.3 | 0.4765 | 27.33 |
| YOLO26n | OpenVINO | FP32 | ✅ | 9.6 | 0.4764 | 3.33 |
| YOLO26n | OpenVINO | FP16 | ✅ | 5.1 | 0.4764 | 3.34 |
| YOLO26n | OpenVINO | INT8 | ✅ | 3.2 | 0.4634 | 3.46 |
| YOLO26s | PyTorch (CPU) | FP32 | ✅ | 19.5 | 0.5703 | 60.2 |
| YOLO26s | OpenVINO | FP32 | ✅ | 36.7 | 0.5616 | 4.66 |
| YOLO26s | OpenVINO | FP16 | ✅ | 18.6 | 0.5616 | 4.65 |
| YOLO26s | OpenVINO | INT8 | ✅ | 10.0 | 0.5462 | 3.99 |
| YOLO26m | PyTorch (CPU) | FP32 | ✅ | 42.2 | 0.6196 | 152.87 |
| YOLO26m | OpenVINO | FP32 | ✅ | 78.4 | 0.6168 | 9.03 |
| YOLO26m | OpenVINO | FP16 | ✅ | 39.5 | 0.6168 | 8.98 |
| YOLO26m | OpenVINO | INT8 | ✅ | 20.6 | 0.6055 | 5.7 |
| YOLO26l | PyTorch (CPU) | FP32 | ✅ | 50.7 | 0.6215 | 201.57 |
| YOLO26l | OpenVINO | FP32 | ✅ | 95.3 | 0.6203 | 11.32 |
| YOLO26l | OpenVINO | FP16 | ✅ | 48.1 | 0.6203 | 11.36 |
| YOLO26l | OpenVINO | INT8 | ✅ | 25.2 | 0.5938 | 7.45 |
| YOLO26x | PyTorch (CPU) | FP32 | ✅ | 113.2 | 0.6512 | 431.04 |
| YOLO26x | OpenVINO | FP32 | ✅ | 213.2 | 0.6568 | 19.08 |
| YOLO26x | OpenVINO | FP16 | ✅ | 107.1 | 0.6568 | 19.63 |
| YOLO26x | OpenVINO | INT8 | ✅ | 54.8 | 0.6385 | 13.69 |

#### Intel® Core™ Ultra™ 7 155H

Benchmarks

![Intel Core Ultra GPU benchmarks](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/openvino-ultra7-155H-gpu.avif)

Detailed Benchmark Results

| Model | Format | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO26n | PyTorch (CPU) | FP32 | ✅ | 5.3 | 0.4765 | 33.19 |
| YOLO26n | OpenVINO | FP32 | ✅ | 9.6 | 0.4758 | 9.13 |
| YOLO26n | OpenVINO | FP16 | ✅ | 5.1 | 0.4758 | 9.07 |
| YOLO26n | OpenVINO | INT8 | ✅ | 3.2 | 0.4573 | 5.79 |
| YOLO26s | PyTorch (CPU) | FP32 | ✅ | 19.5 | 0.5703 | 70.92 |
| YOLO26s | OpenVINO | FP32 | ✅ | 36.7 | 0.5616 | 16.37 |
| YOLO26s | OpenVINO | FP16 | ✅ | 18.6 | 0.5616 | 16.65 |
| YOLO26s | OpenVINO | INT8 | ✅ | 10.0 | 0.545 | 9.27 |
| YOLO26m | PyTorch (CPU) | FP32 | ✅ | 42.2 | 0.6196 | 192.45 |
| YOLO26m | OpenVINO | FP32 | ✅ | 78.4 | 0.6167 | 34.47 |
| YOLO26m | OpenVINO | FP16 | ✅ | 39.5 | 0.6167 | 35.71 |
| YOLO26m | OpenVINO | INT8 | ✅ | 20.6 | 0.6039 | 15.46 |
| YOLO26l | PyTorch (CPU) | FP32 | ✅ | 50.7 | 0.6215 | 239.19 |
| YOLO26l | OpenVINO | FP32 | ✅ | 95.3 | 0.621 | 43.24 |
| YOLO26l | OpenVINO | FP16 | ✅ | 48.1 | 0.621 | 43.4 |
| YOLO26l | OpenVINO | INT8 | ✅ | 25.2 | 0.5999 | 19.72 |
| YOLO26x | PyTorch (CPU) | FP32 | ✅ | 113.2 | 0.6512 | 511.89 |
| YOLO26x | OpenVINO | FP32 | ✅ | 213.2 | 0.6552 | 79.49 |
| YOLO26x | OpenVINO | FP16 | ✅ | 107.1 | 0.6552 | 79.45 |
| YOLO26x | OpenVINO | INT8 | ✅ | 54.8 | 0.642 | 34.66 |

## Reproduce Our Results

To reproduce the Ultralytics benchmarks above on all export [formats](https://docs.ultralytics.com/modes/export) run this code:

Example

```python
from ultralytics import YOLO

# Load a YOLO26n PyTorch model

model = YOLO("yolo26n.pt")

# Benchmark YOLO26n speed and accuracy on the COCO128 dataset for all export formats

results = model.benchmark(data="coco128.yaml")
```

Note that benchmarking results might vary based on the exact hardware and software configuration of a system, as well as the current workload of the system at the time the benchmarks are run. For the most reliable results use a dataset with a large number of images, i.e. `data='coco.yaml'` (5000 val images).

## Conclusion

The benchmarking results clearly demonstrate the benefits of exporting the YOLO26 model to the OpenVINO format. Across different models and hardware platforms, the OpenVINO format consistently outperforms other formats in terms of inference speed while maintaining comparable accuracy.

The benchmarks underline the effectiveness of OpenVINO as a tool for deploying deep learning models. By converting models to the OpenVINO format, developers can achieve significant performance improvements, making it easier to deploy these models in real-world applications.

For more detailed information and instructions on using OpenVINO, refer to the [official OpenVINO documentation](https://docs.openvino.ai/).

---

> **📌 系列导航**：[📖 返回导读目录](README.md) · [中文实操：yolo模型推理与部署详解 →](YOLO模型推理与部署详解.md)
