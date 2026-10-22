# LiteRT (TensorFlow Lite) 部署指南（Ultralytics 官方文档归档）

> **系列定位**：本文是「部署格式专题」之一 —— Android / Google TPU 生态。内容为 Ultralytics 官方英文文档完整归档，与中文实操文章互补。总目录见 [README 导读](README.md)。
>
> - 中文实操路线：[yolo模型推理与部署详解](YOLO模型推理与部署详解.md)
> - 关联阅读：[模型量化深度解析](模型量化深度解析.md) · [onnx.md](onnx.md)（转换链路起点）

---

（以下为官方文档原文归档）

## Export YOLO Models to LiteRT for Edge and Web Deployment

![LiteRT edge deployment framework](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/tflite-logo.avif)

[LiteRT](https://developers.google.com/edge/litert/overview) (short for *Lite Runtime*) is Google's high-performance runtime for on-device AI. It is the next generation and the new name for TensorFlow Lite (TFLite), and it runs the same `.tflite` model format. With LiteRT, a single exported [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) model deploys across **mobile, embedded, edge, and the browser** — covering everything that the older `tflite` and `tfjs` export formats handled separately, now under one umbrella.

**Watch:** How to Export Ultralytics YOLO26 to Google LiteRT | Deploy Vision AI on Android & iOS | Mobile AI 📱🚀

The LiteRT export format optimizes your models for tasks like [object detection](https://www.ultralytics.com/glossary/object-detection), [segmentation](https://www.ultralytics.com/glossary/image-segmentation), [pose estimation](https://docs.ultralytics.com/tasks/pose), and [classification](https://www.ultralytics.com/glossary/image-classification) so they run fast and offline on a wide range of devices.

Run YOLO on Android with LiteRT today via the official Flutter plugin

The official [Ultralytics YOLO Flutter plugin](https://github.com/ultralytics/yolo-flutter-app) runs LiteRT `.tflite` exports on Android out of the box — real-time camera inference, single-image prediction, GPU acceleration, and automatic model download for all seven YOLO26 tasks, including Depth. For Apple devices use the [CoreML export](https://docs.ultralytics.com/integrations/coreml); for Qualcomm Snapdragon NPUs see the [Qualcomm QNN integration](https://docs.ultralytics.com/integrations/qnn).

Official mobile input sizes

Export classification models at `imgsz=224`. Export detect, segment, semantic, depth, pose, and OBB models at
`imgsz=640`. This 224/640 standard is shared by the official LiteRT, CoreML, and QNN mobile assets.

Run YOLO on Web with LiteRT.js today via the official @ultralytics/yolo npm package

The official [Ultralytics YOLO NPM package](https://www.npmjs.com/package/@ultralytics/yolo) runs LiteRT `.tflite` exports directly in the browser via [LiteRT.js](https://developers.google.com/edge/litert/web) no server or Python required — with real-time webcam inference, single-image prediction, and WebGPU acceleration (automatic CPU/WASM fallback) across all six YOLO26 tasks (detect, segment, pose, OBB, classify, semantic). On WebGPU it's often ~2× faster than ONNX Runtime Web.

```
npm i @ultralytics/yolo @litertjs/core
```

## Why Should You Export to LiteRT?

[LiteRT](https://developers.google.com/edge/litert/overview) is an open-source framework designed for on-device inference, also known as [edge computing](https://www.ultralytics.com/glossary/edge-computing). It gives developers the tools to execute trained models on mobile, embedded, and IoT devices, traditional computers, and — through [LiteRT.js](https://developers.google.com/edge/litert/web) — directly in web browsers and Node.js.

One model format, every target:

- **Mobile & Embedded**: Android, iOS, embedded Linux, and microcontrollers (MCUs).
- **Edge accelerators**: Compatible with the [Coral Edge TPU](https://docs.ultralytics.com/integrations/edge-tpu) for further acceleration.
- **Browser & Node.js**: [LiteRT.js](https://developers.google.com/edge/litert/web) runs the same `.tflite` model on the web with WebGPU/WASM acceleration — replacing the need for a separate TensorFlow.js export.

## Key Features of LiteRT Models

- **On-device Optimization**: Reduces latency by processing data locally, enhances privacy by not transmitting personal data, and minimizes model size to save space.
- **Multiple Platform Support**: Runs on Android, iOS, embedded Linux, microcontrollers, and modern web browsers.
- **Hardware Acceleration**: Leverages XNNPACK on CPU, and GPU acceleration via OpenCL, Metal, and WebGPU. The GPU delegate runs in FP16 by default for additional speed.
- **Quantization**: Supports FP32, static INT8 (`quantize=8`, int8 weights + int8 activations), static INT16-activation (`quantize="w8a16"`, int8 weights + int16 activations for higher accuracy), and dynamic INT8 (`quantize="w8a32"`, int8 weights + FP32 activations, no calibration data needed) to compress models and speed up inference with minimal [accuracy](https://www.ultralytics.com/glossary/accuracy) loss.
- **Diverse Language Support**: Compatible with Java/Kotlin, Swift, Objective-C, C++, Python, and JavaScript.

## Measured Performance

**Hardware:** [Xiaomi 17](https://www.mi.com/global/product/xiaomi-17/) with 12 GB LPDDR5X memory and Android 16 /
API 36. Its 3 nm [Snapdragon 8 Elite Gen 5](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-elite-gen-5)
(SM8850) has an 8-core Qualcomm Oryon CPU (2 Prime cores up to 4.6 GHz and 6 Performance cores up to 3.62 GHz),
Adreno GPU, and Hexagon NPU.

| Model | Task | size (pixels) | CPU w8a32 LiteRT (ms) | GPU w8a32 LiteRT (ms) |
| --- | --- | --- | --- | --- |
| YOLO26n | Detect | 640 | 52.2 1.8 / 48.1 / 2.4 | **15.8** 2.3 / 8.9 / 4.6 |
| YOLO26n-seg | Segment | 640 | 73.4 1.8 / 65.6 / 6.0 | **33.2** 1.8 / 23.8 / 7.6 |
| YOLO26n-sem | Semantic | 640 | 61.2 1.8 / 51.1 / 8.3 | **34.2** 1.8 / 24.0 / 8.3 |
| YOLO26n-depth | Depth | 640 | 124.4 1.9 / 115.1 / 7.4 | **23.0** 1.8 / 13.5 / 7.7 |
| YOLO26n-cls | Classify | 224 | 4.4 0.4 / 4.0 / 0.0 | **3.1** 0.8 / 2.1 / 0.2 |
| YOLO26n-pose | Pose | 640 | 57.4 1.8 / 53.8 / 1.8 | **16.6** 2.7 / 10.1 / 3.9 |
| YOLO26n-obb | OBB | 640 | 50.3 1.8 / 47.2 / 1.4 | **11.7** 1.8 / 7.8 / 2.0 |

- **Speed** values are **single-image burst latencies** — the mean of 15 runs after 3 warmup runs on `bus.jpg`, measured with the [Ultralytics Flutter plugin](https://github.com/ultralytics/yolo-flutter-app) `0.6.10` and the standardized `v0.6.6` assets. CPU/GPU order alternated between tasks in one sequential sweep. Native logs confirmed that every CPU row used LiteRT CPU/XNNPACK and every GPU row delegated the complete graph to LiteRT OpenCL (`LITERT_CL`).
- The LiteRT export traces the PyTorch model directly, producing an **NCHW** `.tflite` with a float input — the GPU delegate compiles the whole graph (all seven tasks run on the Adreno GPU here), and `w8a32` needs no calibration data. Consumers should read tensor shapes and signature names instead of assuming the legacy onnx2tf NHWC layout or `Identity` output names; pack RGB data directly as planar CHW or transpose it before inference. Semantic exports return NCHW logits and require a host-side class argmax. The official Android assets are hosted on the [yolo-flutter-app `v0.6.6` release](https://github.com/ultralytics/yolo-flutter-app/releases/tag/v0.6.6), with the detailed benchmark record in [the Flutter performance doc](https://github.com/ultralytics/yolo-flutter-app/blob/main/doc/performance.md).
- The matching Snapdragon **Hexagon NPU** and LiteRT CPU/GPU numbers are in the [Qualcomm QNN integration](https://docs.ultralytics.com/integrations/qnn).
- Compare the Apple CPU/accelerator results in the [CoreML integration](https://docs.ultralytics.com/integrations/coreml#measured-performance).

The following device sweeps use the same standardized `v0.6.6` assets.

### Google Pixel 10

**Hardware:** [Google Pixel 10](https://store.google.com/product/pixel_10_specs) with 12 GB memory and Android 16 /
API 36. Its 3 nm [Google Tensor G5](https://blog.google/products-and-platforms/devices/pixel/tensor-g5-pixel-10/) has
an 8-core CPU (1 Prime core up to 3.78 GHz, 5 Performance cores up to 3.05 GHz, and 2 Efficiency cores up to
2.25 GHz), PowerVR D-Series GPU, and Google TPU. Core clocks and the GPU driver name were read from the benchmark
device because Google does not publish them in the linked specifications.

| Model | Task | size (pixels) | CPU w8a32 LiteRT (ms) | GPU w8a32 LiteRT (ms) |
| --- | --- | --- | --- | --- |
| YOLO26n | Detect | 640 | 53.3 1.5 / 50.2 / 1.6 | **45.5** 3.8 / 37.7 / 4.0 |
| YOLO26n-seg | Segment | 640 | 87.7 1.8 / 78.5 / 7.5 | **50.9** 3.0 / 36.9 / 10.9 |
| YOLO26n-sem | Semantic | 640 | **68.6** 1.5 / 59.0 / 8.0 | 71.6 1.5 / 59.5 / 10.6 |
| YOLO26n-depth | Depth | 640 | 120.3 1.5 / 112.5 / 6.3 | **52.5** 2.0 / 37.5 / 13.0 |
| YOLO26n-cls | Classify | 224 | **4.0** 0.3 / 3.4 / 0.2 | 17.6 0.9 / 16.7 / 0.1 |
| YOLO26n-pose | Pose | 640 | 59.7 1.5 / 57.0 / 1.2 | **46.6** 3.8 / 39.2 / 3.5 |
| YOLO26n-obb | OBB | 640 | 52.0 1.5 / 48.9 / 1.7 | **45.5** 4.0 / 38.5 / 2.9 |

**Benchmark:** Mean of 15 `predict()` calls after 3 warmups on `bus.jpg`, using `ultralytics_yolo` `0.6.10` and the
official `v0.6.6` assets. CPU/GPU order alternates between tasks in one sequential sweep. Native logs confirmed that
every CPU row used LiteRT CPU/XNNPACK and every GPU row delegated the complete graph to LiteRT OpenCL (`LITERT_CL`).

### Samsung Galaxy S26

**Hardware:** [Samsung Galaxy S26](https://www.samsung.com/uk/smartphones/galaxy-s26/) (SM-S942B) with 12 GB memory
and Android 16 / API 36. Its 2 nm [Exynos 2600](https://semiconductor.samsung.com/processor/mobile-processor/exynos-2600/)
has a 10-core Armv9.3 CPU (1 C1-Ultra core up to 3.8 GHz, 3 performance C1-Pro cores up to 3.26 GHz, and 6 efficiency
C1-Pro cores up to 2.76 GHz), Xclipse 960 GPU, and Samsung NPU.

| Model | Task | size (pixels) | CPU w8a32 LiteRT (ms) | GPU w8a32 LiteRT (ms) |
| --- | --- | --- | --- | --- |
| YOLO26n | Detect | 640 | 36.7 1.3 / 33.8 / 1.7 | **16.4** 1.4 / 12.3 / 2.6 |
| YOLO26n-seg | Segment | 640 | 54.6 1.2 / 48.0 / 5.3 | **32.8** 1.3 / 24.5 / 7.0 |
| YOLO26n-sem | Semantic | 640 | 47.8 1.2 / 38.4 / 8.1 | **34.2** 1.3 / 24.9 / 8.0 |
| YOLO26n-depth | Depth | 640 | 92.9 1.2 / 84.8 / 6.9 | **33.5** 1.3 / 22.4 / 9.8 |
| YOLO26n-cls | Classify | 224 | 2.7 0.2 / 2.3 / 0.2 | **2.6** 0.2 / 2.4 / 0.0 |
| YOLO26n-pose | Pose | 640 | 42.8 1.3 / 40.5 / 1.0 | **18.4** 1.4 / 14.1 / 2.9 |
| YOLO26n-obb | OBB | 640 | 37.5 1.3 / 35.1 / 1.2 | **18.8** 2.5 / 14.6 / 1.8 |

**Benchmark:** Mean of 15 `predict()` calls after 3 warmups on `bus.jpg`, using `ultralytics_yolo` `0.6.10` and the
official `v0.6.6` assets. CPU/GPU order alternates between tasks in one sequential sweep. Native logs confirmed that
every CPU row used LiteRT CPU/XNNPACK and every GPU row delegated the complete graph to LiteRT OpenCL (`LITERT_CL`).

### Xiaomi 17T Pro

**Hardware:** [Xiaomi 17T Pro](https://www.mi.com/global/product/xiaomi-17t-pro/specs/) (2602EPTC0G) with 12 GB
LPDDR5X memory and Android 16 / API 36. Its 3 nm
[MediaTek Dimensity 9500](https://www.mediatek.com/products/smartphones/mediatek-dimensity-9500) (MT6993) has an
8-core Armv9.3 CPU (1 C1-Ultra core up to 4.21 GHz, 3 C1-Premium cores up to 3.5 GHz, and 4 C1-Pro cores up to
2.7 GHz), Mali-G1 Ultra MC12 GPU, and MediaTek NPU 990.

| Model | Task | size (pixels) | CPU w8a32 LiteRT (ms) | GPU w8a32 LiteRT (ms) |
| --- | --- | --- | --- | --- |
| YOLO26n | Detect | 640 | 45.4 1.3 / 42.1 / 2.0 | **26.6** 1.9 / 22.0 / 2.7 |
| YOLO26n-seg | Segment | 640 | 126.2 2.6 / 113.9 / 9.7 | **46.7** 2.6 / 33.3 / 10.8 |
| YOLO26n-sem | Semantic | 640 | 117.9 2.6 / 98.8 / 16.5 | **74.3** 2.6 / 54.7 / 17.0 |
| YOLO26n-depth | Depth | 640 | 182.4 2.5 / 167.6 / 12.3 | **47.8** 2.5 / 32.3 / 12.9 |
| YOLO26n-cls | Classify | 224 | **6.2** 0.4 / 5.3 / 0.4 | 7.4 0.4 / 6.9 / 0.1 |
| YOLO26n-pose | Pose | 640 | 97.6 2.5 / 93.3 / 1.8 | **28.6** 2.5 / 23.3 / 2.8 |
| YOLO26n-obb | OBB | 640 | 91.5 2.6 / 85.8 / 3.2 | **27.5** 2.7 / 21.8 / 2.9 |

**Benchmark:** Mean of 15 `predict()` calls after 3 warmups on `bus.jpg`, using `ultralytics_yolo` `0.6.10` and the
official `v0.6.6` assets. CPU/GPU order alternates between tasks in one sequential sweep. Native logs confirmed that
every CPU row used LiteRT CPU/XNNPACK and every GPU row delegated the complete graph to LiteRT OpenCL (`LITERT_CL`).

## Supported Tasks

LiteRT export supports all seven Ultralytics tasks. Semantic segmentation and depth estimation are available only with YOLO26, the only family that ships those heads.

| Task | [YOLOv8](https://docs.ultralytics.com/models/yolov8) | [YOLO11](https://docs.ultralytics.com/models/yolo11) | [YOLO26](https://docs.ultralytics.com/models/yolo26) |
| --- | --- | --- | --- |
| [Detect](https://docs.ultralytics.com/tasks/detect) | ✅ | ✅ | ✅ |
| [Segment](https://docs.ultralytics.com/tasks/segment) | ✅ | ✅ | ✅ |
| [Semantic](https://docs.ultralytics.com/tasks/semantic) | ❌ | ❌ | ✅ |
| [Depth](https://docs.ultralytics.com/tasks/depth) | ❌ | ❌ | ✅ |
| [Classify](https://docs.ultralytics.com/tasks/classify) | ✅ | ✅ | ✅ |
| [Pose](https://docs.ultralytics.com/tasks/pose) | ✅ | ✅ | ✅ |
| [OBB](https://docs.ultralytics.com/tasks/obb) | ✅ | ✅ | ✅ |

## Export to LiteRT: Converting Your YOLO Model

You can improve on-device execution efficiency and broaden deployment options by converting your models to the LiteRT format.

### Installation

To install the required package, run:

Installation

```bash
# Install the required package for YOLO
pip install ultralytics
```

For detailed instructions and best practices, check our [Ultralytics Installation guide](https://docs.ultralytics.com/quickstart). If you encounter any difficulties, consult our [Common Issues guide](https://docs.ultralytics.com/guides/yolo-common-issues).

Platform support

LiteRT **export** is currently supported on **Linux x86\_64** and **macOS**. The exported `.tflite` model itself runs on all LiteRT-supported platforms (mobile, embedded, edge, and the browser).

### Usage

All [Ultralytics YOLO models](https://docs.ultralytics.com/models) support export out of the box. The LiteRT format supports the [Export](https://docs.ultralytics.com/modes/export), [Predict](https://docs.ultralytics.com/modes/predict), and [Validate](https://docs.ultralytics.com/modes/val) modes, so you can export a model, then load it to run inference or validate its accuracy locally.

Export

```python
from ultralytics import YOLO

# Load a YOLO26 model

model = YOLO("yolo26n.pt")

# Export the model to LiteRT format

model.export(format="litert", imgsz=640)  # use imgsz=224 for classification
```

Quantized export

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Dynamic INT8: int8 weights, FP32 activations - no calibration data needed

model.export(format="litert", quantize="w8a32", imgsz=640)  # use imgsz=224 for classification

# Static INT8: int8 weights + int8 activations - needs calibration data

model.export(format="litert", quantize=8, data="coco8.yaml", imgsz=640)  # use 224 for classification

# Static w8a16: int8 weights + int16 activations (higher accuracy) - needs calibration data

model.export(format="litert", quantize="w8a16", data="coco8.yaml", imgsz=640)  # use 224 for classification
```

Predict

```python
from ultralytics import YOLO

# Load the exported LiteRT model

model = YOLO("yolo26n.tflite")

# Run inference

results = model("https://ultralytics.com/images/bus.jpg")
```

Validate

```python
from ultralytics import YOLO

# Load the exported LiteRT model

model = YOLO("yolo26n.tflite")

# Validate accuracy on the COCO8 dataset

metrics = model.val(data="coco8.yaml")
```

### Export Arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `format` | `str` | `'litert'` | Target format for the exported model, defining compatibility with various deployment environments. |
| `imgsz` | `int` or `tuple` | `640` | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions. |
| `quantize` | `int` or `str` | `None` | Quantization precision: `8` (static INT8, int8 weights + int8 activations; needs calibration `data`/`fraction`), `'w8a16'` (static, int8 weights + int16 activations; needs calibration `data`/`fraction`), `'w8a32'` (dynamic INT8, int8 weights + FP32 activations; no calibration needed), or `32`/unset (FP32). FP16 is not exported separately (see note below). Replaces the deprecated `half`/`int8` flags. |
| `batch` | `int` | `1` | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode. |
| `data` | `str` | `None` | Dataset YAML used for INT8 calibration; classification instead takes a dataset directory or a built-in dataset name. If omitted with `quantize=8` or `'w8a16'`, Ultralytics selects the default calibration dataset for the model task. |
| `device` | `str` | `None` | Specifies the device for exporting. LiteRT export runs on CPU (`device=cpu`). |

FP16 precision

Unlike the legacy `tflite` export, LiteRT does not require a separate FP16 export. An FP32 `.tflite` model runs in **half precision at runtime** when using a GPU delegate (WebGPU, OpenCL, Metal) — this is the official LiteRT approach to FP16 inference.

For more details about the export process, visit the [Ultralytics documentation page on exporting](https://docs.ultralytics.com/modes/export).

## Deploying Exported YOLO LiteRT Models

After exporting your Ultralytics YOLO model to LiteRT, you can deploy it across platforms. The quickest way to verify it locally is the `YOLO("yolo26n.tflite")` method shown above. For deployment in other environments, see the following resources:

### Mobile & Embedded

- **[Android](https://developers.google.com/edge/litert/android)**: A quick-start guide for integrating LiteRT into Android applications.
- **[iOS](https://developers.google.com/edge/litert/ios/quickstart)**: A guide for integrating and deploying LiteRT models in iOS applications.
- **[Embedded Linux & Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi)**: Run LiteRT models on single-board computers, optionally accelerated with a [Coral Edge TPU](https://docs.ultralytics.com/integrations/edge-tpu).
- **[Microcontrollers](https://developers.google.com/edge/litert/microcontrollers/overview)**: Deploy on MCUs with only a few kilobytes of memory — the core runtime fits in roughly 16 KB on an Arm Cortex-M3.

### Browser & Node.js (LiteRT.js)

- **[LiteRT.js overview](https://developers.google.com/edge/litert/web)**: Run the same `.tflite` model directly in the browser with WebGPU/WASM acceleration, eliminating server-side computation and keeping data on the user's device.
- **[End-to-End Examples](https://github.com/google-ai-edge/litert)**: Practical examples and tutorials for implementing LiteRT across mobile, edge, and web.

## Summary

In this guide, we covered how to export Ultralytics YOLO models to the LiteRT format. By consolidating mobile/edge (formerly TFLite) and browser (formerly TF.js) deployment into a single `.tflite` model, LiteRT makes your YOLO models faster, smaller, and portable across virtually every on-device target.

For further details, visit the [LiteRT official documentation](https://developers.google.com/edge/litert/overview).

Also, if you're curious about other Ultralytics YOLO integrations, check out our [integration guide page](https://docs.ultralytics.com/integrations) for plenty of helpful resources.

---

> **📌 系列导航**：[📖 返回导读目录](README.md) · [中文实操：yolo模型推理与部署详解 →](YOLO模型推理与部署详解.md)
