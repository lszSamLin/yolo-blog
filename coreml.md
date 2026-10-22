# CoreML 部署指南（Ultralytics 官方文档归档）

> **系列定位**：本文是「部署格式专题」之一 —— Apple 生态（iOS / macOS / Neural Engine）。内容为 Ultralytics 官方英文文档完整归档，与中文实操文章互补。总目录见 [README 导读](README.md)。
>
> - 中文实操路线：[yolo模型推理与部署详解](YOLO模型推理与部署详解.md)
> - 关联阅读：[模型量化深度解析](模型量化深度解析.md) · [onnx.md](onnx.md)（转换链路起点）

---

（以下为官方文档原文归档）

## CoreML Export for YOLO26 Models

Apple ships dedicated AI silicon — the Neural Engine — in every modern iPhone, iPad, and Mac, and [CoreML](https://developer.apple.com/documentation/coreml) is Ultralytics' supported path for deploying models to it today. Exporting [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) models to CoreML turns a trained `.pt` checkpoint into a native `.mlpackage` that runs all seven YOLO tasks on-device at low latency, with no network connection and no data leaving the device.

Run YOLO on the Apple Neural Engine today with the official mobile apps

The official [Ultralytics YOLO iOS SDK](https://github.com/ultralytics/yolo-ios-app) and [Flutter plugin](https://github.com/ultralytics/yolo-flutter-app) run CoreML exports on the Apple Neural Engine out of the box — real-time camera inference, single-image prediction, and automatic model download for all seven YOLO26 tasks, including Depth. For Android NPU deployment, see the [Qualcomm QNN integration](https://docs.ultralytics.com/integrations/qnn).

Official mobile input sizes

Export classification models at `imgsz=224`. Export detect, segment, semantic, depth, pose, and OBB models at
`imgsz=640`. This 224/640 standard is shared by the official CoreML, LiteRT, and QNN mobile assets.

Apple's future Core AI format

Apple has introduced the new [Core AI framework and `.aimodel` format](https://docs.ultralytics.com/integrations/coreai) for the iOS 27 and macOS 27 generation, but Ultralytics does not currently export it. CoreML remains the supported format for current Ultralytics releases and broader Apple device compatibility.

**Watch:** How to Export Ultralytics YOLO26 to CoreML for 2x Fast Inference on Apple Devices 🚀

## What is CoreML?

![Apple CoreML deployment pipeline](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/coreml-overview.avif)

CoreML (styled "Core ML" by Apple) is Apple's on-device [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) framework. It loads models in the modern **ML Program** format — the `.mlpackage` bundle the Ultralytics exporter produces — and schedules them across the device's CPU, GPU, and **Apple Neural Engine (ANE)**, the dedicated NPU in every Apple-silicon chip. Because everything runs locally, inference works offline, adds no network latency, and keeps user data on the device.

CoreML integrates directly with Apple's [Vision framework](https://developer.apple.com/documentation/vision), which handles image scaling and orientation on the way into the model — this is how the Ultralytics iOS SDK feeds camera frames to YOLO with effectively zero preprocessing cost.

## Why Export YOLO26 to CoreML?

- **Neural Engine speed**: CoreML schedules supported operations on Apple's Neural Engine for low-latency on-device inference. See the physical-device table below, and benchmark your exact export on your target hardware.
- **NMS-free by design**: YOLO26 is [end-to-end](https://www.ultralytics.com/glossary/non-maximum-suppression-nms), so the exported graph needs no NMS pipeline and decode is sub-millisecond. Older detection models like YOLO11 can embed a CoreML NMS pipeline with `nms=True`.
- **Private and offline**: All computation stays on the device — no cloud round-trips, no API keys, full [data privacy](https://www.ultralytics.com/glossary/data-privacy).
- **One export, the whole ecosystem**: The same `.mlpackage` runs on iOS, iPadOS, macOS, watchOS, tvOS, and visionOS, and powers the official Ultralytics [iOS SDK](https://github.com/ultralytics/yolo-ios-app) and [Flutter plugin](https://github.com/ultralytics/yolo-flutter-app).

## Measured Performance

End-to-end single-image inference for the standardized `v8.3.0` YOLO26n INT8 CoreML assets on an
[iPhone 17 Pro](https://support.apple.com/en-us/125090) with 12 GB memory and iOS 26.5.2. Its A19 Pro has a 6-core CPU
(2 Performance and 4 Efficiency cores), 6-core GPU with Neural Accelerators, and 16-core Neural Engine. Each cell
shows the **total time** (preprocessing + inference + postprocessing, excluding
annotation) with the per-stage split beneath it. On iOS, Vision performs input scaling inside the inference request,
so preprocessing is reported as 0 and its cost is included in inference.

| Model | Task | size (pixels) | CPU Core ML `.cpuOnly` (ms) | CPU + ANE preferred Core ML `.cpuAndNeuralEngine` (ms) |
| --- | --- | --- | --- | --- |
| YOLO26n | Detect | 640 | 9.2 0.0 / 9.2 / 0.0 | **3.2** 0.0 / 3.2 / 0.0 |
| YOLO26n-seg | Segment | 640 | 12.6 0.0 / 12.0 / 0.5 | **4.8** 0.0 / 4.2 / 0.6 |
| YOLO26n-sem | Semantic | 640 | 9.7 0.0 / 9.2 / 0.5 | **4.6** 0.0 / 4.2 / 0.5 |
| YOLO26n-depth | Depth | 640 | 25.0 0.0 / 24.1 / 0.9 | **5.3** 0.0 / 4.5 / 0.9 |
| YOLO26n-cls | Classify | 224 | 2.2 0.0 / 2.2 / 0.0 | **1.9** 0.0 / 1.9 / 0.0 |
| YOLO26n-pose | Pose | 640 | 11.9 0.0 / 11.9 / 0.0 | **3.9** 0.0 / 3.9 / 0.0 |
| YOLO26n-obb | OBB | 640 | 10.6 0.0 / 10.6 / 0.0 | **3.4** 0.0 / 3.4 / 0.0 |

- The exact `v8.3.0` release assets declare 224×224 inputs for classification and 640×640 for every other task.
- **Speed** values are **single-image burst latencies** — the mean of 15 runs after 3 warmup runs on `bus.jpg`, measured through the [iOS SDK's](https://github.com/ultralytics/yolo-ios-app) per-stage timing via the [Flutter plugin's](https://github.com/ultralytics/yolo-flutter-app) benchmark harness in profile mode (optimized native code). CPU/accelerator order alternated between tasks in one sequential sweep. CPU rows request Core ML `.cpuOnly`; CPU + ANE preferred rows request `.cpuAndNeuralEngine`, with final operation placement controlled by Core ML. Sustained real-time camera operation runs higher because it includes the capture and scaling pipeline plus thermal settling. A historical pre-standard camera sweep measured 11.3 ms/frame for YOLO26n detect and 16.5 ms/frame for YOLO26n Depth on the same device — see the [iOS SDK performance doc](https://github.com/ultralytics/yolo-ios-app/blob/main/docs/performance.md) for steady-state profiling.
- Compare the Android CPU/GPU results in the [LiteRT integration](https://docs.ultralytics.com/integrations/litert#measured-performance) and Snapdragon NPU results in the [Qualcomm QNN integration](https://docs.ultralytics.com/integrations/qnn#measured-performance).

## Supported Tasks

CoreML export supports all seven Ultralytics tasks. Semantic segmentation and depth estimation are available only with YOLO26, the only family that ships those heads.

| Task | [YOLOv8](https://docs.ultralytics.com/models/yolov8) | [YOLO11](https://docs.ultralytics.com/models/yolo11) | [YOLO26](https://docs.ultralytics.com/models/yolo26) |
| --- | --- | --- | --- |
| [Detect](https://docs.ultralytics.com/tasks/detect) | ✅ | ✅ | ✅ |
| [Segment](https://docs.ultralytics.com/tasks/segment) | ✅ | ✅ | ✅ |
| [Semantic](https://docs.ultralytics.com/tasks/semantic) | ❌ | ❌ | ✅ |
| [Depth](https://docs.ultralytics.com/tasks/depth) | ❌ | ❌ | ✅ |
| [Classify](https://docs.ultralytics.com/tasks/classify) | ✅ | ✅ | ✅ |
| [Pose](https://docs.ultralytics.com/tasks/pose) | ✅ | ✅ | ✅ |
| [OBB](https://docs.ultralytics.com/tasks/obb) | ✅ | ✅ | ✅ |

## Exporting YOLO26 Models to CoreML

### Installation

To install the required package, run:

Installation

```bash
# Install the required package for YOLO26
pip install ultralytics
```

The `coremltools` converter is installed automatically on first export. Export runs on macOS or x86 Linux; for detailed instructions and best practices, check our [installation guide](https://docs.ultralytics.com/quickstart) and the [Common Issues guide](https://docs.ultralytics.com/guides/yolo-common-issues).

### Usage

The CoreML format supports the [Export](https://docs.ultralytics.com/modes/export), [Predict](https://docs.ultralytics.com/modes/predict), and [Validate](https://docs.ultralytics.com/modes/val) modes. Inference and validation with CoreML run on macOS only. Export your model, then load the exported model to run inference or validate its accuracy.

Export

```python
from ultralytics import YOLO

# Load a YOLO26 model

model = YOLO("yolo26n.pt")

# Export to CoreML with INT8 weight quantization, matching the official app models

model.export(format="coreml", quantize=8, imgsz=640)  # use imgsz=224 for classification
```

Predict

```python
from ultralytics import YOLO

# Load the exported CoreML model (macOS)

model = YOLO("yolo26n.mlpackage")

# Run inference

results = model("https://ultralytics.com/images/bus.jpg")
```

Validate

```python
from ultralytics import YOLO

# Load the exported CoreML model (macOS)

model = YOLO("yolo26n.mlpackage")

# Validate accuracy on the COCO8 dataset

metrics = model.val(data="coco8.yaml")
```

### Export Arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `format` | `str` | `'coreml'` | Target format for the exported model, defining compatibility with various deployment environments. |
| `imgsz` | `int` or `tuple` | `640` | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions. |
| `quantize` | `int` or `str` | `None` | Quantization precision (weight-only for CoreML): `16` (FP16), `8` (INT8), `"w8a16"` (INT8 weights with FP16 activations), or `32`/unset (FP32). Unset NMS ML Programs use FP16 for Xcode preview; pass `32` to override. Replaces the deprecated `half`/`int8` flags. |
| `nms` | `bool` | `False` | Embeds a CoreML NMS pipeline. Detection models only (ignored with a warning for other tasks); not needed for NMS-free YOLO26, use for earlier models like YOLO11. |
| `dynamic` | `bool` | `False` | Allows dynamic input sizes, enhancing flexibility in handling varying image dimensions. |
| `batch` | `int` | `1` | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode. |
| `device` | `str` | `None` | Specifies the device for exporting: GPU (`device=0`), CPU (`device=cpu`), MPS for Apple silicon (`device=mps`). |

For more details about the export process, visit the [Ultralytics documentation page on exporting](https://docs.ultralytics.com/modes/export).

### Targeting the Neural Engine

CoreML chooses hardware via `MLModelConfiguration.computeUnits`. The Ultralytics iOS SDK defaults to `.cpuAndNeuralEngine` on iOS 16+ rather than `.all`: in a real-time camera app the GPU is already busy compositing the preview and overlays, so excluding it avoids contention and frame-time jitter while the ANE does the heavy lifting. Pin `.cpuOnly` only for compatibility testing — the table above shows what it costs.

Running a CoreML model from Python on a **Mac host** (via Ultralytics or `coremltools`) follows the same rule: Ultralytics loads with `ComputeUnit.CPU_AND_NE` (macOS 13+, falling back to `CPU_ONLY` on older macOS), keeping inference on the Neural Engine (~3× faster than CPU). This also avoids a current macOS host limitation where the default `ComputeUnit.ALL` / `CPU_AND_GPU` — which add the GPU/MPSGraph compile path — **abort the process** with an `Error: MLIR pass manager failed` assertion on `coremltools` 9.x.

## Deploying Exported YOLO26 CoreML Models

The fastest path is the official [Ultralytics YOLO iOS SDK](https://github.com/ultralytics/yolo-ios-app), the same Swift package that powers the Ultralytics iOS app and the [Flutter plugin](https://github.com/ultralytics/yolo-flutter-app). It resolves official model names automatically, downloads and caches the `.mlpackage`, and returns fully decoded results:

```python
import UltralyticsYOLO

// Loads the official INT8 model (downloaded and cached on first use), then runs inference
let yolo = YOLO("yolo26n", task: .detect) { result in
    if case .success(let model) = result {
        let results = model(uiImage)  // boxes, labels, confidences, timing
    }
}
```

For camera apps, drop in the SDK's `YOLOView` for real-time inference with native overlays, or use the [Flutter plugin](https://github.com/ultralytics/yolo-flutter-app) for cross-platform apps that share one codebase with Android.

Integrating a raw `.mlpackage` yourself is also straightforward with Apple's stack — load it with `MLModel`, wrap it in a `VNCoreMLRequest`, and feed images through `VNImageRequestHandler`. These resources cover the details:

- **[Integrating a Core ML Model into Your App](https://developer.apple.com/documentation/coreml/integrating-a-core-ml-model-into-your-app)**: Apple's guide to bundling and calling a CoreML model.
- **[CoreML Tools](https://apple.github.io/coremltools/docs-guides/)**: Conversion, quantization, and optimization reference for the `coremltools` toolchain that powers this export.
- **[Xcode Core ML Performance Reports](https://developer.apple.com/videos/)**: Per-layer device placement and latency profiling for your exact model and device.

Ship the model either embedded in the app bundle (instant availability, ideal for nano/small models) or downloaded on first run and cached (smaller binary, easy model updates). The official apps combine both approaches: default nano models are bundled for immediate use, while larger variants download on demand and are cached locally.

## Recommended Workflow

1. **Train** your model with Ultralytics [Train mode](https://docs.ultralytics.com/modes/train), or start from the official YOLO26 weights
2. **Export** with `model.export(format="coreml", quantize=8, imgsz=640)` on macOS or x86 Linux (`imgsz=224` for classification)
3. **Verify** accuracy with `model.val()` on a Mac, and profile with an Xcode Core ML Performance Report on your target device
4. **Deploy** with the iOS SDK, the Flutter plugin, or your own Vision integration, targeting `.cpuAndNeuralEngine`

## Summary

In this guide, you learned how to export Ultralytics YOLO26 models to CoreML's `.mlpackage` format, quantize them for the Apple Neural Engine, and deploy them at single-digit-millisecond latencies — either through the official iOS SDK and Flutter plugin or your own Vision integration. For other deployment targets, browse the [integration guide page](https://docs.ultralytics.com/integrations), and compare formats with [Benchmark mode](https://docs.ultralytics.com/modes/benchmark).

---

> **📌 系列导航**：[📖 返回导读目录](README.md) · [中文实操：yolo模型推理与部署详解 →](YOLO模型推理与部署详解.md)
