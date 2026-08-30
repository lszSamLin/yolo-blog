# 模型在NPU的C++部署

## 引言

在嵌入式 AI 部署场景中，C++ 因其高性能和低资源消耗而成为首选语言。本文将详细介绍如何在 Rockchip NPU 上使用 C++ 进行 YOLO 模型推理部署，涵盖从模型准备到完整推理流程的各个环节。

> **参考来源**：[RKNN C++ API Documentation](https://github.com/airockchip/rknn-toolkit2/blob/master/docs/en/03_rknn_runtime/02_rknn_runtime_api.md)

---

## 一、RKNN C++ API 概述

### 1.1 核心 API

```c
// RKNN C++ API 核心函数

// 1. 初始化（加载模型并创建上下文）
// ctx: 输出上下文句柄
// model: 模型数据指针（.rknn 文件内容）
// model_size: 模型数据大小
// flag: 保留参数，通常为 0
// 返回值: RKNN_SUCC 表示成功，其他值为错误码
rknn_init(rknn_context* ctx, uint8_t* model, uint32_t model_size, uint32_t flag)

// 2. 查询输入输出信息
// type: 查询类型，常见值包括:
//   RKNN_QUERY_IN_OUT_NUM   - 获取输入输出数量
//   RKNN_QUERY_INPUT_ATTR   - 获取指定输入张量属性
//   RKNN_QUERY_OUTPUT_ATTR  - 获取指定输出张量属性
//   RKNN_QUERY_NATIVE_INPUT_ATTR  - 获取 NPU 原生输入属性
//   RKNN_QUERY_NATIVE_OUTPUT_ATTR - 获取 NPU 原生输出属性
//   RKNN_QUERY_MODEL_ATTRIBUTE  - 查询模型属性(如量化参数)
//   RKNN_QUERY_PERF_RUN     - 查询推理性能数据
// info: 查询结果缓冲区
// size: 缓冲区大小
rknn_query(rknn_context ctx, rknn_query_type type, void* info, uint32_t size)

// 3. 设置输入数据
// n_inputs: 输入数量（通常为 1）
// inputs: 输入数组，设置方式有两种:
//   方式一 (CPU 拷贝): 设置 pass_through=0, buf 指向 CPU 内存，RKNN 自动拷贝到 NPU
//   方式二 (零拷贝):    设置 pass_through=1, mem 指向 NPU 内存，直接引用，避免拷贝
// 注意: 调用前必须通过 rknn_inputs_set() 设置，推理结束后需调用 rknn_outputs_get() 获取结果
rknn_inputs_set(rknn_context ctx, uint32_t n_inputs, rknn_input* inputs)

// 4. 执行推理
// extend: 扩展参数（异步模式传 &run_extend，之后调用 rknn_wait() 等待完成）
// 同步模式: 传 nullptr，函数阻塞直到推理完成
rknn_run(rknn_context ctx, rknn_run_extend* extend)

// 5. 获取输出结果
// n_outputs: 输出数量
// outputs: 输出数组，每个元素需提前设置 index 和 want_float（是否转 float32）
//          获取到的数据存放在各元素的 buf / size 字段中
// extend: 扩展参数，同步模式传 nullptr
rknn_outputs_get(rknn_context ctx, uint32_t n_outputs, rknn_output* outputs, rknn_output_extend* extend)

// 6. 释放输出缓冲区（每次 rknn_outputs_get() 之后必须调用，防止内存泄漏）
rknn_outputs_release(rknn_context ctx, uint32_t n_outputs, rknn_output* outputs)

// 7. 销毁上下文，释放所有资源
rknn_destroy(rknn_context ctx)

// 8. 异步等待（配合异步推理使用）
// 等待异步推理完成，可在多 NPU 场景下实现流水线并行
rknn_wait(rknn_context ctx, rknn_run_extend* extend)

// 9. 创建/销毁 NPU 内存（用于零拷贝和 DMA 优化）
rknn_tensor_mem* rknn_create_mem(rknn_context ctx, uint32_t size)
void rknn_destroy_mem(rknn_context ctx, rknn_tensor_mem* mem)

```

### 1.2 数据结构

```c
// RKNN 上下文（ opaque handle，不应直接访问内部字段）
typedef void* rknn_context;

// 输入输出数量
typedef struct rknn_io_dimension {
    uint32_t n_input;
    uint32_t n_output;
} rknn_io_dimension;

// 张量属性（用于查询模型的输入/输出张量信息）
typedef struct rknn_tensor_attr {
    uint32_t index;                    // 张量索引
    rknn_tensor_type type;             // 数据类型: RKNN_TENSOR_FLOAT32 / FLOAT16 / INT8 / UINT8
    rknn_tensor_format format;         // 内存布局: RKNN_TENSOR_NCHW / NHWC / RKNN_TENSOR_PLANAR / ...
    char name[RKNN_MAX_TENSOR_NAME_LEN]; // 张量名称
    uint32_t n_dims;                   // 维度数量
    int32_t dims[4];                   // 各维度大小 [N, C, H, W]
    uint32_t size;                     // 数据总字节数 = product(dims) * element_size
    uint32_t w_stride;                 // W 维度步长（NCHW 时通常等于 size/(H*C)）
    uint32_t quant_type;               // 量化类型: 0=FP32, 1=ASYMMETRIC_QUANT, 2=SYMMETRIC_QUANT
    uint32_t quant_method;             // 量化方法: 0=NONE, 1=LINEAR, 2=LOGISTIC
    float zero_point[4];               // 量化零点
    float scale[4];                    // 量化步长
} rknn_tensor_attr;

// 输入项（传递给 rknn_inputs_set）
typedef struct rknn_input {
    uint32_t index;        // 输入张量索引（通常为 0）
    void* buf;             // 数据指针（CPU 内存地址，用于 pass_through=0 模式）
    uint32_t size;         // 数据字节数
    rknn_tensor_mem* mem;  // NPU 内存句柄（用于 pass_through=1 零拷贝模式）
    uint8_t pass_through;  // 0=RKNN 内部拷贝 (CPU->NPU)，1=直接使用 NPU 内存（零拷贝）
    uint8_t type;          // 输入类型: RKNN_INPUT_TENSOR(0) 或 RKNN_INPUT_NONE(1, 仅占位)
} rknn_input;

// 输出项（传递给 rknn_outputs_get）
typedef struct rknn_output {
    uint32_t index;        // 输出张量索引
    void* buf;             // 输出数据指针
    uint8_t want_float;    // 1=转换为 float32 输出，0=保持原始量化格式
    uint8_t repeatable;    // 1=输出缓冲区可重复使用（不自动释放），0=单次有效
} rknn_output;

// NPU 内存（用于零拷贝和 DMA 优化）
typedef struct rknn_tensor_mem {
    void* virt_addr;       // CPU 可访问的虚拟地址（若支持 CPU-NPU 共享内存）
    uint32_t size;         // 内存大小
    uint32_t phys_addr;    // 物理地址（供 NPU DMA 使用）
    void* priv_data;       // 私有数据，由运行时管理
} rknn_tensor_mem;

```

### 1.3 错误处理与常用 API

```c
// 错误码（常见值）
#define RKNN_SUCC               0   // 成功
#define RKNN_ERR_FAIL          -1   // 通用失败
#define RKNN_ERR_CTX_INVALID   -2   // 上下文无效
#define RKNN_ERR_MODEL_INVALID -3   // 模型格式错误
#define RKNN_ERR_PARAM_INVALID -4   // 参数非法
#define RKNN_ERR_API_UNIMPL   -10   // 该接口不支持

// 查询模型量化参数（用于反量化后处理）
// 调用 rknn_query(ctx, RKNN_QUERY_MODEL_ATTRIBUTE, &attr, sizeof(attr))
// attr 结构体中包含 quant_type 字段，指示模型是否量化
// 若为量化模型，需通过 RKNN_QUERY_NATIVE_OUTPUT_ATTR 获取量化后的原始数据

// 异步推理（适用于流水线场景，可同时启动多帧推理）
rknn_async_ctx async_ctx;
rknn_run(ctx, &async_ctx);
// ... 可在此处准备下一帧的预处理 ...
rknn_wait(ctx);
// 推理完成后获取输出

// 性能查询
typedef struct rknn_perf_run {
    int64_t run_time_us;     // 推理耗时（微秒）
    int64_t compile_time_us; // 编译耗时（首次运行）
} rknn_perf_run;
rknn_query(ctx, RKNN_QUERY_PERF_RUN, &perf, sizeof(perf));

```

---

## 二、环境配置

### 2.1 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.10)
project(yolo_rknn)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

# ============================================================
# 交叉编译工具链配置（在目标设备上编译时无需交叉编译链）
# 若需 PC 端交叉编译，请取消下方注释并配置路径:
#
# set(CMAKE_SYSTEM_NAME Linux)
# set(CMAKE_SYSTEM_PROCESSOR aarch64)
# set(CMAKE_C_COMPILER   /path/to/aarch64-linux-gnu-gcc)
# set(CMAKE_CXX_COMPILER /path/to/aarch64-linux-gnu-g++)
# set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
# set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
# ============================================================

# RKNN 库路径（设备上通常为 /usr/lib，RKSDK 安装在 /usr）
# 也可通过环境变量 RKNN_ROOT 指定，例如:
#   export RKNN_ROOT=/home/user/rknn-toolkit2
#   set(RKNN_LIB_PATH  ${RKNN_ROOT}/linux/aarch64/usr/lib)
#   set(RKNN_INCLUDE_PATH ${RKNN_ROOT}/linux/aarch64/usr/include)
set(RKNN_LIB_PATH  /usr/lib/aarch64-linux-gnu)
set(RKNN_INCLUDE_PATH /usr/include)

# OpenCV 路径（可选，适用于从 PC 编译后拷贝二进制）
set(OpenCV_DIR /usr/lib/aarch64-linux-gnu/cmake/opencv4)

# 依赖库
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# 可执行文件
add_executable(yolo_rknn main.cpp yolo_detector.cpp)

# 链接库
target_link_libraries(yolo_rknn
    ${OpenCV_LIBS}
    rknn_runtime
    pthread
    dl
)

# 编译选项：O3 优化 + ARMv8-A 指令集 + 内联优化
target_compile_options(yolo_rknn PRIVATE
    -O3
    -march=armv8-a
    -ftree-vectorize
    -fno-slp-vectorize
    -ffast-math
)

# 链接选项：LTO 可选（首次编译慢，运行更快）
# target_link_options(yolo_rknn PRIVATE -flto)

# 安装规则（可选）
install(TARGETS yolo_rknn DESTINATION bin)
install(FILES ${CMAKE_SOURCE_DIR}/coco.names DESTINATION share/${PROJECT_NAME})

```

### 2.2 交叉编译与依赖管理

```bash
# ===== 方式一：在设备（Rockchip 开发板）上直接编译 =====
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# ===== 方式二：PC 端交叉编译后传输到设备 =====
# 安装交叉编译工具链（Ubuntu/Debian）
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# 在 PC 上编译
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=toolchain-aarch64.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DOpenCV_DIR=/path/to/opencv-aarch64/lib/cmake/opencv4

# 将编译产物拷贝到开发板
scp yolo_rknn root@<board_ip>:/usr/local/bin/
scp best.rknn root@<board_ip>:/usr/local/share/yolo_rknn/

# ===== 依赖管理说明 =====
# 开发板上需要提前安装的运行时依赖:
#   sudo apt-get install libopencv-dev librknn-runtime
# 若使用静态链接，可将 rknn_runtime 和 libopencv 编译进二进制
# 静态编译时需添加:
#   target_link_options(yolo_rknn PRIVATE -static)

```

### 2.3 交叉编译工具链文件

```cmake
# toolchain-aarch64.cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER   /usr/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)

set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)

# 搜索模式：仅查找目标平台的库和头文件
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# 链接器标志：确保生成 aarch64 ELF
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--exclude-libs,ALL")

```

### 2.4 编译与运行

```bash
# 编译
mkdir build && cd build
cmake ..
make -j$(nproc)

# 运行
./yolo_rknn best.rknn test_image.jpg

```

### 2.5 常用编译问题排查

```bash
# 问题1: 找不到 rknn_runtime
# 解决: 在开发板上确认安装: sudo apt-get install librknn-runtime-dev

# 问题2: 链接错误 /usr/bin/ld: cannot find -lrknn_runtime
# 解决: 检查库路径: ls /usr/lib/aarch64-linux-gnu/librknn_runtime*
#       或在 CMakeLists.txt 中指定: link_directories(/usr/lib/aarch64-linux-gnu)

# 问题3: OpenCV 版本不匹配
# 解决: 使用 pkg-config 指定版本
#       export PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH
#       find_package(OpenCV 4.5 REQUIRED)

# 问题4: 交叉编译后运行报 "No such file or directory"
# 解决: 通常是缺少动态链接库，在开发板上执行 ldd 检查:
#       ldd ./yolo_rknn

```

---

## 三、C++ 推理实现

### 3.1 YOLO 检测器类

```cpp
// yolo_detector.h
#pragma once

#include <vector>
#include <string>
#include <mutex>
#include <memory>
#include <thread>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <functional>
#include <opencv2/opencv.hpp>
#include <rknn_api.h>

// 检测结果结构
struct Detection {
    int class_id;
    float confidence;
    int x1, y1, x2, y2;
    std::string class_name;
};

// 内存池：复用推理缓冲，避免频繁 malloc/free
class MemoryPool {
public:
    explicit MemoryPool(size_t size) : buffer_(new uint8_t[size]), size_(size), offset_(0) {}
    ~MemoryPool() { delete[] buffer_; }

    // 从池中分配一块内存（简单连续分配，实际生产可用 free-list）
    uint8_t* allocate(size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (offset_ + bytes > size_) return nullptr;
        uint8_t* ptr = buffer_ + offset_;
        offset_ += bytes;
        return ptr;
    }

    // 重置池（帧间复用无需释放）
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        offset_ = 0;
    }

private:
    uint8_t* buffer_;
    size_t size_;
    size_t offset_;
    std::mutex mutex_;
};

// YOLO 检测器主类
class YOLODetector {
public:
    YOLODetector();
    ~YOLODetector();

    // 禁止拷贝，允许移动
    YOLODetector(const YOLODetector&) = delete;
    YOLODetector& operator=(const YOLODetector&) = delete;
    YOLODetector(YOLODetector&&) = default;
    YOLODetector& operator=(YOLODetector&&) = default;

    // 加载模型，返回 0 表示成功
    int load_model(const std::string& model_path);

    // 单帧推理（线程安全）
    std::vector<Detection> detect(const cv::Mat& image);

    // 性能统计
    double get_avg_inference_time() const;
    uint64_t get_total_inferences() const { return frame_count_.load(); }

    // ===== 多线程支持：批量推理 =====
    // 在多个线程上并行处理不同图像（适用于多路摄像头场景）
    std::vector<std::vector<Detection>> detect_batch(
        const std::vector<cv::Mat>& images);

    // 流水线模式：异步提交任务并收集结果
    void enqueue(const cv::Mat& image,
                 std::function<void(const std::vector<Detection>&)> callback);
    void flush_pipeline();

    // 设置后处理参数（可在推理过程中动态调整）
    void set_conf_threshold(float conf)  { conf_threshold_ = conf; }
    void set_iou_threshold(float iou)   { iou_threshold_ = iou; }
    void set_max_detections(int max)     { max_detections_ = max; }

    // 获取模型输入/输出信息（调试用）
    void print_model_info() const;

private:
    // 预处理：图像缩放、归一化、填充
    cv::Mat preprocess(const cv::Mat& image, float& scale, int& pad_w, int& pad_h);

    // 后处理：解码模型输出、还原坐标、NMS
    // 注意：坐标解码需正确处理模型输出格式（xywh 或 corner 格式）
    std::vector<Detection> postprocess(float* outputs, const cv::Size& orig_size,
                                       float scale, int pad_w, int pad_h,
                                       bool is_quantized);

    // NMS 非极大值抑制
    std::vector<int> nms(const std::vector<Detection>& detections, float iou_threshold);

    // 加载类别名称（可选）
    void load_class_names(const std::string& names_path = "");

    // ===== 多线程内部方法 =====
    // 线程本地推理函数（每线程一个 YOLODetector 实例）
    static std::vector<Detection> worker_detect(YOLODetector& detector, const cv::Mat& image);

    // 流水线队列
    struct PipelineTask {
        cv::Mat image;
        std::function<void(const std::vector<Detection>&)> callback;
    };

    std::mutex pipeline_mutex_;
    std::condition_variable pipeline_cv_;
    std::queue<PipelineTask> pipeline_queue_;
    std::atomic<bool> pipeline_stop_{false};
    std::vector<std::thread> pipeline_threads_;

    // 成员变量
    rknn_context ctx_{nullptr};
    rknn_input_output_num io_num_;
    rknn_tensor_attr* input_attrs_{nullptr};
    rknn_tensor_attr* output_attrs_{nullptr};

    std::vector<std::string> class_names_;
    int img_size_{640};
    double avg_time_ms_{0.0};
    std::atomic<uint64_t> frame_count_{0};

    // 后处理阈值
    float conf_threshold_{0.25f};
    float iou_threshold_{0.45f};
    int max_detections_{300};

    // 内存池（用于预分配推理缓冲）
    std::shared_ptr<MemoryPool> pool_;

    // 推理模式标志
    bool model_quantized_{false};   // 模型是否量化
    bool use_zero_copy_{false};     // 是否启用零拷贝
};

```

```cpp
// yolo_detector.cpp
#include "yolo_detector.h"
#include <algorithm>
#include <chrono>
#include <numeric>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <stdexcept>

// ============================================================
// 构造函数 / 析构函数
// ============================================================
YOLODetector::YOLODetector() {}

YOLODetector::~YOLODetector() {
    // 停止流水线线程
    pipeline_stop_ = true;
    pipeline_cv_.notify_all();
    for (auto& t : pipeline_threads_) {
        if (t.joinable()) t.join();
    }

    // 释放张量属性数组
    delete[] input_attrs_;
    delete[] output_attrs_;

    // 销毁 RKNN 上下文
    if (ctx_ != nullptr) {
        rknn_destroy(ctx_);
        ctx_ = nullptr;
    }
}

// ============================================================
// 模型加载
// ============================================================
int YOLODetector::load_model(const std::string& model_path) {
    // 1. 读取模型文件到内存
    FILE* fp = fopen(model_path.c_str(), "rb");
    if (fp == nullptr) {
        printf("[ERROR] Failed to open model: %s\n", model_path.c_str());
        return -1;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        printf("[ERROR] fseek failed\n");
        return -1;
    }
    long model_size = ftell(fp);
    if (model_size <= 0) {
        fclose(fp);
        printf("[ERROR] Invalid model size\n");
        return -1;
    }
    fseek(fp, 0, SEEK_SET);
    std::unique_ptr<uint8_t[]> model_data(new uint8_t[model_size]);
    size_t bytes_read = fread(model_data.get(), 1, model_size, fp);
    fclose(fp);
    if (bytes_read != (size_t)model_size) {
        printf("[ERROR] Failed to read full model (%zu / %ld bytes)\n", bytes_read, model_size);
        return -1;
    }

    // 2. 初始化 RKNN 上下文
    rknn_init(&ctx_, model_data.get(), (uint32_t)model_size, 0);
    if (ctx_ == nullptr) {
        printf("[ERROR] rknn_init failed\n");
        return -1;
    }

    // 3. 查询输入输出数量
    rknn_init(&ctx_, model_data.get(), (uint32_t)model_size, 0);
    if (rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_)) != RKNN_SUCC) {
        printf("[ERROR] rknn_query IN_OUT_NUM failed\n");
        return -1;
    }
    printf("[INFO] Input num: %d, Output num: %d\n", io_num_.n_input, io_num_.n_output);

    // 4. 获取输入张量属性
    input_attrs_ = new rknn_tensor_attr[io_num_.n_input];
    for (uint32_t i = 0; i < io_num_.n_input; i++) {
        std::memset(&input_attrs_[i], 0, sizeof(rknn_tensor_attr));
        input_attrs_[i].index = i;
        if (rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attrs_[i], sizeof(rknn_tensor_attr)) != RKNN_SUCC) {
            printf("[ERROR] rknn_query INPUT_ATTR[%d] failed\n", i);
            return -1;
        }
        printf("[INFO] Input[%d]: name=%s, dims=[%d,%d,%d,%d], type=%d, format=%d\n",
               i, input_attrs_[i].name,
               input_attrs_[i].dims[0], input_attrs_[i].dims[1],
               input_attrs_[i].dims[2], input_attrs_[i].dims[3],
               input_attrs_[i].type, input_attrs_[i].format);
    }

    // 5. 获取输出张量属性（同时检测模型是否量化）
    output_attrs_ = new rknn_tensor_attr[io_num_.n_output];
    for (uint32_t i = 0; i < io_num_.n_output; i++) {
        std::memset(&output_attrs_[i], 0, sizeof(rknn_tensor_attr));
        output_attrs_[i].index = i;
        if (rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i], sizeof(rknn_tensor_attr)) != RKNN_SUCC) {
            printf("[ERROR] rknn_query OUTPUT_ATTR[%d] failed\n", i);
            return -1;
        }
        printf("[INFO] Output[%d]: name=%s, dims=[%d,%d,%d,%d], type=%d, quant_type=%d\n",
               i, output_attrs_[i].name,
               output_attrs_[i].dims[0], output_attrs_[i].dims[1],
               output_attrs_[i].dims[2], output_attrs_[i].dims[3],
               output_attrs_[i].type, output_attrs_[i].quant_type);
        // 记录量化状态，供后处理使用
        if (output_attrs_[i].quant_type != 0) {
            model_quantized_ = true;
        }
    }

    // 6. 初始化运行时
    rknn_init_runtime_option option;
    std::memset(&option, 0, sizeof(option));
    option.affinity = 0;  // 0=不限 CPU 核，可设为具体核 ID 以绑定性能核
    if (rknn_init_runtime(ctx_, &option) != RKNN_SUCC) {
        printf("[ERROR] rknn_init_runtime failed\n");
        return -1;
    }

    // 7. 预分配内存池（减少运行时 malloc 开销）
    size_t input_size = (size_t)input_attrs_[0].size;
    size_t output_size = 0;
    for (uint32_t i = 0; i < io_num_.n_output; i++) {
        output_size += (size_t)output_attrs_[i].size;
    }
    pool_ = std::make_shared<MemoryPool>(input_size + output_size + 4096); // 留 4KB 余量

    // 8. 加载类别名称
    load_class_names();

    printf("[INFO] Model loaded successfully! quantized=%d, use_zero_copy=%d\n",
           model_quantized_, use_zero_copy_);
    return 0;
}

void YOLODetector::load_class_names(const std::string& names_path) {
    class_names_.clear();
    std::string path = names_path.empty() ? "coco.names" : names_path;
    FILE* fp = fopen(path.c_str(), "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            // 去除末尾换行符
            size_t len = std::strlen(line);
            while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
                line[--len] = '\0';
            }
            if (len > 0) class_names_.push_back(std::string(line));
        }
        fclose(fp);
    }
    // 若文件不存在或为空，使用默认 80 类名称
    if (class_names_.empty()) {
        printf("[WARN] Class names file not found, using default names\n");
        for (int i = 0; i < 80; i++) {
            class_names_.push_back("class_" + std::to_string(i));
        }
    }
}

// ============================================================
// 预处理
// ============================================================
cv::Mat YOLODetector::preprocess(const cv::Mat& image, float& scale, int& pad_w, int& pad_h) {
    int h = image.rows;
    int w = image.cols;

    // 等比缩放，保持宽高比
    scale = (float)img_size_ / (float)std::max(h, w);
    int new_w = (int)(w * scale);
    int new_h = (int)(h * scale);

    // 裁剪到最近偶数（NPU 对尺寸有对齐要求）
    new_w = (new_w / 2) * 2;
    new_h = (new_h / 2) * 2;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h));

    // 计算填充（灰色填充，YOLOv5/v8 标准做法）
    pad_w = (img_size_ - new_w) / 2;
    pad_h = (img_size_ - new_h) / 2;

    cv::Mat padded(img_size_, img_size_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(pad_w, pad_h, new_w, new_h)));

    // BGR -> RGB，并归一化到 [0, 1]
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    return rgb;
}

// ============================================================
// 后处理（已修复坐标解码 bug）
// ============================================================
//
// 坐标解码说明（关键修复）：
// YOLOv5/v8 模型的输出格式为 [bx, by, bw, bh, c1, c2, ..., cN]
//   - (bx, by) 是 anchor 中心相对于网格的偏移，范围通常为 [0, 1]
//   - (bw, bh) 是 anchor 宽高相对于网格的倍数
// 正确解码步骤：
//   1. 将模型输出的 xywh 还原到"缩放+填充后"的图像坐标系
//      x_center = (x - 0.5*w) * 640 / scale  （不是直接乘 scale）
//      注意：若模型输出已经是归一化坐标（0~1），则需先还原到 640x640 空间
//   2. 再减去 padding，映射回原始图像坐标
//   3. 最后除以 scale 映射到原始分辨率
//
// 对于 rknn_toolkit2 导出的 YOLO 模型，输出通常是：
//   output[0]: [1, 4+num_classes, 8400]  (YOLOv5 head)
//   output[0]: [1, 4+num_classes, H*W*A]  (YOLOv8/v10 单头部)
// 这里按 YOLOv5 常见格式处理。
// ============================================================
std::vector<Detection> YOLODetector::postprocess(float* outputs, const cv::Size& orig_size,
                                                   float scale, int pad_w, int pad_h,
                                                   bool is_quantized) {
    std::vector<Detection> detections;

    int num_classes = 80;  // COCO 80 类，可根据模型调整
    int num_anchors = 8400;  // YOLOv5 640x640 的 anchor 数量 (20*20 + 40*40 + 80*80) = 400+1600+6400=8400

    for (int i = 0; i < num_anchors; i++) {
        // 提取 xywh 和类别分数
        float bx = outputs[i * (4 + num_classes) + 0];
        float by = outputs[i * (4 + num_classes) + 1];
        float bw = outputs[i * (4 + num_classes) + 2];
        float bh = outputs[i * (4 + num_classes) + 3];

        float* class_scores = &outputs[i * (4 + num_classes) + 4];
        int class_id = (int)std::distance(class_scores,
            std::max_element(class_scores, class_scores + num_classes));
        float confidence = class_scores[class_id];

        // 过滤低置信度检测
        if (confidence < conf_threshold_) continue;

        // 坐标解码（修复后的版本）：
        // 模型输出的 x,y 是 anchor 中心偏移，w,h 是 anchor 宽高
        // 将 anchor 中心映射到 640x640 图像空间，再还原到原图
        //
        // 注意：原代码的 bug 是将 xywh 直接乘 scale 后减 padding，
        // 导致坐标方向反了且没有正确还原到 640 空间。
        float x_center = (bx + 0.5f) * img_size_;  // 还原到 640 空间
        float y_center = (by + 0.5f) * img_size_;
        float width    = bw * img_size_;
        float height   = bh * img_size_;

        // 从 640 空间映射回原图（先减 padding，再除以 scale）
        float cx = (x_center - (float)pad_w) / scale;
        float cy = (y_center - (float)pad_h) / scale;
        float w  = width  / scale;
        float h  = height / scale;

        // 计算边界框角点并裁剪到图像范围内
        float x1 = std::max(0.0f, cx - w / 2.0f);
        float y1 = std::max(0.0f, cy - h / 2.0f);
        float x2 = std::max(0.0f, cx + w / 2.0f);
        float y2 = std::max(0.0f, cy + h / 2.0f);

        x1 = std::min(x1, (float)orig_size.width);
        y1 = std::min(y1, (float)orig_size.height);
        x2 = std::min(x2, (float)orig_size.width);
        y2 = std::min(y2, (float)orig_size.height);

        // 确保 x1 < x2, y1 < y2
        if (x1 >= x2) std::swap(x1, x2);
        if (y1 >= y2) std::swap(y1, y2);

        Detection det;
        det.class_id     = class_id;
        det.confidence   = confidence;
        det.x1           = (int)x1;
        det.y1           = (int)y1;
        det.x2           = (int)x2;
        det.y2           = (int)y2;
        det.class_name   = class_names_.empty()
            ? "class_" + std::to_string(class_id)
            : (class_id < (int)class_names_.size() ? class_names_[class_id] : "unknown");
        detections.push_back(det);
    }

    // 按置信度降序排序（NMS 前预处理）
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    // NMS 去重
    std::vector<int> indices = nms(detections, iou_threshold_);

    // 限制最大检测数
    std::vector<Detection> result;
    int limit = std::min(max_detections_, (int)indices.size());
    result.reserve(limit);
    for (int k = 0; k < limit; k++) {
        result.push_back(detections[indices[k]]);
    }

    return result;
}

// ============================================================
// NMS（非极大值抑制）
// ============================================================
std::vector<int> YOLODetector::nms(const std::vector<Detection>& detections, float iou_threshold) {
    std::vector<int> indices;
    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;

        indices.push_back((int)i);

        const Detection& det_i = detections[i];
        float i1_x1 = det_i.x1, i1_y1 = det_i.y1;
        float i1_x2 = det_i.x2, i1_y2 = det_i.y2;
        float i1_area = std::max(0.0f, i1_x2 - i1_x1) * std::max(0.0f, i1_y2 - i1_y1);

        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;

            const Detection& det_j = detections[j];
            float j_x1 = std::max(i1_x1, (float)det_j.x1);
            float j_y1 = std::max(i1_y1, (float)det_j.y1);
            float j_x2 = std::min(i1_x2, (float)det_j.x2);
            float j_y2 = std::min(i1_y2, (float)det_j.y2);

            float inter_area = std::max(0.0f, j_x2 - j_x1) * std::max(0.0f, j_y2 - j_y1);
            float j_area = std::max(0.0f, (float)det_j.x2 - det_j.x1) *
                           std::max(0.0f, (float)det_j.y2 - det_j.y1);
            float union_area = i1_area + j_area - inter_area;
            float iou = union_area > 1e-6f ? inter_area / union_area : 0.0f;

            if (iou > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return indices;
}

// ============================================================
// 单帧推理（带完整错误处理和性能统计）
// ============================================================
std::vector<Detection> YOLODetector::detect(const cv::Mat& image) {
    if (image.empty()) {
        printf("[ERROR] Empty input image\n");
        return {};
    }
    if (ctx_ == nullptr) {
        printf("[ERROR] Model not loaded, call load_model() first\n");
        return {};
    }

    auto start = std::chrono::high_resolution_clock::now();

    // 预处理
    float scale;
    int pad_w = 0, pad_h = 0;
    cv::Mat input = preprocess(image, scale, pad_w, pad_h);

    // 从内存池获取输入缓冲（零拷贝优化）
    rknn_input inputs[1];
    std::memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_INPUT_TENSOR;

    if (use_zero_copy_ && pool_) {
        // 零拷贝模式：直接从池分配 NPU 兼容内存
        inputs[0].mem = rknn_create_mem(ctx_, (uint32_t)input.cols * input.rows * input.channels() * sizeof(float));
        if (inputs[0].mem) {
            inputs[0].pass_through = 1;
            // 将预处理数据 memcpy 到 NPU 内存
            std::memcpy(inputs[0].mem->virt_addr, input.data,
                        input.cols * input.rows * input.channels() * sizeof(float));
        } else {
            // NPU 内存分配失败，回退到普通模式
            inputs[0].pass_through = 0;
            inputs[0].buf = input.data;
            inputs[0].size = (uint32_t)(input.cols * input.rows * input.channels() * sizeof(float));
        }
    } else {
        // 普通模式：RKNN 内部拷贝
        inputs[0].pass_through = 0;
        inputs[0].buf = input.data;
        inputs[0].size = (uint32_t)(input.cols * input.rows * input.channels() * sizeof(float));
    }

    // 设置输入并执行推理
    if (rknn_inputs_set(ctx_, 1, inputs) != RKNN_SUCC) {
        printf("[ERROR] rknn_inputs_set failed\n");
        if (use_zero_copy_ && inputs[0].mem) rknn_destroy_mem(ctx_, inputs[0].mem);
        return {};
    }

    if (rknn_run(ctx_, nullptr) != RKNN_SUCC) {
        printf("[ERROR] rknn_run failed\n");
        if (use_zero_copy_ && inputs[0].mem) rknn_destroy_mem(ctx_, inputs[0].mem);
        return {};
    }

    // 获取输出
    rknn_output outputs[1];
    std::memset(outputs, 0, sizeof(outputs));
    outputs[0].index = 0;
    // want_float=1：让 RKNN 运行时自动反量化，返回 float32 数据
    // 若 want_float=0，则返回量化后的原始数据（速度略快，但后处理复杂）
    outputs[0].want_float = 1;
    outputs[0].repeatable = 0;

    if (rknn_outputs_get(ctx_, 1, outputs, NULL) != RKNN_SUCC) {
        printf("[ERROR] rknn_outputs_get failed\n");
        if (use_zero_copy_ && inputs[0].mem) rknn_destroy_mem(ctx_, inputs[0].mem);
        return {};
    }

    // 后处理（注意传递 is_quantized 标志）
    std::vector<Detection> detections = postprocess(
        (float*)outputs[0].buf, image.size(), scale, pad_w, pad_h, model_quantized_);

    // 释放输出缓冲区
    rknn_outputs_release(ctx_, 1, outputs);

    // 释放零拷贝 NPU 内存
    if (use_zero_copy_ && inputs[0].mem) {
        rknn_destroy_mem(ctx_, inputs[0].mem);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // 滑动平均更新
    frame_count_.fetch_add(1);
    uint64_t count = frame_count_.load();
    avg_time_ms_ = avg_time_ms_ * (count - 1) / count + elapsed_ms / count;

    return detections;
}

// ============================================================
// 批量推理（多线程）
// ============================================================
std::vector<std::vector<Detection>> YOLODetector::detect_batch(
    const std::vector<cv::Mat>& images) {
    std::vector<std::future<std::vector<Detection>>> futures;
    futures.reserve(images.size());

    for (const auto& img : images) {
        // std::async 默认策略可能使用线程池，也可指定 std::launch::async 强制新线程
        futures.push_back(std::async(std::launch::async,
            [this, img]() { return worker_detect(*this, img); }));
    }

    std::vector<std::vector<Detection>> results;
    results.reserve(futures.size());
    for (auto& f : futures) {
        results.push_back(f.get());
    }
    return results;
}

std::vector<Detection> YOLODetector::worker_detect(YOLODetector& detector, const cv::Mat& image) {
    return detector.detect(image);
}

// ============================================================
// 流水线模式（异步提交，非阻塞）
// ============================================================
void YOLODetector::enqueue(const cv::Mat& image,
                            std::function<void(const std::vector<Detection>&)> callback) {
    {
        std::lock_guard<std::mutex> lock(pipeline_mutex_);
        pipeline_queue_.push({image, callback});
    }
    pipeline_cv_.notify_one();
}

void YOLODetector::flush_pipeline() {
    pipeline_stop_ = true;
    pipeline_cv_.notify_all();
    for (auto& t : pipeline_threads_) {
        if (t.joinable()) t.join();
    }
    pipeline_threads_.clear();
    pipeline_stop_ = false;
}

// 打印模型信息（调试辅助）
void YOLODetector::print_model_info() const {
    if (!input_attrs_ || !output_attrs_) {
        printf("[WARN] Model not loaded\n");
        return;
    }
    printf("=== Model Info ===\n");
    printf("Input count:  %d\n", io_num_.n_input);
    printf("Output count: %d\n", io_num_.n_output);
    for (uint32_t i = 0; i < io_num_.n_input; i++) {
        printf("  Input[%d]  %s: [%d,%d,%d,%d] type=%d quant=%d\n",
               i, input_attrs_[i].name,
               input_attrs_[i].dims[0], input_attrs_[i].dims[1],
               input_attrs_[i].dims[2], input_attrs_[i].dims[3],
               input_attrs_[i].type, input_attrs_[i].quant_type);
    }
    for (uint32_t i = 0; i < io_num_.n_output; i++) {
        printf("  Output[%d] %s: [%d,%d,%d,%d] type=%d quant=%d\n",
               i, output_attrs_[i].name,
               output_attrs_[i].dims[0], output_attrs_[i].dims[1],
               output_attrs_[i].dims[2], output_attrs_[i].dims[3],
               output_attrs_[i].type, output_attrs_[i].quant_type);
    }
    printf("==================\n");
}

### 3.2 主程序

```cpp
// main.cpp
#include "yolo_detector.h"
#include <iostream>
#include <signal.h>

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " <model.rknn> [image.jpg/video.mp4/camera_id]\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program << " best.rknn test.jpg\n";
    std::cout << "  " << program << " best.rknn video.mp4\n";
    std::cout << "  " << program << " best.rknn 0\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return -1;
    }
    
    std::string model_path = argv[1];
    std::string input_path = argc > 2 ? argv[2] : "";
    
    YOLODetector detector;
    
    if (detector.load_model(model_path) != 0) {
        std::cout << "Failed to load model!" << std::endl;
        return -1;
    }
    
    bool is_image = (input_path.find(".jpg") != std::string::npos || 
                     input_path.find(".png") != std::string::npos);
    bool is_video = (input_path.find(".mp4") != std::string::npos || 
                     input_path.find(".avi") != std::string::npos);
    bool is_camera = !input_path.empty() && input_path.find_first_of("0123456789") == 0;
    
    if (is_image) {
        cv::Mat image = cv::imread(input_path);
        if (image.empty()) {
            std::cout << "Failed to load image: " << input_path << std::endl;
            return -1;
        }
        
        auto detections = detector.detect(image);
        
        for (const auto& det : detections) {
            cv::rectangle(image, cv::Rect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1),
                         cv::Scalar(0, 255, 0), 2);
            std::string label = det.class_name + " " + std::to_string(det.confidence).substr(0, 4);
            cv::putText(image, label, cv::Point(det.x1, det.y1 - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }
        
        cv::imwrite("result.jpg", image);
        std::cout << "Result saved to result.jpg" << std::endl;
        
    } else if (is_video || is_camera) {
        int camera_id = is_camera ? std::stoi(input_path) : -1;
        cv::VideoCapture cap;
        
        if (is_video) {
            cap.open(input_path);
        } else {
            cap.open(camera_id);
        }
        
        if (!cap.isOpened()) {
            std::cout << "Failed to open input!" << std::endl;
            return -1;
        }
        
        std::cout << "Running inference... Press 'q' to quit" << std::endl;
        
        cv::Mat frame;
        int frame_count = 0;
        
        while (cap.read(frame)) {
            frame_count++;
            
            auto detections = detector.detect(frame);
            
            for (const auto& det : detections) {
                cv::rectangle(frame, cv::Rect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1),
                             cv::Scalar(0, 255, 0), 2);
                std::string label = det.class_name + " " + std::to_string(det.confidence).substr(0, 4);
                cv::putText(frame, label, cv::Point(det.x1, det.y1 - 10),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            }
            
            std::string fps_str = "FPS: " + std::to_string(1000.0 / detector.get_avg_inference_time()).substr(0, 4);
            cv::putText(frame, fps_str, cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            
            cv::imshow("YOLO NPU C++", frame);
            
            if (cv::waitKey(1) == 'q') break;
        }
        
        cap.release();
        cv::destroyAllWindows();
        
        std::cout << "\nTotal frames processed: " << frame_count << std::endl;
        std::cout << "Average inference time: " << detector.get_avg_inference_time() << " ms" << std::endl;
        
    } else {
        cv::Mat image = cv::imread("test_image.jpg");
        if (image.empty()) {
            std::cout << "Please provide an image path!" << std::endl;
            return -1;
        }
        
        auto detections = detector.detect(image);
        std::cout << "Detected " << detections.size() << " objects:" << std::endl;
        for (const auto& det : detections) {
            std::cout << "  " << det.class_name << ": " << det.confidence 
                      << " at [" << det.x1 << "," << det.y1 << "," 
                      << det.x2 << "," << det.y2 << "]" << std::endl;
        }
    }
    
    std::cout << "\nDone." << std::endl;
    return 0;
}

```

---

## 四、性能优化

### 4.1 内存预分配详解

内存预分配是降低推理延迟抖动（jitter）最有效的手段之一。在嵌入式平台上，`malloc/free` 的系统调用开销可能达到数微秒到数十微秒，频繁分配会导致帧率波动。

```cpp
// ===== 方案一：RAII 智能指针管理单次分配 =====
class PreAllocatedYOLO {
private:
    // 输入缓冲：640x640x3xfloat32 = 4.9MB
    std::unique_ptr<float[]> input_buf_;
    // 输出缓冲：假设模型输出 [1,84,8400] float32 = 2.8MB
    std::unique_ptr<float[]> output_buf_;
    // 用于 NMS 的临时检测列表（避免每次 detect 重新分配）
    std::vector<Detection> detection_pool_;

public:
    PreAllocatedYOLO()
        : input_buf_(std::make_unique<float[]>(640 * 640 * 3)),
          output_buf_(std::make_unique<float[]>(1 * 84 * 8400)),
          detection_pool_.reserve(500)  // 预分配检测列表容量
    {}

    // 在预处理阶段复用缓冲，避免 realloc
    void prepare_input(const cv::Mat& image, float* buf,
                       float& scale, int& pad_w, int& pad_h) {
        // 直接写入预分配缓冲，preprocess 改为输出到指定 buffer
        // 后续调用 rknn_inputs_set 时传入 buf 指针
        // ... (实现略，参考 preprocess 逻辑)
    }
};

// ===== 方案二：使用 RKNN NPU 内存（rknn_create_mem） =====
// 适合需要 DMA 传输的场景，数据直接存放在 NPU 可访问的物理内存
int setup_dma_memory(YOLODetector& detector) {
    // 分配 NPU 侧内存（页锁定，支持 DMA）
    uint32_t input_size  = 1 * 3 * 640 * 640 * sizeof(float);
    uint32_t output_size = 1 * 84 * 8400 * sizeof(float);

    rknn_tensor_mem* input_mem  = rknn_create_mem(detector.ctx_, input_size);
    rknn_tensor_mem* output_mem = rknn_create_mem(detector.ctx_, output_size);

    if (!input_mem || !output_mem) {
        printf("[ERROR] Failed to allocate NPU memory\n");
        return -1;
    }

    // 零拷贝推理：设置 pass_through=1，RKNN 直接使用 NPU 内存
    rknn_input in = {};
    in.index    = 0;
    in.type     = RKNN_INPUT_TENSOR;
    in.pass_through = 1;   // 关键：启用零拷贝
    in.mem      = input_mem;

    // 注意：需要先将数据拷贝到 input_mem->virt_addr（若支持 CPU 访问）
    // 或通过 DMA buffer API 直接写入（见 4.4 节）

    // 输出侧：同样使用 NPU 内存
    rknn_output out = {};
    out.index    = 0;
    out.want_float = 1;
    out.mem      = output_mem;

    // ... 后续推理流程不变 ...

    // 释放时注意顺序：先 destroy 上下文，再 destroy mem
    // rknn_destroy_mem 不是线程安全的，应在主线程中统一释放
    rknn_destroy_mem(detector.ctx_, input_mem);
    rknn_destroy_mem(detector.ctx_, output_mem);
    return 0;
}

```

### 4.2 多 NPU 核心并行（修复竞态条件）

RK3588 拥有 3 个独立的 NPU 核心，通过为每个核心创建独立的 `rknn_context` 可以实现并行推理。**原代码存在竞态条件**：多个线程共享同一个 `YOLODetector` 实例（共享同一个 `ctx_`），而 `rknn_context` 并非线程安全，并发调用 `rknn_run` 会导致结果混乱。

```cpp
// ===== 修复后的多 NPU 并行推理 =====
// 每个 NPU 核心拥有独立的 Detector 实例，彻底避免竞态

#include <thread>
#include <mutex>
#include <future>
#include <atomic>

class ParallelYOLODetector {
private:
    // 每个 NPU 核心一个独立的检测器实例（关键：不共享 ctx_）
    std::vector<std::unique_ptr<YOLODetector>> detectors_;
    // 结果合并锁（仅在合并阶段持有，不影响推理性能）
    std::mutex results_mutex_;
    // NPU 核心数量（根据硬件自动检测）
    int num_npus_;

public:
    ParallelYOLODetector() : num_npus_(get_npu_core_count()) {
        detectors_.reserve(num_npus_);
        for (int i = 0; i < num_npus_; i++) {
            detectors_.emplace_back(std::make_unique<YOLODetector>());
        }
    }

    // 为每个 NPU 核心加载模型（每个上下文独立，无共享状态）
    int load_model(const std::string& model_path) {
        for (int i = 0; i < num_npus_; i++) {
            int ret = detectors_[i]->load_model(model_path);
            if (ret != 0) {
                printf("[ERROR] Failed to load model on NPU core %d\n", i);
                return ret;
            }
            printf("[INFO] Model loaded on NPU core %d\n", i);
        }
        return 0;
    }

    // 多路并行推理：每路图像分配给不同的 NPU 核心
    // 返回结果顺序与输入图像顺序一一对应
    std::vector<std::vector<Detection>> detect_parallel(
        const std::vector<cv::Mat>& images) {

        if (images.empty()) return {};

        std::vector<std::future<std::vector<Detection>>> futures;
        futures.reserve(images.size());

        for (size_t i = 0; i < images.size(); i++) {
            // 关键修复：使用 i % num_npus_ 将图像轮询分配到各 NPU 核心
            // 每个 future 绑定到独立的 YOLODetector 实例，无共享状态
            int core_idx = i % num_npus_;
            futures.push_back(std::async(std::launch::async,
                [this, core_idx, &images, i]() {
                    return detectors_[core_idx]->detect(images[i]);
                }));
        }

        // 按顺序收集结果（保持输入顺序）
        std::vector<std::vector<Detection>> results;
        results.reserve(futures.size());
        for (auto& f : futures) {
            results.push_back(f.get());
        }
        return results;
    }

    // 三路视频流并发推理（典型嵌入式场景）
    std::tuple<std::vector<Detection>, std::vector<Detection>, std::vector<Detection>>
    detect_triple(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& img3) {
        auto f1 = std::async(std::launch::async,
            [&]() { return detectors_[0]->detect(img1); });
        auto f2 = std::async(std::launch::async,
            [&]() { return detectors_[1]->detect(img2); });
        auto f3 = std::async(std::launch::async,
            [&]() { return detectors_[2]->detect(img3); });

        return {f1.get(), f2.get(), f3.get()};
    }

private:
    // 检测 NPU 核心数（读取 /proc 或 sysfs）
    int get_npu_core_count() {
        // 方法1：尝试读取设备节点
        FILE* fp = fopen("/proc/driver/rockchip/npu/info", "r");
        if (fp) {
            // 统计 "Task" 行数（每行一个任务）
            int count = 0;
            char line[256];
            while (fgets(line, sizeof(line), fp) && count < 8) {
                if (strstr(line, "Task")) count++;
            }
            fclose(fp);
            return std::max(1, count);
        }
        // 方法2：默认假设 3 核（RK3588）
        return 3;
    }
};

```

### 4.3 零拷贝推理（Zero-Copy Inference）

零拷贝技术通过将数据直接写入 NPU 可访问的内存区域，跳过 CPU→NPU 的数据搬运，可将推理延迟降低 1~3ms。

```cpp
// ===== 零拷贝推理完整流程 =====
// 适用场景：输入数据已在 DMA buffer 中（如 camera driver 直接输出到物理内存）

// 步骤 1：创建 NPU 可访问的输入内存
rknn_tensor_mem* input_mem = rknn_create_mem(ctx_,
    (uint32_t)(640 * 640 * 3 * sizeof(float)));
if (!input_mem) {
    printf("[ERROR] rknn_create_mem failed\n");
    return;
}

// 步骤 2：将图像数据直接拷贝到 NPU 内存（virt_addr 可被 CPU 访问）
// 若使用 DMA buffer（phys_addr 模式），需通过 mmap 或 ioctl 写入
std::memcpy(input_mem->virt_addr, cpu_buffer, input_size);

// 步骤 3：配置输入（pass_through=1 启用零拷贝模式）
rknn_input inputs[1] = {};
inputs[0].index         = 0;
inputs[0].type          = RKNN_INPUT_TENSOR;
inputs[0].pass_through  = 1;   // 关键：告知 RKNN 不拷贝，直接使用 NPU 内存
inputs[0].mem           = input_mem;

// 步骤 4：执行推理
rknn_inputs_set(ctx_, 1, inputs);
rknn_run(ctx_, nullptr);

// 步骤 5：获取输出（同样可使用 NPU 内存）
rknn_output outputs[1] = {};
outputs[0].index      = 0;
outputs[0].want_float = 1;   // 0=保留量化格式，1=反量化为 float32
outputs[0].repeatable = 0;
rknn_outputs_get(ctx_, 1, outputs, NULL);

// 步骤 6：清理
rknn_outputs_release(ctx_, 1, outputs);
rknn_destroy_mem(ctx_, input_mem);

```

> **注意事项**：
> - `pass_through=1` 模式下，RKNN **不会**拷贝输入数据，必须确保 `mem` 指向的内存生命周期覆盖整个推理过程
> - `rknn_create_mem` 分配的内存位于 NPU 专属区域，无法直接用 `std::memcpy` 从用户空间写入（取决于平台是否支持 CPU-NPU 共享内存）
> - 对于 RK3588，建议使用 `RKNN_MEM_TYPE_DMA_BUF` 类型创建 DMA 缓冲区（需通过 `rknn_create_mem_attr` 指定）

### 4.4 DMA Buffer 使用

```cpp
// ===== DMA Buffer 分配与使用 =====
// DMA buffer 是 Linux 内核提供的物理连续内存，支持 NPU 通过 DMA 直接访问

// 方式一：通过 rknn_create_mem_attr 创建（推荐）
rknn_tensor_mem_attr mem_attr = {};
mem_attr.size    = 640 * 640 * 3 * sizeof(float);
mem_attr.mem_type = RKNN_MEM_TYPE_DMA_BUF;  // DMA 缓冲模式
mem_attr.flags    = RKNN_MEM_FLAG_READ_WRITE | RKNN_MEM_FLAG_CACHED;

rknn_tensor_mem* dma_mem = rknn_create_mem(ctx_, &mem_attr);
if (!dma_mem) {
    printf("[ERROR] Failed to create DMA buffer\n");
}

// 写入数据（若 cached 模式，CPU 可直接写入 virt_addr）
std::memcpy(dma_mem->virt_addr, input_data, mem_attr.size);

// 步骤二：设置为 pass_through 输入
rknn_input in = {};
in.index         = 0;
in.pass_through  = 1;
in.mem           = dma_mem;
rknn_inputs_set(ctx_, 1, &in);

// 推理完成后释放
rknn_destroy_mem(ctx_, dma_mem);

// ===== 方式二：系统级 DMA buffer（适用于 camera 直连场景） =====
// 对于某些 camera driver，可直接将帧数据映射到 NPU 内存，完全避免拷贝:
//
// 1. 在设备树中配置 camera → NPU DMA channel
// 2. 使用 ion/ionallocator 分配物理连续内存
// 3. camera driver 直接写入该物理内存
// 4. NPU 推理时 pass_through=1 直接引用

// ion alloc 示例（需要 root 权限）：
// int fd = open("/dev/ion", O_RDWR);
// struct ion_allocation_data alloc = {.len = size, .align = 4096, .heap_id_mask = 1, .flags = 0};
// ioctl(fd, ION_IOC_ALLOC, &alloc);
// struct ion_fd_data fd_data = {.handle = alloc.handle};
// ioctl(fd, ION_IOC_MAP, &fd_data);
// void* mapped = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd_data.fd, 0);

```

---

## 五、性能分析

### 5.1 NPU 监控

```bash
# 查看 NPU 使用情况
cat /proc/driver/rockchip/npu/info

# 查看 NPU 频率
cat /sys/class/devfreq/*npu*/freqlist

# 设置 NPU 性能模式
echo "performance" > /sys/class/devfreq/*npu*/governor

# 监控 NPU 利用率
while true; do
    cat /proc/driver/rockchip/npu/info | grep -E "Utilization|Power"
    sleep 1
done

```

### 5.2 系统级性能分析工具

```bash
# ===== top / htop：CPU 和内存概览 =====
htop

# ===== perf：CPU 性能计数器和火焰图 =====
# 安装
sudo apt install linux-perf

# 录制推理过程的 CPU 采样（10秒）
perf record -g -p $(pgrep yolo_rknn) -- sleep 10
# 生成报告
perf report

# 火焰图（可视化调用栈分布）
perf record -g -F 999 -p $(pgrep yolo_rknn) -- ./yolo_rknn model.rknn test.mp4
perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg

# ===== strace：系统调用追踪 =====
strace -e trace=write,mmap,munmap,malloc,free -c ./yolo_rknn model.rknn test.jpg
# 详细追踪（写入文件）
strace -f -o trace.log ./yolo_rknn model.rknn test.mp4

# ===== valgrind：内存泄漏检测 =====
valgrind --leak-check=full --show-leak-kinds=all ./yolo_rknn model.rknn test.jpg

# ===== RKNN 内置性能查询 =====
# 代码中调用：
rknn_perf_run perf;
rknn_query(ctx, RKNN_QUERY_PERF_RUN, &perf, sizeof(perf));
printf("Run time: %.2f us\n", (double)perf.run_time_us / 1000.0);

# ===== 时间戳追踪（代码内埋点） =====
#include <sys/time.h>
struct timeval tv;
gettimeofday(&tv, nullptr);
long long ms = tv.tv_sec * 1000 + tv.tv_usec / 1000;
printf("[T+%lldms] Preprocess done\n", ms - base_ms);

```

### 5.3 功耗与温度监控

```bash
# 查看 NPU 功耗
cat /sys/class/powercap/rockchip-powercap/*/energy

# 查看 SoC 温度
cat /sys/class/thermal/thermal_zone*/temp
# 转换为摄氏度: temp / 1000

# 实时监控
watch -n 1 'echo "NPU Temp: $(cat /sys/class/thermal/thermal_zone0/temp) mC"; cat /proc/driver/rockchip/npu/info | grep -E "Utilization|Power"'

# 设置 NPU 频率档位（高频省电权衡）
# 读取当前频率
cat /sys/class/devfreq/fd8c0000.npu/devfreq/fd8c0000.npu/trans_table
# 查看可用档位
cat /sys/class/devfreq/fd8c0000.npu/freqlist

```

### 5.4 推理性能分析代码模板

```cpp
// 分段计时，定位性能瓶颈
struct TimingTracker {
    std::chrono::high_resolution_clock::time_point start;
    std::vector<std::pair<std::string, double>> stages;

    void mark(const std::string& name) {
        auto now = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - start).count();
        stages.push_back({name, ms});
        start = now;
    }

    void print() const {
        double prev = 0;
        for (auto& [name, accumulated] : stages) {
            double duration = accumulated - prev;
            printf("  %-20s %8.2f ms  (cumulative: %8.2f ms)\n",
                   name.c_str(), duration, accumulated);
            prev = accumulated;
        }
    }
};

// 使用示例
TimingTracker tracker;
auto t0 = std::chrono::high_resolution_clock::now();

// 预处理
cv::Mat input = preprocess(image, scale, pad_w, pad_h);
tracker.mark("preprocess");

// 推理
rknn_inputs_set(ctx_, 1, &in);
rknn_run(ctx_, nullptr);
tracker.mark("rknn_run");

// 后处理
auto results = postprocess(...);
tracker.mark("postprocess");

// 打印
tracker.print();

```

---

## 六、生产环境部署

在生产环境中部署 NPU 推理服务时，除功能正确性外，还需关注错误恢复、日志管理、运行监控和系统稳定性。

### 6.1 错误处理与恢复

```cpp
// 健壮的推理循环：自动处理临时故障并恢复
class RobustYOLOService {
private:
    YOLODetector detector_;
    std::mutex service_mutex_;
    std::atomic<bool> healthy_{false};

    // 指数退避重试策略
    static constexpr int MAX_RETRIES  = 3;
    static constexpr int BASE_RETRY_MS = 100;

public:
    bool start(const std::string& model_path) {
        for (int retry = 0; retry < MAX_RETRIES; retry++) {
            try {
                int ret = detector_.load_model(model_path);
                if (ret == 0) {
                    healthy_.store(true);
                    printf("[INFO] Service started successfully\n");
                    return true;
                }
            } catch (const std::exception& e) {
                printf("[ERROR] load_model exception: %s (retry %d/%d)\n",
                       e.what(), retry + 1, MAX_RETRIES);
            }
            // 指数退避
            std::this_thread::sleep_for(
                std::chrono::milliseconds(BASE_RETRY_MS << retry));
        }
        healthy_.store(false);
        return false;
    }

    // 健康检查：服务是否可用
    bool is_healthy() const { return healthy_.load(); }

    // 带超时和重试的单帧推理
    std::vector<Detection> infer_safe(const cv::Mat& image) {
        if (!healthy_.load()) return {};

        std::vector<Detection> result;
        for (int retry = 0; retry < MAX_RETRIES; retry++) {
            try {
                result = detector_.detect(image);
                return result;  // 成功返回
            } catch (const std::exception& e) {
                printf("[WARN] detect retry %d: %s\n", retry + 1, e.what());
                std::this_thread::sleep_for(std::chrono::milliseconds(50 << retry));
            }
        }
        return {};  // 所有重试失败
    }

    // 定期健康检查（可配合 watchdog 使用）
    void health_check_loop() {
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            // 检查 NPU 状态
            FILE* fp = fopen("/proc/driver/rockchip/npu/info", "r");
            if (!fp) {
                printf("[ALERT] NPU driver not responding!\n");
                healthy_.store(false);
                continue;
            }
            fclose(fp);
            healthy_.store(true);
        }
    }
};

```

### 6.2 结构化日志

```cpp
// 生产级日志：分级、带时间戳、避免频繁 I/O
#include <fstream>
#include <sstream>
#include <iomanip>

class ProductionLogger {
public:
    enum Level { DEBUG, INFO, WARN, ERROR, FATAL };

    static ProductionLogger& instance() {
        static ProductionLogger logger;
        return logger;
    }

    void log(Level level, const char* file, int line, const char* fmt, ...) {
        // 时间戳
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::fprintf(stderr, "[%02d:%02d:%02d.%03lld] [%s] ",
            (int)std::localtime(&time_t)->tm_hour,
            (int)std::localtime(&time_t)->tm_min,
            (int)std::localtime(&time_t)->tm_sec,
            (long long)ms.count(),
            level_name(level));

        va_list args;
        va_start(args, fmt);
        std::vfprintf(stderr, fmt, args);
        va_end(args);

        std::fprintf(stderr, "  (%s:%d)\n", file, line);
        std::fflush(stderr);

        // 同时写入文件日志（缓冲写入，避免频繁 I/O）
        if (level >= WARN) {
            std::lock_guard<std::mutex> lock(file_mutex_);
            if (file_log_.is_open()) {
                file_log_ << std::localtime(&time_t) << " [" << level_name(level) << "] "
                          << fmt << "  (" << file << ":" << line << ")\n";
                file_log_.flush();
            }
        }
    }

    void open_file_log(const std::string& path) {
        file_log_.open(path, std::ios::app);
    }

private:
    ProductionLogger() {}
    static const char* level_name(Level l) {
        static const char* names[] = {"DEBUG", "INFO ", "WARN ", "ERROR", "FATAL"};
        return names[l];
    }
    std::ofstream file_log_;
    std::mutex file_mutex_;
};

#define LOG(level) ProductionLogger::instance().log(level, __FILE__, __LINE__)
#define LOG_I(...) LOG(ProductionLogger::INFO, __VA_ARGS__)
#define LOG_E(...) LOG(ProductionLogger::ERROR, __VA_ARGS__)

```

### 6.3 进程监控与自动恢复

```bash
# 使用 systemd 管理推理服务（支持自动重启）
# /etc/systemd/system/yolo-npu.service
[Unit]
Description=YOLO NPU Inference Service
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/local/bin/yolo_rknn /usr/local/share/yolo_rknn/best.rknn --camera 0
Restart=on-failure
RestartSec=3
StandardOutput=journal
StandardError=journal
# 资源限制（防止 OOM）
MemoryMax=512M
CPUQuota=200%

[Install]
WantedBy=multi-user.target

# 启动与监控
sudo systemctl enable yolo-npu
sudo systemctl start yolo-npu
sudo systemctl status yolo-npu
journalctl -u yolo-npu -f

# 使用 watchdog 监控进程存活
# /etc/watchdog.conf
watchdog-device = /dev/watchdog
interval = 10
retry-on-error = 3
test-binary = /usr/local/bin/yolo_rknn_health_check

```

### 6.4 部署清单

```markdown
- [ ] 模型文件版本化管理（记录模型哈希值，便于回滚）
- [ ] 运行时库版本锁定（锁定 librknn_runtime.so 版本，避免升级后兼容问题）
- [ ] 交叉编译产物验证（在目标设备上运行 `ldd` 确认无缺失库）
- [ ] 温度阈值告警（超过 80°C 时自动降频或告警）
- [ ] 日志轮转（避免日志文件撑满存储）
- [ ] 看门狗定时器（硬件或软件 watchdog 防止死锁）
- [ ] 配置外置化（阈值、路径等参数从配置文件读取，不硬编码）
- [ ] 压力测试（连续运行 24 小时无内存泄漏，fps 波动 < 10%）

```

### 总结

C++ 部署 YOLO 到 Rockchip NPU 涉及多个关键环节，本文系统性地梳理了从模型准备到生产部署的完整流程。

#### 核心要点回顾

1. **模型准备与转换**
   - 将训练好的 ONNX 模型通过 RKNN-Toolkit2 转换为 `.rknn` 格式
   - 选择合适的量化策略（FP16 / INT8），量化模型推理更快但需关注精度损失
   - 转换时配置正确的输入输出节点名称，避免后处理格式不匹配

2. **推理流程规范**
   - `rknn_init` → `rknn_query` → `rknn_init_runtime` → 推理循环 → `rknn_destroy`
   - 每次推理前必须调用 `rknn_inputs_set` 设置输入，推理后调用 `rknn_outputs_get` 获取结果
   - 注意 `pass_through` 模式与零拷贝内存的生命周期管理

3. **坐标解码（常见 Bug）**
   - 模型输出的 `x, y` 是相对于 anchor 网格的偏移量，需要先还原到 640×640 空间再映射回原图
   - 正确顺序：`模型输出 → 640 空间坐标 → 减去 padding → 除以 scale → 原图坐标`
   - 原代码的 bug 是直接将模型输出乘以 scale 后减 padding，忽略了 640 空间还原步骤

4. **性能优化路径**
   - **内存**：预分配缓冲、使用 `rknn_create_mem` NPU 内存、内存池复用
   - **并行**：多 NPU 核心（每个核心独立 `rknn_context`，避免共享竞态）
   - **零拷贝**：`pass_through=1` + `rknn_tensor_mem` 消除 CPU→NPU 数据搬运
   - **编译**：`-O3 -march=armv8-a -ftree-vectorize -ffast-math`

5. **多线程与流水线**
   - 单实例 `YOLODetector` 非线程安全，多线程场景需每个线程独立实例
   - 流水线模式通过 `std::async` + 独立上下文实现帧级并行
   - 批量推理使用 `std::future` 收集结果，保持输入顺序

6. **调试与分析**
   - 使用 `RKNN_QUERY_PERF_RUN` 获取 NPU 推理耗时
   - `perf` / `strace` / `valgrind` 定位 CPU 侧瓶颈
   - `/proc/driver/rockchip/npu/info` 监控 NPU 利用率
   - 分段计时（`TimingTracker`）精确识别预处理、推理、后处理各阶段耗时

7. **生产部署考量**
   - 健壮的异常处理与指数退避重试
   - 结构化日志（分级、缓冲写入、文件日志）
   - systemd 服务管理 + watchdog 自动恢复
   - 温度监控、内存限制、日志轮转等运维基础

#### 参考资源

- [RKNN C++ API Documentation](https://github.com/airockchip/rknn-toolkit2/blob/master/docs/en/03_rknn_runtime/02_rknn_runtime_api.md)
- [RKNN-Toolkit2 Examples](https://github.com/airockchip/rknn-toolkit2/tree/master/examples)
- [RK3588 NPU 开发指南](https://wiki.t-firefly.com/ROC-RK3588-PC/NPU.html)

---

## 七、扩展 API 参考

### 7.1 rknn_init_runtime_option 详解

`rknn_init_runtime` 在模型加载后、推理前调用，用于配置运行时行为。其参数结构 `rknn_init_runtime_option` 包含以下关键字段：

```c
typedef struct rknn_init_runtime_option {
    uint32_t affinity;       // CPU 亲和性掩码
    uint32_t perf_mode;      // 性能模式
    uint32_t mem_alloc_mode; // 内存分配模式
    uint32_t flag;           // 保留，设为 0
} rknn_init_runtime_option;

```

#### 7.1.1 affinity（CPU 亲和性）

`affinity` 是一个位掩码，用于指定 RKNN 运行时线程绑定到哪些 CPU 核心。Rockchip NPU 的驱动会创建多个 worker 线程，绑定到性能核（Big Core）可避免被能效核（LITTLE Core）抢占，降低延迟抖动。

```c
rknn_init_runtime_option option;
std::memset(&option, 0, sizeof(option));

// 方案一：不限制（让内核调度）
option.affinity = 0;

// 方案二：绑定到大核（RK3588 的 CPU 布局：0-3 为 A55 小核，4-7 为 A76 大核）
option.affinity = 0xF0;  // 绑定到 core 4,5,6,7

// 方案三：绑定到全部核心
option.affinity = 0xFF;

rknn_init_runtime(ctx_, &option);

```

> **经验值**：在 RK3588 上，将 `affinity` 设为 `0xF0`（仅大核）可使推理延迟抖动降低约 30%。若同时运行其他服务（如视频解码），建议仅绑定部分核心以避免资源争抢。

#### 7.1.2 perf_mode（性能模式）

```c
// perf_mode 可选值
#define RKNN_PERF_MODE_NORMAL  0  // 默认性能模式
#define RKNN_PERF_MODE_HIGH    1  // 高性能模式（NPU 频率锁定最高档）
#define RKNN_PERF_MODE_LOW     2  // 低功耗模式（NPU 频率降低）

```

```c
option.perf_mode = RKNN_PERF_MODE_HIGH;  // 锁定最高频，适合实时检测
// 或
option.perf_mode = RKNN_PERF_MODE_NORMAL; // 默认，平衡功耗和性能

```

> **注意**：`perf_mode` 的设置效果取决于内核是否支持对应的频率 governors。在部分固件版本中，HIGH 模式需配合 `/sys/class/devfreq/*npu*/governor` 为 `performance` 才能生效。

#### 7.1.3 mem_alloc_mode（内存分配模式）

```c
// mem_alloc_mode 可选值
#define RKNN_MEM_ALLOC_MODE_AUTO   0  // 自动选择（默认）
#define RKNN_MEM_ALLOC_MODE_DMA_BUF 1 // 强制使用 DMA buffer
#define RKNN_MEM_ALLOC_MODE_PHYSICAL 2 // 强制使用物理内存

```

```c
option.mem_alloc_mode = RKNN_MEM_ALLOC_MODE_AUTO;  // 推荐：由运行时自动选择最优方案

```

> **选择建议**：
> - 若需要零拷贝（pass_through=1），`AUTO` 模式通常足够
> - 若使用 DMA buffer（`RKNN_MEM_TYPE_DMA_BUF`），设置为 `DMA_BUF` 可确保分配器使用 dma_buf 分配路径
> - 嵌入式内存紧张时，`PHYSICAL` 模式可避免 dma_buf 的额外映射开销

---

### 7.2 rknn_query 详细类型

`rknn_query` 支持多种查询类型，以下列出所有重要类型及其用法：

```c
// ===== 输入输出数量查询 =====
rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(rknn_io_dimension));
// 结果: io_num.n_input, io_num.n_output

// ===== 输入/输出张量属性查询 =====
rknn_tensor_attr attr;
attr.index = 0;
rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
// 结果: attr.name, attr.dims[], attr.type, attr.format, attr.quant_type, attr.scale[], attr.zero_point[]

rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
// 同上，但查询输出张量

// ===== 原生输入/输出属性（量化模型专用）=====
// 用于获取量化后 NPU 实际使用的属性，与 QUERY_INPUT_ATTR 的区别：
// - INPUT_ATTR 返回的是转换后的属性
// - NATIVE_INPUT_ATTR 返回 NPU 实际运行时使用的属性（如 INT8 量化后的 scale）
rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &attr, sizeof(attr));
rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &attr, sizeof(attr));

// ===== 模型属性查询（量化参数等）=====
// 用于获取模型的量化信息（zero_point 和 scale）
rknn_query(ctx, RKNN_QUERY_MODEL_ATTRIBUTE, &attr, sizeof(attr));
// attr.quant_type: 0=FP32, 1=ASYMMETRIC_QUANT, 2=SYMMETRIC_QUANT
// attr.quant_method: 0=NONE, 1=LINEAR, 2=LOGISTIC
// attr.scale[] 和 attr.zero_point[]: 量化参数

// ===== 性能统计查询 =====
rknn_perf_run perf;
rknn_query(ctx, RKNN_QUERY_PERF_RUN, &perf, sizeof(perf));
// perf.run_time_us: 上次推理耗时（微秒）
// perf.compile_time_us: 首次编译耗时（微秒，仅首次推理有效）

// ===== 内存分配状态查询 =====
rknn_mem_alloc_status mem_status;
rknn_query(ctx, RKNN_QUERY_MEM_ALLOC_STATUS, &mem_status, sizeof(mem_status));
// mem_status.used: 已使用 NPU 内存（字节）
// mem_status.total: 总 NPU 内存（字节）
// mem_status.max_used: 历史峰值使用量

// ===== 自定义字符串查询 =====
// 用于查询任意 RKNN 自定义字符串属性
rknn_query(ctx, RKNN_QUERY_CUSTOM_STRING, &custom_str, sizeof(custom_str));
// custom_str 为 rknn_custom_string 结构体，包含 key 和 value

// ===== 设备信息查询 =====
rknn_query(ctx, RKNN_QUERY_DEVICE_UTILIZATION, &utilization, sizeof(utilization));
// 查询 NPU 当前利用率（百分比）

// ===== 驱动版本查询 =====
rknn_query(ctx, RKNN_QUERY_VERSION, &version, sizeof(version));
// 获取运行时驱动版本号字符串

```

**查询类型枚举完整列表：**

```c
typedef enum rknn_query_type {
    RKNN_QUERY_MODEL_ATTRIBUTE       = 0,  // 模型属性（量化参数）
    RKNN_QUERY_PERF_RUN             = 1,  // 推理性能统计
    RKNN_QUERY_IN_OUT_NUM           = 2,  // 输入输出数量
    RKNN_QUERY_INPUT_ATTR           = 3,  // 输入张量属性
    RKNN_QUERY_OUTPUT_ATTR          = 4,  // 输出张量属性
    RKNN_QUERY_NATIVE_INPUT_ATTR    = 5,  // 原生输入属性
    RKNN_QUERY_NATIVE_OUTPUT_ATTR   = 6,  // 原生输出属性
    RKNN_QUERY_MEM_ALLOC_STATUS     = 7,  // 内存分配状态
    RKNN_QUERY_CUSTOM_STRING        = 8,  // 自定义字符串查询
    RKNN_QUERY_DEVICE_UTILIZATION   = 9,  // 设备利用率
    RKNN_QUERY_VERSION              = 10, // 版本信息
    RKNN_QUERY_GPU_MEM_ALLOC_STATUS = 11, // GPU 内存状态（部分平台）
} rknn_query_type;

```

---

### 7.3 rknn_set_io_mem 详解

`rknn_set_io_mem` 用于在推理前显式绑定输入/输出缓冲区到 RKNN 上下文。它与 `rknn_inputs_set` 配合使用，适用于需要精确控制内存布局的高级场景。

```c
int rknn_set_io_mem(rknn_context ctx, rknn_tensor_mem *mem, rknn_input *input)

```

**参数说明：**
- `ctx`：RKNN 上下文句柄
- `mem`：通过 `rknn_create_mem` 创建的 NPU 内存句柄
- `input`：输入描述符，指定哪个输入张量使用该内存

**典型用法：**

```c
// 1. 创建 NPU 内存
rknn_tensor_mem* input_mem = rknn_create_mem(ctx, input_size);
rknn_tensor_mem* output_mem = rknn_create_mem(ctx, output_size);

// 2. 设置输入内存绑定
rknn_input in = {};
in.index         = 0;
in.type          = RKNN_INPUT_TENSOR;
in.pass_through  = 1;  // 零拷贝模式
in.mem           = input_mem;

// 3. 调用 rknn_inputs_set 完成绑定
rknn_inputs_set(ctx, 1, &in);

// 4. 推理
rknn_run(ctx, nullptr);

// 5. 获取输出
rknn_output out = {};
out.index       = 0;
out.want_float  = 1;
rknn_outputs_get(ctx, 1, &out, NULL);

// 6. 清理
rknn_outputs_release(ctx, 1, &out);
rknn_destroy_mem(ctx, input_mem);
rknn_destroy_mem(ctx, output_mem);

```

> **与 rknn_create_mem 的区别**：`rknn_set_io_mem` 是显式绑定，适合需要多次复用同一块内存的场景（如双缓冲）。`rknn_inputs_set` 的 `mem` 字段则适合简单的一次性推理。

---

### 7.4 rknn_create_mem / rknn_destroy_mem 完整参数文档

#### rknn_create_mem

```c
// 基础版本：仅指定大小
rknn_tensor_mem* rknn_create_mem(rknn_context ctx, uint32_t size);

// 增强版本：指定内存属性（支持 DMA buffer、物理内存等）
rknn_tensor_mem* rknn_create_mem_attr(rknn_context ctx, rknn_tensor_mem_attr* attr);

```

**rknn_tensor_mem_attr 结构体：**

```c
typedef struct rknn_tensor_mem_attr {
    uint32_t size;              // 请求分配的内存大小（字节）
    uint32_t mem_type;          // 内存类型
    uint32_t flags;             // 内存标志位
} rknn_tensor_mem_attr;

```

**mem_type 可选值：**

```c
#define RKNN_MEM_TYPE_NORMAL  0  // 普通内存（默认，由运行时自动选择）
#define RKNN_MEM_TYPE_DMA_BUF 1  // DMA buffer（物理连续，支持 NPU DMA 直接访问）
#define RKNN_MEM_TYPE_PHYSICAL 2 // 物理内存（页锁定，适合零拷贝）
#define RKNN_MEM_TYPE_CACHE    3 // 缓存内存（CPU 高速缓存可见）

```

**flags 可选值（可按位或组合）：**

```c
#define RKNN_MEM_FLAG_READ_WRITE      (1 << 0)  // 读写权限
#define RKNN_MEM_FLAG_READ_ONLY       (1 << 1)  // 只读权限
#define RKNN_MEM_FLAG_CACHED          (1 << 2)  // CPU 缓存可见（提升 CPU 访问速度）
#define RKNN_MEM_FLAG_NON_CACHED      (1 << 3)  // CPU 缓存不可见（避免 cache coherency 问题）
#define RKNN_MEM_FLAG_DMA_COHERENT    (1 << 4)  // DMA 一致性（硬件保障 CPU/NPU 数据同步）

```

**使用示例：**

```c
// 示例 1：普通 NPU 内存（适合 CPU 直接写入 + NPU 直接读取）
rknn_tensor_mem* mem = rknn_create_mem(ctx, 640 * 640 * 3 * sizeof(float));
if (mem) {
    std::memcpy(mem->virt_addr, input_data, mem->size);
    // 使用 mem
    rknn_destroy_mem(ctx, mem);
}

// 示例 2：DMA buffer（适合 camera driver 直连场景）
rknn_tensor_mem_attr attr = {};
attr.size     = 640 * 640 * 3 * sizeof(float);
attr.mem_type = RKNN_MEM_TYPE_DMA_BUF;
attr.flags    = RKNN_MEM_FLAG_READ_WRITE | RKNN_MEM_FLAG_CACHED;

rknn_tensor_mem* dma_mem = rknn_create_mem_attr(ctx, &attr);
if (dma_mem) {
    // DMA buffer 可能需要通过 phys_addr 访问，或使用 virt_addr（若 cached）
    if (dma_mem->virt_addr) {
        std::memcpy(dma_mem->virt_addr, input_data, attr.size);
    }
    // 使用 dma_mem
    rknn_destroy_mem(ctx, dma_mem);
}

```

#### rknn_destroy_mem

```c
void rknn_destroy_mem(rknn_context ctx, rknn_tensor_mem *mem);

```

- 释放通过 `rknn_create_mem` 或 `rknn_create_mem_attr` 分配的 NPU 内存
- **必须在 `rknn_destroy(ctx)` 之前调用**，否则会导致内存泄漏
- 不是线程安全的，应在单一线程中统一释放
- 若 `mem` 为 `nullptr`，函数不做任何操作（安全调用）

---

## 八、详细构建流程

### 8.1 各场景构建命令

#### 场景一：设备本地编译（最简单）

```bash
# 1. 确认交叉编译工具链不可用（或在设备本地）
# 2. 设置环境变量
export RKNN_ROOT=/usr
export OpenCV_DIR=/usr/lib/aarch64-linux-gnu/cmake/opencv4

# 3. 编译
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 4. 验证
ls -lh yolo_rknn
file yolo_rknn  # 应显示 ARM aarch64 ELF
ldd yolo_rknn   # 确认无缺失库

```

#### 场景二：PC 端交叉编译后部署

```bash
# ===== 1. 安装交叉编译工具链（Ubuntu 20.04+）=====
sudo apt update
sudo apt install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
                    cmake make git

# ===== 2. 获取 RKSDK（从 Rockchip 官方或设备拷贝）=====
# 方法 A：从开发板拷贝 RKNN runtime 库和头文件
scp root@<board_ip>:/usr/include/rknn_api.h ./include/
scp root@<board_ip>:/usr/lib/aarch64-linux-gnu/librknn_runtime.so* ./lib/
scp -r root@<board_ip>:/usr/lib/aarch64-linux-gnu/cmake/opencv4 ./cmake/

# 方法 B：从 RKSDK 安装包提取
# tar -xzf RK3588_Linux_Release_V*.tar.gz
# cp -r RK3588_Linux_Release/usr/aarch64-linux-gnu/include/rknn_api.h ./include/
# cp -r RK3588_Linux_Release/usr/aarch64-linux-gnu/lib/* ./lib/

# ===== 3. 编译（交叉编译）=====
mkdir -p build && cd build
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../toolchain-aarch64.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DRKNN_INCLUDE_DIR=$PWD/../include \
    -DRKNN_LIB_DIR=$PWD/../lib \
    -DOpenCV_DIR=$PWD/../cmake/opencv4

make -j$(nproc)

# ===== 4. 部署到设备=====
scp yolo_rknn root@<board_ip>:/usr/local/bin/
scp best.rknn root@<board_ip>:/usr/local/share/yolo_rknn/
scp coco.names root@<board_ip>:/usr/local/share/yolo_rknn/

# 在设备上验证
ssh root@<board_ip> "ldd /usr/local/bin/yolo_rknn && /usr/local/bin/yolo_rknn /usr/local/share/yolo_rknn/best.rknn test.jpg"

```

#### 场景三：静态链接编译（最大便携性）

```bash
# 静态编译 eliminates 所有动态库依赖
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../toolchain-aarch64.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_STATIC=ON  # 在 CMakeLists.txt 中控制

# CMakeLists.txt 中添加：
# if(BUILD_STATIC)
#     target_link_options(yolo_rknn PRIVATE -static)
#     # 静态链接时需排除部分不兼容的库
#     target_link_libraries(yolo_rknn
#         ${OpenCV_LIBS}
#         rknn_runtime
#         pthread dl rt z
#     )
# endif()

# 静态编译后无需拷贝 .so 文件，但二进制体积增大 2-3 倍

```

---

### 8.2 CI/CD Pipeline 示例（GitHub Actions）

```yaml
# .github/workflows/build-npu.yml
name: NPU C++ Build

on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]
  workflow_dispatch:  # 允许手动触发

jobs:
  build-aarch64:
    runs-on: ubuntu-22.04
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Install cross-compile toolchain
        run: |
          sudo apt update
          sudo apt install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
                              cmake make

      - name: Setup build environment
        run: |
          # 下载 RKNN runtime 库（从内部 artifact 或 Rockchip SDK）
          mkdir -p rknn_sdk/include rknn_sdk/lib
          # 从 Release asset 或内部仓库获取
          curl -L "${{ secrets.RKNN_RUNTIME_URL }}" -o rknn_sdk.tar.gz
          tar -xzf rknn_sdk.tar.gz -C rknn_sdk/

      - name: Configure
        run: |
          mkdir -p build && cd build
          cmake .. \
            -DCMAKE_TOOLCHAIN_FILE=../toolchain-aarch64.cmake \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_PREFIX_PATH=$PWD/../rknn_sdk

      - name: Build
        run: |
          cd build
          make -j$(nproc)
          ls -lh yolo_rknn

      - name: Run static analysis
        run: |
          cd build
          # 静态链接检查
          ldd yolo_rknn || true
          # 符号检查
          nm -D yolo_rknn | grep " U " | wc -l

      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: yolo_rknn-aarch64
          path: build/yolo_rknn
          retention-days: 7

  deploy-to-device:
    needs: build-aarch64
    runs-on: ubuntu-22.04
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Download artifact
        uses: actions/download-artifact@v4
        with:
          name: yolo_rknn-aarch64

      - name: Deploy to target device
        run: |
          scp yolo_rknn root@${{ secrets.BOARD_IP }}:/usr/local/bin/
          ssh root@${{ secrets.BOARD_IP }} "chmod +x /usr/local/bin/yolo_rknn"
          ssh root@${{ secrets.BOARD_IP }} "/usr/local/bin/yolo_rknn --version"

```

---

### 8.3 Docker 交叉编译环境

```dockerfile
# Dockerfile.cross_compile
FROM ubuntu:22.04

# 安装交叉编译工具链和构建依赖
RUN apt-get update && apt-get install -y \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    cmake \
    make \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 创建 RKNN SDK 挂载点（从宿主机或内部分发）
RUN mkdir -p /opt/rknn-sdk/{include,lib,cmake}

WORKDIR /workspace

# 默认命令
CMD ["/bin/bash"]

```

```bash
# 构建 Docker 镜像
docker build -t yolo-rknn-cross -f Dockerfile.cross_compile .

# 运行交叉编译
docker run --rm \
    -v $(pwd):/workspace \
    -v /path/to/rknn-sdk:/opt/rknn-sdk \
    yolo-rknn-cross \
    bash -c "
        cd /workspace &&
        mkdir -p build && cd build &&
        cmake .. \
            -DCMAKE_TOOLCHAIN_FILE=../toolchain-aarch64.cmake \
            -DCMAKE_BUILD_TYPE=Release \
            -DRKNN_INCLUDE_DIR=/opt/rknn-sdk/include \
            -DRKNN_LIB_DIR=/opt/rknn-sdk/lib &&
        make -j$(nproc) &&
        ls -lh yolo_rknn
    "

```

**Docker 多阶段构建（更小体积）：**

```dockerfile
# 阶段 1：编译
FROM ubuntu:22.04 AS builder
RUN apt-get update && apt-get install -y \
    gcc-aarch64-linux-gnu g++-aarch64-linux-gnu cmake make git
WORKDIR /build
COPY . .
RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchain-aarch64.cmake \
             -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# 阶段 2：运行时镜像（极小）
FROM ubuntu:22.04-slim
COPY --from=builder /build/build/yolo_rknn /usr/local/bin/
COPY --from=builder /build/best.rknn /usr/local/share/yolo_rknn/
COPY --from=builder /build/coco.names /usr/local/share/yolo_rknn/
RUN apt-get update && apt-get install -y libopencv-dev && rm -rf /var/lib/apt/lists/*
ENTRYPOINT ["/usr/local/bin/yolo_rknn"]
CMD ["best.rknn", "camera:0"]

```

---

## 九、扩展 YOLO 检测器：多引擎支持

### 9.1 推理引擎抽象接口（策略模式）

为了在 RKNN、ONNX Runtime、TensorRT 之间灵活切换，设计一个通用的推理引擎接口：

```cpp
// inference_engine.h
#pragma once
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include "yolo_detector.h"  // 定义 Detection 结构

// 通用推理引擎接口
class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;

    // 加载模型，返回 0 表示成功
    virtual int load_model(const std::string& model_path) = 0;

    // 单帧推理
    virtual std::vector<Detection> detect(const cv::Mat& image) = 0;

    // 批量推理
    virtual std::vector<std::vector<Detection>> detect_batch(
        const std::vector<cv::Mat>& images) = 0;

    // 获取推理耗时（毫秒）
    virtual double get_last_inference_time() const = 0;

    // 获取平均推理耗时
    virtual double get_avg_inference_time() const = 0;

    // 打印模型信息
    virtual void print_model_info() const = 0;
};

// 工厂函数：根据后端名称创建对应的引擎实例
std::unique_ptr<InferenceEngine> create_engine(const std::string& backend);

```

```cpp
// inference_engine_factory.cpp
#include "inference_engine.h"
#include "rknn_engine.h"       // RKNN 后端实现
#include "onnx_engine.h"       // ONNX Runtime 后端实现
#include "tensorrt_engine.h"   // TensorRT 后端实现（可选）

std::unique_ptr<InferenceEngine> create_engine(const std::string& backend) {
    if (backend == "rknn") {
        return std::make_unique<RKNNEngine>();
    } else if (backend == "onnx") {
        return std::make_unique<ONNXEngine>();
    } else if (backend == "tensorrt") {
        return std::make_unique<TensorRTEngine>();
    } else {
        throw std::invalid_argument("Unknown backend: " + backend);
    }
}

```

使用示例：

```cpp
// main.cpp - 通过命令行参数选择后端
std::string backend = (argc > 3) ? argv[3] : "rknn";
auto engine = create_engine(backend);

if (engine->load_model(model_path) != 0) {
    std::cerr << "Failed to load model with " << backend << " backend\n";
    return -1;
}

// 后续推理逻辑完全与后端无关
auto detections = engine->detect(image);

```

---

### 9.2 ONNX Runtime C++ 推理路径

参考 Ultralytics 官方示例，ONNX Runtime 提供了与 RKNN 并行的推理后端。以下是关键实现：

**核心类定义（inference.h）：**

```cpp
// onnx_engine.h
#pragma once
#include "inference_engine.h"
#include "onnxruntime_cxx_api.h"

class ONNXEngine : public InferenceEngine {
public:
    ONNXEngine();
    ~ONNXEngine();

    int load_model(const std::string& model_path) override;
    std::vector<Detection> detect(const cv::Mat& image) override;
    std::vector<std::vector<Detection>> detect_batch(
        const std::vector<cv::Mat>& images) override;
    double get_last_inference_time() const override;
    double get_avg_inference_time() const override;
    void print_model_info() const override;

    // ONNX Runtime 特有功能
    char* warmup();

private:
    Ort::Env env_;
    Ort::Session* session_ = nullptr;
    Ort::RunOptions options_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<int64_t> input_dims_;

    // 模型配置
    MODEL_TYPE model_type_;
    std::vector<int> img_size_;
    float conf_threshold_;
    float iou_threshold_;
    float resize_scale_;

    // 性能统计
    double last_time_ms_ = 0.0;
    double avg_time_ms_ = 0.0;
    uint64_t frame_count_ = 0;

    char* pre_process(const cv::Mat& img, cv::Mat& out);
    char* tensor_process(float* blob, std::vector<Detection>& results);
    char* tensor_process_half(half* blob, std::vector<Detection>& results);
    void load_classes();
};

```

**预处理与推理（inference.cpp 核心逻辑）：**

```cpp
// ONNX 预处理：LetterBox 缩放
char* ONNXEngine::pre_process(const cv::Mat& img, cv::Mat& out) {
    if (img.channels() == 3) {
        out = img.clone();
        cv::cvtColor(out, out, cv::COLOR_BGR2RGB);
    } else {
        cv::cvtColor(img, out, cv::COLOR_GRAY2RGB);
    }

    switch (model_type_) {
        case YOLO_DETECT_V8:
        case YOLO_DETECT_V8_HALF:
        {
            // LetterBox 保持宽高比
            if (img.cols >= img.rows) {
                resize_scale_ = img.cols / (float)img_size_[0];
                cv::resize(out, out, cv::Size(img_size_[0],
                    static_cast<int>(img.rows / resize_scale_)));
            } else {
                resize_scale_ = img.rows / (float)img_size_[0];
                cv::resize(out, out, cv::Size(
                    static_cast<int>(img.cols / resize_scale_),
                    img_size_[1]));
            }
            // 填充灰色背景
            cv::Mat temp(img_size_[0], img_size_[1], CV_8UC3, cv::Scalar(114, 114, 114));
            out.copyTo(temp(cv::Rect(0, 0, out.cols, out.rows)));
            out = temp;
            break;
        }
        case YOLO_CLS:
        {
            // 分类模型：中心裁剪
            int h = img.rows, w = img.cols;
            int m = std::min(h, w);
            cv::resize(out(cv::Rect((w-m)/2, (h-m)/2, m, m)),
                       out, cv::Size(img_size_[0], img_size_[1]));
            break;
        }
    }
    return RET_OK;
}

// ONNX 推理主流程
std::vector<Detection> ONNXEngine::detect(const cv::Mat& img) {
    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat processed;
    pre_process(img, processed);

    std::vector<Detection> results;

    if (model_type_ == YOLO_DETECT_V8 || model_type_ == YOLO_DETECT_V8_HALF) {
        if (model_type_ == YOLO_DETECT_V8) {
            float* blob = new float[processed.total() * 3];
            // 手动归一化到 [0,1]
            processed.convertTo(processed, CV_32FC3, 1.0/255.0);
            std::memcpy(blob, processed.data, processed.total() * 3 * sizeof(float));
            tensor_process(blob, results);
            delete[] blob;
        } else {
            // FP16 模式需要 CUDA 支持，此处略
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    last_time_ms_ = std::chrono::duration<double, std::milli>(end - start).count();
    frame_count_++;
    avg_time_ms_ = avg_time_ms_ * (frame_count_ - 1) / frame_count_ + last_time_ms_ / frame_count_;

    return results;
}

// ONNX 后处理
char* ONNXEngine::tensor_process(float* blob, std::vector<Detection>& results) {
    // 创建输入 tensor
    std::vector<int64_t> dims = {1, 3, img_size_[0], img_size_[1]};
    Ort::Value input = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
        blob, 3 * img_size_[0] * img_size_[1], dims.data(), dims.size());

    // 推理
    auto outputs = session_->Run(options_, input_names_.data(), &input, 1,
                                  output_names_.data(), output_names_.size());

    // 获取输出 shape
    auto type_info = outputs[0].GetTypeInfo();
    auto shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
    // shape: [1, 84, 8400] for YOLOv8

    // reshape 为 [8400, 84]
    cv::Mat rawData(84, 8400, CV_32F, blob);
    // 注意：ONNX Runtime 输出是 C-contiguous，需要正确 reshape
    cv::Mat outMat(shape[1], shape[2], CV_32F, outputs[0].GetTensorMutableData<float>());
    cv::transpose(outMat, outMat);  // [8400, 84]

    float* data = (float*)outMat.data;
    int num_classes = static_cast<int>(classes_.size());
    int num_anchors = static_cast<int>(shape[2]);

    for (int i = 0; i < num_anchors; i++) {
        float* class_scores = data + 4;
        cv::Point class_id;
        double max_score;
        cv::Mat scores(1, num_classes, CV_32FC1, class_scores);
        cv::minMaxLoc(scores, 0, &max_score, 0, &class_id);

        if (max_score > conf_threshold_) {
            float x = data[0], y = data[1], w = data[2], h = data[3];

            float x_factor = img.cols / (float)img_size_[0];
            float y_factor = img.rows / (float)img_size_[1];

            int left   = static_cast<int>((x - 0.5f * w) * x_factor);
            int top    = static_cast<int>((y - 0.5f * h) * y_factor);
            int width  = static_cast<int>(w * x_factor);
            int height = static_cast<int>(h * y_factor);

            Detection det;
            det.class_id     = class_id.x;
            det.confidence   = static_cast<float>(max_score);
            det.x1           = left;
            det.y1           = top;
            det.x2           = left + width;
            det.y2           = top + height;
            det.class_name   = (class_id.x < static_cast<int>(classes_.size()))
                ? classes_[class_id.x] : "unknown";
            results.push_back(det);
        }
        data += 4 + num_classes;
    }

    // NMS
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    for (auto& d : results) {
        boxes.push_back(cv::Rect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1));
        confs.push_back(d.confidence);
    }
    std::vector<int> nms_idx;
    cv::dnn::NMSBoxes(boxes, confs, conf_threshold_, iou_threshold_, nms_idx);

    std::vector<Detection> final_results;
    for (int idx : nms_idx) {
        if (idx < static_cast<int>(results.size()))
            final_results.push_back(results[idx]);
    }
    results = final_results;
    return RET_OK;
}

```

**CMakeLists.txt 配置 ONNX Runtime：**

```cmake
# 查找 ONNX Runtime
find_package(OnnxRuntime REQUIRED)

target_link_libraries(yolo_rknn
    ${OpenCV_LIBS}
    rknn_runtime          # RKNN 后端
    OnnxRuntime::OnnxRuntime  # ONNX Runtime 后端
    pthread dl
)

# 启用 ONNX 后端（通过宏控制）
target_compile_definitions(yolo_rknn PRIVATE USE_ONNX)

```

---

### 9.3 TensorRT 推理路径（对比参考）

TensorRT 在 NVIDIA GPU 上提供极致性能，以下是与 RKNN 的关键对比：

| 特性 | RKNN (NPU) | TensorRT (GPU) | ONNX Runtime (CPU/GPU) |
|------|-----------|----------------|----------------------|
| 目标平台 | Rockchip SoC | NVIDIA GPU | 跨平台 |
| 模型格式 | .rknn | .engine | .onnx |
| 量化支持 | FP16/INT8 | FP16/FP32/INT8 | FP16/FP32/INT8 |
| 推理延迟 | 3-10ms (RK3588) | 1-5ms (T4) | 10-50ms (CPU) |
| 功耗 | 2-5W | 30-100W | 10-50W |
| 部署复杂度 | 中 | 高 | 低 |

```cpp
// tensorrt_engine.h（概念性代码，NVIDIA 平台）
#pragma once
#include "inference_engine.h"
#include <NvInfer.h>

class TensorRTEngine : public InferenceEngine {
public:
    int load_model(const std::string& engine_path) override;
    std::vector<Detection> detect(const cv::Mat& image) override;
    // ... 其他纯虚函数实现

private:
    nvinfer1::ICudaEngine* engine_ = nullptr;
    cudaStream_t stream_ = nullptr;
    void* bindings_[4];  // input, output, workspace, etc.
};

```

**TensorRT 引擎构建（Python，在 PC 端完成）：**

```python
import tensorrt as trt
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path, fp16=True):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    engine = builder.build_serialized_network(network, config)
    with open(engine_path, 'wb') as f:
        f.write(engine)
    print(f"Engine saved to {engine_path}")

build_engine("yolov8s.onnx", "yolov8s.trt", fp16=True)

```

---

### 9.4 YOLOv8（无锚框）与 YOLOv5（有锚框）后处理详解

YOLOv5 和 YOLOv8 的输出格式有本质区别，后处理逻辑不同：

#### YOLOv5（有锚框，Anchor-Based）

```
输出格式: [1, 25, 8400] 或 [1, 84, 8400]
  - 前 4 个值: cx, cy, cw, ch（相对于网格的偏移）
  - 第 5 个值: objectness 置信度
  - 后 80 个值: 各类别条件概率 P(class|object)

解码逻辑:
  1. 对每个 anchor 点:
     x = (sigmoid(cx)) * 2 - 0.5 + grid_x   # 中心 x（网格坐标）
     y = (sigmoid(cy)) * 2 - 0.5 + grid_y   # 中心 y
     w = anchor_w * exp(cw)                  # 宽高（乘以 anchor）
     h = anchor_h * exp(ch)
  2. 乘以 stride 映射到原图
  3. 类别分数 = objectness * max(class_probs)

```

```cpp
// YOLOv5 后处理（有锚框）
std::vector<Detection> postprocess_yolov5(float* outputs, int num_anchors,
                                           const std::vector<float>& anchor_w,
                                           const std::vector<float>& anchor_h,
                                           const std::vector<int>& strides,
                                           const cv::Size& orig_size,
                                           float scale, int pad_w, int pad_h) {
    std::vector<Detection> detections;
    int num_classes = 80;

    // 3 个尺度的 anchor: [8,10], [16,20], ... 对应 stride [8,16,32]
    int anchor_per_scale = num_anchors / 3;
    int anchor_idx = 0;

    for (int s = 0; s < 3; s++) {  // 3 个尺度
        int stride = strides[s];
        int grid_size = 640 / stride;  // 网格大小

        for (int i = 0; i < grid_size * grid_size; i++) {
            float cx = outputs[anchor_idx * (5 + num_classes) + 0];
            float cy = outputs[anchor_idx * (5 + num_classes) + 1];
            float cw = outputs[anchor_idx * (5 + num_classes) + 2];
            float ch = outputs[anchor_idx * (5 + num_classes) + 3];
            float obj_score = outputs[anchor_idx * (5 + num_classes) + 4];

            // sigmoid 解码
            cx = (float)1.0 / (1.0 + std::exp(-cx));
            cy = (float)1.0 / (1.0 + std::exp(-cy));
            cw = std::exp(cw);
            ch = std::exp(ch);

            // 网格坐标
            int grid_x = i % grid_size;
            int grid_y = i / grid_size;

            // 还原到 640 空间
            float x_center = (cx * 2.0f - 0.5f + (float)grid_x) * stride;
            float y_center = (cy * 2.0f - 0.5f + (float)grid_y) * stride;
            float width    = anchor_w[anchor_idx % 3] * cw;
            float height   = anchor_h[anchor_idx % 3] * ch;

            // 类别
            float* class_scores = &outputs[anchor_idx * (5 + num_classes) + 5];
            int class_id = std::distance(class_scores,
                std::max_element(class_scores, class_scores + num_classes));
            float confidence = obj_score * class_scores[class_id];

            if (confidence < conf_threshold_) continue;

            // 映射回原图
            float cx_final = (x_center - (float)pad_w) / scale;
            float cy_final = (y_center - (float)pad_h) / scale;
            float w_final  = width  / scale;
            float h_final  = height / scale;

            Detection det;
            det.class_id = class_id;
            det.confidence = confidence;
            det.x1 = std::max(0, (int)(cx_final - w_final / 2));
            det.y1 = std::max(0, (int)(cy_final - h_final / 2));
            det.x2 = std::min(orig_size.width, (int)(cx_final + w_final / 2));
            det.y2 = std::min(orig_size.height, (int)(cy_final + h_final / 2));
            detections.push_back(det);

            anchor_idx++;
        }
    }
    return detections;
}

```

#### YOLOv8（无锚框，Anchor-Free）

```
输出格式: [1, 4+num_classes, H*W*A]  或经过 transpose 后 [1, H*W*A, 4+num_classes]
  - 前 4 个值: cx, cy, cw, ch（中心 + 宽高，无需 anchor）
  - 后 num_classes 个值: 各类别概率（无需 objectness）

解码逻辑:
  1. 模型输出已经是归一化坐标（相对于输入尺寸）
  2. 直接乘输入尺寸得到 640x640 空间坐标
  3. 减去 padding，除以 scale 映射回原图
  4. 类别分数 = max(class_probs)

```

```cpp
// YOLOv8 后处理（无锚框，Anchor-Free）
std::vector<Detection> postprocess_yolov8(float* outputs, int num_anchors,
                                           const cv::Size& orig_size,
                                           float scale, int pad_w, int pad_h) {
    std::vector<Detection> detections;
    int num_classes = 80;

    for (int i = 0; i < num_anchors; i++) {
        float cx = outputs[i * (4 + num_classes) + 0];
        float cy = outputs[i * (4 + num_classes) + 1];
        float cw = outputs[i * (4 + num_classes) + 2];
        float ch = outputs[i * (4 + num_classes) + 3];

        float* class_scores = &outputs[i * (4 + num_classes) + 4];
        int class_id = std::distance(class_scores,
            std::max_element(class_scores, class_scores + num_classes));
        float confidence = class_scores[class_id];

        if (confidence < conf_threshold_) continue;

        // YOLOv8 输出坐标解码（anchor-free）
        // cx,cy 是中心点，cw,ch 是宽高
        float x_center = cx * img_size_;    // 还原到 640 空间
        float y_center = cy * img_size_;
        float width    = cw * img_size_;
        float height   = ch * img_size_;

        // 映射回原图
        float cx_final = (x_center - (float)pad_w) / scale;
        float cy_final = (y_center - (float)pad_h) / scale;
        float w_final  = width  / scale;
        float h_final  = height / scale;

        Detection det;
        det.class_id = class_id;
        det.confidence = confidence;
        det.x1 = std::max(0, (int)(cx_final - w_final / 2));
        det.y1 = std::max(0, (int)(cy_final - h_final / 2));
        det.x2 = std::min(orig_size.width, (int)(cx_final + w_final / 2));
        det.y2 = std::min(orig_size.height, (int)(cy_final + h_final / 2));
        detections.push_back(det);
    }

    // NMS
    std::sort(detections.begin(), detections.end(),
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });

    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    for (auto& d : detections) {
        boxes.push_back(cv::Rect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1));
        confs.push_back(d.confidence);
    }
    std::vector<int> nms_idx;
    cv::dnn::NMSBoxes(boxes, confs, conf_threshold_, iou_threshold_, nms_idx);

    std::vector<Detection> result;
    for (int idx : nms_idx) {
        result.push_back(detections[idx]);
    }
    return result;
}

```

#### 自动检测模型格式

```cpp
// 根据输出张量 shape 自动判断 YOLOv5 vs YOLOv8
bool detect_model_format(const rknn_tensor_attr& output_attr) {
    // YOLOv5: 输出 shape [1, 25+num_classes, 8400] 或 [1, 8400, 25+num_classes]
    // YOLOv8: 输出 shape [1, 4+num_classes, 8400] 或 [1, 8400, 4+num_classes]
    // 关键区别：YOLOv5 有 objectness 通道（第 5 维），YOLOv8 没有

    int num_classes = 80;  // 可根据模型调整
    int total_channels = output_attr.dims[1];  // NCHW 格式

    // 判断：若第 2 维 = 4+num_classes，则是 YOLOv8（anchor-free）
    // 若第 2 维 = 5+num_classes（含 objectness），则是 YOLOv5
    if (total_channels == 4 + num_classes) {
        return true;  // YOLOv8
    } else if (total_channels == 5 + num_classes) {
        return false;  // YOLOv5
    } else {
        // 尝试从名称判断
        std::string name(output_attr.name);
        return name.find("v8") != std::string::npos || name.find("v10") != std::string::npos;
    }
}

```

---

### 9.5 批量后处理向量化优化

```cpp
// 使用 SIMD 友好的数据结构进行批量后处理
struct BatchDetection {
    float cx, cy, cw, ch;       // 4 个浮点
    float confidence;           // 1 个浮点
    int class_id;               // 4 字节对齐
    char pad[3];                // 对齐到 32 字节
};

// 批量后处理：一次性处理整个 batch
std::vector<std::vector<Detection>> batch_postprocess(
    float* batch_outputs,    // [batch_size, 4+num_classes, num_anchors]
    int batch_size,
    int num_anchors,
    int num_classes,
    const std::vector<cv::Size>& orig_sizes,
    const std::vector<float>& scales,
    const std::vector<int>& pad_ws,
    const std::vector<int>& pad_hs) {

    std::vector<std::vector<Detection>> all_results(batch_size);

    #pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        auto& results = all_results[b];
        float* out = batch_outputs + b * (4 + num_classes) * num_anchors;

        for (int i = 0; i < num_anchors; i++) {
            float cx = out[i * (4 + num_classes) + 0];
            float cy = out[i * (4 + num_classes) + 1];
            float cw = out[i * (4 + num_classes) + 2];
            float ch = out[i * (4 + num_classes) + 3];

            float* scores = &out[i * (4 + num_classes) + 4];
            int cid = std::distance(scores, std::max_element(scores, scores + num_classes));
            float conf = scores[cid];

            if (conf < conf_threshold_) continue;

            float x_center = cx * img_size_;
            float y_center = cy * img_size_;
            float width    = cw * img_size_;
            float height   = ch * img_size_;

            float fx = (x_center - (float)pad_ws[b]) / scales[b];
            float fy = (y_center - (float)pad_hs[b]) / scales[b];
            float fw = width  / scales[b];
            float fh = height / scales[b];

            Detection det;
            det.class_id = cid;
            det.confidence = conf;
            det.x1 = std::max(0, (int)(fx - fw / 2));
            det.y1 = std::max(0, (int)(fy - fh / 2));
            det.x2 = std::min(orig_sizes[b].width, (int)(fx + fw / 2));
            det.y2 = std::min(orig_sizes[b].height, (int)(fy + fh / 2));
            results.push_back(det);
        }

        // 当前 batch 内的 NMS
        std::sort(results.begin(), results.end(),
            [](const Detection& a, const Detection& b) {
                return a.confidence > b.confidence;
            });

        std::vector<cv::Rect> boxes;
        std::vector<float> confs;
        for (auto& d : results) {
            boxes.push_back(cv::Rect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1));
            confs.push_back(d.confidence);
        }
        std::vector<int> nms_idx;
        cv::dnn::NMSBoxes(boxes, confs, conf_threshold_, iou_threshold_, nms_idx);

        std::vector<Detection> final;
        for (int idx : nms_idx) final.push_back(results[idx]);
        results = final;
    }

    return all_results;
}

```

---

## 十、内存管理深度剖析

### 10.1 NPU 内存类型详解

Rockchip NPU 支持多种内存类型，每种类型有不同的访问特性和性能特征：

#### 10.1.1 DMA_BUF 类型

```
DMA_BUF（Direct Memory Access Buffer）
├── 物理连续内存，由 Linux 内核分配
├── 支持 CPU 通过 mmap 映射到用户空间
├── NPU 可直接通过 DMA 通道访问
├── 适合大尺寸张量（输入/输出缓冲）
└── 创建: rknn_create_mem_attr with mem_type=RKNN_MEM_TYPE_DMA_BUF

```

```c
// DMA_BUF 内存的完整生命周期
rknn_tensor_mem_attr attr = {};
attr.size     = 640 * 640 * 3 * sizeof(float);  // ~4.9MB
attr.mem_type = RKNN_MEM_TYPE_DMA_BUF;
attr.flags    = RKNN_MEM_FLAG_READ_WRITE | RKNN_MEM_FLAG_CACHED;

rknn_tensor_mem* mem = rknn_create_mem_attr(ctx, &attr);

// virt_addr: CPU 可直接访问（若 cached）
if (mem->virt_addr) {
    std::memcpy(mem->virt_addr, cpu_data, attr.size);
}

// phys_addr: NPU DMA 使用的物理地址
printf("Physical address: 0x%08X\n", mem->phys_addr);

// 推理...
rknn_destroy_mem(ctx, mem);

```

#### 10.1.2 PHYSICAL 类型

```
PHYSICAL（物理内存）
├── 页锁定内存（page-locked），不可被 swap
├── CPU 和 NPU 均可通过物理地址访问
├── 适合需要 CPU 直接写入的场景
├── 创建: rknn_create_mem_attr with mem_type=RKNN_MEM_TYPE_PHYSICAL
└── 注意：物理内存总量受限（通常 128MB-256MB）

```

#### 10.1.3 CPU_accessible 类型

```
CPU_accessible（CPU 可访问内存）
├── 系统 RAM 中的普通内存
├── RKNN 运行时自动拷贝到 NPU（pass_through=0）
├── 最简单但延迟最高（多一次拷贝）
└── 适合小模型或对延迟不敏感的场景

```

#### 内存类型选择指南

```
┌─────────────────┬──────────────┬──────────────┬─────────────────┐
│     特性        │  DMA_BUF     │  PHYSICAL    │ CPU_accessible  │
├─────────────────┼──────────────┼──────────────┼─────────────────┤
│ 分配速度        │  慢 (ms级)   │  中等        │  快 (us级)      │
│ CPU 访问延迟    │  中          │  快          │  最快           │
│ NPU DMA 访问    │  最优        │  优          │  需拷贝         │
│ 内存开销        │  中          │  高 (锁定)   │  低             │
│ 适用场景        │  零拷贝推理  │  高频推理    │  原型/调试      │
└─────────────────┴──────────────┴──────────────┴─────────────────┘

```

---

### 10.2 内存池设计模式

#### 模式一：Buddy Allocator（伙伴分配器）

```cpp
class BuddyAllocator {
private:
    static constexpr size_t MIN_BLOCK = 64;      // 最小分配单元
    static constexpr size_t MAX_ORDER = 20;      // 最大 2^20 = 1MB 块
    static constexpr size_t TOTAL_SIZE = 64 * 1024 * 1024;  // 64MB 池

    struct Block {
        size_t size;
        bool allocated;
        Block* next;
    };

    std::vector<char> pool_;
    Block* free_list_[MAX_ORDER + 1];  // 按大小分级的空闲链表

public:
    BuddyAllocator() {
        pool_.resize(TOTAL_SIZE);
        std::memset(free_list_, 0, sizeof(free_list_));
        // 初始化：将整个池作为一个最大块插入空闲链表
        free_list_[MAX_ORDER] = reinterpret_cast<Block*>(pool_.data());
        free_list_[MAX_ORDER]->size = TOTAL_SIZE;
        free_list_[MAX_ORDER]->allocated = false;
    }

    void* allocate(size_t bytes) {
        if (bytes == 0) return nullptr;
        size_t order = std::max(0, (int)std::ceil(std::log2(bytes / (float)MIN_BLOCK)));
        order = std::min(order, MAX_ORDER);

        // 查找合适的空闲块
        while (order <= MAX_ORDER && !free_list_[order]) order++;
        if (order > MAX_ORDER) return nullptr;

        // 分裂大块
        Block* block = free_list_[order];
        free_list_[order] = block->next;

        while (block->size > (MIN_BLOCK << order) && order > 0) {
            block->size >>= 1;
            order--;
            block->next = free_list_[order];
            block->next->allocated = false;
            free_list_[order] = block->next;
        }

        block->allocated = true;
        return static_cast<char*>(block) + sizeof(Block);
    }

    void deallocate(void* ptr) {
        // 将块合并回空闲链表（伙伴合并）
        // ... 实现略，核心思想是找到相邻伙伴并递归合并
    }
};

```

#### 模式二：Free-List 池（简单高效）

```cpp
class FreeListPool {
private:
    struct Chunk {
        size_t size;
        Chunk* next;
    };

    std::vector<char> memory_;
    Chunk* free_head_ = nullptr;
    size_t chunk_count_ = 0;

public:
    FreeListPool(size_t total_size, size_t chunk_size = 4096)
        : memory_(total_size), chunk_count_(total_size / chunk_size) {
        // 初始化 free list
        char* ptr = memory_.data() + sizeof(Chunk);
        for (size_t i = 0; i < chunk_count_ - 1; i++) {
            reinterpret_cast<Chunk*>(ptr)->next =
                reinterpret_cast<Chunk*>(ptr + chunk_size);
            ptr += chunk_size;
        }
        reinterpret_cast<Chunk*>(ptr)->next = nullptr;
        free_head_ = reinterpret_cast<Chunk*>(memory_.data() + sizeof(Chunk));
    }

    void* allocate() {
        if (!free_head_) return nullptr;
        Chunk* chunk = free_head_;
        free_head_ = free_head_->next;
        return chunk;
    }

    void deallocate(void* ptr) {
        Chunk* chunk = static_cast<Chunk*>(ptr);
        chunk->next = free_head_;
        free_head_ = chunk;
    }
};

```

---

### 10.3 进程间共享内存

在多进程架构中（如主进程负责推理，子进程负责后处理/显示），共享内存可减少数据拷贝：

```cpp
// shared_memory.h
#pragma once
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>

class SharedMemory {
public:
    SharedMemory(const std::string& name, size_t size, bool create = true)
        : name_(name), size_(size), fd_(-1), addr_(MAP_FAILED) {
        if (create) {
            fd_ = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
            if (fd_ == -1) throw std::runtime_error("shm_open failed");
            ftruncate(fd_, size);
        } else {
            fd_ = shm_open(name.c_str(), O_RDWR, 0666);
            if (fd_ == -1) throw std::runtime_error("shm_open failed: not found");
        }
        addr_ = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (addr_ == MAP_FAILED) throw std::runtime_error("mmap failed");
    }

    ~SharedMemory() {
        if (addr_ != MAP_FAILED) munmap(addr_, size_);
        if (fd_ != -1) close(fd_);
    }

    // 禁止拷贝
    SharedMemory(const SharedMemory&) = delete;
    SharedMemory& operator=(const SharedMemory&) = delete;

    void* data() { return addr_; }
    const void* data() const { return addr_; }
    size_t size() const { return size_; }

    // 删除共享内存（仅在创建者进程中调用）
    static void unlink(const std::string& name) {
        shm_unlink(name.c_str());
    }

private:
    std::string name_;
    size_t size_;
    int fd_;
    void* addr_;
};

```

**使用示例：主进程写入，子进程读取**

```cpp
// 生产者进程（推理主进程）
SharedMemory shm("yolo_shm", 10 * 1024 * 1024);  // 10MB 共享内存
auto* buf = static_cast<float*>(shm.data());
// 将预处理后的输入写入共享内存
std::memcpy(buf, preprocessed_data, input_size * sizeof(float));
// 通过 posix semaphore 通知消费者
sem_post(&data_ready_sem);

// 消费者进程（后处理/显示）
SharedMemory shm("yolo_shm", 10 * 1024 * 1024, false);  // 不创建，只附加
auto* buf = static_cast<float*>(shm.data());
// 等待数据就绪
sem_wait(&data_ready_sem);
// 直接使用共享内存中的数据进行后处理

```

---

### 10.4 跨进程模型共享

多个推理进程共享同一份模型文件，避免重复加载：

```bash
# 方案一：只读 mmap 共享（操作系统自动处理）
# 多个进程 mmap 同一 .rknn 文件，内核自动共享物理页
# 无需额外编程，操作系统内核处理

# 方案二：通过共享内存传递模型句柄
# 主进程加载模型后，将 rknn_context 传递给子进程
# 注意：rknn_context 本身不可直接共享，需通过 IPC 传递推理结果

```

```cpp
// 跨进程推理架构
// 主进程（模型加载 + 推理）
class ModelServer {
private:
    YOLODetector detector_;
    SharedMemory result_shm_;
    std::mutex result_mutex_;
    std::condition_variable result_cv_;

public:
    ModelServer(const std::string& model_path) {
        detector_.load_model(model_path);
        result_shm_.resize(1024 * 1024);  // 1MB 结果缓冲
    }

    // 接收请求，返回结果（通过共享内存）
    void process_request(const float* input, size_t input_size) {
        // 1. 预处理
        cv::Mat img = preprocess_from_buffer(input, input_size);

        // 2. 推理（单线程，避免并发）
        auto results = detector_.detect(img);

        // 3. 写入共享内存
        {
            std::lock_guard<std::mutex> lock(result_mutex_);
            std::memcpy(result_shm_.data(), results.data(),
                        results.size() * sizeof(Detection));
            result_cv_.notify_all();
        }
    }
};

// 工作进程（仅需读取结果）
class WorkerProcess {
private:
    SharedMemory result_shm_;
    std::mutex result_mutex_;
    std::condition_variable result_cv_;

public:
    void wait_and_process() {
        std::unique_lock<std::mutex> lock(result_mutex_);
        result_cv_.wait(lock, [&]() {
            return /* 有新数据 */ true;
        });
        auto* results = static_cast<Detection*>(result_shm_.data());
        // 处理后处理逻辑
    }
};

```

---

## 十一、高级性能优化

### 11.1 CPU-NPU 流水线重叠（双缓冲）

在单 NPU 场景下，可以通过双缓冲技术让 CPU 预处理和 NPU 推理并行执行：

```cpp
// double_buffer.h
#pragma once
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>

class DoubleBufferPipeline {
private:
    static constexpr int BUFFER_COUNT = 2;

    struct BufferSlot {
        cv::Mat input;           // 预处理后的输入
        rknn_tensor_mem* mem;   // NPU 内存
        bool ready;             // 是否已填充
        std::mutex mutex;       // 槽位锁
        std::condition_variable cv;
    };

    BufferSlot slots_[BUFFER_COUNT];
    int next_slot_ = 0;
    std::atomic<bool> running_{false};
    std::thread pipeline_thread_;

public:
    DoubleBufferPipeline(rknn_context ctx, size_t input_size) {
        for (int i = 0; i < BUFFER_COUNT; i++) {
            slots_[i].ready = false;
            slots_[i].mem = rknn_create_mem(ctx, (uint32_t)input_size);
        }
    }

    ~DoubleBufferPipeline() {
        running_ = false;
        if (pipeline_thread_.joinable()) pipeline_thread_.join();
        for (int i = 0; i < BUFFER_COUNT; i++) {
            if (slots_[i].mem) rknn_destroy_mem(nullptr, slots_[i].mem);
        }
    }

    // 提交帧（非阻塞，立即返回）
    void submit(const cv::Mat& raw_image) {
        int slot = next_slot_;
        next_slot_ = (next_slot_ + 1) % BUFFER_COUNT;

        // 预处理并写入该槽位
        float scale;
        int pad_w, pad_h;
        cv::Mat processed = preprocess(raw_image, scale, pad_w, pad_h);

        std::lock_guard<std::mutex> lock(slots_[slot].mutex);
        slots_[slot].input = processed;
        slots_[slot].ready = true;
        slots_[slot].cv.notify_one();
    }

    // 获取推理结果（阻塞，直到当前槽位完成推理）
    std::vector<Detection> fetch_results(rknn_context ctx) {
        int slot = next_slot_;  // 等待上一个 slot 完成

        // 等待该槽位被填充
        std::unique_lock<std::mutex> lock(slots_[slot].mutex);
        slots_[slot].cv.wait(lock, [&]() { return slots_[slot].ready; });

        // 设置零拷贝输入
        rknn_input in = {};
        in.index         = 0;
        in.type          = RKNN_INPUT_TENSOR;
        in.pass_through  = 1;
        in.mem           = slots_[slot].mem;
        std::memcpy(slots_[slot].mem->virt_addr,
                    slots_[slot].input.data,
                    slots_[slot].input.total() * slots_[slot].input.elemSize());
        rknn_inputs_set(ctx, 1, &in);

        // 启动推理
        rknn_output out = {};
        out.index       = 0;
        out.want_float  = 1;
        rknn_run(ctx, nullptr);
        rknn_outputs_get(ctx, 1, &out, NULL);

        // 后处理
        auto results = postprocess(
            (float*)out.buf, slots_[slot].input.size(), /*参数略*/);
        rknn_outputs_release(ctx, 1, &out);

        // 标记为可用（供下一轮复用）
        slots_[slot].ready = false;
        return results;
    }
};

```

> **效果**：双缓冲可将有效吞吐量提升 30-50%，因为预处理和推理在时间上重叠。

---

### 11.2 多摄像头多流处理

```cpp
// multi_camera.h
#pragma once
#include <thread>
#include <queue>
#include <atomic>
#include <functional>

class MultiCameraPipeline {
private:
    static constexpr int MAX_CAMERAS = 4;
    YOLODetector detectors_[MAX_CAMERAS];
    cv::VideoCapture cams_[MAX_CAMERAS];
    std::atomic<bool> running_{false};
    std::vector<std::thread> threads_;
    std::function<void(int, const std::vector<Detection>&)> result_callback_;

public:
    MultiCameraPipeline(int num_cameras,
                        const std::vector<std::string>& model_paths,
                        std::function<void(int, const std::vector<Detection>&)> callback)
        : result_callback_(callback) {
        // 为每个摄像头加载独立模型实例
        for (int i = 0; i < num_cameras; i++) {
            detectors_[i].load_model(model_paths[i]);
            cams_[i].open(i);  // camera ID
            if (!cams_[i].isOpened()) {
                printf("[ERROR] Failed to open camera %d\n", i);
            }
        }
    }

    void start() {
        running_ = true;
        for (int i = 0; i < MAX_CAMERAS; i++) {
            if (cams_[i].isOpened()) {
                threads_.emplace_back([this, i]() { worker_thread(i); });
            }
        }
    }

    void stop() {
        running_ = false;
        for (auto& t : threads_) {
            if (t.joinable()) t.join();
        }
    }

private:
    void worker_thread(int cam_id) {
        cv::Mat frame;
        while (running_) {
            if (!cams_[cam_id].read(frame)) continue;
            if (frame.empty()) continue;

            auto results = detectors_[cam_id].detect(frame);
            result_callback_(cam_id, results);
        }
    }
};

```

---

### 11.3 帧率自适应帧跳过策略

在 NPU 处理速度跟不上输入帧率时，智能跳过帧以避免延迟累积：

```cpp
class AdaptiveFrameSkipper {
private:
    std::atomic<double> target_fps_{30.0};
    std::atomic<double> min_fps_{10.0};
    std::atomic<double> current_interval_ms_{33.3};  // 30fps 对应 33.3ms
    std::atomic<int> frames_skipped_{0};

public:
    // 判断是否应跳过当前帧
    bool should_skip(double inference_time_ms) {
        double required_interval = 1000.0 / target_fps_.load();

        // 如果推理耗时超过目标间隔，进入自适应模式
        if (inference_time_ms > required_interval) {
            // 根据负载动态调整跳过策略
            double overhead_ratio = inference_time_ms / required_interval;
            if (overhead_ratio > 2.0) {
                // 严重过载：每 3 帧处理 1 帧
                int skip = (frames_skipped_.fetch_add(1) + 1) % 3;
                return skip != 0;
            } else if (overhead_ratio > 1.5) {
                // 中度过载：每 2 帧处理 1 帧
                int skip = (frames_skipped_.fetch_add(1) + 1) % 2;
                return skip != 0;
            }
        }
        return false;
    }

    void set_target_fps(double fps) {
        target_fps_.store(fps);
        current_interval_ms_.store(1000.0 / fps);
    }

    int get_skipped_count() const {
        return frames_skipped_.load();
    }
};

```

---

### 11.4 NPU 频率缩放集成

动态调整 NPU 频率以平衡性能和功耗：

```cpp
class NPUClockManager {
public:
    // 设置 NPU 频率档位
    static bool set_frequency(const std::string& governor) {
        std::string path = "/sys/class/devfreq/fd8c0000.npu/governor";
        std::ofstream f(path);
        if (!f.is_open()) return false;
        f << governor;
        return f.good();
    }

    // 读取当前频率
    static int get_current_freq() {
        std::string path = "/sys/class/devfreq/fd8c0000.npu/cur_freq";
        std::ifstream f(path);
        int freq = 0;
        f >> freq;
        return freq;
    }

    // 根据负载动态调整
    static void adaptive_scaling(double load, int max_freq, int min_freq) {
        if (load > 0.9) {
            set_frequency("performance");  // 全速
        } else if (load < 0.3) {
            set_frequency("powersave");    // 省电
        } else {
            set_frequency("interactive");  // 动态
        }
    }

    // 获取可用频率档位
    static std::vector<int> get_freq_table() {
        std::vector<int> freqs;
        std::string path = "/sys/class/devfreq/fd8c0000.npu/freqlist";
        std::ifstream f(path);
        int freq;
        while (f >> freq) freqs.push_back(freq);
        return freqs;
    }
};

```

---

### 11.5 功耗感知调度

```cpp
class PowerAwareScheduler {
private:
    double max_power_watts_;
    double current_power_watts_;
    int max_npu_cores_;

public:
    PowerAwareScheduler(double max_power = 5.0, int max_cores = 3)
        : max_power_watts_(max_power), max_npu_cores_(max_cores) {}

    // 根据功耗预算决定启用多少个 NPU 核心
    int get_active_core_count() {
        current_power_watts_ = read_power_sensor();
        if (current_power_watts_ > max_power_watts_) {
            // 超额：减少核心数
            return std::max(1, max_npu_cores_ - 1);
        }
        return max_npu_cores_;
    }

    // 读取 NPU 功耗（瓦特）
    double read_power_sensor() {
        // Rockchip 平台的功耗读数
        std::string path = "/sys/class/powercap/rockchip-powercap/fd8c0000.npu/power";
        std::ifstream f(path);
        double power = 0;
        f >> power;
        return power / 1000000.0;  // 转换为瓦特
    }

    // 整合到并行推理中
    std::vector<std::vector<Detection>> detect_power_aware(
        const std::vector<cv::Mat>& images,
        std::vector<YOLODetector>& detectors) {
        int active_cores = get_active_core_count();
        std::vector<std::future<std::vector<Detection>>> futures;

        for (size_t i = 0; i < images.size(); i++) {
            if (i >= (size_t)active_cores) break;  // 超出功耗预算则跳过
            futures.push_back(std::async(std::launch::async,
                [&detectors, &images, i]() {
                    return detectors[i % active_cores].detect(images[i]);
                }));
        }

        std::vector<std::vector<Detection>> results;
        for (auto& f : futures) results.push_back(f.get());
        return results;
    }
};

```

---

## 十二、跨平台编译考虑

### 12.1 Windows (WSL2) 交叉编译

```bash
# WSL2 环境配置
sudo apt update
sudo apt install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu cmake

# 从 WSL2 编译
cd /mnt/c/Users/ASUS/Desktop/技术博客
mkdir -p build && cd build
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../toolchain-aarch64.cmake \
    -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)

# 通过 SCP 传输到开发板
scp yolo_rknn root@192.168.1.100:/usr/local/bin/

```

### 12.2 macOS 交叉编译

```bash
# macOS 上使用 Homebrew 安装交叉工具链
brew install FiloSottile/musl-cross/musl-cross  # 通用 ARM 交叉编译
# 或
brew install aarch64-linux-gnu-gcc

# 配置 CMake 工具链
cat > toolchain-macos-aarch64.cmake << 'EOF'
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# macOS 上的交叉编译工具
set(CMAKE_C_COMPILER   /opt/homebrew/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /opt/homebrew/bin/aarch64-linux-gnu-g++)

set(CMAKE_FIND_ROOT_PATH /opt/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
EOF

# 编译
cmake .. -DCMAKE_TOOLCHAIN_FILE=toolchain-macos-aarch64.cmake -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)

```

### 12.3 静态链接 vs 动态链接权衡

```
┌──────────────┬─────────────────────┬─────────────────────┐
│    特性       │   动态链接           │   静态链接           │
├──────────────┼─────────────────────┼─────────────────────┤
│ 二进制体积    │ 小 (1-5MB)          │ 大 (10-30MB)        │
│ 部署复杂度    │ 需确保目标机有 .so   │ 单文件即运行         │
│ 更新灵活性    │ 可单独升级 .so       │ 需重新编译           │
│ 启动速度      │ 稍慢 (动态链接解析)  │ 稍快                 │
│ 内存占用      │ 共享 .so 节省内存    │ 每个进程独立拷贝     │
│ 适用场景      │ 开发/测试/多应用共享 │ 交付/嵌入式/单应用   │
└──────────────┴─────────────────────┴─────────────────────┘

```

**动态链接（推荐开发阶段）：**

```cmake
# CMakeLists.txt 默认即可
target_link_libraries(yolo_rknn
    ${OpenCV_LIBS}
    rknn_runtime
    pthread dl
)

```

**静态链接（推荐交付阶段）：**

```cmake
# 静态链接需要额外处理
target_link_options(yolo_rknn PRIVATE -static)
# 排除不兼容静态链接的库
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--exclude-libs,ALL")
# 手动链接必要库
target_link_libraries(yolo_rknn
    ${OpenCV_LIBS}
    rknn_runtime
    pthread dl rt z
)

```

### 12.4 打包分发

```bash
# ===== DEB 包（Debian/Ubuntu/Rockchip Linux）=====
# debian/control
Package: yolo-rknn
Version: 1.0.0
Section: utils
Priority: optional
Architecture: arm64
Depends: libc6, libopencv-core4, libopencv-dnn4, librknn-runtime
Description: YOLO NPU inference service for Rockchip

# 构建
dpkg-buildpackage -us -uc -b

# ===== RPM 包（Rockchip Yocto 发行版）=====
# yolo-rknn.spec
Name: yolo-rknn
Version: 1.0.0
Release: 1
Summary: YOLO NPU inference
BuildArch: aarch64
Requires: libopencv, librknn-runtime

# ===== AppImage（通用 Linux）=====
# 需要包含所有依赖的 .so
appimagetool-x86_64.AppImage yolo-rknn/

# ===== 简单 tarball 分发 =====
tar -czf yolo-rknn-1.0.0-aarch64.tar.gz \
    yolo_rknn \
    best.rknn \
    coco.names \
    README.md

```

---

## 十三、生产部署增强

### 13.1 HTTP 健康检查端点

```cpp
// health_check.h
#pragma once
#include <thread>
#include <atomic>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cstring>

class HealthCheckServer {
private:
    int server_fd_;
    std::atomic<bool> running_{false};
    std::thread server_thread_;
    YOLODetector& detector_;

    static const char* respond(int client_fd, const char* body, int status_code) {
        std::string response = "HTTP/1.1 " + std::to_string(status_code) + "\r\n"
                             + "Content-Type: application/json\r\n"
                             + "Connection: close\r\n"
                             + "Content-Length: " + std::to_string(std::strlen(body)) + "\r\n"
                             + "\r\n" + body;
        send(client_fd, response.c_str(), response.size(), MSG_NOSIGNAL);
        return "";
    }

public:
    HealthCheckServer(YOLODetector& detector, uint16_t port = 8080)
        : detector_(detector), server_fd_(-1) {
        server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        int opt = 1;
        setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port);
        bind(server_fd_, (struct sockaddr*)&addr, sizeof(addr));
        listen(server_fd_, 5);
    }

    void start() {
        running_ = true;
        server_thread_ = std::thread([this]() {
            while (running_) {
                struct sockaddr_in client_addr;
                socklen_t client_len = sizeof(client_addr);
                int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
                if (client_fd < 0) break;

                // 读取请求
                char buffer[1024] = {};
                recv(client_fd, buffer, sizeof(buffer) - 1, 0);

                // 简单路由
                std::string request(buffer);
                std::string body;
                int status = 200;

                if (request.find("/health") != std::string::npos) {
                    body = "{\"status\":\"healthy\",\"npu_load\":"
                           + get_npu_load() + "}";
                } else if (request.find("/metrics") != std::string::npos) {
                    body = "# HELP inference_total Total inferences\n"
                           "# TYPE inference_total counter\n"
                           "inference_total " + std::to_string(detector_.get_total_inferences()) + "\n";
                } else {
                    body = "{\"service\":\"yolo-rknn\",\"version\":\"1.0.0\"}";
                }

                respond(client_fd, body.c_str(), status);
                close(client_fd);
            }
        });
    }

    void stop() {
        running_ = false;
        if (server_thread_.joinable()) server_thread_.join();
        if (server_fd_ >= 0) close(server_fd_);
    }

private:
    std::string get_npu_load() {
        // 读取 NPU 利用率
        std::string path = "/proc/driver/rockchip/npu/info";
        std::ifstream f(path);
        std::string line;
        while (std::getline(f, line)) {
            if (line.find("Utilization") != std::string::npos) {
                // 解析利用率百分比
                auto pos = line.find(':');
                if (pos != std::string::npos) {
                    return line.substr(pos + 1).substr(0, 4);
                }
            }
        }
        return "0";
    }
};

```

### 13.2 gRPC 服务封装

```cpp
// 使用 protobuf 定义服务接口
// yolo_service.proto
syntax = "proto3";
package yolo;

service YOLOService {
    rpc Detect (DetectRequest) returns (DetectResponse);
    rpc DetectBatch (DetectBatchRequest) returns (DetectBatchResponse);
    rpc GetModelInfo (Empty) returns (ModelInfo);
}

message DetectRequest {
    bytes image_data = 1;      // JPEG/PNG 编码的图像
    int32  image_width = 2;
    int32  image_height = 3;
    int32  image_channels = 4;
}

message BoundingBox {
    int32  x1 = 1;
    int32  y1 = 2;
    int32  x2 = 3;
    int32  y2 = 4;
    int32  class_id = 5;
    float  confidence = 6;
    string class_name = 7;
}

message DetectResponse {
    repeated BoundingBox boxes = 1;
    float inference_time_ms = 2;
}

message DetectBatchRequest {
    repeated DetectRequest requests = 1;
}

message DetectBatchResponse {
    repeated DetectResponse responses = 1;
}

message ModelInfo {
    string model_name = 1;
    int32  input_width = 2;
    int32  input_height = 3;
    int32  num_classes = 4;
    bool   quantized = 5;
}

message Empty {}

```

**gRPC 服务实现：**

```cpp
// grpc_service.h
#pragma once
#include <grpcpp/grpcpp.h>
#include "yolo_service.grpc.pb.h"
#include "yolo_detector.h"

class YOLOServiceImpl final : public yolo::YOLOService::Service {
public:
    YOLOServiceImpl(YOLODetector& detector) : detector_(detector) {}

    grpc::Status Detect(grpc::ServerContext* context,
                        const yolo::DetectRequest* request,
                        yolo::DetectResponse* response) override {
        // 解码图像
        cv::Mat img = decode_image(request->image_data(),
                                    request->image_width(),
                                    request->image_height(),
                                    request->image_channels());
        if (img.empty()) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Invalid image");
        }

        // 推理
        auto start = std::chrono::high_resolution_clock::now();
        auto detections = detector_.detect(img);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        // 填充响应
        for (const auto& det : detections) {
            auto* box = response->add_boxes();
            box->set_x1(det.x1);
            box->set_y1(det.y1);
            box->set_x2(det.x2);
            box->set_y2(det.y2);
            box->set_class_id(det.class_id);
            box->set_confidence(det.confidence);
            box->set_class_name(det.class_name);
        }
        response->set_inference_time_ms(ms);
        return grpc::Status::OK;
    }

    grpc::Status GetModelInfo(grpc::ServerContext* context,
                              const yolo::Empty* request,
                              yolo::ModelInfo* response) override {
        response->set_model_name("yolov8s-rk3588");
        response->set_input_width(640);
        response->set_input_height(640);
        response->set_num_classes(80);
        response->set_quantized(true);
        return grpc::Status::OK;
    }

private:
    YOLODetector& detector_;
    cv::Mat decode_image(const char* data, int width, int height, int channels) {
        cv::Mat img(height, width, CV_MAKETYPE(CV_8U, channels),
                    (unsigned char*)data);
        return img;
    }
};

```

### 13.3 Prometheus 指标导出

```cpp
// prometheus_metrics.h
#pragma once
#include <prometheus/metric.h>
#include <prometheus/counter.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <prometheus/server/register_center.h>
#include <prometheus/server.h>

class PrometheusExporter {
public:
    PrometheusExporter(uint16_t port = 9090)
        : registry_(std::make_shared<prometheus::Registry>()) {
        // 创建指标
        inference_count_ = &prometheus::BuildCounter()
            .Name("yolo_inference_total")
            .Help("Total number of inferences")
            .Register(*registry_)
            .Add({});

        inference_latency_ = &prometheus::BuildHistogram()
            .Name("yolo_inference_latency_seconds")
            .Help("Inference latency distribution")
            .Register(*registry_)
            .Add({});

        npu_utilization_ = &prometheus::BuildGauge()
            .Name("yolo_npu_utilization_percent")
            .Help("Current NPU utilization")
            .Register(*registry_)
            .Add({});

        frame_skip_count_ = &prometheus::BuildCounter()
            .Name("yolo_frames_skipped_total")
            .Help("Total frames skipped due to overload")
            .Register(*registry_)
            .Add({});

        // 启动 HTTP 服务器
        server_ = std::make_unique<prometheus::HttpServer>(
            prometheus::HttpServer::CreateBare("0.0.0.0", port, 1));
    }

    void record_inference(double latency_ms, int num_detections) {
        inference_count_->Increment();
        inference_latency_->Observe(latency_ms / 1000.0);
    }

    void set_npu_utilization(double percent) {
        npu_utilization_->Set(percent);
    }

    void record_frame_skip() {
        frame_skip_count_->Increment();
    }

private:
    std::shared_ptr<prometheus::Registry> registry_;
    std::unique_ptr<prometheus::HttpServer> server_;
    prometheus::Counter* inference_count_;
    prometheus::Histogram* inference_latency_;
    prometheus::Gauge* npu_utilization_;
    prometheus::Counter* frame_skip_count_;
};

```

### 13.4 模型热加载

```cpp
// hot_reload.h
#pragma once
#include <thread>
#include <atomic>
#include <mutex>

class HotReloadableDetector {
private:
    YOLODetector detector_;
    std::atomic<bool> reload_requested_{false};
    std::string current_model_path_;
    std::mutex reload_mutex_;
    std::thread reload_watcher_;

public:
    bool load_model(const std::string& model_path) {
        std::lock_guard<std::mutex> lock(reload_mutex_);
        if (model_path == current_model_path_) return true;

        printf("[INFO] Reloading model: %s\n", model_path.c_str());
        detector_.~YOLODetector();  // 销毁旧实例
        new (&detector_) YOLODetector();  // 原地构造新实例
        int ret = detector_.load_model(model_path);
        if (ret == 0) {
            current_model_path_ = model_path;
            printf("[INFO] Model reloaded successfully\n");
        }
        return ret == 0;
    }

    // 后台监控模型文件变更
    void start_watcher(const std::string& model_path, int check_interval_s = 10) {
        current_model_path_ = model_path;
        reload_watcher_ = std::thread([this, model_path, check_interval_s]() {
            long long last_mtime = get_file_mtime(model_path);
            while (true) {
                std::this_thread::sleep_for(std::chrono::seconds(check_interval_s));
                long long current_mtime = get_file_mtime(model_path);
                if (current_mtime != last_mtime) {
                    printf("[INFO] Model file changed, reloading...\n");
                    load_model(model_path);
                    last_mtime = current_mtime;
                }
                if (reload_requested_.load()) {
                    printf("[INFO] Manual reload requested\n");
                    load_model(model_path);
                    reload_requested_.store(false);
                }
            }
        });
    }

    void request_reload() { reload_requested_.store(true); }

    ~HotReloadableDetector() {
        reload_requested_.store(true);
        if (reload_watcher_.joinable()) reload_watcher_.join();
    }

private:
    long long get_file_mtime(const std::string& path) {
        struct stat st;
        if (stat(path.c_str(), &st) == 0) return st.st_mtime;
        return 0;
    }
};

```

### 13.5 YAML 配置文件管理

```yaml
# config.yaml
model:
  path: "/usr/local/share/yolo_rknn/best.rknn"
  input_size: 640
  backend: "rknn"  # rknn | onnx | tensorrt

input:
  source: "camera:0"
  width: 1920
  height: 1080
  fps: 30

postprocess:
  conf_threshold: 0.25
  iou_threshold: 0.45
  max_detections: 300
  nms_method: "dnn"  # dnn | cpu | gpu

performance:
  num_npu_cores: 3
  zero_copy: true
  double_buffer: true
  frame_skip: true
  target_fps: 30
  min_fps: 10

monitoring:
  http_port: 8080
  prometheus_port: 9090
  log_level: "INFO"
  log_file: "/var/log/yolo_rknn.log"

power:
  npu_governor: "performance"
  max_power_watts: 5.0
  thermal_threshold_c: 80

```

```cpp
// config_manager.h
#pragma once
#include <yaml-cpp/yaml.h>
#include <nlohmann/json.hpp>

class ConfigManager {
public:
    static ConfigManager& instance() {
        static ConfigManager config;
        return config;
    }

    bool load(const std::string& path) {
        try {
            YAML::Node node = YAML::LoadFile(path);
            model_path_ = node["model"]["path"].as<std::string>();
            input_size_ = node["model"]["input_size"].as<int>();
            backend_ = node["model"]["backend"].as<std::string>();
            conf_threshold_ = node["postprocess"]["conf_threshold"].as<float>();
            iou_threshold_ = node["postprocess"]["iou_threshold"].as<float>();
            num_npu_cores_ = node["performance"]["num_npu_cores"].as<int>();
            zero_copy_ = node["performance"]["zero_copy"].as<bool>();
            return true;
        } catch (const YAML::Exception& e) {
            printf("[ERROR] Failed to load config: %s\n", e.what());
            return false;
        }
    }

    std::string model_path_;
    int input_size_ = 640;
    std::string backend_ = "rknn";
    float conf_threshold_ = 0.25f;
    float iou_threshold_ = 0.45f;
    int num_npu_cores_ = 3;
    bool zero_copy_ = true;
};

```

---

## 十四、完整工作示例

### 14.1 完整项目结构

```
yolo-npu-deploy/
├── CMakeLists.txt                    # 主构建配置
├── toolchain-aarch64.cmake          # 交叉编译工具链
├── config.yaml                       # 配置文件
├── src/
│   ├── main.cpp                      # 入口
│   ├── yolo_detector.h               # 检测器头文件
│   ├── yolo_detector.cpp             # 检测器实现
│   ├── inference_engine.h            # 引擎抽象接口
│   ├── rknn_engine.h                 # RKNN 后端
│   ├── rknn_engine.cpp
│   ├── onnx_engine.h                 # ONNX 后端
│   ├── onnx_engine.cpp
│   ├── health_check.h                # HTTP 健康检查
│   ├── prometheus_metrics.h          # Prometheus 指标
│   ├── config_manager.h              # 配置管理
│   └── hot_reload.h                  # 热加载
├── include/
│   └── rknn_api.h                   # RKNN API 头文件
├── lib/
│   └── librknn_runtime.so           # RKNN 运行时库
├── models/
│   └── best.rknn                    # 转换后的模型
├── data/
│   └── coco.names                   # 类别名称
├── scripts/
│   │   ├── build.sh                  # 构建脚本
│   │   ├── deploy.sh                 # 部署脚本
│   │   ├── benchmark.sh              # 性能测试脚本
│   │   └── integration_test.sh       # 集成测试脚本
│   └── docker/
│       └── Dockerfile.cross_compile  # Docker 交叉编译
├── tests/
│   ├── test_detector.cpp             # 单元测试
│   └── test_postprocess.cpp          # 后处理测试
└── README.md

```

### 14.2 分步编译和部署指南

```bash
# ===== Step 1: 环境准备 =====
# 方式 A: 在开发板上直接编译
ssh root@<board_ip>
cd /opt/yolo-npu-deploy
mkdir -p build && cd build

# 方式 B: 在 PC 上交叉编译
cd ~/yolo-npu-deploy
mkdir -p build && cd build

# ===== Step 2: CMake 配置 =====
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=../toolchain-aarch64.cmake \
    -DOpenCV_DIR=/usr/lib/aarch64-linux-gnu/cmake/opencv4 \
    -DRKNN_ROOT=/usr \
    -DCONFIG_FILE=../config.yaml

# ===== Step 3: 编译 =====
make -j$(nproc)

# ===== Step 4: 验证二进制 =====
file yolo_rknn
# 应输出: ELF 64-bit LSB executable, ARM aarch64

ldd yolo_rknn
# 确认无 "not found" 条目

# ===== Step 5: 部署到设备 =====
scp yolo_rknn root@<board_ip>:/usr/local/bin/
scp models/best.rknn root@<board_ip>:/usr/local/share/yolo_rknn/
scp data/coco.names root@<board_ip>:/usr/local/share/yolo_rknn/
scp config.yaml root@<board_ip>:/usr/local/share/yolo_rknn/

# ===== Step 6: 在设备上验证 =====
ssh root@<board_ip>
chmod +x /usr/local/bin/yolo_rknn
/usr/local/bin/yolo_rknn --model /usr/local/share/yolo_rknn/best.rknn --image test.jpg

```

### 14.3 集成测试脚本

```bash
#!/bin/bash
# scripts/integration_test.sh
set -e

MODEL_PATH="/usr/local/share/yolo_rknn/best.rknn"
TEST_IMAGE="/path/to/test.jpg"
BINARY="./yolo_rknn"

echo "=== Integration Test ==="

# Test 1: 模型加载
echo "[Test 1] Model loading..."
timeout 30 $BINARY --model $MODEL_PATH --image $TEST_IMAGE --dry-run
if [ $? -eq 0 ]; then
    echo "PASS: Model loaded successfully"
else
    echo "FAIL: Model loading failed"
    exit 1
fi

# Test 2: 单帧推理
echo "[Test 2] Single frame inference..."
$BINARY --model $MODEL_PATH --image $TEST_IMAGE --output result.json
if [ -f result.json ] && [ -s result.json ]; then
    DET_COUNT=$(cat result.json | jq '.detections | length')
    echo "PASS: Detected $DET_COUNT objects"
else
    echo "FAIL: No output generated"
    exit 1
fi

# Test 3: 批量推理
echo "[Test 3] Batch inference..."
$BINARY --model $MODEL_PATH --batch test_images/ --output batch_result.json
BATCH_COUNT=$(cat batch_result.json | jq '.total_frames')
if [ "$BATCH_COUNT" -gt 0 ]; then
    echo "PASS: Processed $BATCH_COUNT frames"
else
    echo "FAIL: Batch inference failed"
    exit 1
fi

# Test 4: 性能基准
echo "[Test 4] Performance benchmark..."
$BINARY --model $MODEL_PATH --image $TEST_IMAGE --benchmark --iterations 100
echo "PASS: Benchmark complete"

# Test 5: 健康检查
echo "[Test 5] Health check..."
curl -s http://localhost:8080/health | jq .status
if [ $? -eq 0 ]; then
    echo "PASS: Health check OK"
else
    echo "FAIL: Health check failed"
fi

echo "=== All tests passed ==="

```

### 14.4 基准测试脚本

```bash
#!/bin/bash
# scripts/benchmark.sh
set -e

MODEL_PATH="${1:-/usr/local/share/yolo_rknn/best.rknn}"
TEST_DIR="${2:-test_images/}"
ITERATIONS="${3:-100}"
BINARY="./yolo_rknn"

echo "=== Benchmark: $MODEL_PATH ==="
echo "Iterations: $ITERATIONS"
echo "Test images: $TEST_DIR"

# 单帧延迟分布
echo "--- Single Frame Latency ---"
for img in $TEST_DIR/*.jpg; do
    echo "Testing: $(basename $img)"
    $BINARY --model $MODEL_PATH --image $img --benchmark --iterations $ITERATIONS
done

# 吞吐测试
echo "--- Throughput Test ---"
$BINARY --model $MODEL_PATH --batch $TEST_DIR --benchmark --iterations $ITERATIONS

# NPU 利用率
echo "--- NPU Utilization ---"
cat /proc/driver/rockchip/npu/info | grep -E "Utilization|Power|Temp"

# 内存使用
echo "--- Memory Usage ---"
cat /proc/driver/rockchip/npu/info | grep -i mem

# 系统资源
echo "--- System Resources ---"
top -bn1 | grep "Cpu(s)"
free -h | grep Mem

```

```python
# scripts/benchmark_report.py - 生成基准测试报告
import json
import statistics
import sys

def analyze_results(log_file):
    with open(log_file) as f:
        lines = f.readlines()

    latencies = []
    for line in lines:
        if "inference_time" in line.lower():
            try:
                lat = float(line.split(":")[1].strip().split()[0])
                latencies.append(lat)
            except:
                pass

    if not latencies:
        print("No latency data found")
        return

    print(f"Total samples: {len(latencies)}")
    print(f"Mean:   {statistics.mean(latencies):.2f} ms")
    print(f"Median: {statistics.median(latencies):.2f} ms")
    print(f"Min:    {min(latencies):.2f} ms")
    print(f"Max:    {max(latencies):.2f} ms")
    print(f"Std:    {statistics.stdev(latencies):.2f} ms")
    print(f"P95:    {sorted(latencies[int(len(latencies)*0.95)]:.2f} ms")
    print(f"P99:    {sorted(latencies[int(len(latencies)*0.99)]:.2f} ms")
    print(f"FPS:    {1000/statistics.mean(latencies):.1f}")

if __name__ == "__main__":
    analyze_results(sys.argv[1])

```

---

## 总结

C++ 部署 YOLO 到 Rockchip NPU 涉及多个关键环节，本文系统性地梳理了从模型准备到生产部署的完整流程。

### 核心要点回顾

1. **模型准备与转换**
   - 将训练好的 ONNX 模型通过 RKNN-Toolkit2 转换为 `.rknn` 格式
   - 选择合适的量化策略（FP16 / INT8），量化模型推理更快但需关注精度损失
   - 转换时配置正确的输入输出节点名称，避免后处理格式不匹配

2. **推理流程规范**
   - `rknn_init` → `rknn_query` → `rknn_init_runtime` → 推理循环 → `rknn_destroy`
   - 每次推理前必须调用 `rknn_inputs_set` 设置输入，推理后调用 `rknn_outputs_get` 获取结果
   - 注意 `pass_through` 模式与零拷贝内存的生命周期管理

3. **坐标解码（常见 Bug）**
   - 模型输出的 `x, y` 是相对于 anchor 网格的偏移量，需要先还原到 640×640 空间再映射回原图
   - 正确顺序：`模型输出 → 640 空间坐标 → 减去 padding → 除以 scale → 原图坐标`
   - 原代码的 bug 是直接将模型输出乘以 scale 后减 padding，忽略了 640 空间还原步骤

4. **多引擎支持**
   - 通过策略模式设计 `InferenceEngine` 接口，支持 RKNN / ONNX Runtime / TensorRT 后端切换
   - YOLOv8（anchor-free）与 YOLOv5（anchor-based）的后处理格式差异需分别处理
   - 批量后处理可通过 OpenMP 并行化提升吞吐

5. **性能优化路径**
   - **内存**：预分配缓冲、使用 `rknn_create_mem` NPU 内存、内存池复用
   - **并行**：多 NPU 核心（每个核心独立 `rknn_context`，避免共享竞态）
   - **零拷贝**：`pass_through=1` + `rknn_tensor_mem` 消除 CPU→NPU 数据搬运
   - **流水线**：双缓冲实现 CPU-NPU 时间重叠，帧率自适应跳过过载帧
   - **编译**：`-O3 -march=armv8-a -ftree-vectorize -ffast-math`

6. **调试与分析**
   - 使用 `RKNN_QUERY_PERF_RUN` 获取 NPU 推理耗时
   - `perf` / `strace` / `valgrind` 定位 CPU 侧瓶颈
   - `/proc/driver/rockchip/npu/info` 监控 NPU 利用率
   - 分段计时（`TimingTracker`）精确识别预处理、推理、后处理各阶段耗时

7. **生产部署考量**
   - 健壮的异常处理与指数退避重试
   - 结构化日志（分级、缓冲写入、文件日志）
   - systemd 服务管理 + watchdog 自动恢复
   - HTTP 健康检查 + gRPC 服务封装 + Prometheus 指标导出
   - 模型热加载、YAML 配置外置、CI/CD 自动化构建

### 参考资源

- [RKNN C++ API Documentation](https://github.com/airockchip/rknn-toolkit2/blob/master/docs/en/03_rknn_runtime/02_rknn_runtime_api.md)
- [RKNN-Toolkit2 Examples](https://github.com/airockchip/rknn-toolkit2/tree/master/examples)
- [RK3588 NPU 开发指南](https://wiki.t-firefly.com/ROC-RK3588-PC/NPU.html)
- [Ultralytics YOLOv8 C++ Examples](https://github.com/ultralytics/ultralytics/tree/main/examples)
- [ONNX Runtime C++ API](https://onnxruntime.ai/docs/api/cxx/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Prometheus C++ Client](https://github.com/jupp0r/prometheus-cpp)

---

*本文基于 RKNN-Toolkit2 官方 API 编写，适用于 Rockchip RK3568 / RK3568S / RK3588 系列 NPU 平台。*

---

> **📌 系列导航**：[← 上一篇：yolo模型在npu的python部署](yolo模型在npu的python部署.md) · [📖 导读目录](README.md) · [下一篇：yolo模型推理与部署详解 →](yolo模型推理与部署详解.md)
