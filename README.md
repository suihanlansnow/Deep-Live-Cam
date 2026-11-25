<h1 align="center">Deep-Live-Cam · 工程化改造版</h1>

<p align="center">
  <img src="media/demo.gif" alt="Demo GIF" width="640">
</p>

> 本仓库是对 [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) 的工程化改造与性能优化。所有上游功能保留，新增的优化都是**可选运行时增强**——缺失 TensorRT 或 cuDNN 时自动回退。

## 免责声明

本软件用于 AI 媒体生产行业。用户须对真实人脸获得明确授权，并在发布时标注为 deepfake。详情见上游仓库 Disclaimer。

## 性能

> RTX 3070 8GB / Windows 10 / Python 3.9 / ONNX Runtime 1.19.2。insightface 检测每 2 帧一次，swap 每帧一次。

| 场景 | CPU-only | CUDA graph | TensorRT FP32 |
|---|---|---|---|
| 实时摄像头 · 640×480 | 0.6 fps · swap 1500 ms | 27.5 fps · swap 19 ms | **29.9 fps · swap 10 ms** |
| 实时摄像头 · 1280×720 | ~0.4 fps | 20.1 fps | **26.9 fps** |

### 核心改动

1. **DLL 多路注册** — `run.py` 同时注册 `sys.prefix/Lib/site-packages/nvidia/*/bin`、`torch/lib`、`%CUDA_PATH%/bin`、`third_party/TensorRT/bin`；`requirements.txt` 固定 `nvidia-cudnn-cu12`。解决 Windows 上 onnxruntime-gpu **静默回退 CPU** 导致 live 只有 0.5 fps 的问题。
2. **TensorRT FP32 EP** — `face_swapper.py` 的 `_tensorrt_providers()` 探测 DLL、校验 `ort.get_available_providers()`、缓存 engine 到 `models/trt_cache/`。**FP32 而非 FP16**（inswapper FP16 会出红斑伪影），与 CUDA 输出 ≤2/255。`DLC_DISABLE_TENSORRT=1` 强制关闭。
3. **CUDA graph replay** — TRT 不可用时，ORT session 开 `enable_cuda_graph`，每帧推理退化为 `update_inplace + run_with_iobinding`。与 TRT **互斥**。
4. **map_faces 流式编码** — 预处理直接 `cv2.imencode` → ffmpeg stdin，零磁盘往返；编码同趟混音。
5. **检测 ROI 追踪** — `ROIDetector` 用上一帧 bbox 外扩 1.6× 裁剪送检，在专用 384 会话上跑（时序抖动 rms 0.56px，比全帧 640 的 1.98px 更稳），CPU 上检测 113ms → 49ms；丢失/贴边自动回落全帧重锁。离线管线的检测线程同样受益。
6. **检测模型 TensorRT 化** — det 输入固定 letterbox 到 det_size，按 `1x3x640x640` 静态 profile 构建 TRT engine，推理约减半；ROI 会话按 `384` 静态 profile 独立构建。
7. **Live 检测线程分离** — 检测挪到独立 `QThread`：处理线程投递最新帧并直接取结果，永不等待。60fps 摄像头 16.7ms 帧预算不再被内联检测挤爆，消除周期性顿挫。
8. **Live 卡死修复** — 三个 enhancer（GFPGAN / GPEN-BFR-256 / 512）模型缺失时：
   - `pre_check()` 和 `get_*()` 都用 `ensure_model(..., download=False)` **只查磁盘，绝不联网阻塞**
   - `get_*()` return None 不 raise，`process_frame` 直接返回原帧；一次性日志抑制重复刷屏
9. **启动后台预热** — `core.run()` 解析完 providers 后，后台线程预构建 swap / 检测 / ROI 三个引擎，首次 Live 不再卡 60 秒。`DLC_NO_WARMUP=1` 可关。

### 回退优先级

```
TensorRT FP32 (engine cache hit)
  ↓
CUDA graph replay
  ↓
Plain CUDA EP
  ↓
CPU EP (last resort)
```

## 快速开始

**1. 装依赖**

```powershell
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**2. 放模型**

| 模型 | 放至 `models/` |
|---|---|
| `inswapper_128_fp16.onnx` | 必须 |
| `gfpgan-1024.onnx` | 可选（人脸增强；缺失时自动跳过，不卡死） |
| insightface buffalo_l | 首次启动自动下到 `~/.insightface/models/buffalo_l/` |

**3. 启动**

```powershell
py run.py
```

UI：选源人脸 → 选摄像头 → 点 **Live**。启动日志应出现 `Face swapper running on TensorRT.` 即 TRT 通路生效。

## 可选：TensorRT 运行时

TRT EP 需要 NVIDIA TensorRT 运行时 DLL：

```
h:\project\deeplivecam\third_party\TensorRT\bin\
├── nvinfer_10.dll               (395 MB)
├── nvinfer_plugin_10.dll
├── nvonnxparser_10.dll
├── nvinfer_lean_10.dll
├── nvinfer_dispatch_10.dll
└── nvinfer_builder_resource_sm86_10.dll   (RTX 30xx sm_86)
```

也可用 `TENSORRT_PATH` 环境变量指向别处。DLL 齐全后首次启动构建 engine ~60 s，后续命中缓存 ~4 s。**engine 是 sm 架构特定的**，换 GPU 会重建。

## 命令行

```powershell
py run.py -s source.jpg -t target.mp4 -o out.mp4
py run.py --frame-processor face_swapper face_enhancer
py run.py --execution-provider cuda
```

## 性能分析

设置 `DLC_PROFILE=1` 后，Live 预览和视频处理会每 2 秒打印一次分段耗时：

```powershell
$env:DLC_PROFILE='1'; py run.py
# [DLC.PROFILE:live] fps=29.8 frames=60 queue_wait=0.12ms detect=5.41ms swap=10.03ms
# [DLC.PROFILE:display] fps=29.8 frames=60 display=1.42ms
# [DLC.PROFILE:video] fps=48.2 frames=96 detect=6.10ms process=12.40ms write_wait=0.31ms
```

`queue_wait` 长时间接近 0 说明 GPU 是瓶颈；`write_wait` 偏高说明编码器跟不上。

## 致谢

- [hacksider/Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) — 上游
- [deepinsight/insightface](https://github.com/deepinsight/insightface) — 人脸检测/识别

见上游仓库 Credits 与 Disclaimer。
