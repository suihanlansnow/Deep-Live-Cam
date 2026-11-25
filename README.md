<h1 align="center">Deep-Live-Cam 2.1.6</h1>

<p align="center">
  Real-time face swap and video deepfake with a single click and only a single image.
</p>

<p align="center">
<a href="https://trendshift.io/repositories/11395" target="_blank"><img src="https://trendshift.io/api/badge/repositories/11395" alt="hacksider%2FDeep-Live-Cam | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <img src="media/demo.gif" alt="Demo GIF" width="800">
</p>

##  Disclaimer

This deepfake software is designed to be a productive tool for the AI-generated media industry. It can assist artists in animating custom characters, creating engaging content, and even using models for clothing design.

We are aware of the potential for unethical applications and are committed to preventative measures. A built-in check prevents the program from processing inappropriate media (nudity, graphic content, sensitive material like war footage, etc.). We will continue to develop this project responsibly, adhering to the law and ethics. We may shut down the project or add watermarks if legally required.

- Ethical Use: Users are expected to use this software responsibly and legally. If using a real person's face, obtain their consent and clearly label any output as a deepfake when sharing online.

- Content Restrictions: The software includes built-in checks to prevent processing inappropriate media, such as nudity, graphic content, or sensitive material.

- Legal Compliance: We adhere to all relevant laws and ethical guidelines. If legally required, we may shut down the project or add watermarks to the output.

- User Responsibility: We are not responsible for end-user actions. Users must ensure their use of the software aligns with ethical standards and legal requirements.

By using this software, you agree to these terms and commit to using it in a manner that respects the rights and dignity of others.

Users are expected to use this software responsibly and legally. If using a real person's face, obtain their consent and clearly label any output as a deepfake when sharing online. We are not responsible for end-user actions.

## 项目目的 · What this project is for

本项目是一次对 [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) 的**工程化改造与性能优化**。原版代码以 CPU 运行为主，在 Windows + NVIDIA GPU 环境下 onnxruntime-gpu 常常因为 cuDNN / CUDA 运行时缺失而**静默回退到 CPU**，导致实时摄像头换脸实际只有 0.5–1 fps。我们的目标是**保证实时摄像头模式稳定 ≥ 20 fps**，并对视频离线管线做同量级的吞吐提升。

> 原版功能保留不变：UI live 摄像头、CLI 视频/图片换脸、map_faces 多脸映射、mouth mask、poisson blend 等。所有优化都是**可选的运行时增强**——缺失 TensorRT 或缺少 cuDNN 时自动回退到 CUDA graph / CPU，不增加运行时依赖的强约束。

This fork is an engineering pass and performance overhaul of [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam). The upstream code was tuned around a CPU baseline; on Windows + NVIDIA, onnxruntime-gpu would silently fall back to CPU whenever a cuDNN / CUDA runtime DLL was missing, dropping live webcam swap to ~0.5 fps. Our deliverable is a **20 fps guarantee for the live webcam path** and comparable throughput gains for the offline video pipeline.

> Every upstream feature is preserved: UI live webcam, CLI image/video swap, map_faces multi-face mapping, mouth mask, poisson blend, and so on. All new optimizations are **opt-in runtime enhancements** — if TensorRT or cuDNN are absent, the code falls back to CUDA graph replay or plain CPU. No hard new runtime dependencies were introduced.

## Pre-built Deep-Live-Cam 2.8 Ultimate!

<p align="center">
  <a href="https://deeplivecam.net/index.php/quickstart">
    <img src="https://github.com/user-attachments/assets/fa2cdf79-c933-4b93-844a-b087192261ed" width="100%" alt="Lite / Ultimate Download Banner">
  </a>
</p>

<p align="center">
<a href="https://deeplivecam.net/index.php/plans/nvidia-gpu?plan_id=0&group_id=1">
  <img src="https://github.com/user-attachments/assets/56b61811-3a1e-4672-9b50-cf7f6e8e6852" width="40" alt="Windows">
</a>
  &nbsp;&nbsp;&nbsp;
<a href="https://deeplivecam.net/index.php/plans/nvidia-gpu?plan_id=0&group_id=2">
  <img src="https://github.com/user-attachments/assets/6538e3a6-c957-431a-b586-2d6abcf534dc" width="34" alt="Mac Silicon">
</a>
  &nbsp;&nbsp;&nbsp;
<a href="https://deeplivecam.net/index.php/plans/nvidia-gpu?plan_id=0&group_id=3">
  <img src="https://github.com/user-attachments/assets/ad45142e-426c-4364-a2a9-a512670cc62c" width="40" alt="CPU">
</a>
</p>

<p align="center">
  <strong>Windows • Mac Silicon • CPU • NVIDIA • AMD</strong>
</p>

<p align="center">
  Builds optimized for your hardware.
</p>

<p align="center">
  <a href="https://deeplivecam.net/index.php/quickstart">
    <img src="media/Download.png" width="280" alt="Download">
  </a>
</p>

> **Ultimate** includes **30+ exclusive features**, performance optimizations, and **priority support** We only have a single official website which is https://deeplivecam.net . Please be careful on where you download other versions of this application aside from that website and this github repo.

Perfect if you want the fastest setup with **zero manual installation**, pre-configured dependencies, and optimized builds for every supported platform.

## TLDR; Live Deepfake in just 3 Clicks
![easysteps](https://github.com/user-attachments/assets/af825228-852c-411b-b787-ffd9aac72fc6)
1. Select a face
2. Select which camera to use
3. Press live!

## Features & Uses - Everything is in real-time

### Mouth Mask

**Retain your original mouth for accurate movement using Mouth Mask**

<p align="center">
  <img src="media/ludwig.gif" alt="resizable-gif">
</p>

### Face Mapping

**Use different faces on multiple subjects simultaneously**

<p align="center">
  <img src="media/streamers.gif" alt="face_mapping_source">
</p>

### Your Movie, Your Face

**Watch movies with any face in real-time**

<p align="center">
  <img src="media/movie.gif" alt="movie">
</p>

### Live Show

**Run Live shows and performances**

<p align="center">
  <img src="media/live_show.gif" alt="show">
</p>

### Memes

**Create Your Most Viral Meme Yet**

<p align="center">
  <img src="media/meme.gif" alt="show" width="450"> 
  <br>
  <sub>Created using Many Faces feature in Deep-Live-Cam</sub>
</p>

### Omegle

**Surprise people on Omegle**

<p align="center">
  <video src="https://github.com/user-attachments/assets/2e9b9b82-fa04-4b70-9f56-b1f68e7672d0" width="450" controls></video>
</p>

## 性能优化 · Performance optimizations

> 全部在 RTX 3070 8GB / Windows 10 / Python 3.9 上实测。live 循环每 2 帧检测一次人脸（30 fps 摄像头），swap 每帧执行一次。基准为 CPU EP、运行时依赖齐全、TensorRT 已缓存引擎。

> All numbers measured on an RTX 3070 8GB / Windows 10 / Python 3.9. The live loop runs face detection every 2 frames on a 30 fps webcam, swap every frame. Baseline uses the CPU EP; runtime deps are fully installed; TensorRT engine is from cache.

| 路径 / Path | CPU-only | CUDA graph | TensorRT FP32 |
|---|---|---|---|
| 实时摄像头 live · 640×480 | **0.6 fps** · swap 1500 ms | **27.5 fps** · swap 19 ms | **29.9 fps** · swap 10 ms |
| 实时摄像头 live · 1280×720 | ~0.4 fps | 20.1 fps | **26.9 fps** |
| map_faces 视频管线 · 1080p / 5 min | 吞吐低、磁盘 I/O 重 | 消除 PNG 往返、同趟混音 | 引擎缓存命中后秒级加载 |

### 1. 修复：CUDA 执行器加载失败（根因） · CUDA EP restoration

原版 `run.py` 只注册了 pip `nvidia-*` wheels 里的 DLL 目录。当用户只装了 CUDA Toolkit 而没装 pip wheel 时（或反过来），onnxruntime-gpu **静默**回退到 CPU，无任何警告——这是"live 卡死"的直接原因。

- `run.py` 在启动时同时注册：`sys.prefix/Lib/site-packages/nvidia/*/bin`、`torch/lib`、`%CUDA_PATH%/bin` 兜底，并对每个目录执行 `PATH` 前置 + `os.add_dll_directory()`（Python 3.8+ 扩展模块依赖需后者）。
- `requirements.txt` 声明 `nvidia-cudnn-cu12>=9.1,<10`（Windows），避免重装环境后再次静默回退。
- 验证标准：启动日志中应出现 `Applied providers: ['CUDAExecutionProvider', ...]`。

**Fix: the upstream `run.py` only registered DLL directories from pip `nvidia-*` wheels.** Users with only the CUDA Toolkit installed (or vice versa) got a **silent** CPU fallback — the direct cause of "live is painfully slow".

- `run.py` now registers, at startup: `sys.prefix/Lib/site-packages/nvidia/*/bin`, `torch/lib`, and `%CUDA_PATH%/bin` as a fallback, with both `PATH` prepend and `os.add_dll_directory()` (required by Python ≥3.8 for extension-module native deps).
- `requirements.txt` pins `nvidia-cudnn-cu12>=9.1,<10` on Windows, so a fresh environment won't silently drop to CPU.
- Smoke check: the startup log should print `Applied providers: ['CUDAExecutionProvider', ...]`.

### 2. map_faces 流式编码 · Streaming encoder for map_faces

原版离线管线在预处理阶段把每一帧写成 PNG 到磁盘，再在处理阶段逐帧读回做换脸——整段 I/O 是瓶颈。

**三项改动 / Three changes:**

A. **零 PNG 直写 ffmpeg stdin** — 预处理阶段直接 `cv2.imencode('.jpg', frame)` → `ffmpeg -i pipe:0 -c:v libx264`，不经过磁盘。
B. **编码同趟混音** — 把换脸后的裸视频流和原始音频**一次性**交给 ffmpeg `ffmpeg -i pipe:0 -i audio.mp3 -c:v copy -c:a aac -map 0:v -map 1:a out.mp4`，避免先写无音频文件再 mux。
C. **帧索引 O(1)** — `target_faces_in_frame` 用 `{frame_index: [...faces...]}` 字典查找，替代按文件名字符串排序后 O(N) 扫描。

### 3. TensorRT FP32 执行器 · TensorRT FP32 EP

inswapper 是一个 1×3×128×128 → 1×3×128×128 的 U-Net 结构，完全静态 shape，恰好是 TensorRT 最擅长的场景。

**为什么选 FP32 而不是 FP16？** 先用 FP16 引擎试跑了——推理虽快，但输出出现**明显的红斑块状伪影**（最大像素差 64/255 vs CUDA），原因是该模型部分层激活溢出 fp16 累加器。FP32 引擎输出与 CUDA 最大差 2/255、平均差 0.22，肉眼无差异，且仍有约 2 倍加速（19 ms → 10 ms）。

- 运行时：`run.py` 注册 `third_party/TensorRT/bin`（或 `TENSORRT_PATH` 环境变量）。
- 代码：`modules/processors/frame/face_swapper.py` 新增 `_tensorrt_providers()`，DLL 齐全时 TRT 优先、CUDA 次之、CPU 兜底；任何环节失败自动回退。
- 缓存：引擎和 timing cache 放在 `models/trt_cache/`。首次启动构建引擎 ~60 秒（RTX 3070）；后续命中缓存 ~4 秒。
- 回退开关：`DLC_DISABLE_TENSORRT=1` 强制走 CUDA graph 回放。
- 注意：引擎是 **sm86 (RTX 30xx)** 特定的。换 GPU/换算力架构后首次启动会重建。

> `third_party/TensorRT/bin`（~1.2 GB）和 `models/trt_cache`（~265 MB）默认 `.gitignore`，需要自己下载/生成。

**Why FP32 and not FP16?** The FP16 engine produced visible blotchy colour artifacts (max pixel diff 64/255 vs CUDA) because certain layer activations overflow the fp16 accumulator. FP32 matches CUDA within 2/255 (mean 0.22) and is still ~2× faster (19 ms → 10 ms per swap).

- Runtime: `run.py` registers `third_party/TensorRT/bin` (or `TENSORRT_PATH` env var).
- Code: `modules/processors/frame/face_swapper.py` adds `_tensorrt_providers()` that puts TRT first, CUDA second, CPU last; each layer is independent so any missing piece causes a transparent fallback.
- Caching: engine + timing cache under `models/trt_cache/`. First launch builds the engine (~60 s on RTX 3070); subsequent launches load from cache (~4 s).
- Escape hatch: set `DLC_DISABLE_TENSORRT=1` to force CUDA graph replay.
- Caveat: the engine is **sm86 (RTX 30xx)** specific — a different GPU / compute capability will trigger a rebuild on first run.

> `third_party/TensorRT/bin` (~1.2 GB) and `models/trt_cache` (~265 MB) are `.gitignore`d by default — download/generate them locally.

### 4. CUDA graph session · CUDA graph replay

当 TensorRT 不可用时（例如没装运行时 DLL），`_init_cuda_graph_session()` 会在启动时用 `enable_cuda_graph: '1'` 创建一个带 CUDA graph 的 ORT session，并预先分配好输入输出的 GPU buffer（inswapper 静态 1×3×128×128 + 1×512）。每帧推理变成一次 `update_inplace + run_with_iobinding`，省去了每次 kernel launch 的 CPU 同步开销。CUDA graph 与 TensorRT **互斥**——TensorRT EP 自带 graph 级优化，再叠一层 ORT 级的 graph replay 不会更快。

### 5. Live 检测频率 · Live detection cadence

`WebcamPreviewWindow` 的 worker 用 `det_interval = max(1, round(self._fps * 0.08))` 决定每隔多少帧检测一次人脸。30 fps 摄像头 → 每 2 帧检测一次，swap 每帧执行。这既减少了检测开销（buffalo_l 的 det_model 约 24 ms/次），又避免每帧都重算仿射变换导致贴回位置抖动。

### 回退优先级 · Fallback precedence

```
TensorRT FP32 (引擎缓存命中)
  ↓ 缺失 DLL / 引擎构建失败
CUDA graph replay (enable_cuda_graph + 预分配 GPU buffer)
  ↓ CUDA EP 不可用
Plain CUDA EP (无 graph)
  ↓
CPU EP (兜底)
```

```
TensorRT FP32 (engine cache hit)
  ↓ DLL missing / engine build fails
CUDA graph replay (enable_cuda_graph + pre-allocated GPU buffers)
  ↓ CUDA EP unavailable
Plain CUDA EP (no graph)
  ↓
CPU EP (last resort)
```

## Installation (Manual)

**Please be aware that the installation requires technical skills and is not for beginners. Consider downloading the quickstart version.**

<details>
<summary>Click to see the process</summary>

### Installation

This is more likely to work on your computer but will be slower as it utilizes the CPU.

**1. Set up Your Platform**

-   Python (3.14 recommended; 3.11-3.14 supported)
-   pip
-   git
-   [ffmpeg](https://www.youtube.com/watch?v=OlNWCpFdVMA) - ```iex (irm ffmpeg.tc.ht)```
-   [Visual Studio 2022 Runtimes (Windows)](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

**2. Clone the Repository**

```bash
git clone --depth 1 https://github.com/hacksider/Deep-Live-Cam.git
cd Deep-Live-Cam
```

**3. Download the Models**

1. [gfpgan-1024.onnx](https://huggingface.co/hacksider/deep-live-cam/resolve/main/gfpgan-1024.onnx)
2. [inswapper\_128\_fp16.onnx](https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx)

Place these files in the "**models**" folder.

**4. Install Dependencies**

We highly recommend using a `venv` to avoid issues.


For Windows:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
For Linux:
```bash
# Ensure you use the installed Python 3.14
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**For macOS:**

Apple Silicon (M1 through M5) requires specific setup:

```bash
# Install Python 3.14
brew install python@3.14

# Install tkinter package (required for the GUI)
brew install python-tk@3.14

# Create and activate virtual environment with Python 3.14
python3.14 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

** In case something goes wrong and you need to reinstall the virtual environment **

```bash
# Deactivate the virtual environment
rm -rf venv

# Reinstall the virtual environment
python -m venv venv
source venv/bin/activate

# install the dependencies again
pip install -r requirements.txt

# gfpgan and basicsrs issue fix
pip install git+https://github.com/xinntao/BasicSR.git@master
pip uninstall gfpgan -y
pip install git+https://github.com/TencentARC/GFPGAN.git@master
```

**Run:** If you don't have a GPU, you can run Deep-Live-Cam using `python run.py`. Note that initial execution will download models (~300MB).

### GPU Acceleration

**CUDA Execution Provider (Nvidia)**

1. Install [CUDA Toolkit 12.8.0](https://developer.nvidia.com/cuda-12-8-0-download-archive)
2. Install [cuDNN v8.9.7 for CUDA 12.x](https://developer.nvidia.com/rdp/cudnn-archive) (required for onnxruntime-gpu):
   - Download cuDNN v8.9.7 for CUDA 12.x
   - Make sure the cuDNN bin directory is in your system PATH
3. Install dependencies:

```bash
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu==1.26.0
```

3. Usage:

```bash
python run.py --execution-provider cuda
```

**CoreML Execution Provider (Apple Silicon)**

Apple Silicon (M1 through M5) specific installation:

1. Make sure you've completed the macOS setup above using Python 3.14.
2. No extra install step is needed — `requirements.txt` pulls the official
   `onnxruntime` build, whose macOS wheels ship the CoreML execution provider.
   If you previously installed the unmaintained `onnxruntime-silicon` fork,
   remove it first, as it shadows the real package:

```bash
pip uninstall onnxruntime-silicon
pip install -r requirements.txt
```

3. Usage:

```bash
python3.14 run.py --execution-provider coreml
```

**Important Notes for macOS:**
- Python 3.11 is the minimum (onnxruntime dropped 3.10); 3.14 is recommended
- Always run with `python3.14` command not just `python` if you have multiple Python versions installed
- If you get error about `_tkinter` missing, reinstall the tkinter package: `brew reinstall python-tk@3.14`
- If you get model loading errors, check that your models are in the correct folder
- If you encounter conflicts with other Python versions, consider uninstalling them:
  ```bash
  # List all installed Python versions
  brew list | grep python

  # Uninstall conflicting versions if needed
  brew uninstall --ignore-dependencies python@3.11

  # Keep only Python 3.14
  brew cleanup
  ```

**CoreML Execution Provider (Apple Legacy)**

1. Install dependencies:

```bash
pip uninstall onnxruntime onnxruntime-coreml
pip install onnxruntime-coreml==1.21.0
```

2. Usage:

```bash
python run.py --execution-provider coreml
```

**DirectML Execution Provider (Windows)**

1. Install dependencies:

```bash
pip uninstall onnxruntime onnxruntime-directml
pip install onnxruntime-directml==1.21.0
```

2. Usage:

```bash
python run.py --execution-provider dml
```

**OpenVINO™ Execution Provider (Intel)**

1. Install dependencies:

```bash
pip uninstall onnxruntime onnxruntime-openvino
pip install onnxruntime-openvino==1.21.0
```

**Note:** `onnxruntime-openvino` newer than 1.21.0 must be installed together with `openvino`, and the two versions must correspond one-to-one. The supported pairings are:

| onnxruntime-openvino | OpenVINO |
| --- | --- |
| 1.24.1 | 2025.4.1 |
| 1.23.0 | 2025.3 |
| 1.22.0 | 2025.1 |

```bash
# Example: onnxruntime-openvino 1.24.1 pairs with OpenVINO 2025.4.1
pip install openvino==2025.4.1
pip install onnxruntime-openvino==1.24.1
```

See the [OpenVINO Execution Provider requirements](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#requirements) for the full version-mapping details.

2. Usage:

```bash
python run.py --execution-provider openvino
```
</details>

## Usage

**1. Image/Video Mode**

-   Execute `python run.py`.
-   Choose a source face image and a target image/video.
-   Click "Start".
-   The output will be saved in a directory named after the target video.

**2. Webcam Mode**

-   Execute `python run.py`.
-   Select a source face image.
-   Click "Live".
-   Wait for the preview to appear (10-30 seconds).
-   Use a screen capture tool like OBS to stream.
-   To change the face, select a new source image.

## Download all models in this huggingface link
- [**Download models here**](https://huggingface.co/hacksider/deep-live-cam/tree/main)

## Command Line Arguments (Unmaintained)

```
options:
  -h, --help                                               show this help message and exit
  -s SOURCE_PATH, --source SOURCE_PATH                     select a source image
  -t TARGET_PATH, --target TARGET_PATH                     select a target image or video
  -o OUTPUT_PATH, --output OUTPUT_PATH                     select output file or directory
  --frame-processor FRAME_PROCESSOR [FRAME_PROCESSOR ...]  frame processors (choices: face_swapper, face_enhancer, ...)
  --keep-fps                                               keep original fps
  --keep-audio                                             keep original audio
  --keep-frames                                            keep temporary frames
  --many-faces                                             process every face
  --map-faces                                              map source target faces
  --mouth-mask                                             mask the mouth region
  --video-encoder {libx264,libx265,libvpx-vp9}             adjust output video encoder
  --video-quality [0-51]                                   adjust output video quality
  --live-mirror                                            the live camera display as you see it in the front-facing camera frame
  --live-resizable                                         the live camera frame is resizable
  --max-memory MAX_MEMORY                                  maximum amount of RAM in GB
  --execution-provider {cpu} [{cpu} ...]                   available execution provider (choices: cpu, ...)
  --execution-threads EXECUTION_THREADS                    number of execution threads
  -v, --version                                            show program's version number and exit
```

Looking for a CLI mode? Using the -s/--source argument will make the run program in cli mode.

## Press

 - [**Ars Technica**](https://arstechnica.com/information-technology/2024/08/new-ai-tool-enables-real-time-face-swapping-on-webcams-raising-fraud-concerns/) - *"Deep-Live-Cam goes viral, allowing anyone to become a digital doppelganger"*
 - [**Yahoo!**](https://www.yahoo.com/tech/ok-viral-ai-live-stream-080041056.html) - *"OK, this viral AI live stream software is truly terrifying"*
 - [**CNN Brasil**](https://www.cnnbrasil.com.br/tecnologia/ia-consegue-clonar-rostos-na-webcam-entenda-funcionamento/) - *"AI can clone faces on webcam; understand how it works"*
 - [**Bloomberg Technoz**](https://www.bloombergtechnoz.com/detail-news/71032/kenalan-dengan-teknologi-deep-live-cam-bisa-jadi-alat-menipu) - *"Get to know Deep Live Cam technology, it can be used as a tool for deception."*
 - [**TrendMicro**](https://www.trendmicro.com/vinfo/gb/security/news/cyber-attacks/ai-vs-ai-deepfakes-and-ekyc) - *"AI vs AI: DeepFakes and eKYC"*
 - [**PetaPixel**](https://petapixel.com/2024/08/14/deep-live-cam-deepfake-ai-tool-lets-you-become-anyone-in-a-video-call-with-single-photo-mark-zuckerberg-jd-vance-elon-musk/) - *"Deepfake AI Tool Lets You Become Anyone in a Video Call With Single Photo"*
 - [**SomeOrdinaryGamers**](https://www.youtube.com/watch?time_continue=1074&v=py4Tc-Y8BcY) - *"That's Crazy, Oh God. That's Fucking Freaky Dude... That's So Wild Dude"*
 - [**IShowSpeed**](https://www.youtube.com/live/mFsCe7AIxq8?feature=shared&t=2686) - *"Alright look look look, now look chat, we can do any face we want to look like chat"*
 - [**TechLinked (Linus Tech Tips)**](https://www.youtube.com/watch?v=wnCghLjqv3s&t=551s) - *"They do a pretty good job matching poses, expression and even the lighting"*
 - [**IShowSpeed**](https://youtu.be/JbUPRmXRUtE?t=3964) - *"What the F***! Why do I look like Vinny Jr? I look exactly like Vinny Jr!? No, this shit is crazy! Bro This is F*** Crazy!"*


## Credits

-   [ffmpeg](https://ffmpeg.org/): for making video-related operations easy
-   [Henry](https://github.com/henryruhs): One of the major contributor in this repo
-   [deepinsight](https://github.com/deepinsight): for their [insightface](https://github.com/deepinsight/insightface) project which provided a well-made library and models. Please be reminded that the [use of the model is for non-commercial research purposes only](https://github.com/deepinsight/insightface?tab=readme-ov-file#license).
-   [havok2-htwo](https://github.com/havok2-htwo): for sharing the code for webcam
-   [GosuDRM](https://github.com/GosuDRM): for the open version of roop
-   [pereiraroland26](https://github.com/pereiraroland26): Multiple faces support
-   [vic4key](https://github.com/vic4key): For supporting/contributing to this project
-   [kier007](https://github.com/kier007): for improving the user experience
-   [qitianai](https://github.com/qitianai): for multi-lingual support
-   [laurigates](https://github.com/laurigates): Decoupling stuffs to make everything faster!
-   [maxwbuckley](https://github.com/maxwbuckley): For making the effort to optimize this for mac!
-   and [all developers](https://github.com/hacksider/Deep-Live-Cam/graphs/contributors) behind libraries used in this project.
-   Footnote: Please be informed that the base author of the code is [s0md3v](https://github.com/s0md3v/roop)
-   All the wonderful users who helped make this project go viral by starring the repo ❤️

[![Stargazers](https://reporoster.com/stars/hacksider/Deep-Live-Cam)](https://github.com/hacksider/Deep-Live-Cam/stargazers)

## Contributions

![Alt](https://repobeats.axiom.co/api/embed/fec8e29c45dfdb9c5916f3a7830e1249308d20e1.svg "Repobeats analytics image")

## Stars to the Moon 🚀

<a href="https://star-history.com/#hacksider/deep-live-cam&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=hacksider/deep-live-cam&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=hacksider/deep-live-cam&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=hacksider/deep-live-cam&type=Date" />
 </picture>
</a>
