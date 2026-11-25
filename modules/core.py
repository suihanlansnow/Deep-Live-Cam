import os
import sys
import threading
import time
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '6'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import importlib.util
import onnxruntime
try:
    import tensorflow
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    HAS_TORCH = importlib.util.find_spec("torch") is not None
except (ImportError, ValueError):
    HAS_TORCH = False

import modules.globals
import modules.metadata
import modules.ui as ui
from modules.processors.frame.core import get_frame_processors_modules, process_video_in_memory
from modules.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
if HAS_TORCH:
    warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser()
    program.add_argument('-s', '--source', help='select an source image', dest='source_path')
    program.add_argument('-t', '--target', help='select an target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    program.add_argument('--frame-processor', help='pipeline of frame processors', dest='frame_processor', default=['face_swapper'], choices=['face_swapper', 'face_enhancer', 'face_enhancer_gpen256', 'face_enhancer_gpen512'], nargs='+')
    program.add_argument('--keep-fps', help='keep original fps', dest='keep_fps', action='store_true', default=False)
    program.add_argument('--keep-audio', help='keep original audio', dest='keep_audio', action='store_true', default=True)
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true', default=False)
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true', default=False)
    program.add_argument('--nsfw-filter', help='filter the NSFW image or video', dest='nsfw_filter', action='store_true', default=False)
    program.add_argument('--map-faces', help='map source target faces', dest='map_faces', action='store_true', default=False)
    program.add_argument('--mouth-mask', help='mask the mouth region', dest='mouth_mask', action='store_true', default=False)
    program.add_argument('--video-encoder', help='adjust output video encoder', dest='video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9'])
    program.add_argument('--video-quality', help='adjust output video quality', dest='video_quality', type=int, default=18, choices=range(52), metavar='[0-51]')
    program.add_argument('-l', '--lang', help='Ui language', default="en")
    program.add_argument('--live-mirror', help='The live camera display as you see it in the front-facing camera frame', dest='live_mirror', action='store_true', default=False)
    program.add_argument('--live-resizable', help='The live camera frame is resizable', dest='live_resizable', action='store_true', default=False)
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int, default=suggest_max_memory())
    program.add_argument('--execution-provider', help='execution provider', dest='execution_provider', default=[suggest_default_execution_provider()], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=None)
    program.add_argument('--det-size', help='face detection input size (160, 320, or 640)', dest='det_size', type=int, default=modules.globals.DEFAULT_DET_SIZE, choices=[160, 320, 640])
    program.add_argument('-v', '--version', action='version', version=f'{modules.metadata.name} {modules.metadata.version}')

    # register deprecated args
    program.add_argument('-f', '--face', help=argparse.SUPPRESS, dest='source_path_deprecated')
    program.add_argument('--cpu-cores', help=argparse.SUPPRESS, dest='cpu_cores_deprecated', type=int)
    program.add_argument('--gpu-vendor', help=argparse.SUPPRESS, dest='gpu_vendor_deprecated')
    program.add_argument('--gpu-threads', help=argparse.SUPPRESS, dest='gpu_threads_deprecated', type=int)

    args = program.parse_args()

    modules.globals.source_path = args.source_path
    modules.globals.target_path = args.target_path
    modules.globals.output_path = normalize_output_path(modules.globals.source_path, modules.globals.target_path, args.output_path)
    modules.globals.frame_processors = args.frame_processor
    modules.globals.headless = args.source_path or args.target_path or args.output_path
    modules.globals.keep_fps = args.keep_fps
    modules.globals.keep_audio = args.keep_audio
    modules.globals.keep_frames = args.keep_frames
    modules.globals.many_faces = args.many_faces
    modules.globals.mouth_mask = args.mouth_mask
    modules.globals.nsfw_filter = args.nsfw_filter
    modules.globals.map_faces = args.map_faces
    modules.globals.video_encoder = args.video_encoder
    modules.globals.video_quality = args.video_quality
    modules.globals.live_mirror = args.live_mirror
    modules.globals.live_resizable = args.live_resizable
    modules.globals.max_memory = args.max_memory
    modules.globals.execution_providers = decode_execution_providers(args.execution_provider)
    modules.globals.execution_threads = args.execution_threads
    modules.globals.det_size = args.det_size
    modules.globals.lang = args.lang

    # The argparse default (None) avoids evaluating suggest_execution_threads()
    # before providers are decoded, and deprecated-arg overrides above may
    # have already set execution_threads.
    if modules.globals.execution_threads is None:
        modules.globals.execution_threads = suggest_execution_threads()

    #for ENHANCER tumblers:
    for enhancer_key in ('face_enhancer', 'face_enhancer_gpen256', 'face_enhancer_gpen512'):
        modules.globals.fp_ui[enhancer_key] = enhancer_key in args.frame_processor

    # translate deprecated args
    if args.source_path_deprecated:
        print('\033[33mArgument -f and --face are deprecated. Use -s and --source instead.\033[0m')
        modules.globals.source_path = args.source_path_deprecated
        modules.globals.output_path = normalize_output_path(args.source_path_deprecated, modules.globals.target_path, args.output_path)
    if args.cpu_cores_deprecated:
        print('\033[33mArgument --cpu-cores is deprecated. Use --execution-threads instead.\033[0m')
        modules.globals.execution_threads = args.cpu_cores_deprecated
    if args.gpu_vendor_deprecated == 'apple':
        print('\033[33mArgument --gpu-vendor apple is deprecated. Use --execution-provider coreml instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['coreml'])
    if args.gpu_vendor_deprecated == 'nvidia':
        print('\033[33mArgument --gpu-vendor nvidia is deprecated. Use --execution-provider cuda instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['cuda'])
    if args.gpu_vendor_deprecated == 'amd':
        print('\033[33mArgument --gpu-vendor amd is deprecated. Use --execution-provider cuda instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['rocm'])
    if args.gpu_threads_deprecated:
        print('\033[33mArgument --gpu-threads is deprecated. Use --execution-threads instead.\033[0m')
        modules.globals.execution_threads = args.gpu_threads_deprecated


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 4
    return 16


def suggest_default_execution_provider() -> str:
    """Pick the best available provider: cuda > rocm > coreml > openvino > dml > cpu."""
    available = encode_execution_providers(onnxruntime.get_available_providers())
    for pref in ('cuda', 'rocm', 'coreml', 'openvino', 'dml'):
        if pref in available:
            return pref
    return 'cpu'


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    """Suggest optimal thread count based on hardware and execution provider."""
    import os
    
    # Get CPU count
    cpu_count = os.cpu_count() or 4
    
    if 'DmlExecutionProvider' in modules.globals.execution_providers:
        return 1
    if 'ROCMExecutionProvider' in modules.globals.execution_providers:
        return 1
    if 'CUDAExecutionProvider' in modules.globals.execution_providers:
        return 2
    if 'OpenVINOExecutionProvider' in modules.globals.execution_providers:
        return 1
    
    # For CPU execution, use most cores but leave some for system
    return max(4, min(cpu_count - 2, 16))


def limit_resources() -> None:
    # prevent tensorflow memory leak
    if HAS_TENSORFLOW:
        gpus = tensorflow.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tensorflow.config.experimental.set_memory_growth(gpu, True)
    # limit memory usage
    if modules.globals.max_memory:
        # setrlimit(RLIMIT_DATA) fails with EINVAL on macOS, crashing on launch.
        # See https://github.com/hacksider/Deep-Live-Cam/issues/1848
        if platform.system().lower() == 'darwin':
            return
        memory = modules.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # The working-set minimum is what Windows tries to keep resident.
            # Setting it to the limit itself asked the OS to pin max_memory
            # gigabytes — the opposite of limiting.  Only the maximum acts as
            # a cap; keep the minimum small so idle pages can still be trimmed.
            min_memory = min(memory, 512 * 1024 ** 2)
            kernel32.SetProcessWorkingSetSize(
                -1, ctypes.c_size_t(min_memory), ctypes.c_size_t(memory),
            )
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def release_resources() -> None:
    if 'CUDAExecutionProvider' in modules.globals.execution_providers and HAS_TORCH:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg is not installed.')
        return False
    return True


def update_status(message: str, scope: str = 'DLC.CORE') -> None:
    print(f'[{scope}] {message}')
    if not modules.globals.headless:
        ui.update_status(message)

def process_video_map_in_memory(source_path: str, target_path: str, fps: float) -> bool:
    """map_faces video pipeline: decode -> swap -> stream straight to FFmpeg.

    Decodes input frames directly with cv2 (pixel-identical to the legacy
    PNG round-trip) and swaps each frame via face_swapper.process_frame_v2
    keyed by frame index.  Processed frames are written to the FFmpeg encoder's
    stdin in strict frame order, so the whole run touches zero PNGs (the old
    path wrote one PNG per output frame and read them all back to encode).

    The encoder also muxes the source audio in the same pass with an optional
    stream map (`-map 1:a:0?`), so a source without audio no longer errors and
    the separate restore_audio copy is skipped.

    A worker pool overlaps the swap/GPU work with the blocking stdin writes,
    and ordered flushing keeps frame order.  A sliding window bounds how many
    decoded frames are held pending, so memory stays flat regardless of video
    length.  Falls back to the disk pipeline in start() on any failure.
    """
    import subprocess
    import time
    import cv2
    import numpy as np
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor
    from modules.utilities import (
        create_temp, get_temp_output_path, resolve_video_encoder, StageProfiler,
    )
    from modules.processors.frame import face_swapper

    create_temp(target_path)
    temp_output_path = get_temp_output_path(target_path)

    cap = cv2.VideoCapture(target_path)
    if not cap.isOpened():
        return False
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if width <= 0 or height <= 0:
        cap.release()
        return False

    num_threads = max(1, int(getattr(modules.globals, 'execution_threads', 1)))
    prof = StageProfiler('video-map')
    # Bound in-flight frames so a long video never materializes its whole
    # length in memory; 2 frames per worker is enough to keep the GPU fed.
    max_pending = num_threads * 2
    encoder, encoder_options = resolve_video_encoder()
    keep_audio = bool(getattr(modules.globals, 'keep_audio', False))

    # --- Writer: encode raw BGR24 from stdin (and mux source audio in-pass) ---
    writer_cmd = [
        'ffmpeg', '-hide_banner',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}',
        '-r', str(fps),
        '-i', '-',
    ]
    if keep_audio:
        writer_cmd.extend(['-i', target_path])
    writer_cmd.extend(['-c:v', encoder])
    writer_cmd.extend(encoder_options)
    if keep_audio:
        # Optional map: silently produces a video-only file when the source
        # has no audio stream instead of failing the whole run.
        writer_cmd.extend(['-map', '0:v:0', '-map', '1:a:0?', '-c:a', 'copy'])
    writer_cmd.extend([
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1',
        '-v', 'error',
        '-y', temp_output_path,
    ])

    writer = None
    try:
        writer = subprocess.Popen(
            writer_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except Exception as e:
        print(f"[DLC.CORE] Failed to start FFmpeg encoder: {e}")
        cap.release()
        return False

    def _process_one(frame_index, frame):
        process_start = time.perf_counter() if prof.enabled else 0.0
        try:
            processed = face_swapper.process_frame_v2(frame, frame_index=frame_index)
        except Exception as e:
            print(f"{face_swapper.NAME}: Error processing frame {frame_index}: {e}")
            processed = frame
        if processed is None:
            processed = frame
        if prof.enabled:
            prof.add('process', time.perf_counter() - process_start)
        return processed

    written = 0
    broken_pipe = False

    def _write_frame(processed):
        nonlocal written, broken_pipe
        if processed.shape[0] != height or processed.shape[1] != width:
            processed = cv2.resize(processed, (width, height))
        write_start = time.perf_counter() if prof.enabled else 0.0
        try:
            # .data hands FFmpeg the frame buffer directly; .tobytes() copied
            # the whole frame (~6 MB at 1080p) on every write for no reason.
            buffer = np.ascontiguousarray(processed, dtype=np.uint8)
            writer.stdin.write(buffer.data)
        except (BrokenPipeError, OSError):
            broken_pipe = True
            return
        if prof.enabled:
            prof.add('write_wait', time.perf_counter() - write_start)
        written += 1

    try:
        with ThreadPoolExecutor(max_workers=num_threads) as executor, \
             tqdm(total=total if total > 0 else None, desc='Processing', unit='frame',
                  dynamic_ncols=True) as progress:
            progress.set_postfix({
                'mode': 'in-memory(map,stream)',
                'threads': num_threads,
                'execution_providers': modules.globals.execution_providers,
            })
            by_index = {}
            next_write = 0
            frame_index = 0
            while not broken_pipe:
                ok, frame = cap.read()
                if not ok:
                    break
                # No copy: cv2.VideoCapture.read() allocates a fresh buffer on
                # every call, so the next read cannot alias this frame.
                by_index[frame_index] = executor.submit(_process_one, frame_index, frame)
                frame_index += 1
                # Flush in order once the window is full; waiting on
                # next_write (not as_completed) keeps output frame order.
                while len(by_index) > max_pending and not broken_pipe:
                    _write_frame(by_index.pop(next_write).result())
                    next_write += 1
                    progress.update(1)
                    prof.tick()

            # Drain the remaining in-flight frames in order.
            while by_index and not broken_pipe:
                _write_frame(by_index.pop(next_write).result())
                next_write += 1
                progress.update(1)
                prof.tick()
    finally:
        cap.release()

    try:
        writer.stdin.close()
        stderr_output = writer.stderr.read()
        returncode = writer.wait()
    except Exception as e:
        print(f"[DLC.CORE] Failed to finalize encoder: {e}")
        try:
            writer.kill()
        except Exception:
            pass
        return False

    if broken_pipe or returncode != 0 or written == 0:
        message = stderr_output.decode(errors='ignore').strip()
        print(f"[DLC.CORE] Streaming encode failed (rc={returncode}, frames={written})"
              + (f": {message}" if message else ''))
        return False
    return os.path.isfile(temp_output_path)


def start() -> None:
    """Start processing with performance monitoring."""
    import time
    
    start_time = time.time()
    
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        if not frame_processor.pre_start():
            return
    update_status('Processing...')
    
    # process image to image
    if has_image_extension(modules.globals.target_path):
        if modules.globals.nsfw_filter and ui.check_and_ignore_nsfw(modules.globals.target_path, destroy):
            return
        try:
            shutil.copy2(modules.globals.target_path, modules.globals.output_path)
        except Exception as e:
            print("Error copying file:", str(e))
        for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_image(modules.globals.source_path, modules.globals.output_path, modules.globals.output_path)
            release_resources()
        if is_image(modules.globals.target_path):
            elapsed = time.time() - start_time
            update_status(f'Processing to image succeed! (Time: {elapsed:.2f}s)')
        else:
            update_status('Processing to image failed!')
        return
    
    # process image to videos
    if modules.globals.nsfw_filter and ui.check_and_ignore_nsfw(modules.globals.target_path, destroy):
        return

    # Detect FPS early (needed by both pipelines)
    if modules.globals.keep_fps:
        update_status('Detecting fps...')
        fps = detect_fps(modules.globals.target_path)
    else:
        fps = 30.0

    video_created = False
    # The streaming map pipeline muxes the source audio in the same encode
    # pass, so start() must not run a second full-file copy for it.
    audio_muxed_in_stream = False

    # --- In-memory pipeline ---
    # map_faces: decode input frames directly (no PNG extraction), swap keyed
    # by frame index, write processed PNGs, encode via create_video (same
    # encode as the disk path => bit-comparable output).
    # non-map: reads frames from FFmpeg pipe, processes in memory, encodes
    # directly (eliminates all per-frame PNG disk I/O).
    if modules.globals.map_faces:
        update_status(f'Processing video in-memory (map_faces) at {fps} fps...')
        processing_start = time.time()
        video_created = process_video_map_in_memory(
            modules.globals.source_path,
            modules.globals.target_path,
            fps,
        )
        processing_time = time.time() - processing_start
        release_resources()

        if video_created:
            audio_muxed_in_stream = bool(modules.globals.keep_audio)
            update_status(f'In-memory map processing + encoding completed in {processing_time:.2f}s')
    else:
        update_status(f'Processing video in-memory at {fps} fps...')
        create_temp(modules.globals.target_path)

        processing_start = time.time()
        video_created = process_video_in_memory(
            modules.globals.source_path,
            modules.globals.target_path,
            fps,
        )
        processing_time = time.time() - processing_start
        release_resources()

        if video_created:
            # process_video_in_memory muxes the source audio in the same
            # encode pass, so start() must not run restore_audio() on top.
            audio_muxed_in_stream = bool(modules.globals.keep_audio)
            update_status(f'In-memory processing + encoding completed in {processing_time:.2f}s')

    # --- Disk-based fallback (required for map_faces, or if pipe failed) ---
    if not video_created:
        if not modules.globals.map_faces:
            update_status('Falling back to disk-based processing...')

        extraction_start = time.time()
        create_temp(modules.globals.target_path)
        update_status('Extracting frames...')
        if not extract_frames(modules.globals.target_path):
            update_status('Extracting frames failed!')
            clean_temp(modules.globals.target_path)
            return
        extraction_time = time.time() - extraction_start

        temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)
        total_frames = len(temp_frame_paths)
        update_status(f'Processing {total_frames} frames with {modules.globals.execution_threads} threads...')

        processing_start = time.time()
        for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_video(modules.globals.source_path, temp_frame_paths)
            release_resources()
        processing_time = time.time() - processing_start
        fps_processing = total_frames / processing_time if processing_time > 0 else 0
        update_status(f'Frame processing completed in {processing_time:.2f}s ({fps_processing:.2f} fps)')

        encoding_start = time.time()
        update_status(f'Creating video with {fps} fps...')
        video_created = create_video(modules.globals.target_path, fps)
        encoding_time = time.time() - encoding_start
        if video_created:
            update_status(f'Video encoding completed in {encoding_time:.2f}s')

    if not video_created:
        update_status('Video encoding failed. No temporary output video was created.')
        clean_temp(modules.globals.target_path)
        return
    
    # handle audio
    if modules.globals.keep_audio:
        if audio_muxed_in_stream:
            update_status('Audio already muxed during encoding.')
            move_temp(modules.globals.target_path, modules.globals.output_path)
        elif modules.globals.keep_fps:
            update_status('Restoring audio...')
            restore_audio(modules.globals.target_path, modules.globals.output_path)
        else:
            update_status('Restoring audio might cause issues as fps are not kept...')
            restore_audio(modules.globals.target_path, modules.globals.output_path)
    else:
        move_temp(modules.globals.target_path, modules.globals.output_path)
    
    # clean and validate
    clean_temp(modules.globals.target_path)
    
    total_time = time.time() - start_time
    if is_video(modules.globals.target_path) and modules.globals.output_path and os.path.isfile(modules.globals.output_path):
        update_status(f'Video processing succeeded! Total time: {total_time:.2f}s')
    else:
        update_status('Processing to video failed!')


def destroy(to_quit=True) -> None:
    if modules.globals.target_path:
        clean_temp(modules.globals.target_path)
    if to_quit:
        quit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        if not frame_processor.pre_check():
            return
    limit_resources()
    if not modules.globals.headless and os.environ.get("DLC_NO_WARMUP") != "1":
        # Warm the GPU engines in the background now that the execution
        # providers are resolved: first Live otherwise stalls on TensorRT
        # engine construction (~60 s cold cache) while the UI is already up.
        def _warm_up() -> None:
            time.sleep(2.0)  # let the UI paint first; warm-up competes for CPU/GPU
            try:
                from modules.face_analyser import get_face_analyser, _get_roi_det_model
                from modules.processors.frame.face_swapper import get_face_swapper

                print("[startup] Warming up models in background...", flush=True)
                get_face_analyser()      # buffalo_l incl. TRT detection engine
                _get_roi_det_model()     # small-shape ROI detector
                get_face_swapper()       # inswapper incl. TRT engine
                print("[startup] Model warm-up complete.", flush=True)
            except Exception as e:
                print(f"[startup] Model warm-up skipped: {e}", flush=True)

        threading.Thread(target=_warm_up, name="DLC.WARMUP", daemon=True).start()
    if modules.globals.headless:
        start()
    else:
        window = ui.init(start, destroy, modules.globals.lang)
        window.mainloop()