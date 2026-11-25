import os
import queue
import subprocess
import sys
import threading
import time
import importlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import ModuleType
from typing import Any, List, Callable

import numpy as np
from tqdm import tqdm

import modules
import modules.globals
from modules.face_analyser import get_one_face

FRAME_PROCESSORS_MODULES: List[ModuleType] = []
FRAME_PROCESSORS_INTERFACE = [
    'pre_check',
    'pre_start',
    'process_frame',
    'process_image',
    'process_video'
]

ALLOWED_PROCESSORS = {
    'face_swapper',
    'face_enhancer',
    'face_enhancer_gpen256',
    'face_enhancer_gpen512'
}

def load_frame_processor_module(frame_processor: str) -> Any:
    if frame_processor not in ALLOWED_PROCESSORS:
        print(f"Frame processor {frame_processor} is not allowed")
        sys.exit()
    try:
        frame_processor_module = importlib.import_module(f'modules.processors.frame.{frame_processor}')
        for method_name in FRAME_PROCESSORS_INTERFACE:
            if not hasattr(frame_processor_module, method_name):
                print(f"Frame processor {frame_processor} is missing required method {method_name}")
                sys.exit()
    except ImportError:
        print(f"Frame processor {frame_processor} not found")
        sys.exit()
    return frame_processor_module


def get_frame_processors_modules(frame_processors: List[str]) -> List[ModuleType]:
    global FRAME_PROCESSORS_MODULES

    if not FRAME_PROCESSORS_MODULES:
        for frame_processor in frame_processors:
            frame_processor_module = load_frame_processor_module(frame_processor)
            FRAME_PROCESSORS_MODULES.append(frame_processor_module)
    set_frame_processors_modules_from_ui(frame_processors)
    return FRAME_PROCESSORS_MODULES

def set_frame_processors_modules_from_ui(frame_processors: List[str]) -> None:
    global FRAME_PROCESSORS_MODULES
    current_processor_names = [proc.__name__.split('.')[-1] for proc in FRAME_PROCESSORS_MODULES]

    for frame_processor, state in modules.globals.fp_ui.items():
        if state and frame_processor not in current_processor_names:
            try:
                frame_processor_module = load_frame_processor_module(frame_processor)
                FRAME_PROCESSORS_MODULES.append(frame_processor_module)
                if frame_processor not in modules.globals.frame_processors:
                     modules.globals.frame_processors.append(frame_processor)
            except SystemExit:
                 print(f"Warning: Failed to load frame processor {frame_processor} requested by UI state.")
            except Exception as e:
                 print(f"Warning: Error loading frame processor {frame_processor} requested by UI state: {e}")

        elif not state and frame_processor in current_processor_names:
            try:
                module_to_remove = next((mod for mod in FRAME_PROCESSORS_MODULES if mod.__name__.endswith(f'.{frame_processor}')), None)
                if module_to_remove:
                    FRAME_PROCESSORS_MODULES.remove(module_to_remove)
                if frame_processor in modules.globals.frame_processors:
                    modules.globals.frame_processors.remove(frame_processor)
            except Exception as e:
                 print(f"Warning: Error removing frame processor {frame_processor}: {e}")

def multi_process_frame(source_path: str, temp_frame_paths: List[str], process_frames: Callable[[str, List[str], Any], None], progress: Any = None) -> None:
    """Process frames in parallel without batch barriers."""
    max_workers = modules.globals.execution_threads

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_frames, source_path, [path], progress)
            for path in temp_frame_paths
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing frame: {e}")


def process_video(source_path: str, frame_paths: list[str], process_frames: Callable[[str, List[str], Any], None]) -> None:
    progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    total = len(frame_paths)
    with tqdm(total=total, desc='Processing', unit='frame', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
        progress.set_postfix({'execution_providers': modules.globals.execution_providers, 'execution_threads': modules.globals.execution_threads, 'max_memory': modules.globals.max_memory})
        multi_process_frame(source_path, frame_paths, process_frames, progress)


def _detect_target_face(frame: np.ndarray, detector: Any = None) -> Any:
    """Detection-only target lookup for the pipelined in-memory video path.

    The swap model needs only a bbox and 5 keypoints from the target face, so
    running the full analysis (recognition + 2d106 landmarks) on every frame
    was pure overhead.  Recognition only matters for map_faces embedding
    matching, and landmarks are still computed when mouth masking needs them.

    ``detector`` optionally carries per-pipeline ROI tracking state; the
    single-threaded detect executor owns it, so the tracker sees frames in
    order across the pipeline.
    """
    from modules.face_analyser import detect_one_face_fast, ensure_landmarks

    face = detector.detect(frame) if detector is not None else detect_one_face_fast(frame)
    if face is not None and getattr(modules.globals, 'mouth_mask', False):
        ensure_landmarks(frame, face)
    return face


def process_video_in_memory(source_path: str, target_path: str, fps: float) -> bool:
    """Process video frames in-memory using FFmpeg pipes, eliminating disk I/O.

    Reads raw frames from the source video via an FFmpeg decoder pipe, runs each
    frame through all active frame processors sequentially, and writes the
    result directly to an FFmpeg encoder pipe.  This avoids extracting frames to
    PNG on disk, which is the biggest I/O bottleneck in the disk-based pipeline.

    Returns True on success, False on failure (caller should fall back to the
    disk-based pipeline).
    """
    from modules import imread_unicode
    from modules.face_analyser import get_one_face
    from modules.utilities import (
        _ffmpeg_has_encoder,
        get_video_dimensions,
        estimate_frame_count,
        get_temp_output_path,
    )

    temp_output_path = get_temp_output_path(target_path)

    # --- Pre-load source face (needed by face_swapper in simple mode) ---
    source_face = None
    if source_path and os.path.exists(source_path):
        source_img = imread_unicode(source_path)
        if source_img is not None:
            source_face = get_one_face(source_img)
            del source_img
        if source_face is None:
            print("[DLC.CORE] Warning: No face detected in source image. "
                  "Face swapping will be skipped.")

    # --- Collect frame processors & reset per-video state ---
    frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
    for fp in frame_processors:
        if hasattr(fp, 'PREVIOUS_FRAME_RESULT'):
            fp.PREVIOUS_FRAME_RESULT = None

    # --- Video metadata ---
    try:
        width, height = get_video_dimensions(target_path)
    except Exception as e:
        print(f"[DLC.CORE] Failed to get video dimensions: {e}")
        return False

    total_frames = estimate_frame_count(target_path, fps)
    frame_size = width * height * 3

    # --- Build encoder arguments ---
    encoder = modules.globals.video_encoder
    encoder_options: List[str] = []
    is_hw_encoder = False

    if 'CUDAExecutionProvider' in modules.globals.execution_providers:
        if encoder == 'libx264':
            encoder = 'h264_nvenc'
            is_hw_encoder = True
            encoder_options = [
                '-preset', 'p4', '-tune', 'hq', '-rc', 'vbr',
                '-cq', str(modules.globals.video_quality), '-b:v', '0',
            ]
        elif encoder == 'libx265':
            encoder = 'hevc_nvenc'
            is_hw_encoder = True
            encoder_options = [
                '-preset', 'p4', '-tune', 'hq', '-rc', 'vbr',
                '-cq', str(modules.globals.video_quality), '-b:v', '0',
            ]
    elif 'DmlExecutionProvider' in modules.globals.execution_providers:
        if encoder == 'libx264':
            encoder = 'h264_amf'
            is_hw_encoder = True
            encoder_options = [
                '-quality', 'quality', '-rc', 'vbr_latency',
                '-qp_i', str(modules.globals.video_quality),
                '-qp_p', str(modules.globals.video_quality),
            ]
        elif encoder == 'libx265':
            encoder = 'hevc_amf'
            is_hw_encoder = True
            encoder_options = [
                '-quality', 'quality', '-rc', 'vbr_latency',
                '-qp_i', str(modules.globals.video_quality),
                '-qp_p', str(modules.globals.video_quality),
            ]

    # Skip a doomed hardware attempt: if this ffmpeg build lacks the encoder,
    # _run_pipe_pipeline would process the whole video before failing and the
    # retry below would run the entire pass a second time.  Probe up front.
    if is_hw_encoder and not _ffmpeg_has_encoder(encoder):
        print(f"[DLC.CORE] {encoder} is not available in this ffmpeg build; "
              f"using software encoding.")
        encoder = modules.globals.video_encoder
        is_hw_encoder = False

    if not is_hw_encoder:
        if encoder == 'libx264':
            encoder_options = [
                '-preset', 'medium',
                '-crf', str(modules.globals.video_quality),
                '-tune', 'film',
            ]
        elif encoder == 'libx265':
            encoder_options = [
                '-preset', 'medium',
                '-crf', str(modules.globals.video_quality),
                '-x265-params', 'log-level=error',
            ]
        elif encoder == 'libvpx-vp9':
            encoder_options = [
                '-crf', str(modules.globals.video_quality),
                '-b:v', '0', '-cpu-used', '2',
            ]

    # --- Attempt pipeline (hw encoder first, then sw fallback) ---
    encoders_to_try = [(encoder, encoder_options)]
    if is_hw_encoder:
        # Software fallback
        sw_encoder = 'libx264'
        sw_options = [
            '-preset', 'medium',
            '-crf', str(modules.globals.video_quality),
            '-tune', 'film',
        ]
        encoders_to_try.append((sw_encoder, sw_options))

    for attempt, (enc, enc_opts) in enumerate(encoders_to_try):
        # Reset interpolation state on retry
        if attempt > 0:
            for fp in frame_processors:
                if hasattr(fp, 'PREVIOUS_FRAME_RESULT'):
                    fp.PREVIOUS_FRAME_RESULT = None

        success = _run_pipe_pipeline(
            target_path, temp_output_path, fps,
            source_face, frame_processors,
            width, height, frame_size, total_frames,
            enc, enc_opts,
        )
        if success:
            return True

        if attempt == 0 and is_hw_encoder:
            print(f"[DLC.CORE] Hardware encoder '{enc}' failed, "
                  f"retrying with software encoder...")

    return False


def _run_pipe_pipeline(
    target_path: str,
    temp_output_path: str,
    fps: float,
    source_face: Any,
    frame_processors: List[Any],
    width: int,
    height: int,
    frame_size: int,
    total_frames: int,
    encoder: str,
    encoder_options: List[str],
) -> bool:
    """Run the FFmpeg-pipe read → process → encode pipeline once."""

    # --- Reader: decode source video to raw BGR24 on stdout ---
    reader_cmd = [
        'ffmpeg', '-hide_banner',
        '-hwaccel', 'auto',
        '-i', target_path,
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-v', 'error',
        '-',
    ]

    # --- Writer: encode raw BGR24 from stdin ---
    # When audio is kept, mux the source's audio stream in this same pass (an
    # optional map, so a source without audio still produces a video-only
    # file).  The alternative was a second full-file remux pass in
    # restore_audio() that copied every frame of the completed video again.
    keep_audio = bool(getattr(modules.globals, 'keep_audio', False))
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
        writer_cmd.extend(['-map', '0:v:0', '-map', '1:a:0?', '-c:a', 'copy'])
    writer_cmd.extend([
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1',
        '-v', 'error',
        '-y', temp_output_path,
    ])

    reader = None
    writer = None
    try:
        reader = subprocess.Popen(
            reader_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        writer = subprocess.Popen(
            writer_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except Exception as e:
        print(f"[DLC.CORE] Failed to start FFmpeg pipes: {e}")
        for proc in (reader, writer):
            if proc:
                try:
                    proc.kill()
                except Exception:
                    pass
        return False

    processed_count = 0
    detect_executor = None
    write_error: List[str] = []

    # Decouple the blocking FFmpeg stdin write from the GPU pipeline.  Before
    # this, writer.stdin.write() ran inline: whenever the encoder's pipe was
    # full it blocked the loop, so the GPU sat idle for the duration of the
    # write.  A two-deep hand-off queue plus a three-buffer pool lets the
    # encoder drain in the background while the next frame is processed.
    # Frames stay zero-copy: a buffer only returns to the pool after FFmpeg
    # has fully consumed it.
    write_queue: queue.Queue = queue.Queue(maxsize=2)
    free_buffers: queue.Queue = queue.Queue()
    # Four buffers: one being filled by the reader thread, up to two in the
    # read queue, and the rest recycling through the writer.
    for _ in range(4):
        free_buffers.put(bytearray(frame_size))

    def _writer_loop() -> None:
        while True:
            item = write_queue.get()
            if item is None:
                return
            frame, buf = item
            try:
                writer.stdin.write(frame.data)
            except (BrokenPipeError, OSError) as exc:
                try:
                    err = writer.stderr.read().decode(errors='ignore').strip()
                except Exception:
                    err = ''
                write_error.append(f"{exc} {err}".strip())
                return
            finally:
                free_buffers.put(buf)

    from modules.utilities import StageProfiler

    prof = StageProfiler('video')
    writer_thread = threading.Thread(
        target=_writer_loop, name='DLC.VIDEO-WRITER', daemon=True
    )
    writer_thread.start()

    # Decode-ahead thread.  A frame is 6 MB at 1080p but the OS pipe buffer is
    # a few dozen KB, so ffmpeg cannot run ahead of us: reading inline made
    # every frame wait for a full decode (8 ms at 1080p, 43 ms at 4K) with the
    # GPU idle.  Buffers cycle free_buffers -> reader -> frame_queue -> main
    # loop -> write_queue -> writer -> free_buffers, still zero-copy.
    frame_queue: queue.Queue = queue.Queue(maxsize=2)
    read_error: List[str] = []
    reader_stop = threading.Event()

    def _reader_loop() -> None:
        try:
            while not reader_stop.is_set():
                buf = free_buffers.get()
                if reader_stop.is_set():
                    free_buffers.put(buf)
                    return
                read_mv = memoryview(buf)
                total_read = 0
                try:
                    while total_read < frame_size:
                        chunk = reader.stdout.readinto(read_mv[total_read:])
                        if not chunk:
                            break
                        total_read += chunk
                finally:
                    read_mv.release()
                if total_read != frame_size:
                    free_buffers.put(buf)
                    return
                while not reader_stop.is_set():
                    try:
                        frame_queue.put((buf, total_read), timeout=0.2)
                        break
                    except queue.Full:
                        continue
        except Exception as exc:
            read_error.append(str(exc))
        finally:
            try:
                frame_queue.put(None, timeout=1.0)
            except queue.Full:
                pass

    reader_thread = threading.Thread(
        target=_reader_loop, name='DLC.VIDEO-READER', daemon=True
    )
    reader_thread.start()

    bar_fmt = ('{l_bar}{bar}| {n_fmt}/{total_fmt} '
               '[{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    try:
        with tqdm(total=total_frames if total_frames > 0 else None,
                  desc='Processing', unit='frame',
                  dynamic_ncols=True, bar_format=bar_fmt) as progress:
            progress.set_postfix({
                'execution_providers': modules.globals.execution_providers,
                'threads': modules.globals.execution_threads,
                'mode': 'in-memory',
            })

            # Pipelined detection: while processing frame N (swap on
            # ANE), start detecting the face in the next frame
            # (detection on GPU).  They use different hardware units
            # so the work overlaps.
            detect_executor = ThreadPoolExecutor(max_workers=1)
            pending_detect = None
            use_pipeline = not modules.globals.many_faces
            # Per-pipeline ROI tracker: the detect executor is single-threaded
            # and receives frames in order, so the tracker is safe here.
            roi_detector = None
            if use_pipeline:
                from modules.face_analyser import ROIDetector
                roi_detector = ROIDetector()

            while True:
                read_start = time.perf_counter() if prof.enabled else 0.0
                item = frame_queue.get()
                if item is None:
                    break
                read_buf, _ = item
                if prof.enabled:
                    prof.add('read_wait', time.perf_counter() - read_start)
                frame = np.frombuffer(read_buf, dtype=np.uint8).reshape(
                    (height, width, 3)
                )

                # Get the detection result for THIS frame
                detect_start = time.perf_counter() if prof.enabled else 0.0
                if use_pipeline:
                    if pending_detect is not None:
                        target_face = pending_detect.result()
                    else:
                        target_face = _detect_target_face(frame, roi_detector)
                    # Start detecting on THIS frame eagerly — the result
                    # will be used for the next iteration.  At video
                    # frame rates the face barely moves between frames.
                    # Hand the detector its own copy: the frame processors
                    # below mutate `frame` in place (paste-back), which
                    # would otherwise race with detection.
                    pending_detect = detect_executor.submit(
                        _detect_target_face, frame.copy(), roi_detector)
                else:
                    target_face = None
                if prof.enabled:
                    prof.add('detect', time.perf_counter() - detect_start)

                # Run frame through every active processor
                process_start = time.perf_counter() if prof.enabled else 0.0
                for fp in frame_processors:
                    try:
                        frame = fp.process_frame(source_face, frame, target_face=target_face)
                    except TypeError as e:
                        if "unexpected keyword argument" not in str(e):
                            raise
                        frame = fp.process_frame(source_face, frame)
                if prof.enabled:
                    prof.add('process', time.perf_counter() - process_start)

                if not frame.flags['C_CONTIGUOUS']:
                    frame = np.ascontiguousarray(frame)
                # Hand the frame to the writer thread: the encoder pipe never
                # blocks this loop, and the pooled buffer returns only after
                # FFmpeg has consumed it.
                write_start = time.perf_counter() if prof.enabled else 0.0
                # Blocking hand-off, but never wait forever if the encoder
                # thread died on a broken pipe — surface the error instead.
                while True:
                    try:
                        write_queue.put((frame, read_buf), timeout=0.5)
                        break
                    except queue.Full:
                        if not writer_thread.is_alive():
                            free_buffers.put(read_buf)
                            detail = write_error[0] if write_error else 'encoder thread exited'
                            raise BrokenPipeError(detail)
                if prof.enabled:
                    prof.add('write_wait', time.perf_counter() - write_start)
                processed_count += 1
                progress.update(1)
                prof.tick()

            if writer_thread.is_alive():
                try:
                    write_queue.put(None, timeout=5)
                except queue.Full:
                    pass
            writer_thread.join(timeout=10)

        if write_error:
            raise BrokenPipeError(write_error[0])

        # Graceful shutdown
        writer.stdin.close()
        writer.wait()
        reader.wait()
        reader_thread.join(timeout=5)
        if read_error:
            print(f"[DLC.CORE] Video reader error: {read_error[0]}")

        if writer.returncode != 0:
            stderr_out = writer.stderr.read().decode(errors='ignore').strip()
            if stderr_out:
                print(f"[DLC.CORE] FFmpeg encoder error: {stderr_out}")
            return False

        return processed_count > 0 and os.path.isfile(temp_output_path)

    except BrokenPipeError as e:
        print(f"[DLC.CORE] FFmpeg pipe broken (encoder may not be available): {e}")
        return False
    except Exception as e:
        print(f"[DLC.CORE] In-memory processing error: {e}")
        return False
    finally:
        # If the loop exited abnormally the writer thread is still waiting on
        # the queue — release it before killing the FFmpeg processes it feeds.
        reader_stop.set()
        if writer_thread is not None and writer_thread.is_alive():
            try:
                write_queue.put(None, timeout=1)
            except Exception:
                pass
            writer_thread.join(timeout=5)
        if detect_executor is not None:
            detect_executor.shutdown(wait=True)
        for proc in (reader, writer):
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass
        for proc in (reader, writer):
            if proc is not None:
                for pipe in (proc.stdin, proc.stdout, proc.stderr):
                    if pipe is not None:
                        try:
                            pipe.close()
                        except Exception:
                            pass
        # Killing the reader process above releases a blocked readinto().
        if reader_thread is not None:
            reader_thread.join(timeout=5)
