import os
import shutil
from typing import Any
import insightface
import threading
import cv2
import numpy as np

import modules.globals
from modules import imread_unicode, imwrite_unicode
from tqdm import tqdm
from modules.typing import Frame
from modules.cluster_analysis import find_cluster_centroids, find_closest_centroid
from modules.utilities import get_temp_directory_path, create_temp, extract_frames, clean_temp, get_temp_frame_paths
from pathlib import Path

FACE_ANALYSER = None
FACE_ANALYSER_LOCK = threading.Lock()


def _current_det_size() -> tuple:
    s = int(getattr(modules.globals, 'det_size', modules.globals.DEFAULT_DET_SIZE))
    return (s, s)


def reset_face_analyser() -> None:
    """Force re-init on next get_face_analyser() call.

    Called when det_size changes — insightface FaceAnalysis fixes its
    detection input shape at prepare() time, so a new size requires a
    fresh prepare() call.
    """
    global FACE_ANALYSER
    with FACE_ANALYSER_LOCK:
        FACE_ANALYSER = None


def get_face_analyser() -> Any:
    """Get face analyser with thread-safe initialization."""
    global FACE_ANALYSER

    # Snapshot the global so a concurrent reset_face_analyser() (e.g. a
    # det_size change from the UI thread mid-Live) can't null it between our
    # check and the return.
    analyser = FACE_ANALYSER
    if analyser is None:
        with FACE_ANALYSER_LOCK:
            # Double-check after acquiring lock
            if FACE_ANALYSER is None:
                from modules.processors.frame._onnx_enhancer import (
                    build_provider_config,
                )
                from modules.model_downloader import ensure_insightface_pack

                ensure_insightface_pack('buffalo_l')
                providers = build_provider_config()
                analyser = insightface.app.FaceAnalysis(
                    name='buffalo_l',
                    providers=providers,
                    allowed_modules=['detection', 'recognition', 'landmark_2d_106']
                )
                det_size = _current_det_size()
                analyser.prepare(ctx_id=0, det_size=det_size)
                _optimize_det_model(analyser, providers, det_size)
                # Publish only after prepare()/optimize() complete, so a
                # lock-free reader can never observe a half-built analyser
                # (insightface fixes det shape at prepare() time).
                FACE_ANALYSER = analyser
            else:
                analyser = FACE_ANALYSER
    return analyser


def _optimize_det_model(fa: Any, providers, det_size: tuple) -> None:
    """Replace the detection model's ONNX session with a faster one.

    Apple Silicon: CoreML-optimized model folding dynamic Shape→Gather chains
    into constants (the input size is fixed at det_size), eliminating CPU↔ANE
    partition boundaries in the RetinaFace FPN upsampling path.  21ms → 4ms on
    M3 Max.

    Windows/Linux: fixed-shape TensorRT engine.  The detector letterboxes
    every input to exactly det_size, so the dynamic-shape ONNX can be pinned
    to 1x3x{d}x{d} — TRT builds a static engine, roughly halving CUDA time.
    """
    det_model = fa.det_model
    model_path = getattr(det_model, 'model_file', None)
    if model_path is None or not os.path.exists(model_path):
        return

    from modules.onnx_optimize import optimize_for_coreml, IS_APPLE_SILICON

    if not IS_APPLE_SILICON:
        if _is_dml():
            return
        session = _build_det_trt_session(model_path, det_model.input_name, det_size[0])
        if session is not None:
            det_model.session = session
            print(f"[DLC.FACE-ANALYSER] Detector running on TensorRT "
                  f"(static {det_size[0]}x{det_size[0]}).")
        return

    ds = det_size
    input_shape = (1, 3, ds[1], ds[0])
    optimized_path = optimize_for_coreml(model_path, input_shape=input_shape)
    if optimized_path == model_path:
        return

    import onnxruntime
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    # Route detection to GPU shader cores (CPUAndGPU) instead of ANE.
    # This lets detection run concurrently with the swap model on the
    # ANE, overlapping the two inference calls.  Detection is fast
    # enough on GPU (~4ms) and this frees ANE for the heavier swap.
    det_providers = []
    for p in providers:
        name = p[0] if isinstance(p, tuple) else p
        if name == "CoreMLExecutionProvider":
            det_providers.append((
                "CoreMLExecutionProvider",
                {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU"},
            ))
        else:
            det_providers.append(p)

    det_model.session = onnxruntime.InferenceSession(
        optimized_path, sess_options=session_options, providers=det_providers,
    )


def _needs_landmark() -> bool:
    """Check whether any active feature requires 106-point landmarks.

    Landmarks are needed by face enhancers and mouth masking, but not
    by the face swapper alone.
    """
    if getattr(modules.globals, "mouth_mask", False):
        return True
    processors = getattr(modules.globals, "frame_processors", [])
    return any(p in processors for p in
               ("face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512"))


def _is_dml() -> bool:
    return any("DmlExecutionProvider" in p for p in modules.globals.execution_providers)


def _analyse_faces(frame: Frame) -> list:
    """Run face detection, then recognition (and optionally landmark).

    Replaces InsightFace's ``FaceAnalysis.get()`` to skip the
    landmark_2d_106 model when only face_swapper is active (saves ~1ms
    per face and avoids an unnecessary ONNX session call).
    """
    fa = get_face_analyser()

    bboxes, kpss = fa.det_model.detect(frame, max_num=0, metric="default")
    if bboxes.shape[0] == 0:
        return []

    need_landmark = _needs_landmark()
    rec_model = fa.models.get("recognition")
    lmk_model = fa.models.get("landmark_2d_106") if need_landmark else None

    from insightface.app.common import Face

    faces = []
    for i in range(bboxes.shape[0]):
        face = Face(bbox=bboxes[i, 0:4],
                    kps=kpss[i] if kpss is not None else None,
                    det_score=bboxes[i, 4])
        if rec_model is not None:
            rec_model.get(frame, face)
        if lmk_model is not None:
            lmk_model.get(frame, face)
        faces.append(face)

    return faces


def get_one_face(frame: Frame, faces: Any = None) -> Any:
    if faces is None:
        if _is_dml():
            with modules.globals.dml_lock:
                faces = _analyse_faces(frame)
        else:
            faces = _analyse_faces(frame)
    try:
        return min(faces, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(frame: Frame) -> Any:
    try:
        if _is_dml():
            with modules.globals.dml_lock:
                return _analyse_faces(frame)
        else:
            return _analyse_faces(frame)
    except IndexError:
        return None


def _detect_only(frame: Frame) -> tuple:
    """Run the detection model, serializing on DirectML if needed."""
    fa = get_face_analyser()
    if _is_dml():
        with modules.globals.dml_lock:
            return fa.det_model.detect(frame, max_num=0, metric="default")
    return fa.det_model.detect(frame, max_num=0, metric="default")


def _build_det_trt_session(model_path: str, input_name: str, det_size: int):
    """Fixed-shape TensorRT session for a detection model, or None.

    The detector letterboxes every input to exactly det_size, so the ONNX
    input can be pinned: min=opt=max=1x3x{det_size}x{det_size} gives TRT a
    static engine (no dynamic-shape overhead).  Returns None when TRT is
    unavailable so the caller keeps the existing session.
    """
    import sys as _sys
    if _sys.platform != "win32":
        return None
    try:
        from modules.processors.frame.face_swapper import _tensorrt_providers
        trt_providers = _tensorrt_providers()
        if not trt_providers:
            return None
        import onnxruntime
        shape = f"{input_name}:1x3x{det_size}x{det_size}"
        trt_name, trt_opts = trt_providers[0]
        opts = dict(trt_opts)
        opts["trt_profile_min_shapes"] = shape
        opts["trt_profile_opt_shapes"] = shape
        opts["trt_profile_max_shapes"] = shape
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options,
            providers=[(trt_name, opts)] + list(trt_providers[1:]),
        )
        if "TensorrtExecutionProvider" not in session.get_providers()[0]:
            return None
        return session
    except Exception as e:
        print(f"[DLC.FACE-ANALYSER] TensorRT detection session unavailable: {e}")
        return None


_ROI_DET_MODEL = None       # False = build failed, retry is pointless
_ROI_DET_LOCK = threading.Lock()
ROI_DET_SIZE = 384          # det size for ROI crop passes

# det 384 was chosen by sweep on a synthetic moving-face sequence
# (jitter_det.py): temporal kps jitter rms 0.56px — smoother than the
# full-frame 640 pass (1.98px) — at ~2.5x lower CPU cost (45ms vs 113ms).
# 320 was faster still (32.5ms) but jittered 4.66px, enough to wobble the
# swap alignment; 416/512 were slower without a jitter win.


def _get_roi_det_model():
    """Dedicated RetinaFace for ROI crop passes, run at ROI_DET_SIZE.

    The full-frame detector letterboxes everything to det_size (640), so its
    inference cost is fixed no matter what input size it gets — cropping alone
    saves almost nothing.  The real win is a smaller det size: the tracked
    face fills the crop, and at 384 the face is still 200+ px in detection
    space, well above the ~60 px RetinaFace needs.  Cost drops by
    (384/640)^2 ≈ 2.8x, more on TensorRT (static 384 engine).

    Built lazily on first ROI use; session.run() is thread-safe and the
    postprocess is stateless, so the singleton can be shared.  Returns None
    when the pack or session cannot be built (callers fall back to the
    full-size detector).
    """
    global _ROI_DET_MODEL
    m = _ROI_DET_MODEL
    if m is None:
        with _ROI_DET_LOCK:
            if _ROI_DET_MODEL is None:
                try:
                    fa = get_face_analyser()  # downloads the pack if needed
                    det_path = getattr(fa.det_model, "model_file", None)
                    if not det_path or not os.path.exists(det_path):
                        _ROI_DET_MODEL = False
                        return None
                    session = _build_det_trt_session(
                        det_path, fa.det_model.input_name, ROI_DET_SIZE
                    )
                    if session is not None:
                        m = insightface.model_zoo.get_model(
                            det_path,
                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                        )
                        m.session = session
                    else:
                        from modules.processors.frame._onnx_enhancer import (
                            build_provider_config,
                        )
                        m = insightface.model_zoo.get_model(
                            det_path, providers=build_provider_config()
                        )
                    m.prepare(ctx_id=0, input_size=(ROI_DET_SIZE, ROI_DET_SIZE))
                    _ROI_DET_MODEL = m
                    provider = (
                        m.session.get_providers()[0]
                        if hasattr(m.session, "get_providers") else "?"
                    )
                    print(
                        f"[DLC.FACE-ANALYSER] ROI detector ready at "
                        f"{ROI_DET_SIZE}x{ROI_DET_SIZE} on {provider}"
                    )
                except Exception as e:
                    print(f"[DLC.FACE-ANALYSER] ROI detector unavailable: {e}")
                    _ROI_DET_MODEL = False
                    return None
    return m or None


def _roi_detect(crop: Frame) -> tuple:
    """Detect faces inside an ROI crop with the small-shape model.

    SCRFD.detect takes a per-call input_size override; pass it explicitly so
    the crop always runs at ROI_DET_SIZE regardless of prepare() semantics.
    """
    m = _get_roi_det_model()
    if m is None:
        return _detect_only(crop)
    roi_size = (ROI_DET_SIZE, ROI_DET_SIZE)
    if _is_dml():
        with modules.globals.dml_lock:
            return m.detect(crop, input_size=roi_size, max_num=0, metric="default")
    return m.detect(crop, input_size=roi_size, max_num=0, metric="default")


def detect_one_face_fast(frame: Frame) -> Any:
    """Detection-only — skips landmark and recognition models.

    Returns a Face with bbox, kps, det_score (enough for face swap).
    ~10ms vs ~16ms for full get_one_face() at 1080p.
    """
    from insightface.app.common import Face
    bboxes, kpss = _detect_only(frame)
    if bboxes.shape[0] == 0:
        return None
    idx = int(bboxes[:, 0].argmin())
    return Face(bbox=bboxes[idx, :4], kps=kpss[idx], det_score=bboxes[idx, 4])


def detect_many_faces_fast(frame: Frame) -> Any:
    """Detection-only multi-face — skips landmark and recognition."""
    from insightface.app.common import Face
    bboxes, kpss = _detect_only(frame)
    if bboxes.shape[0] == 0:
        return None
    return [Face(bbox=bboxes[i, :4], kps=kpss[i], det_score=bboxes[i, 4])
            for i in range(bboxes.shape[0])]


class ROIDetector:
    """Single-face detector that tracks the previous bbox and crops the
    detection input around it.

    The full-frame detector letterboxes everything to det_size (640), so its
    inference cost barely changes with input size — cropping alone saves
    almost nothing.  The real win comes from running crop passes on a
    dedicated small-shape detector (ROI_DET_SIZE=320, see _get_roi_det_model):
    the tracked face fills the crop, so a 4x smaller det grid loses nothing
    in accuracy.  When the tracked face is lost, scores low, or touches the
    crop border (it may have moved out), the detector falls back to one
    full-frame pass on the 640 model and the tracker re-locks.

    Returned Face objects carry full-frame coordinates — the crop offset is
    added back before returning — so callers need no changes.  Each pipeline
    owns its own instance; instances are not thread-safe (the live detection
    worker and the offline detect executor are both single-threaded).
    """

    EXPAND = 1.6          # ROI margin around the last bbox, per axis
    MIN_CROP = 128        # never shrink the crop below this side
    EDGE_FALLBACK = 8     # bbox within this many px of the crop border -> full-frame retry
    MIN_SCORE = 0.4       # detection below this score is treated as a miss

    def __init__(self) -> None:
        self._last_bbox = None  # np.ndarray [x1, y1, x2, y2] in frame coords

    def reset(self) -> None:
        self._last_bbox = None

    def detect(self, frame: Frame) -> Any:
        from insightface.app.common import Face

        h, w = frame.shape[:2]
        if self._last_bbox is not None:
            x1, y1, x2, y2 = self._roi(w, h)
            crop = frame[y1:y2, x1:x2]
            if crop.size and crop.shape[0] >= 32 and crop.shape[1] >= 32:
                bboxes, kpss = _roi_detect(crop)
                if bboxes.shape[0] > 0:
                    idx = int(bboxes[:, 0].argmin())
                    score = float(bboxes[idx, 4])
                    bb = bboxes[idx, :4]
                    kps = kpss[idx]
                    # bb/kps are crop-local; the ROI origin is (x1, y1). A bbox
                    # hugging the crop border may have moved out of the ROI —
                    # re-detect on the full frame instead of locking onto a
                    # partially visible face.
                    near_edge = (
                        bb[0] <= self.EDGE_FALLBACK
                        or bb[1] <= self.EDGE_FALLBACK
                        or (x2 - x1) - bb[2] <= self.EDGE_FALLBACK
                        or (y2 - y1) - bb[3] <= self.EDGE_FALLBACK
                    )
                    if score >= self.MIN_SCORE and not near_edge:
                        offset = (x1, y1, x1, y1)
                        face = Face(bbox=bb + offset,
                                    kps=kps + (x1, y1),
                                    det_score=score)
                        self._remember(face.bbox, w, h)
                        return face
        # Tracker empty, crop missed, score too low, or face hugging the
        # crop border — one full-frame pass re-locks the tracker.
        face = detect_one_face_fast(frame)
        if face is not None:
            self._remember(face.bbox, w, h)
        else:
            self._last_bbox = None
        return face

    def _roi(self, w: int, h: int) -> tuple:
        x1, y1, x2, y2 = (float(v) for v in self._last_bbox[:4])
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        side = max(x2 - x1, y2 - y1, self.MIN_CROP) * self.EXPAND
        half = side / 2.0
        rx1 = int(max(0, cx - half))
        ry1 = int(max(0, cy - half))
        rx2 = int(min(w, cx + half))
        ry2 = int(min(h, cy + half))
        return rx1, ry1, rx2, ry2

    def _remember(self, bbox, w: int, h: int) -> None:
        bb = np.asarray(bbox, dtype=np.float32)[:4]
        # Clamp into the frame; a bbox poking outside corrupts the next ROI.
        bb[0::2] = np.clip(bb[0::2], 0, w - 1)
        bb[1::2] = np.clip(bb[1::2], 0, h - 1)
        self._last_bbox = bb


def ensure_landmarks(frame: Frame, faces: Any) -> None:
    """Run the 2d106 landmark model in-place on faces that lack it.

    The fast webcam path (detect_one_face_fast / detect_many_faces_fast)
    produces detection-only Face objects with no ``landmark_2d_106``.
    Mouth masking needs those landmarks, so add them on demand only when
    the feature is active — keeping the fast path fast otherwise.
    """
    if faces is None:
        return
    if not isinstance(faces, (list, tuple)):
        faces = [faces]

    fa = get_face_analyser()
    lmk_model = fa.models.get("landmark_2d_106")
    if lmk_model is None:
        return

    for face in faces:
        if face is None:
            continue
        # insightface Face is a dict; missing keys raise AttributeError,
        # so getattr(..., None) is the safe presence check.
        if getattr(face, "landmark_2d_106", None) is None:
            try:
                lmk_model.get(frame, face)
            except Exception as e:  # pragma: no cover - never break the swap
                print(f"Error computing 2d106 landmarks: {e}")


def has_valid_map() -> bool:
    for map in modules.globals.source_target_map:
        if "source" in map and "target" in map:
            return True
    return False

def default_source_face() -> Any:
    for map in modules.globals.source_target_map:
        if "source" in map:
            return map['source']['face']
    return None

def simplify_maps() -> Any:
    centroids = []
    faces = []
    for map in modules.globals.source_target_map:
        if "source" in map and "target" in map:
            centroids.append(map['target']['face'].normed_embedding)
            faces.append(map['source']['face'])

    modules.globals.simple_map = {'source_faces': faces, 'target_embeddings': centroids}
    return None

def add_blank_map() -> Any:
    try:
        max_id = -1
        if len(modules.globals.source_target_map) > 0:
            max_id = max(modules.globals.source_target_map, key=lambda x: x['id'])['id']

        modules.globals.source_target_map.append({
                'id' : max_id + 1
                })
    except ValueError:
        return None
    
def get_unique_faces_from_target_image() -> Any:
    try:
        modules.globals.source_target_map = []
        target_frame = imread_unicode(modules.globals.target_path)
        many_faces = get_many_faces(target_frame)
        if many_faces is None:
            return None
        i = 0

        for face in many_faces:
            x_min, y_min, x_max, y_max = face['bbox']
            modules.globals.source_target_map.append({
                'id' : i, 
                'target' : {
                            'cv2' : target_frame[int(y_min):int(y_max), int(x_min):int(x_max)],
                            'face' : face
                            }
                })
            i = i + 1
    except ValueError:
        return None
    
    
def get_unique_faces_from_target_video() -> Any:
    try:
        modules.globals.source_target_map = []
        frame_face_embeddings = []
        face_embeddings = []
    
        target_path = modules.globals.target_path
        print('Decoding and analysing frames in-memory...')
        cap = cv2.VideoCapture(target_path)
        if not cap.isOpened():
            print(f'Could not open target video for frame analysis: {target_path}')
            return modules.globals.source_target_map

        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            i = 0
            with tqdm(total=total if total > 0 else None, desc='Extracting face embeddings from frames') as progress:
                while True:
                    ok, temp_frame = cap.read()
                    if not ok:
                        break
                    # i is the real 0-based video frame index, incremented for
                    # every frame so it lines up with the processing pass that
                    # decodes the same video sequentially.
                    many_faces = get_many_faces(temp_frame)
                    if many_faces is not None:
                        for face in many_faces:
                            face_embeddings.append(face.normed_embedding)
                        frame_face_embeddings.append({
                            'frame': i,
                            'frame_index': i,
                            'faces': many_faces,
                        })
                    i += 1
                    progress.update(1)
        finally:
            cap.release()

        centroids = find_cluster_centroids(face_embeddings)

        for frame in frame_face_embeddings:
            for face in frame['faces']:
                closest_centroid_index, _ = find_closest_centroid(centroids, face.normed_embedding)
                face['target_centroid'] = closest_centroid_index

        # Bucket frames per centroid in a single pass, instead of rescanning
        # every frame once per centroid (was O(centroids x frames)). Frames
        # with no face for a centroid are skipped: downstream consumers treat
        # a missing frame entry exactly like an empty 'faces' list, so this
        # also shrinks the map without changing swap behaviour.
        buckets = [[] for _ in range(len(centroids))]
        for frame in frame_face_embeddings:
            by_centroid = {}
            for face in frame['faces']:
                by_centroid.setdefault(face['target_centroid'], []).append(face)
            for centroid_index, faces in by_centroid.items():
                buckets[centroid_index].append({
                    'frame': frame['frame'],
                    'frame_index': frame['frame_index'],
                    'faces': faces,
                })

        for i in range(len(centroids)):
            modules.globals.source_target_map.append({
                'id' : i,
                'target_faces_in_frame': buckets[i],
                # O(1) lookup for the per-frame pipeline: the old lookup
                # linearly scanned this bucket list on every frame.
                '_frame_index': {entry['frame_index']: entry for entry in buckets[i]},
            })

        # dump_faces(centroids, frame_face_embeddings)
        default_target_face()
    except ValueError:
        return None
    

def _read_video_frame(target_path: str, frame_index: int):
    """Decode a single video frame by 0-based index.

    Used to crop the default-target thumbnail without keeping every decoded
    frame resident in memory (only a handful of best frames are sought).
    """
    cap = cv2.VideoCapture(target_path)
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def default_target_face():
    target_path = modules.globals.target_path
    for map in modules.globals.source_target_map:
        best_face = None
        best_frame = None
        for frame in map['target_faces_in_frame']:
            if len(frame['faces']) > 0:
                best_face = frame['faces'][0]
                best_frame = frame
                break

        if best_face is None:
            continue  # No faces detected in this cluster — skip

        for frame in map['target_faces_in_frame']:
            for face in frame['faces']:
                if face['det_score'] > best_face['det_score']:
                    best_face = face
                    best_frame = frame

        x_min, y_min, x_max, y_max = best_face['bbox']
        frame_index = best_frame.get('frame_index', best_frame.get('frame'))
        target_frame = _read_video_frame(target_path, frame_index)
        if target_frame is None:
            continue
        map['target'] = {
                        'cv2' : target_frame[int(y_min):int(y_max), int(x_min):int(x_max)],
                        'face' : best_face
                        }


def dump_faces(centroids: Any, frame_face_embeddings: list):
    temp_directory_path = get_temp_directory_path(modules.globals.target_path)

    for i in range(len(centroids)):
        if os.path.exists(temp_directory_path + f"/{i}") and os.path.isdir(temp_directory_path + f"/{i}"):
            shutil.rmtree(temp_directory_path + f"/{i}")
        Path(temp_directory_path + f"/{i}").mkdir(parents=True, exist_ok=True)

        for frame in tqdm(frame_face_embeddings, desc=f"Copying faces to temp/./{i}"):
            temp_frame = _read_video_frame(modules.globals.target_path, frame.get('frame_index', frame['frame']))

            j = 0
            for face in frame['faces']:
                if face['target_centroid'] == i:
                    x_min, y_min, x_max, y_max = face['bbox']

                    if temp_frame[int(y_min):int(y_max), int(x_min):int(x_max)].size > 0:
                        imwrite_unicode(temp_directory_path + f"/{i}/{frame['frame']}_{j}.png", temp_frame[int(y_min):int(y_max), int(x_min):int(x_max)])
                j += 1
