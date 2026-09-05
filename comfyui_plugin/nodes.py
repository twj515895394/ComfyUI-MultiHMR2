"""ComfyUI adapters for Multi-HMR 2 video analysis and rendering."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

LOGGER = logging.getLogger("ComfyUI-MultiHMR2")
PLUGIN_ROOT = Path(__file__).resolve().parent
COMFY_ROOT = PLUGIN_ROOT.parents[1]
MODEL_PATH = COMFY_ROOT / "models" / "multihmr2" / "multihmr2.pt"
CACHE_ROOT = COMFY_ROOT / "temp" / "multihmr2"
_SESSION_CACHE: dict[tuple[str, bool], Any] = {}
_SEGMENTER_CACHE: Any = None


def _use_software_mesh() -> bool:
    """Allow an explicit software-only mode for troubleshooting."""
    return os.environ.get("MULTIHMR2_FORCE_SOFTWARE") == "1"


def _ensure_native_opengl() -> None:
    """Repair a previously imported EGL platform on Windows."""
    if os.name != "nt":
        return
    platform_module = sys.modules.get("OpenGL.platform")
    platform_obj = getattr(platform_module, "PLATFORM", None) if platform_module else None
    if platform_obj is None or type(platform_obj).__name__ == "Win32Platform":
        return
    from OpenGL.platform.win32 import Win32Platform

    native_platform = Win32Platform()
    native_platform.install(platform_module.__dict__)
    LOGGER.warning(
        "MultiHMR2 switched preloaded PyOpenGL platform from %s to Win32Platform",
        type(platform_obj).__name__,
    )


def _backend():
    """Import the bundled backend lazily so missing optional deps do not block ComfyUI."""
    package_root = str(PLUGIN_ROOT)
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    try:
        # ComfyUI video/render plugins may globally force EGL.  The Windows
        # EGL path on this installation fails with EGL_BAD_PARAMETER; letting
        # pyrender select native WGL is the reliable offscreen path here.
        if os.name == "nt" and os.environ.get("PYOPENGL_PLATFORM") in {"egl", "osmesa"}:
            LOGGER.warning("Ignoring incompatible PYOPENGL_PLATFORM=%s on Windows; using native WGL", os.environ["PYOPENGL_PLATFORM"])
            os.environ.pop("PYOPENGL_PLATFORM", None)
        _ensure_native_opengl()
        from multihmr2 import init_hmr_session, infer_image
        from multihmr2.utils.render import render_meshes
        from multihmr2.tracker import FeatPelvisTracker
        return init_hmr_session, infer_image, render_meshes, FeatPelvisTracker
    except Exception as exc:
        raise RuntimeError(
            "MultiHMR2 依赖未就绪。请使用 ComfyUI 自带 Python 安装 "
            "anny==0.6.0、roma、warp-lang；需要 3D 网格渲染时再安装 pyrender。"
        ) from exc


def _session(compile_model: bool):
    key = (str(MODEL_PATH), bool(compile_model))
    if key not in _SESSION_CACHE:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"找不到模型文件：{MODEL_PATH}")
        # ComfyUI installations may have a read-only user cache.  Keep the
        # torch.hub trust list and repository cache inside ComfyUI temp.
        hub_root = CACHE_ROOT / "torch_hub"
        hub_root.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(hub_root))
        init_hmr_session, _, _, _ = _backend()
        LOGGER.info("Loading Multi-HMR2 checkpoint: %s", MODEL_PATH)
        _SESSION_CACHE[key] = init_hmr_session(MODEL_PATH, compile_model=compile_model)
    return _SESSION_CACHE[key]


def _as_uint8(frame: torch.Tensor) -> np.ndarray:
    array = frame.detach().cpu().float().clamp(0, 1).numpy()
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"期望 IMAGE 为 HWC 3/4 通道，实际为 {array.shape}")
    return (array[..., :3] * 255.0 + 0.5).astype(np.uint8)


def _cache_key(images: torch.Tensor, fps: float, conf: float, nms: float, lowres: bool) -> str:
    digest = hashlib.sha256()
    for frame in images:
        digest.update(_as_uint8(frame).tobytes())
    digest.update(json.dumps([fps, conf, nms, lowres, "0.1.0"], sort_keys=True).encode())
    return digest.hexdigest()


def _track_frame(tracker, session, frame_pred, time):
    if len(frame_pred) > 0:
        trackfeat = frame_pred.persons.trackfeat.clone().float()
        pelvis_xyz = frame_pred.persons.transl_pelvis.clone().float()
        pelvis_ijn = torch.cat([
            frame_pred.persons.j2d[:, 0] / session.img_size,
            torch.log(1 / pelvis_xyz[:, 2:3].clamp_min(1e-5)),
        ], dim=-1).clone().float()
        pelvis_ori = frame_pred.persons.bone_poses[:, 0, :3, :3].clone().float()
        human_ids = tracker.track_next_frame(time, trackfeat, pelvis_xyz, pelvis_ijn, pelvis_ori)
    else:
        human_ids = torch.empty((0,), dtype=torch.long)
    frame_pred.persons.track_id = human_ids
    return frame_pred


def _load_or_analyze(images, fps, conf_thresh, dist_thresh_nms, lowres, compile_model, segment_size):
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    key = _cache_key(images, fps, conf_thresh, dist_thresh_nms, lowres)
    cache_path = CACHE_ROOT / f"{key}.pt"
    if cache_path.is_file():
        LOGGER.info("Multi-HMR2 cache hit: %s", cache_path)
        return torch.load(cache_path, map_location="cpu", weights_only=False), key

    _, infer_image, _, FeatPelvisTracker = _backend()
    session = _session(compile_model)
    tracker = FeatPelvisTracker()
    tracker.reset()
    preds = []
    step = int(segment_size) if int(segment_size) > 0 else len(images)
    for start in range(0, len(images), step):
        end = min(start + step, len(images))
        LOGGER.info("MultiHMR2 analyzing segment %s-%s/%s", start, end, len(images))
        for time in range(start, end):
            frame_pred = infer_image(
                session, _as_uint8(images[time]),
                conf_thresh=float(conf_thresh),
                dist_thresh_nms=float(dist_thresh_nms),
                lowres=bool(lowres),
            )
            preds.append(_track_frame(tracker, session, frame_pred, time))
    torch.save(preds, cache_path)
    return preds, key


def _birefnet_masks(images):
    """Reuse the installed ComfyUI-RMBG BiRefNet node for high-quality alpha."""
    global _SEGMENTER_CACHE
    try:
        rmbg_dir = COMFY_ROOT / "custom_nodes" / "ComfyUI-RMBG" / "py"
        if str(rmbg_dir) not in sys.path:
            sys.path.insert(0, str(rmbg_dir))
        if _SEGMENTER_CACHE is None:
            module = importlib.import_module("AILab_BiRefNet")
            _SEGMENTER_CACHE = module.BiRefNetRMBG()
        _, masks, _ = _SEGMENTER_CACHE.process_image(
            images, model="BiRefNet-portrait", sensitivity=1.0,
            mask_blur=0, mask_offset=0, invert_output=False,
            refine_foreground=True, background="Alpha", background_color="#000000",
        )
        masks = masks.detach().cpu().float().clamp(0, 1).numpy()
        return masks[:, 0] if masks.ndim == 4 else masks
    except Exception as exc:
        LOGGER.warning("BiRefNet unavailable; transparent mode will use mesh alpha: %s", exc)
        return None


def _model_points_to_frame(points: np.ndarray, frame_shape, model_size=768, patch_size=16) -> np.ndarray:
    """Undo Multi-HMR2's longest-side resize and centered patch padding."""
    height, width = frame_shape[:2]
    scale = min(model_size / float(width), model_size / float(height))
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    padded_width = int(np.ceil(resized_width / patch_size) * patch_size)
    padded_height = int(np.ceil(resized_height / patch_size) * patch_size)
    pad_left = (padded_width - resized_width) // 2
    pad_top = (padded_height - resized_height) // 2
    mapped = points.astype(np.float32).copy()
    mapped[..., 0] = (mapped[..., 0] - pad_left) / scale
    mapped[..., 1] = (mapped[..., 1] - pad_top) / scale
    return mapped


def _camera_to_frame(k: np.ndarray, frame_shape, model_size=768, patch_size=16) -> tuple[np.ndarray, np.ndarray]:
    height, width = frame_shape[:2]
    scale = min(model_size / float(width), model_size / float(height))
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    padded_width = int(np.ceil(resized_width / patch_size) * patch_size)
    padded_height = int(np.ceil(resized_height / patch_size) * patch_size)
    pad_left = (padded_width - resized_width) // 2
    pad_top = (padded_height - resized_height) // 2
    focal = k[[0, 1], [0, 1]].astype(np.float32) / scale
    princpt = np.array([(k[0, 2] - pad_left) / scale, (k[1, 2] - pad_top) / scale], dtype=np.float32)
    return focal, princpt


def _fallback_overlay(image: np.ndarray, pred, show_skeleton: bool, show_id: bool) -> np.ndarray:
    import cv2

    out = image.copy()
    if len(pred) == 0:
        return out
    colors = [(40, 120, 255), (70, 210, 100), (220, 90, 180), (255, 180, 40)]
    joints = _model_points_to_frame(pred.persons.j2d.detach().cpu().numpy(), image.shape)
    track_id = getattr(pred.persons, "track_id", None)
    ids = track_id.detach().cpu().numpy() if track_id is not None else None
    for i, pts in enumerate(joints):
        color = colors[int(ids[i]) % len(colors)] if ids is not None else colors[i % len(colors)]
        if show_skeleton:
            for x, y in pts:
                cv2.circle(out, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)
            for a, b in zip(range(len(pts) - 1), range(1, len(pts))):
                cv2.line(out, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), color, 2, cv2.LINE_AA)
        if show_id:
            label = f"ID {int(ids[i])}" if ids is not None else f"P{i}"
            x, y = pts[0].astype(int)
            cv2.putText(out, label, (int(x), int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out


def _software_mesh_overlay(image: np.ndarray, pred, focal: np.ndarray, princpt: np.ndarray, opacity: float, faces):
    """Rasterize the actual body triangles without requiring an OpenGL context.

    ComfyUI often runs headless on Windows, where pyrender cannot create a WGL
    context.  This keeps the real body topology visible instead of drawing only
    a convex-hull silhouette.
    """
    import cv2

    overlay = image.copy()
    alpha_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    for vertices in pred.persons.v3d.detach().cpu().numpy():
        verts = vertices.reshape(-1, 3)
        z = verts[:, 2]
        valid = np.isfinite(verts).all(axis=1) & (z > 1e-5)
        if valid.sum() < 3:
            continue
        pts = np.empty((len(verts), 2), dtype=np.float32)
        pts[:, 0] = focal[0] * verts[:, 0] / np.maximum(verts[:, 2], 1e-5) + princpt[0]
        pts[:, 1] = focal[1] * verts[:, 1] / np.maximum(verts[:, 2], 1e-5) + princpt[1]
        pts = np.round(pts).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, image.shape[1] - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, image.shape[0] - 1)
        # Solid underlay prevents tiny gaps between projected triangles from
        # revealing the source image.  The real triangles below still provide
        # the mesh topology and shading.
        hull = cv2.convexHull(pts[valid])
        cv2.fillConvexPoly(overlay, hull, (225, 225, 225), cv2.LINE_AA)
        cv2.fillConvexPoly(alpha_mask, hull, 255, cv2.LINE_AA)
        valid_faces = faces[(faces >= 0).all(axis=1) & (faces < len(verts)).all(axis=1)]
        valid_faces = valid_faces[np.isfinite(verts[valid_faces]).all(axis=(1, 2)) & (verts[valid_faces, 2] > 1e-5).all(axis=1)]
        # Painter's algorithm: far triangles first, near triangles last.
        depths = verts[valid_faces, 2].mean(axis=1)
        for tri, depth in sorted(zip(valid_faces, depths), key=lambda item: item[1], reverse=True):
            polygon = pts[tri]
            edge1 = verts[tri[1]] - verts[tri[0]]
            edge2 = verts[tri[2]] - verts[tri[0]]
            normal = np.cross(edge1, edge2)
            light = float(np.clip(abs(normal[1]) / (np.linalg.norm(normal) + 1e-6), 0.0, 1.0))
            shade = int(225 + 30 * light)
            cv2.fillConvexPoly(overlay, polygon, (shade, shade, shade), cv2.LINE_AA)
            cv2.fillConvexPoly(alpha_mask, polygon, 255, cv2.LINE_AA)
    amount = float(np.clip(opacity, 0.0, 1.0))
    blended = (image * (1.0 - amount) + overlay * amount).clip(0, 255).astype(np.uint8)
    return blended, alpha_mask


class MultiHMR2VideoAnalyze:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "来自 VHS Load Video 的视频帧序列；建议保持原始帧顺序。"}),
            "frame_count": ("INT", {"default": 1, "min": 1, "tooltip": "视频帧数，通常直接连接 VHS 的 frame_count 输出。"}),
            "conf_thresh": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "人物检测置信度阈值。越高误检越少，但远处或遮挡人物可能被过滤；推荐 0.40。"}),
            "dist_thresh_nms": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 2.0, "step": 0.01, "tooltip": "3D 骨盆距离 NMS 阈值（米）。用于合并过近的重复检测；推荐 0.25。"}),
            "lowres": ("BOOLEAN", {"default": False, "tooltip": "使用低分辨率 Anny 身体模型。速度更快、显存更低，但网格细节较少。"}),
            "compile_model": ("BOOLEAN", {"default": False, "tooltip": "启用 torch.compile。视频重复推理可能更快，但首次运行会额外编译且占用更多显存。"}),
        }, "optional": {
            "audio": ("AUDIO", {"tooltip": "可选音频透传；最终请将原始音频直接连接到 VHS Video Combine。"}),
            "video_info": ("VHS_VIDEOINFO", {"tooltip": "VHS 视频元数据，用于读取真实 FPS；建议连接 VHS Load Video 的 video_info。"}),
            "segment_size": ("INT", {"default": 120, "min": 0, "max": 10000, "step": 1, "tooltip": "每段处理的帧数。120 适合长视频；0 表示不主动分段。分段之间仍共享 tracker，Track ID 保持连续。"}),
        }}

    RETURN_TYPES = ("MULTI_HMR2_ANALYSIS", "INT", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("analysis", "frame_count", "audio", "video_info")
    FUNCTION = "analyze"
    CATEGORY = "MultiHMR2/Video"

    def analyze(self, images, frame_count, conf_thresh, dist_thresh_nms, lowres, compile_model, segment_size=120, audio=None, video_info=None):
        fps = float((video_info or {}).get("loaded_fps", 30.0))
        preds, key = _load_or_analyze(images, fps, conf_thresh, dist_thresh_nms, lowres, compile_model, segment_size)
        return ({"preds": preds, "cache_key": key, "fps": fps, "lowres": bool(lowres)}, int(frame_count), audio, video_info or {})


class MultiHMR2VideoRender:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "原始视频帧；必须与 Analyze 节点使用同一批帧。"}),
            "analysis": ("MULTI_HMR2_ANALYSIS", {"tooltip": "MultiHMR2 Video Analyze 输出的人体 3D 分析缓存。"}),
            "background": (["original", "green_screen", "transparent"], {"default": "original", "tooltip": "背景模式：original 保留原背景；green_screen 输出纯绿色背景；transparent 输出 RGBA 透明背景。"}),
            "show_mesh": ("BOOLEAN", {"default": True, "tooltip": "显示 3D 人体网格。关闭后仍可显示骨骼和 Track ID。"}),
            "show_skeleton": ("BOOLEAN", {"default": True, "tooltip": "显示 2D 骨骼连接线，用于检查姿态和坐标对齐。"}),
            "show_track_id": ("BOOLEAN", {"default": True, "tooltip": "在人物附近显示跨帧跟踪 ID。"}),
            "mesh_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "3D 白模不透明度。1.0 为完全不透明（推荐）；较低值会让原视频背景透过网格。"}),
        }, "optional": {
            "track_id": ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1, "tooltip": "只渲染指定 Track ID；-1 表示渲染全部人物。Track ID 来自 Analyze 节点。"}),
        }}

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("images", "frame_rate")
    FUNCTION = "render"
    CATEGORY = "MultiHMR2/Video"

    def render(self, images, analysis, background, show_mesh, show_skeleton, show_track_id, mesh_opacity, track_id=-1, background_image=None, **kwargs):
        _, _, render_meshes, _ = _backend()
        preds = analysis["preds"]
        high_quality_masks = _birefnet_masks(images) if background == "transparent" else None
        output = []
        for index, (frame, pred) in enumerate(zip(images, preds)):
            current_track_id = getattr(pred.persons, "track_id", None)
            if track_id >= 0 and current_track_id is not None:
                pred = type(pred)(K=pred.K, persons=pred.persons[current_track_id == track_id])
            base = _as_uint8(frame)
            transparent = background == "transparent"
            if background == "green_screen" or transparent:
                base = np.zeros_like(base)
                if background == "green_screen":
                    base[...] = (0, 255, 0)
            alpha = (high_quality_masks[index] * 255).astype(np.uint8) if high_quality_masks is not None else (np.zeros(base.shape[:2], dtype=np.uint8) if transparent else None)
            try:
                if show_mesh and len(pred) and _use_software_mesh():
                    session = _session(False)
                    body_model = session.model.full_body_decoder.lowres_body_model if analysis.get("lowres") else session.model.full_body_decoder.body_model
                    focal, k = _camera_to_frame(pred.K.numpy(), base.shape)
                    base, software_alpha = _software_mesh_overlay(base, pred, focal, k, mesh_opacity, body_model.faces.cpu().numpy())
                    if transparent:
                        alpha = np.maximum(alpha, software_alpha)
                elif show_mesh and len(pred) and hasattr(render_meshes, "__call__"):
                    body_model = analysis.get("body_model")
                    # The current backend's render helper needs the loaded session body faces.
                    session = _session(False)
                    body_model = session.model.full_body_decoder.lowres_body_model if analysis.get("lowres") else session.model.full_body_decoder.body_model
                    verts = [v.reshape(-1, 3).numpy() for v in pred.persons.v3d]
                    faces = [body_model.faces.numpy() for _ in verts]
                    focal, k = _camera_to_frame(pred.K.numpy(), base.shape)
                    render_result = render_meshes(
                        base, verts, faces, {"focal": focal, "princpt": k},
                        # Stable light material makes the reconstructed body
                        # visibly read as a white model over the source footage.
                        color=(0.92, 0.92, 0.92), return_mask=transparent,
                    )
                    if transparent:
                        rendered, alpha = render_result
                        base = rendered.astype(np.uint8)
                    else:
                        base = (base * (1.0 - float(mesh_opacity)) + render_result * float(mesh_opacity)).clip(0, 255).astype(np.uint8)
            except Exception as exc:
                LOGGER.warning("Mesh rendering failed on frame %s; using software mesh fallback: %s", index, exc)
                if show_mesh and len(pred):
                    try:
                        session = _session(False)
                        body_model = session.model.full_body_decoder.lowres_body_model if analysis.get("lowres") else session.model.full_body_decoder.body_model
                        focal, k = _camera_to_frame(pred.K.numpy(), base.shape)
                        base, software_alpha = _software_mesh_overlay(base, pred, focal, k, mesh_opacity, body_model.faces.cpu().numpy())
                        if transparent:
                            alpha = np.maximum(alpha, software_alpha)
                    except Exception as fallback_exc:
                        LOGGER.warning("Software mesh fallback unavailable on frame %s: %s", index, fallback_exc)
            if show_skeleton or show_track_id:
                base = _fallback_overlay(base, pred, show_skeleton, show_track_id)
            if transparent:
                rgba = np.concatenate([base, alpha[..., None]], axis=-1)
                output.append(torch.from_numpy(rgba.astype(np.float32) / 255.0))
            else:
                output.append(torch.from_numpy(base.astype(np.float32) / 255.0))
        return (torch.stack(output, dim=0), float(analysis.get("fps", 30.0)))


NODE_CLASS_MAPPINGS = {
    "MultiHMR2VideoAnalyze": MultiHMR2VideoAnalyze,
    "MultiHMR2VideoRender": MultiHMR2VideoRender,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiHMR2VideoAnalyze": "MultiHMR2 Video Analyze",
    "MultiHMR2VideoRender": "MultiHMR2 Video Render",
}
