"""ComfyUI adapters for Multi-HMR 2 video analysis and rendering."""

from __future__ import annotations

import hashlib
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


def _backend():
    """Import the bundled backend lazily so missing optional deps do not block ComfyUI."""
    package_root = str(PLUGIN_ROOT)
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    try:
        from multihmr2 import init_hmr_session, infer_video
        from multihmr2.utils.render import render_meshes
        return init_hmr_session, infer_video, render_meshes
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
        init_hmr_session, _, _ = _backend()
        LOGGER.info("Loading Multi-HMR2 checkpoint: %s", MODEL_PATH)
        _SESSION_CACHE[key] = init_hmr_session(MODEL_PATH, compile_model=compile_model)
    return _SESSION_CACHE[key]


def _as_uint8(frame: torch.Tensor) -> np.ndarray:
    array = frame.detach().cpu().float().clamp(0, 1).numpy()
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"期望 IMAGE 为 HWC 3/4 通道，实际为 {array.shape}")
    return (array[..., :3] * 255.0 + 0.5).astype(np.uint8)


def _write_temp_video(images: torch.Tensor, fps: float) -> Path:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("需要 ComfyUI 环境中的 OpenCV 来桥接 VHS 帧和 Multi-HMR2") from exc

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    handle, raw_path = tempfile.mkstemp(prefix="input_", suffix=".mp4", dir=CACHE_ROOT)
    os.close(handle)
    path = Path(raw_path)
    height, width = images.shape[1:3]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(float(fps), 1.0), (width, height)
    )
    if not writer.isOpened():
        path.unlink(missing_ok=True)
        raise RuntimeError("无法创建临时视频，请确认 FFmpeg/OpenCV 编码器可用")
    try:
        for frame in images:
            rgb = _as_uint8(frame)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return path


def _cache_key(images: torch.Tensor, fps: float, conf: float, nms: float, lowres: bool) -> str:
    digest = hashlib.sha256()
    for frame in images:
        digest.update(_as_uint8(frame).tobytes())
    digest.update(json.dumps([fps, conf, nms, lowres, "0.1.0"], sort_keys=True).encode())
    return digest.hexdigest()


def _load_or_analyze(images, fps, conf_thresh, dist_thresh_nms, lowres, compile_model):
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    key = _cache_key(images, fps, conf_thresh, dist_thresh_nms, lowres)
    cache_path = CACHE_ROOT / f"{key}.pt"
    if cache_path.is_file():
        LOGGER.info("Multi-HMR2 cache hit: %s", cache_path)
        return torch.load(cache_path, map_location="cpu", weights_only=False), key

    _, infer_video, _ = _backend()
    session = _session(compile_model)
    input_video = _write_temp_video(images, fps)
    frame_dir = CACHE_ROOT / f"frames_{key}"
    try:
        preds = infer_video(
            session,
            str(input_video),
            str(frame_dir),
            conf_thresh=float(conf_thresh),
            dist_thresh_nms=float(dist_thresh_nms),
            lowres=bool(lowres),
        )
        torch.save(preds, cache_path)
        return preds, key
    finally:
        input_video.unlink(missing_ok=True)


def _fallback_overlay(image: np.ndarray, pred, show_skeleton: bool, show_id: bool) -> np.ndarray:
    import cv2

    out = image.copy()
    if len(pred) == 0:
        return out
    colors = [(40, 120, 255), (70, 210, 100), (220, 90, 180), (255, 180, 40)]
    joints = pred.persons.j2d.detach().cpu().numpy()
    track_id = getattr(pred.persons, "track_id", None)
    ids = track_id.detach().cpu().numpy() if track_id is not None else None
    for i, pts in enumerate(joints):
        color = colors[int(ids[i]) % len(colors)] if ids is not None else colors[i % len(colors)]
        for x, y in pts:
            cv2.circle(out, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)
        if show_skeleton:
            for a, b in zip(range(len(pts) - 1), range(1, len(pts))):
                cv2.line(out, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), color, 2, cv2.LINE_AA)
        if show_id:
            label = f"ID {int(ids[i])}" if ids is not None else f"P{i}"
            x, y = pts[0].astype(int)
            cv2.putText(out, label, (int(x), int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out


class MultiHMR2VideoAnalyze:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "frame_count": ("INT", {"default": 1, "min": 1}),
            "conf_thresh": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
            "dist_thresh_nms": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 2.0, "step": 0.01}),
            "lowres": ("BOOLEAN", {"default": False}),
            "compile_model": ("BOOLEAN", {"default": False}),
        }, "optional": {
            "audio": ("AUDIO",),
            "video_info": ("VHS_VIDEOINFO",),
        }}

    RETURN_TYPES = ("MULTI_HMR2_ANALYSIS", "INT", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("analysis", "frame_count", "audio", "video_info")
    FUNCTION = "analyze"
    CATEGORY = "MultiHMR2/Video"

    def analyze(self, images, frame_count, conf_thresh, dist_thresh_nms, lowres, compile_model, audio=None, video_info=None):
        fps = float((video_info or {}).get("loaded_fps", 30.0))
        preds, key = _load_or_analyze(images, fps, conf_thresh, dist_thresh_nms, lowres, compile_model)
        return ({"preds": preds, "cache_key": key, "fps": fps, "lowres": bool(lowres)}, int(frame_count), audio, video_info or {})


class MultiHMR2VideoRender:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "analysis": ("MULTI_HMR2_ANALYSIS",),
            "background": (["original", "black", "image"], {"default": "original"}),
            "show_mesh": ("BOOLEAN", {"default": True}),
            "show_skeleton": ("BOOLEAN", {"default": True}),
            "show_track_id": ("BOOLEAN", {"default": True}),
            "mesh_opacity": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
        }, "optional": {"background_image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("images", "frame_rate")
    FUNCTION = "render"
    CATEGORY = "MultiHMR2/Video"

    def render(self, images, analysis, background, show_mesh, show_skeleton, show_track_id, mesh_opacity, background_image=None):
        _, _, render_meshes = _backend()
        preds = analysis["preds"]
        output = []
        for index, (frame, pred) in enumerate(zip(images, preds)):
            base = _as_uint8(frame)
            if background == "black":
                base = np.zeros_like(base)
            elif background == "image" and background_image is not None:
                base = _as_uint8(background_image[index % len(background_image)])
            try:
                if show_mesh and len(pred) and hasattr(render_meshes, "__call__"):
                    body_model = analysis.get("body_model")
                    # The current backend's render helper needs the loaded session body faces.
                    session = _session(False)
                    body_model = session.model.full_body_decoder.lowres_body_model if analysis.get("lowres") else session.model.full_body_decoder.body_model
                    verts = [v.reshape(-1, 3).numpy() for v in pred.persons.v3d]
                    faces = [body_model.faces.numpy() for _ in verts]
                    k = pred.K[[0, 1], [-1, -1]].numpy()
                    focal = pred.K[[0, 1], [0, 1]].numpy()
                    rendered = render_meshes(base, verts, faces, {"focal": focal, "princpt": k}, color=None)
                    base = (base * (1.0 - float(mesh_opacity)) + rendered * float(mesh_opacity)).clip(0, 255).astype(np.uint8)
            except Exception as exc:
                LOGGER.warning("3D mesh rendering unavailable on frame %s; using 2D fallback: %s", index, exc)
            if show_skeleton or show_track_id:
                base = _fallback_overlay(base, pred, show_skeleton, show_track_id)
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
