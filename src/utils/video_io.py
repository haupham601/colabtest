"""Video input and output utilities.

Provides functions for loading, saving, sampling, and extracting video frames
using decord and imageio ffmpeg with PyTorch tensor integration.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import decord
    from decord import VideoReader, cpu
    _DECORD_AVAILABLE = True
except ImportError:
    _DECORD_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


def _natural_sort_key(s: str) -> List[Union[int, str]]:
    """Sort strings containing numbers in human natural order."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def load_video(
    path: Union[str, Path],
    max_frames: Optional[int] = None,
    fps: Optional[float] = None,
    resolution: Optional[Union[Tuple[int, int], List[int]]] = None,
    to_float: bool = True,
    device: str = "cpu",
) -> Tuple[torch.Tensor, float]:
    """Load video frames from disk into a PyTorch tensor.

    Uses decord for accelerated video decoding when available, with an automatic
    OpenCV fallback. Supports frame rate resampling, frame truncation, and spatial
    resizing.

    Args:
        path: Path to the video file.
        max_frames: Maximum number of frames to load. If None, load all sampled frames.
        fps: Target frame rate for temporal resampling. If specified and different
            from original fps, frames are sampled uniformly to match this fps.
        resolution: Target spatial resolution as (height, width) or (width, height).
            If (width, height) is passed where width > height for landscape video,
            dimensions are matched accordingly.
        to_float: If True, returns tensor with values scaled to [0.0, 1.0] as float32.
            If False, returns uint8 tensor with values in [0, 255].
        device: Device to place the resulting tensor on (default: 'cpu').

    Returns:
        Tuple containing:
            - frames: PyTorch tensor of shape [T, C, H, W].
            - original_fps: Frame rate of the source video as float.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If decoding fails with all available backends.
    """
    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path.resolve()}")

    frames_np: Optional[np.ndarray] = None
    original_fps: float = 30.0

    # 1. Try loading with decord
    if _DECORD_AVAILABLE:
        try:
            vr = VideoReader(str(video_path), ctx=cpu(0))
            original_fps = float(vr.get_avg_fps())
            if original_fps <= 0:
                original_fps = 30.0

            total_frames = len(vr)
            if total_frames == 0:
                raise ValueError("Video reader returned 0 frames.")

            # Calculate frame sampling indices
            if fps is not None and fps > 0 and abs(original_fps - fps) > 1e-2:
                duration_sec = total_frames / original_fps
                num_target_frames = max(1, int(round(duration_sec * fps)))
                sample_indices = np.linspace(0, total_frames - 1, num_target_frames, dtype=int)
            else:
                sample_indices = np.arange(total_frames, dtype=int)

            if max_frames is not None and max_frames > 0:
                sample_indices = sample_indices[:max_frames]

            frames_np = vr.get_batch(sample_indices).asnumpy()  # [T, H, W, C]
            logger.debug(
                "Loaded %d frames using decord from %s (orig fps: %.2f)",
                len(sample_indices),
                video_path.name,
                original_fps,
            )
        except Exception as exc:
            logger.warning("Decord failed to read %s: %s. Falling back to OpenCV.", video_path, exc)
            frames_np = None

    # 2. Fallback to OpenCV if decord unavailable or failed
    if frames_np is None:
        if not _CV2_AVAILABLE:
            raise RuntimeError(
                "Failed to decode video: decord failed and cv2 (opencv-python) is not installed."
            )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open video file: {video_path}")

        try:
            detected_fps = float(cap.get(cv2.CAP_PROP_FPS))
            original_fps = detected_fps if detected_fps > 0 else 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            all_frames: List[np.ndarray] = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                all_frames.append(frame_rgb)

            if not all_frames:
                raise RuntimeError(f"No frames could be read from {video_path}")

            all_frames_np = np.stack(all_frames, axis=0)  # [T_all, H, W, C]
            total_read = len(all_frames_np)

            # Sample frames based on fps
            if fps is not None and fps > 0 and abs(original_fps - fps) > 1e-2:
                duration_sec = total_read / original_fps
                num_target_frames = max(1, int(round(duration_sec * fps)))
                sample_indices = np.linspace(0, total_read - 1, num_target_frames, dtype=int)
                sampled_np = all_frames_np[sample_indices]
            else:
                sampled_np = all_frames_np

            if max_frames is not None and max_frames > 0:
                sampled_np = sampled_np[:max_frames]

            frames_np = sampled_np
            logger.debug(
                "Loaded %d frames using OpenCV fallback from %s (orig fps: %.2f)",
                len(frames_np),
                video_path.name,
                original_fps,
            )
        finally:
            cap.release()

    if frames_np is None or len(frames_np) == 0:
        raise RuntimeError(f"Could not extract any frames from {video_path}")

    # Convert numpy [T, H, W, C] to torch tensor [T, C, H, W]
    tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2)

    # Convert data type
    if to_float:
        tensor = tensor.to(dtype=torch.float32) / 255.0
    else:
        tensor = tensor.to(dtype=torch.uint8)

    # Resize if requested
    if resolution is not None:
        orig_h, orig_w = tensor.shape[2], tensor.shape[3]
        target_dim1, target_dim2 = int(resolution[0]), int(resolution[1])

        # Resolve (width, height) vs (height, width) ambiguity:
        # If input aspect is landscape (orig_w >= orig_h) and target_dim1 > target_dim2,
        # interpret resolution as [target_w, target_h].
        if orig_w >= orig_h and target_dim1 > target_dim2:
            target_w, target_h = target_dim1, target_dim2
        elif orig_h > orig_w and target_dim2 > target_dim1:
            target_w, target_h = target_dim1, target_dim2
        else:
            # Standard PyTorch order (height, width)
            target_h, target_w = target_dim1, target_dim2

        if (orig_h, orig_w) != (target_h, target_w):
            float_tensor = tensor.float() if tensor.dtype != torch.float32 else tensor
            resized = F.interpolate(
                float_tensor,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            if not to_float:
                tensor = resized.clamp(0, 255).to(torch.uint8)
            else:
                tensor = resized

    if device != "cpu":
        tensor = tensor.to(device)

    return tensor, original_fps


def save_video(
    frames: Union[torch.Tensor, np.ndarray],
    path: Union[str, Path],
    fps: int = 15,
    codec: str = "h264",
    pixelformat: str = "yuv420p",
) -> None:
    """Save video frames to disk using imageio ffmpeg backend.

    Args:
        frames: Video frames tensor of shape [T, C, H, W] or [T, H, W, C],
            or numpy array of shape [T, H, W, C]. Floats are assumed to be
            in [0.0, 1.0] or [-1.0, 1.0] and will be converted to uint8 [0, 255].
        path: Output file path for the video.
        fps: Playback frame rate (frames per second).
        codec: Video codec to encode with (default: 'h264').
        pixelformat: Pixel format for broad player compatibility (default: 'yuv420p').

    Raises:
        ValueError: If input tensor dimensions are invalid.
        IOError: If saving the video fails.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu()
        if frames.ndim == 4:
            # Detect [T, C, H, W] vs [T, H, W, C]
            if frames.shape[1] in (1, 3, 4) and frames.shape[1] < frames.shape[3]:
                frames = frames.permute(0, 2, 3, 1)
        elif frames.ndim != 3:
            raise ValueError(f"Expected 4D or 3D tensor, got shape {frames.shape}")

        if frames.is_floating_point():
            min_val = float(frames.min())
            max_val = float(frames.max())
            # Check if normalized in [-1, 1]
            if min_val < -0.01:
                frames = (frames + 1.0) / 2.0
            frames = (frames * 255.0).clamp(0, 255).to(torch.uint8)
        else:
            frames = frames.clamp(0, 255).to(torch.uint8)

        frames_np = frames.numpy()
    elif isinstance(frames, np.ndarray):
        if frames.ndim == 4 and frames.shape[1] in (1, 3, 4) and frames.shape[1] < frames.shape[3]:
            frames_np = np.transpose(frames, (0, 2, 3, 1))
        else:
            frames_np = frames

        if np.issubdtype(frames_np.dtype, np.floating):
            if frames_np.min() < -0.01:
                frames_np = (frames_np + 1.0) / 2.0
            frames_np = np.clip(frames_np * 255.0, 0, 255).astype(np.uint8)
        else:
            frames_np = np.clip(frames_np, 0, 255).astype(np.uint8)
    else:
        raise TypeError(f"Unsupported frames type: {type(frames)}")

    # Repeat single-channel grayscale to 3 channels for ffmpeg yuv420p compatibility
    if frames_np.ndim == 4 and frames_np.shape[-1] == 1:
        frames_np = np.repeat(frames_np, 3, axis=-1)

    try:
        writer = imageio.get_writer(
            str(output_path),
            fps=fps,
            codec=codec,
            pixelformat=pixelformat,
        )
        for frame in frames_np:
            writer.append_data(frame)
        writer.close()
        logger.info("Saved video with %d frames to %s", len(frames_np), output_path)
    except Exception as exc:
        logger.error("Failed to save video to %s: %s", output_path, exc)
        raise IOError(f"Failed to save video to {output_path}") from exc


def extract_frames(
    video_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    fps: Optional[float] = None,
    image_format: str = "png",
) -> Union[List[str], List[np.ndarray]]:
    """Extract frames from a video file and save them as individual images or return as list.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory where extracted frame images will be stored (optional).
        fps: Target frame rate to sample at. If None, extracts all frames.
        image_format: Format extension for saved images ('png', 'jpg').

    Returns:
        List of absolute file paths if output_dir is specified, otherwise list of RGB numpy arrays [H, W, C].
    """
    frames_tensor, _ = load_video(
        path=video_path,
        fps=fps,
        to_float=False,
    )  # [T, C, H, W], uint8

    num_frames = frames_tensor.shape[0]

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: List[str] = []
        for idx in range(num_frames):
            frame = frames_tensor[idx].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
            if frame.shape[2] == 1:
                frame = frame.squeeze(2)
            img = Image.fromarray(frame)
            out_file = out_dir / f"frame_{idx:06d}.{image_format}"
            img.save(out_file)
            saved_paths.append(str(out_file.resolve()))
        logger.info("Extracted %d frames from %s to %s", len(saved_paths), video_path, out_dir)
        return saved_paths
    else:
        frames_list: List[np.ndarray] = []
        for idx in range(num_frames):
            frame = frames_tensor[idx].permute(1, 2, 0).cpu().numpy()
            if frame.shape[2] == 1:
                frame = frame.squeeze(2)
            frames_list.append(frame)
        return frames_list



def frames_to_video(
    frame_dir: Union[str, Path],
    output_path: Union[str, Path],
    fps: int = 15,
    codec: str = "h264",
    pattern: str = "*.[pP][nN][gG]",
) -> str:
    """Combine individual image frames from a directory into a video file.

    Frames are sorted in natural numerical order (e.g., frame_1, frame_2, ..., frame_10).

    Args:
        frame_dir: Directory containing frame images.
        output_path: Destination path for the generated video file.
        fps: Playback frame rate.
        codec: Video codec (default: 'h264').
        pattern: Glob pattern to identify frame files.

    Returns:
        Absolute path to the created video file as a string.

    Raises:
        FileNotFoundError: If frame_dir does not exist or contains no matching images.
        IOError: If video writing fails.
    """
    dir_path = Path(frame_dir)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Frame directory does not exist: {dir_path.resolve()}")

    supported_exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    frame_files: List[Path] = []
    for ext in supported_exts:
        frame_files.extend(dir_path.glob(ext))
        frame_files.extend(dir_path.glob(ext.upper()))

    # Deduplicate and sort naturally
    unique_files = sorted(list({p.resolve() for p in frame_files}), key=lambda p: _natural_sort_key(p.name))

    if not unique_files:
        raise FileNotFoundError(f"No image frames found in directory: {dir_path.resolve()}")

    # Read all frames into numpy array
    frames_list: List[np.ndarray] = []
    for frame_file in unique_files:
        with Image.open(frame_file) as img:
            rgb_img = img.convert("RGB")
            frames_list.append(np.array(rgb_img, dtype=np.uint8))

    stacked_frames = np.stack(frames_list, axis=0)  # [T, H, W, C]
    save_video(frames=stacked_frames, path=output_path, fps=fps, codec=codec)

    logger.info("Combined %d frames into video %s", len(unique_files), output_path)
    return str(Path(output_path).resolve())
