"""Image processing utilities for AI Motion Transfer.

Provides functions for loading, resizing, cropping, normalizing, denormalizing,
and creating visual comparison grids for diffusion models.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def load_image(
    path: Union[str, Path],
    resolution: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    """Load an image from disk, convert to RGB, and optionally resize.

    Handles EXIF orientation tags automatically to prevent accidental rotations.

    Args:
        path: Path to the image file.
        resolution: Optional target size as (width, height) to resize into.

    Returns:
        PIL Image in RGB mode.

    Raises:
        FileNotFoundError: If the image path does not exist.
        IOError: If the image cannot be opened or decoded.
    """
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path.resolve()}")

    try:
        with Image.open(image_path) as img:
            # Correct EXIF orientation if present
            img = ImageOps.exif_transpose(img)
            rgb_img = img.convert("RGB")

            if resolution is not None:
                target_w, target_h = int(resolution[0]), int(resolution[1])
                rgb_img = rgb_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

            return rgb_img
    except Exception as exc:
        logger.error("Failed to load image from %s: %s", image_path, exc)
        raise IOError(f"Could not load image: {image_path}") from exc


def resize_and_pad(
    image: Image.Image,
    target_size: Tuple[int, int],
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Resize image to fit within target size while preserving aspect ratio, then pad.

    Args:
        image: Source PIL Image.
        target_size: Target dimensions as (width, height).
        pad_color: RGB tuple for padding fill color (default: black (0, 0, 0)).

    Returns:
        New PIL Image with dimensions matching target_size.
    """
    orig_w, orig_h = image.size
    target_w, target_h = int(target_size[0]), int(target_size[1])

    if orig_w == target_w and orig_h == target_h:
        return image.copy()

    # Calculate uniform scaling factor to fit within bounding box
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    resized_img = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Create padded background and paste resized image in the center
    padded_img = Image.new("RGB", (target_w, target_h), color=pad_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    padded_img.paste(resized_img, (paste_x, paste_y))

    return padded_img


def center_crop(
    image: Image.Image,
    size: Tuple[int, int],
) -> Image.Image:
    """Resize image preserving aspect ratio to cover target size, then center crop.

    Args:
        image: Source PIL Image.
        size: Target crop dimensions as (width, height).

    Returns:
        Center-cropped PIL Image of exact dimensions size.
    """
    orig_w, orig_h = image.size
    target_w, target_h = int(size[0]), int(size[1])

    if orig_w == target_w and orig_h == target_h:
        return image.copy()

    # Scale so that both dimensions are at least target dimensions
    scale = max(target_w / orig_w, target_h / orig_h)
    scaled_w = max(target_w, int(round(orig_w * scale)))
    scaled_h = max(target_h, int(round(orig_h * scale)))

    scaled_img = image.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

    # Compute crop coordinates
    left = (scaled_w - target_w) // 2
    top = (scaled_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    return scaled_img.crop((left, top, right, bottom))


def normalize_image(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
) -> torch.Tensor:
    """Convert an image to a PyTorch tensor normalized to [-1.0, 1.0].

    Args:
        image: Input image as PIL Image, numpy array [H, W, C], or torch Tensor
            [C, H, W] / [B, C, H, W].

    Returns:
        Float32 PyTorch tensor normalized to [-1.0, 1.0].
        Single images have shape [C, H, W]; batched images retain [B, C, H, W].

    Raises:
        TypeError: If input type is unsupported.
        ValueError: If tensor or array dimensions are invalid.
    """
    if isinstance(image, Image.Image):
        rgb_img = image.convert("RGB")
        np_img = np.array(rgb_img, dtype=np.float32)  # [H, W, 3] in [0, 255]
        tensor = torch.from_numpy(np_img).permute(2, 0, 1)  # [3, H, W]
        tensor = (tensor / 127.5) - 1.0
        return tensor

    if isinstance(image, np.ndarray):
        np_arr = image.astype(np.float32)
        if np_arr.ndim == 2:  # Grayscale [H, W]
            np_arr = np.expand_dims(np_arr, axis=-1)

        if np_arr.ndim == 3:  # [H, W, C]
            tensor = torch.from_numpy(np_arr).permute(2, 0, 1)
        elif np_arr.ndim == 4:  # [B, H, W, C]
            tensor = torch.from_numpy(np_arr).permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Expected 2D, 3D, or 4D numpy array, got ndim={np_arr.ndim}")

        # Check existing range
        val_min = float(tensor.min())
        val_max = float(tensor.max())
        if val_min >= -1.01 and val_max <= 1.01 and val_min < -0.01:
            return tensor.clamp(-1.0, 1.0)
        elif val_max > 1.0:
            return (tensor / 127.5) - 1.0
        else:
            # Assume [0, 1]
            return (tensor * 2.0) - 1.0

    if isinstance(image, torch.Tensor):
        tensor = image.detach().float()
        val_min = float(tensor.min())
        val_max = float(tensor.max())

        if val_min >= -1.01 and val_max <= 1.01 and val_min < -0.01:
            return tensor.clamp(-1.0, 1.0)
        elif val_max > 1.0:
            return (tensor / 127.5) - 1.0
        else:
            # Assume [0, 1]
            return (tensor * 2.0) - 1.0

    raise TypeError(f"Unsupported input type for normalize_image: {type(image)}")


def denormalize_image(
    tensor: Union[torch.Tensor, np.ndarray],
) -> Image.Image:
    """Convert a normalized PyTorch tensor or numpy array to a PIL Image.

    Supports inputs in [-1.0, 1.0] or [0.0, 1.0] with shapes [C, H, W], [1, C, H, W],
    or [H, W, C].

    Args:
        tensor: Normalized image tensor or array.

    Returns:
        PIL Image in RGB mode.

    Raises:
        ValueError: If input dimensions are invalid.
    """
    if isinstance(tensor, torch.Tensor):
        t = tensor.detach().cpu().float()
        if t.ndim == 4:
            if t.shape[0] != 1:
                logger.warning("denormalize_image received batch size %d; taking first image.", t.shape[0])
            t = t[0]
        if t.ndim != 3:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(tensor.shape)}")

        # Check if [C, H, W]
        if t.shape[0] in (1, 3, 4) and t.shape[0] < t.shape[2]:
            t = t.permute(1, 2, 0)  # [H, W, C]

        val_min = float(t.min())
        if val_min < -0.01:
            # Scale from [-1, 1] to [0, 255]
            t = ((t + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        elif float(t.max()) <= 1.01:
            # Scale from [0, 1] to [0, 255]
            t = (t * 255.0).clamp(0, 255).to(torch.uint8)
        else:
            t = t.clamp(0, 255).to(torch.uint8)

        np_img = t.numpy()
    elif isinstance(tensor, np.ndarray):
        arr = tensor.astype(np.float32)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[2]:
            arr = np.transpose(arr, (1, 2, 0))

        if arr.min() < -0.01:
            arr = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
        elif arr.max() <= 1.01:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        np_img = arr
    else:
        raise TypeError(f"Unsupported tensor type: {type(tensor)}")

    if np_img.ndim == 3 and np_img.shape[-1] == 1:
        np_img = np_img.squeeze(-1)
        return Image.fromarray(np_img, mode="L").convert("RGB")
    elif np_img.ndim == 3 and np_img.shape[-1] == 3:
        return Image.fromarray(np_img, mode="RGB")
    elif np_img.ndim == 2:
        return Image.fromarray(np_img, mode="L").convert("RGB")
    else:
        raise ValueError(f"Cannot convert array of shape {np_img.shape} to PIL Image.")


def create_grid(
    images: Union[Sequence[Union[Image.Image, torch.Tensor, np.ndarray]], torch.Tensor],
    nrow: int = 4,
    padding: int = 4,
    pad_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Arrange multiple images into a grid image for visual comparison.

    Converts heterogeneous inputs (PIL Images, tensors, arrays) to standard PIL Images,
    resizes to a uniform tile size, and arranges them in rows and columns.

    Args:
        images: Sequence of PIL images, tensors, or numpy arrays; or a 4D PyTorch tensor.
        nrow: Number of images per row in the grid.
        padding: Padding width in pixels between tiles.
        pad_color: RGB tuple for the grid border/padding color (default: white (255, 255, 255)).

    Returns:
        PIL Image containing the assembled grid.

    Raises:
        ValueError: If images sequence is empty.
    """
    pil_images: List[Image.Image] = []

    if isinstance(images, torch.Tensor) and images.ndim == 4:
        for idx in range(images.shape[0]):
            pil_images.append(denormalize_image(images[idx]))
    elif isinstance(images, Sequence):
        for item in images:
            if isinstance(item, Image.Image):
                pil_images.append(item.convert("RGB"))
            elif isinstance(item, (torch.Tensor, np.ndarray)):
                pil_images.append(denormalize_image(item))
            else:
                raise TypeError(f"Unsupported item in images sequence: {type(item)}")
    else:
        raise TypeError(f"Unsupported images collection type: {type(images)}")

    if not pil_images:
        raise ValueError("Cannot create grid from empty image sequence.")

    num_images = len(pil_images)
    nrow = max(1, min(nrow, num_images))
    ncol = math.ceil(num_images / nrow)

    # Determine uniform tile size using maximum dimensions among inputs
    tile_w = max(img.width for img in pil_images)
    tile_h = max(img.height for img in pil_images)

    # Compute overall canvas size including padding
    grid_w = nrow * tile_w + (nrow + 1) * padding
    grid_h = ncol * tile_h + (ncol + 1) * padding

    grid_canvas = Image.new("RGB", (grid_w, grid_h), color=pad_color)

    for idx, img in enumerate(pil_images):
        col_idx = idx % nrow
        row_idx = idx // nrow

        # Resize image to tile dimensions if necessary
        if img.size != (tile_w, tile_h):
            tile_img = resize_and_pad(img, (tile_w, tile_h), pad_color=pad_color)
        else:
            tile_img = img

        x_offset = padding + col_idx * (tile_w + padding)
        y_offset = padding + row_idx * (tile_h + padding)
        grid_canvas.paste(tile_img, (x_offset, y_offset))

    return grid_canvas
