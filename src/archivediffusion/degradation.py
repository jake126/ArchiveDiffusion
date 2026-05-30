"""Synthetic degradation functions for ArchiveDiffusion.

These are placeholders for the first implementation pass.
"""

from __future__ import annotations

from PIL import Image, ImageFilter
import numpy as np


def add_gaussian_noise(image: Image.Image, sigma: float = 20.0) -> Image.Image:
    """Add Gaussian noise to a PIL image."""
    arr = np.asarray(image).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    degraded = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(degraded)


def add_blur(image: Image.Image, radius: float = 1.5) -> Image.Image:
    """Apply Gaussian blur to a PIL image."""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def reduce_contrast(image: Image.Image, factor: float = 0.7) -> Image.Image:
    """Reduce contrast around the image mean."""
    arr = np.asarray(image).astype(np.float32)
    mean = arr.mean(axis=(0, 1), keepdims=True)
    degraded = np.clip((arr - mean) * factor + mean, 0, 255).astype(np.uint8)
    return Image.fromarray(degraded)
