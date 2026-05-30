"""Evaluation metrics for ArchiveDiffusion."""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(clean: np.ndarray, restored: np.ndarray) -> float:
    """Compute PSNR between clean and restored images."""
    return float(peak_signal_noise_ratio(clean, restored, data_range=255))


def ssim(clean: np.ndarray, restored: np.ndarray) -> float:
    """Compute SSIM between clean and restored images."""
    channel_axis = -1 if clean.ndim == 3 else None
    return float(structural_similarity(clean, restored, data_range=255, channel_axis=channel_axis))
