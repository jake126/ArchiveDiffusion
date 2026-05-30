"""Dataset utilities for ArchiveDiffusion."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image


def list_images(root: str | Path, suffixes: Iterable[str] = (".png", ".jpg", ".jpeg")) -> list[Path]:
    """Return image paths under a directory."""
    root = Path(root)
    suffixes = tuple(s.lower() for s in suffixes)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in suffixes)


def load_image(path: str | Path, image_size: int = 64, grayscale: bool = True) -> Image.Image:
    """Load and resize an image for training."""
    mode = "L" if grayscale else "RGB"
    image = Image.open(path).convert(mode)
    return image.resize((image_size, image_size), Image.Resampling.LANCZOS)
