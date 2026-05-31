from pathlib import Path
import argparse
import random
import csv

import cv2
import numpy as np
from PIL import Image, ImageDraw


def load_gray(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def add_scratches_and_blotches(img: Image.Image, seed: int = 42):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    arr = np.array(img).astype(np.float32)
    h, w = arr.shape

    mask = Image.new("L", (w, h), 0)
    mask_draw = ImageDraw.Draw(mask)

    # Add vertical / diagonal scratch lines.
    n_scratches = rng.randint(3, 6)
    for _ in range(n_scratches):
        x = rng.randint(int(0.05 * w), int(0.95 * w))
        y0 = rng.randint(0, int(0.20 * h))
        y1 = rng.randint(int(0.70 * h), h)
        drift = rng.randint(-20, 20)
        width = rng.choice([1, 1, 2, 2, 3])
        brightness = rng.randint(190, 245)

        points = []
        steps = 8
        for i in range(steps):
            t = i / (steps - 1)
            y = int(y0 * (1 - t) + y1 * t)
            x_t = int(x + drift * t + rng.randint(-4, 4))
            points.append((x_t, y))

        mask_draw.line(points, fill=brightness, width=width)

    # Add dust / blotches.
    n_blotches = rng.randint(8, 18)
    for _ in range(n_blotches):
        cx = rng.randint(0, w - 1)
        cy = rng.randint(0, h - 1)
        r = rng.randint(2, 9)
        fill = rng.randint(150, 235)

        mask_draw.ellipse(
            (cx - r, cy - r, cx + r, cy + r),
            fill=fill,
        )

    mask_arr = np.array(mask).astype(np.float32)

    # Blur mask slightly so blemishes are not perfectly digital.
    mask_arr = cv2.GaussianBlur(mask_arr, (3, 3), 0)

    # Add slight extra grain.
    grain = np_rng.normal(0, 8, size=arr.shape)

    degraded = arr.copy()
    degraded = degraded + grain

    # Blend blemishes in. Bright scratches/dust are common in old film scans.
    alpha = mask_arr / 255.0
    degraded = degraded * (1 - alpha) + mask_arr * alpha

    degraded = np.clip(degraded, 0, 255).astype(np.uint8)
    mask_arr = np.clip(mask_arr, 0, 255).astype(np.uint8)

    return Image.fromarray(degraded, mode="L"), Image.fromarray(mask_arr, mode="L")


def choose_example(cleanish_dir: Path, index: int) -> Path:
    paths = sorted(cleanish_dir.glob("*.png"))
    if not paths:
        raise ValueError(f"No PNG files found in {cleanish_dir}")
    index = min(max(index, 0), len(paths) - 1)
    return paths[index]


def main(cleanish_dir, output_dir, example_index, seed):
    cleanish_dir = Path(cleanish_dir)
    output_dir = Path(output_dir)

    original_dir = output_dir / "original"
    degraded_dir = output_dir / "degraded"
    mask_dir = output_dir / "masks"

    original_dir.mkdir(parents=True, exist_ok=True)
    degraded_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    source_path = choose_example(cleanish_dir, example_index)
    original = load_gray(source_path)
    degraded, mask = add_scratches_and_blotches(original, seed=seed)

    stem = source_path.stem
    original_path = original_dir / f"{stem}_original.png"
    degraded_path = degraded_dir / f"{stem}_synthetic_blemish.png"
    mask_path = mask_dir / f"{stem}_blemish_mask.png"

    original.save(original_path)
    degraded.save(degraded_path)
    mask.save(mask_path)

    manifest_path = output_dir / "synthetic_blemish_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_frame",
            "original_path",
            "degraded_path",
            "mask_path",
            "seed",
            "degradation_type",
        ])
        writer.writerow([
            str(source_path),
            str(original_path),
            str(degraded_path),
            str(mask_path),
            seed,
            "scratches; dust; mild grain",
        ])

    print(f"Source frame: {source_path}")
    print(f"Original target: {original_path}")
    print(f"Synthetic blemish input: {degraded_path}")
    print(f"Blemish mask: {mask_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanish_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--example_index", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        cleanish_dir=args.cleanish_dir,
        output_dir=args.output_dir,
        example_index=args.example_index,
        seed=args.seed,
    )