from pathlib import Path
import argparse
import csv
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm


def load_image(path: Path, image_size: int) -> Image.Image:
    img = Image.open(path).convert("L")
    img = ImageOps.autocontrast(img)
    img = ImageOps.fit(img, (image_size, image_size), method=Image.Resampling.LANCZOS)
    return img


def add_scratch_and_dust(img: Image.Image, seed: int):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape

    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    # Scratches.
    for _ in range(rng.randint(2, 5)):
        x = rng.randint(5, w - 6)
        drift = rng.randint(-12, 12)
        y0 = rng.randint(0, int(0.25 * h))
        y1 = rng.randint(int(0.65 * h), h - 1)
        width = rng.choice([1, 1, 2, 2, 3])

        points = []
        for i in range(8):
            t = i / 7
            y = int(y0 * (1 - t) + y1 * t)
            x_t = int(x + drift * t + rng.randint(-3, 3))
            points.append((x_t, y))

        draw.line(points, fill=rng.randint(180, 255), width=width)

    # Dust / blotches.
    for _ in range(rng.randint(6, 16)):
        cx = rng.randint(0, w - 1)
        cy = rng.randint(0, h - 1)
        r = rng.randint(1, 6)
        fill = rng.randint(150, 240)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill)

    mask_arr = np.asarray(mask).astype(np.float32)
    mask_arr = cv2.GaussianBlur(mask_arr, (3, 3), 0)

    alpha = mask_arr / 255.0

    # Mild extra grain.
    grain = np_rng.normal(0, 6, size=arr.shape)
    degraded = arr + grain

    # Blend bright scratches/dust.
    degraded = degraded * (1 - alpha) + mask_arr * alpha
    degraded = np.clip(degraded, 0, 255).astype(np.uint8)

    binary_mask = (mask_arr > 20).astype(np.uint8) * 255

    return Image.fromarray(degraded, "L"), Image.fromarray(binary_mask, "L")


def add_grain(img: Image.Image, seed: int):
    np_rng = np.random.default_rng(seed)
    arr = np.asarray(img).astype(np.float32)
    noise = np_rng.normal(0, 18, size=arr.shape)
    degraded = np.clip(arr + noise, 0, 255).astype(np.uint8)

    # For grain, mask is the whole image.
    mask = np.ones_like(degraded, dtype=np.uint8) * 255
    return Image.fromarray(degraded, "L"), Image.fromarray(mask, "L")


def add_blur_contrast_loss(img: Image.Image, seed: int):
    rng = random.Random(seed)
    arr = np.asarray(img).astype(np.float32)

    k = rng.choice([3, 5, 7])
    degraded = cv2.GaussianBlur(arr, (k, k), 0)

    # Reduce contrast and add slight brightness shift.
    degraded = (degraded - 127.5) * 0.72 + 127.5
    degraded = degraded + rng.randint(-8, 8)
    degraded = np.clip(degraded, 0, 255).astype(np.uint8)

    mask = np.ones_like(degraded, dtype=np.uint8) * 255
    return Image.fromarray(degraded, "L"), Image.fromarray(mask, "L")


DEGRADATIONS = {
    "scratch_dust": add_scratch_and_dust,
    "grain": add_grain,
    "blur_contrast": add_blur_contrast_loss,
}


def main(
    cleanish_dir,
    output_dir,
    n_images,
    image_size,
    seed,
    degradations,
):
    cleanish_dir = Path(cleanish_dir)
    output_dir = Path(output_dir)

    original_dir = output_dir / "original"
    degraded_dir = output_dir / "degraded"
    mask_dir = output_dir / "masks"

    original_dir.mkdir(parents=True, exist_ok=True)
    degraded_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(cleanish_dir.glob("*.png"))
    if not paths:
        raise ValueError(f"No PNG files found in {cleanish_dir}")

    rng = random.Random(seed)
    rng.shuffle(paths)
    paths = paths[: min(n_images, len(paths))]

    selected_degradations = [d.strip() for d in degradations.split(",")]
    for d in selected_degradations:
        if d not in DEGRADATIONS:
            raise ValueError(f"Unknown degradation '{d}'. Choose from {list(DEGRADATIONS)}")

    manifest_path = output_dir / "metadata.csv"

    rows = []
    example_id = 0

    for image_path in tqdm(paths, desc="Creating synthetic restoration dataset"):
        original = load_image(image_path, image_size=image_size)

        for degradation_name in selected_degradations:
            example_id += 1
            example_seed = seed + example_id

            degraded, mask = DEGRADATIONS[degradation_name](original, seed=example_seed)

            base = f"example_{example_id:05d}_{image_path.stem}_{degradation_name}"

            original_path = original_dir / f"{base}_target.png"
            degraded_path = degraded_dir / f"{base}_input.png"
            mask_path = mask_dir / f"{base}_mask.png"

            original.save(original_path)
            degraded.save(degraded_path)
            mask.save(mask_path)

            rows.append({
                "example_id": f"example_{example_id:05d}",
                "source_frame": str(image_path),
                "degradation_type": degradation_name,
                "seed": example_seed,
                "target_path": str(original_path),
                "input_path": str(degraded_path),
                "mask_path": str(mask_path),
                "image_size": image_size,
            })

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "example_id",
            "source_frame",
            "degradation_type",
            "seed",
            "target_path",
            "input_path",
            "mask_path",
            "image_size",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {len(rows)} examples")
    print(f"Manifest: {manifest_path}")
    print(f"Original targets: {original_dir}")
    print(f"Degraded inputs: {degraded_dir}")
    print(f"Masks: {mask_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanish_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_images", type=int, default=50)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--degradations",
        default="scratch_dust,grain,blur_contrast",
        help="Comma-separated list: scratch_dust,grain,blur_contrast",
    )
    args = parser.parse_args()

    main(
        cleanish_dir=args.cleanish_dir,
        output_dir=args.output_dir,
        n_images=args.n_images,
        image_size=args.image_size,
        seed=args.seed,
        degradations=args.degradations,
    )