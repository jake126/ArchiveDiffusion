from pathlib import Path
import argparse
import csv
import math
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm


def load_gray(path: Path, size=(256, 192)) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = ImageOps.fit(img, size, method=Image.Resampling.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0


def high_frequency_score(arr: np.ndarray) -> float:
    blurred = cv2.GaussianBlur(arr, (5, 5), 0)
    residual = arr - blurred
    return float(np.mean(np.abs(residual)))


def laplacian_score(arr: np.ndarray) -> float:
    lap = cv2.Laplacian(arr, cv2.CV_32F)
    return float(np.var(lap))


def scratch_score(arr: np.ndarray) -> float:
    # Bright thin vertical-ish structures are a rough proxy for scratches.
    uint = (arr * 255).astype(np.uint8)
    edges = cv2.Canny(uint, 80, 160)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)

    return float(np.mean(vertical > 0))


def exposure_penalty(arr: np.ndarray) -> float:
    mean = float(np.mean(arr))
    std = float(np.std(arr))

    too_dark = max(0.0, 0.20 - mean)
    too_bright = max(0.0, mean - 0.85)
    too_flat = max(0.0, 0.10 - std)

    return too_dark + too_bright + too_flat


def score_frame(path: Path) -> dict:
    arr = load_gray(path)

    hf = high_frequency_score(arr)
    lap = laplacian_score(arr)
    scratch = scratch_score(arr)
    exposure = exposure_penalty(arr)
    brightness = float(np.mean(arr))
    contrast = float(np.std(arr))

    # Higher score means more degraded / less clean-ish.
    # We downweight Laplacian slightly because detail and damage can both increase it.
    degradation_score = (
        1.00 * hf
        + 0.15 * lap
        + 2.00 * scratch
        + 1.50 * exposure
    )

    return {
        "frame_path": str(path),
        "degradation_score": round(degradation_score, 6),
        "high_frequency_score": round(hf, 6),
        "laplacian_score": round(lap, 6),
        "scratch_score": round(scratch, 6),
        "exposure_penalty": round(exposure, 6),
        "brightness": round(brightness, 6),
        "contrast": round(contrast, 6),
    }


def copy_ranked(rows, output_dir: Path, limit: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    for row in rows[:limit]:
        src = Path(row["frame_path"])
        dst = output_dir / src.name
        shutil.copy2(src, dst)


def make_contact_sheet(rows, output_path: Path, title: str, max_images=100, thumb_width=180):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected = rows[:max_images]
    if not selected:
        print(f"No rows for {title}")
        return

    thumbs = []

    for rank, row in enumerate(selected, start=1):
        path = Path(row["frame_path"])
        img = Image.open(path).convert("L")
        img = ImageOps.autocontrast(img)

        w, h = img.size
        new_h = int(h * thumb_width / w)
        img = img.resize((thumb_width, new_h))

        label_h = 56
        canvas = Image.new("L", (thumb_width, new_h + label_h), color=255)
        canvas.paste(img, (0, 0))

        draw = ImageDraw.Draw(canvas)
        draw.text((4, new_h + 4), f"{rank}: {path.stem}", fill=0)
        draw.text((4, new_h + 20), f"deg={row['degradation_score']}", fill=0)
        draw.text((4, new_h + 36), f"hf={row['high_frequency_score']} sc={row['scratch_score']}", fill=0)

        thumbs.append(canvas)

    cols = 4
    rows_n = math.ceil(len(thumbs) / cols)
    cell_h = max(t.height for t in thumbs)

    sheet = Image.new("L", (cols * thumb_width, rows_n * cell_h + 40), color=255)
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill=0)

    y_offset = 40
    for i, thumb in enumerate(thumbs):
        x = (i % cols) * thumb_width
        y = (i // cols) * cell_h + y_offset
        sheet.paste(thumb, (x, y))

    sheet.save(output_path)


def main(
    input_dir,
    output_csv,
    cleanish_dir,
    degraded_dir,
    cleanish_sheet,
    degraded_sheet,
    n_cleanish,
    n_degraded,
):
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)
    cleanish_dir = Path(cleanish_dir)
    degraded_dir = Path(degraded_dir)
    cleanish_sheet = Path(cleanish_sheet)
    degraded_sheet = Path(degraded_sheet)

    frame_paths = sorted(input_dir.glob("*.png"))
    if not frame_paths:
        raise ValueError(f"No PNG files found in {input_dir}")

    rows = []
    for path in tqdm(frame_paths, desc="Scoring non-text frames"):
        rows.append(score_frame(path))

    rows_low_to_high = sorted(rows, key=lambda r: r["degradation_score"])
    rows_high_to_low = list(reversed(rows_low_to_high))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank_clean_to_degraded",
        "rank_degraded_to_clean",
        "frame_path",
        "degradation_score",
        "high_frequency_score",
        "laplacian_score",
        "scratch_score",
        "exposure_penalty",
        "brightness",
        "contrast",
        "manual_keep",
        "manual_notes",
    ]

    high_rank_lookup = {
        row["frame_path"]: idx + 1
        for idx, row in enumerate(rows_high_to_low)
    }

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(rows_low_to_high, start=1):
            writer.writerow({
                "rank_clean_to_degraded": idx,
                "rank_degraded_to_clean": high_rank_lookup[row["frame_path"]],
                **row,
                "manual_keep": "",
                "manual_notes": "",
            })

    copy_ranked(rows_low_to_high, cleanish_dir, n_cleanish)
    copy_ranked(rows_high_to_low, degraded_dir, n_degraded)

    make_contact_sheet(
        rows_low_to_high[:n_cleanish],
        cleanish_sheet,
        title=f"Clean-ish candidates: lowest {n_cleanish} degradation scores",
        max_images=n_cleanish,
    )

    make_contact_sheet(
        rows_high_to_low[:n_degraded],
        degraded_sheet,
        title=f"Naturally degraded candidates: highest {n_degraded} degradation scores",
        max_images=n_degraded,
    )

    print(f"Scored {len(rows)} frames")
    print(f"Wrote ranking CSV: {output_csv}")
    print(f"Copied clean-ish candidates to: {cleanish_dir}")
    print(f"Copied degraded candidates to: {degraded_dir}")
    print(f"Wrote clean-ish contact sheet: {cleanish_sheet}")
    print(f"Wrote degraded contact sheet: {degraded_sheet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--cleanish_dir", required=True)
    parser.add_argument("--degraded_dir", required=True)
    parser.add_argument("--cleanish_sheet", required=True)
    parser.add_argument("--degraded_sheet", required=True)
    parser.add_argument("--n_cleanish", type=int, default=100)
    parser.add_argument("--n_degraded", type=int, default=100)

    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        cleanish_dir=args.cleanish_dir,
        degraded_dir=args.degraded_dir,
        cleanish_sheet=args.cleanish_sheet,
        degraded_sheet=args.degraded_sheet,
        n_cleanish=args.n_cleanish,
        n_degraded=args.n_degraded,
    )