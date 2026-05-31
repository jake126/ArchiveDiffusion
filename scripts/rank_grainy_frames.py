from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import csv
import math
import argparse


def grain_score(image_path):
    img = Image.open(image_path).convert("L")
    arr = np.asarray(img).astype(np.float32) / 255.0

    # Simple high-frequency proxy:
    # compare image to a local mean-smoothed version.
    padded = np.pad(arr, 1, mode="reflect")
    smooth = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
        padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
        padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    ) / 9.0

    residual = arr - smooth

    # Avoid selecting almost-black or almost-white frames only because of compression artefacts.
    contrast = arr.std()
    brightness = arr.mean()

    if brightness < 0.08 or brightness > 0.92 or contrast < 0.04:
        return -1.0

    return float(np.mean(np.abs(residual)))


def make_ranked_sheet(scored_paths, output_path, thumb_width=160, cols=5):
    thumbs = []

    for rank, (path, score) in enumerate(scored_paths, start=1):
        img = Image.open(path).convert("L")
        img = ImageOps.autocontrast(img)

        aspect = img.height / img.width
        thumb_height = int(thumb_width * aspect)
        img = img.resize((thumb_width, thumb_height))

        canvas = Image.new("L", (thumb_width, thumb_height + 36), color=255)
        canvas.paste(img, (0, 0))

        draw = ImageDraw.Draw(canvas)
        draw.text((4, thumb_height + 4), f"{rank}: {path.stem}", fill=0)
        draw.text((4, thumb_height + 18), f"grain={score:.5f}", fill=0)

        thumbs.append(canvas)

    rows = math.ceil(len(thumbs) / cols)
    cell_w = thumb_width
    cell_h = max(t.height for t in thumbs)

    sheet = Image.new("L", (cols * cell_w, rows * cell_h), color=255)

    for i, thumb in enumerate(thumbs):
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h
        sheet.paste(thumb, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main(input_dir, output_csv, output_sheet, top_k):
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)
    output_sheet = Path(output_sheet)

    rows = []
    for path in sorted(input_dir.glob("*.png")):
        rows.append((path, grain_score(path)))

    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    top = rows[:top_k]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "frame_path", "grain_score", "manual_keep", "notes"])
        for i, (path, score) in enumerate(rows, start=1):
            writer.writerow([i, str(path), score, "", ""])

    make_ranked_sheet(top, output_sheet)
    print(f"Saved ranked CSV to {output_csv}")
    print(f"Saved top-{top_k} contact sheet to {output_sheet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_sheet", required=True)
    parser.add_argument("--top_k", type=int, default=100)
    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        output_sheet=args.output_sheet,
        top_k=args.top_k,
    )