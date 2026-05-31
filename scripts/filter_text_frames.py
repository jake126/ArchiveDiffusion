from pathlib import Path
import argparse
import csv
import shutil
import re

import cv2
import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm


def preprocess_for_ocr(image_path: Path) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    arr = np.asarray(img)

    # Crop away outer border; old films often have black frame edges.
    h, w = arr.shape
    y0, y1 = int(0.04 * h), int(0.96 * h)
    x0, x1 = int(0.04 * w), int(0.96 * w)
    arr = arr[y0:y1, x0:x1]

    # Upscale for OCR.
    arr = cv2.resize(arr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # Normalize contrast.
    arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)

    # EasyOCR accepts numpy arrays. RGB is safest.
    arr_rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    return arr_rgb


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def useful_token_count(text: str) -> int:
    tokens = re.findall(r"[A-Za-z0-9]{2,}", text)
    return len(tokens)


def ocr_score(reader, image_path: Path, min_confidence: float) -> dict:
    image = preprocess_for_ocr(image_path)

    # detail=1 returns bounding boxes, text, confidence.
    results = reader.readtext(
        image,
        detail=1,
        paragraph=False,
        decoder="greedy",
        batch_size=1,
    )

    kept_texts = []
    confidences = []
    total_box_area = 0.0
    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w

    for box, text, conf in results:
        text = clean_text(text)

        if not text:
            continue

        if conf < min_confidence:
            continue

        token_count = useful_token_count(text)
        if token_count == 0:
            continue

        kept_texts.append(text)
        confidences.append(float(conf))

        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        box_area = max(0, max(xs) - min(xs)) * max(0, max(ys) - min(ys))
        total_box_area += box_area

    joined_text = " ".join(kept_texts)
    word_count = useful_token_count(joined_text)
    char_count = len(re.sub(r"[^A-Za-z0-9]", "", joined_text))
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    max_conf = float(np.max(confidences)) if confidences else 0.0
    text_area_fraction = float(total_box_area / max(1, img_area))

    # Title cards tend to have multiple confident words and visible text area.
    # This is intentionally conservative to avoid removing cinematic frames.
    auto_text_frame = int(
        (word_count >= 3 and char_count >= 10 and mean_conf >= min_confidence)
        or (word_count >= 2 and char_count >= 16 and max_conf >= 0.55)
        or (text_area_fraction >= 0.035 and word_count >= 2)
    )

    return {
        "joined_text": joined_text,
        "char_count": char_count,
        "word_count": word_count,
        "mean_conf": round(mean_conf, 4),
        "max_conf": round(max_conf, 4),
        "text_area_fraction": round(text_area_fraction, 5),
        "auto_text_frame": auto_text_frame,
    }


def make_review_sheet(rows, output_path: Path, max_images=120, thumb_width=180):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected = rows[:max_images]
    if not selected:
        print("No OCR-flagged frames to put in review sheet.")
        return

    thumbs = []

    for row in selected:
        path = Path(row["frame_path"])
        img = Image.open(path).convert("L")
        img = ImageOps.autocontrast(img)

        w, h = img.size
        new_h = int(h * thumb_width / w)
        img = img.resize((thumb_width, new_h))

        canvas = Image.new("L", (thumb_width, new_h + 64), color=255)
        canvas.paste(img, (0, 0))

        draw = ImageDraw.Draw(canvas)
        draw.text((4, new_h + 4), path.stem, fill=0)
        draw.text(
            (4, new_h + 20),
            f"words={row['word_count']} conf={row['mean_conf']}",
            fill=0,
        )
        draw.text(
            (4, new_h + 36),
            f"area={row['text_area_fraction']}",
            fill=0,
        )

        preview = str(row["joined_text"])[:28]
        draw.text((4, new_h + 50), preview, fill=0)

        thumbs.append(canvas)

    cols = 4
    rows_n = int(np.ceil(len(thumbs) / cols))
    cell_h = max(t.height for t in thumbs)

    sheet = Image.new("L", (cols * thumb_width, rows_n * cell_h), color=255)

    for i, thumb in enumerate(thumbs):
        x = (i % cols) * thumb_width
        y = (i // cols) * cell_h
        sheet.paste(thumb, (x, y))

    sheet.save(output_path)


def main(
    input_dir,
    output_csv,
    review_sheet,
    keep_dir,
    text_dir,
    min_confidence,
    gpu,
):
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)
    review_sheet = Path(review_sheet)

    if keep_dir:
        keep_dir = Path(keep_dir)
        keep_dir.mkdir(parents=True, exist_ok=True)

    if text_dir:
        text_dir = Path(text_dir)
        text_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(input_dir.glob("*.png"))
    if not frame_paths:
        raise ValueError(f"No PNG frames found in {input_dir}")

    print("Loading EasyOCR reader...")
    reader = easyocr.Reader(["en"], gpu=gpu)

    rows = []

    for path in tqdm(frame_paths, desc="OCR filtering frames"):
        result = ocr_score(reader, path, min_confidence=min_confidence)

        row = {
            "frame_path": str(path),
            **result,
            "manual_keep": "",
            "manual_notes": "",
        }
        rows.append(row)

        if keep_dir and not result["auto_text_frame"]:
            shutil.copy2(path, keep_dir / path.name)

        if text_dir and result["auto_text_frame"]:
            shutil.copy2(path, text_dir / path.name)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "frame_path",
        "joined_text",
        "char_count",
        "word_count",
        "mean_conf",
        "max_conf",
        "text_area_fraction",
        "auto_text_frame",
        "manual_keep",
        "manual_notes",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    text_rows = [r for r in rows if r["auto_text_frame"] == 1]
    text_rows = sorted(
        text_rows,
        key=lambda r: (
            r["word_count"],
            r["char_count"],
            r["text_area_fraction"],
            r["mean_conf"],
        ),
        reverse=True,
    )

    make_review_sheet(text_rows, review_sheet)

    n_text = sum(r["auto_text_frame"] for r in rows)
    n_keep = len(rows) - n_text

    print(f"Scanned {len(rows)} frames")
    print(f"Auto-flagged text/title frames: {n_text}")
    print(f"Remaining frames: {n_keep}")
    print(f"Wrote CSV: {output_csv}")
    print(f"Wrote review sheet: {review_sheet}")

    if keep_dir:
        print(f"Copied non-text frames to: {keep_dir}")
    if text_dir:
        print(f"Copied text frames to: {text_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--review_sheet", required=True)
    parser.add_argument("--keep_dir", default=None)
    parser.add_argument("--text_dir", default=None)
    parser.add_argument("--min_confidence", type=float, default=0.35)
    parser.add_argument("--gpu", action="store_true")

    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        review_sheet=args.review_sheet,
        keep_dir=args.keep_dir,
        text_dir=args.text_dir,
        min_confidence=args.min_confidence,
        gpu=args.gpu,
    )