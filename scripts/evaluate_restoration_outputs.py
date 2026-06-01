from pathlib import Path
import argparse
import csv
import math
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from tqdm import tqdm


def load_gray_float(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L")).astype(np.float32)
    return arr / 255.0


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    value = mse(a, b)
    if value <= 1e-12:
        return float("inf")
    return float(20 * math.log10(1.0 / math.sqrt(value)))


def ssim_simple(a: np.ndarray, b: np.ndarray) -> float:
    # Lightweight global SSIM approximation to avoid extra dependencies.
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_a = float(np.mean(a))
    mu_b = float(np.mean(b))
    var_a = float(np.var(a))
    var_b = float(np.var(b))
    cov = float(np.mean((a - mu_a) * (b - mu_b)))

    numerator = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    denominator = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float(numerator / denominator)


def high_frequency_energy(arr: np.ndarray) -> float:
    blurred = cv2.GaussianBlur(arr, (5, 5), 0)
    residual = arr - blurred
    return float(np.mean(np.abs(residual)))


def edge_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_uint = np.clip(a * 255, 0, 255).astype(np.uint8)
    b_uint = np.clip(b * 255, 0, 255).astype(np.uint8)

    edges_a = cv2.Canny(a_uint, 80, 160) > 0
    edges_b = cv2.Canny(b_uint, 80, 160) > 0

    intersection = np.logical_and(edges_a, edges_b).sum()
    union = np.logical_or(edges_a, edges_b).sum()

    if union == 0:
        return 1.0
    return float(intersection / union)


def masked_metric(metric_fn, pred, target, mask, use_mask=True):
    if use_mask:
        region = mask > 0.5
    else:
        region = mask <= 0.5

    if region.sum() == 0:
        return float("nan")

    return metric_fn(pred[region], target[region])


def evaluate_row(row):
    pred = load_gray_float(Path(row["prediction_path"]))
    target = load_gray_float(Path(row["target_path"]))
    inp = load_gray_float(Path(row["input_path"]))
    mask = load_gray_float(Path(row["mask_path"]))

    pred_hf = high_frequency_energy(pred)
    target_hf = high_frequency_energy(target)

    return {
        "example_id": row["example_id"],
        "method": row["method"],
        "degradation_type": row["degradation_type"],
        "mse": mse(pred, target),
        "mae": mae(pred, target),
        "psnr": psnr(pred, target),
        "ssim_simple": ssim_simple(pred, target),
        "edge_similarity": edge_similarity(pred, target),
        "masked_mae": masked_metric(mae, pred, target, mask, use_mask=True),
        "outside_mask_mae": masked_metric(mae, pred, target, mask, use_mask=False),
        "input_mae": mae(inp, target),
        "input_psnr": psnr(inp, target),
        "high_frequency_ratio": pred_hf / target_hf if target_hf > 1e-8 else float("nan"),
    }


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["degradation_type"])].append(row)

    metric_names = [
        "mse",
        "mae",
        "psnr",
        "ssim_simple",
        "edge_similarity",
        "masked_mae",
        "outside_mask_mae",
        "input_mae",
        "input_psnr",
        "high_frequency_ratio",
    ]

    summary = []

    for (method, degradation_type), group_rows in grouped.items():
        out = {
            "method": method,
            "degradation_type": degradation_type,
            "n_examples": len(group_rows),
        }

        for metric in metric_names:
            values = []
            for r in group_rows:
                try:
                    v = float(r[metric])
                    if not math.isnan(v) and not math.isinf(v):
                        values.append(v)
                except Exception:
                    pass

            if values:
                out[f"{metric}_mean"] = float(np.mean(values))
                out[f"{metric}_median"] = float(np.median(values))
            else:
                out[f"{metric}_mean"] = ""
                out[f"{metric}_median"] = ""

        summary.append(out)

    summary = sorted(summary, key=lambda r: (r["degradation_type"], r["method"]))
    return summary


def make_grid(prediction_rows, output_path: Path, max_examples=6):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Choose first few examples and show each method.
    example_ids = []
    for row in prediction_rows:
        if row["example_id"] not in example_ids:
            example_ids.append(row["example_id"])
        if len(example_ids) >= max_examples:
            break

    rows_by_example = defaultdict(list)
    for row in prediction_rows:
        if row["example_id"] in example_ids:
            rows_by_example[row["example_id"]].append(row)

    methods = sorted(set(r["method"] for r in prediction_rows))

    thumb = 128
    label_h = 34
    cols = 2 + len(methods)  # input, target, methods
    rows_n = len(example_ids)

    canvas = Image.new("RGB", (cols * thumb, rows_n * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(canvas)

    headers = ["input", "target"] + methods
    for c, header in enumerate(headers):
        draw.text((c * thumb + 4, 4), header, fill=(0, 0, 0))

    for r_idx, example_id in enumerate(example_ids):
        rows = rows_by_example[example_id]
        first = rows[0]

        images = [
            ("input", Path(first["input_path"])),
            ("target", Path(first["target_path"])),
        ]

        method_lookup = {r["method"]: r for r in rows}
        for method in methods:
            images.append((method, Path(method_lookup[method]["prediction_path"])))

        for c_idx, (_, path) in enumerate(images):
            img = Image.open(path).convert("L")
            img = ImageOps.autocontrast(img)
            img = img.resize((thumb, thumb), Image.Resampling.LANCZOS).convert("RGB")

            x = c_idx * thumb
            y = r_idx * (thumb + label_h) + label_h
            canvas.paste(img, (x, y))

        draw.text((4, r_idx * (thumb + label_h) + 18), example_id, fill=(80, 80, 80))

    canvas.save(output_path)


def main(prediction_manifest, output_metrics, output_summary, output_grid):
    prediction_manifest = Path(prediction_manifest)
    rows = read_csv(prediction_manifest)

    metrics_rows = []
    for row in tqdm(rows, desc="Evaluating restoration outputs"):
        metrics_rows.append(evaluate_row(row))

    write_csv(Path(output_metrics), metrics_rows)

    summary_rows = summarise(metrics_rows)
    write_csv(Path(output_summary), summary_rows)

    make_grid(rows, Path(output_grid))

    print(f"Wrote per-example metrics: {output_metrics}")
    print(f"Wrote summary metrics: {output_summary}")
    print(f"Wrote example grid: {output_grid}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_manifest", required=True)
    parser.add_argument("--output_metrics", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--output_grid", required=True)
    args = parser.parse_args()

    main(
        prediction_manifest=args.prediction_manifest,
        output_metrics=args.output_metrics,
        output_summary=args.output_summary,
        output_grid=args.output_grid,
    )