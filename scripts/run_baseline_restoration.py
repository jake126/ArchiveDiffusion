from pathlib import Path
import argparse
import csv

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")).astype(np.uint8)


def save_gray(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "L").save(path)


def baseline_unchanged(input_arr: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
    return input_arr.copy()


def baseline_median(input_arr: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
    return cv2.medianBlur(input_arr, 3)


def baseline_inpaint(input_arr: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
    # OpenCV inpainting expects non-zero mask where pixels need restoration.
    mask = (mask_arr > 0).astype(np.uint8) * 255

    # If mask is basically the whole image, inpainting is inappropriate.
    # Fall back to denoising for whole-image degradations like grain/blur.
    if np.mean(mask > 0) > 0.40:
        return cv2.fastNlMeansDenoising(input_arr, None, h=8, templateWindowSize=7, searchWindowSize=21)

    return cv2.inpaint(input_arr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def baseline_denoise(input_arr: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(input_arr, None, h=8, templateWindowSize=7, searchWindowSize=21)


BASELINES = {
    "unchanged": baseline_unchanged,
    "median": baseline_median,
    "inpaint_or_denoise": baseline_inpaint,
    "nl_means_denoise": baseline_denoise,
}


def read_metadata(metadata_path: Path):
    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(dataset_dir, output_dir, methods):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    metadata_path = dataset_dir / "metadata.csv"

    rows = read_metadata(metadata_path)

    selected_methods = [m.strip() for m in methods.split(",")]
    for method in selected_methods:
        if method not in BASELINES:
            raise ValueError(f"Unknown method '{method}'. Choose from {list(BASELINES)}")

    prediction_manifest = []

    for row in tqdm(rows, desc="Running baseline restoration"):
        input_arr = load_gray(Path(row["input_path"]))
        mask_arr = load_gray(Path(row["mask_path"]))

        for method in selected_methods:
            pred_arr = BASELINES[method](input_arr, mask_arr)

            pred_path = output_dir / method / f"{row['example_id']}_{row['degradation_type']}_prediction.png"
            save_gray(pred_arr, pred_path)

            prediction_manifest.append({
                "example_id": row["example_id"],
                "method": method,
                "degradation_type": row["degradation_type"],
                "input_path": row["input_path"],
                "target_path": row["target_path"],
                "mask_path": row["mask_path"],
                "prediction_path": str(pred_path),
            })

    manifest_path = output_dir / "prediction_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "example_id",
            "method",
            "degradation_type",
            "input_path",
            "target_path",
            "mask_path",
            "prediction_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_manifest)

    print(f"Wrote predictions to: {output_dir}")
    print(f"Wrote prediction manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--methods",
        default="unchanged,median,inpaint_or_denoise,nl_means_denoise",
    )
    args = parser.parse_args()

    main(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        methods=args.methods,
    )