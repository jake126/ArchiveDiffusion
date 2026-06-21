from pathlib import Path
import argparse
import csv
import random
from collections import defaultdict


def infer_source_frame(row):
    for key in ["source_frame", "source_frame_id", "original_frame", "frame_id"]:
        if key in row and row[key]:
            return row[key]

    target_path = row.get("target_path", "")
    if target_path:
        return Path(target_path).stem

    example_id = row.get("example_id", "")
    if example_id:
        # Fallback: keep this conservative.
        # If example_id already encodes a source frame, this will group correctly.
        return example_id.split("_scratch_dust")[0].split("_grain")[0].split("_blur_contrast")[0]

    raise ValueError(f"Could not infer source frame from row: {row}")


def main(
    metadata_path,
    output_path,
    train_fraction,
    val_fraction,
    test_fraction,
    seed,
):
    metadata_path = Path(metadata_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if abs((train_fraction + val_fraction + test_fraction) - 1.0) > 1e-6:
        raise ValueError("Train/val/test fractions must sum to 1.0")

    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in {metadata_path}")

    grouped = defaultdict(list)

    for row in rows:
        source_frame = infer_source_frame(row)
        row["source_frame"] = source_frame
        grouped[source_frame].append(row)

    source_frames = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(source_frames)

    n_total = len(source_frames)
    n_train = int(round(n_total * train_fraction))
    n_val = int(round(n_total * val_fraction))

    train_frames = set(source_frames[:n_train])
    val_frames = set(source_frames[n_train:n_train + n_val])
    test_frames = set(source_frames[n_train + n_val:])

    output_rows = []

    for source_frame, frame_rows in grouped.items():
        if source_frame in train_frames:
            split = "train"
        elif source_frame in val_frames:
            split = "val"
        elif source_frame in test_frames:
            split = "test"
        else:
            raise RuntimeError(f"Frame not assigned to split: {source_frame}")

        for row in frame_rows:
            row["split"] = split
            output_rows.append(row)

    fieldnames = list(output_rows[0].keys())

    # Put useful columns near the front if present.
    preferred = [
        "example_id",
        "source_frame",
        "split",
        "degradation_type",
        "input_path",
        "target_path",
        "mask_path",
    ]
    fieldnames = preferred + [c for c in fieldnames if c not in preferred]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    counts = defaultdict(int)
    source_counts = defaultdict(set)

    for row in output_rows:
        counts[row["split"]] += 1
        source_counts[row["split"]].add(row["source_frame"])

    print(f"Wrote split metadata to: {output_path}")
    print("Example counts:")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {counts[split]} examples from {len(source_counts[split])} source frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--train_fraction", type=float, default=0.70)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    main(
        metadata_path=args.metadata_path,
        output_path=args.output_path,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )