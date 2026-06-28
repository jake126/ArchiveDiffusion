from pathlib import Path
import argparse
import csv
import random
import time

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

from diffusers import UNet2DModel, DDPMScheduler


def load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("L")
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().squeeze().clamp(-1, 1)
    arr = (tensor.numpy() + 1.0) / 2.0
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "L")


def read_metadata(metadata_path: Path):
    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@torch.no_grad()
def restore_image(
    model,
    scheduler,
    condition,
    num_inference_steps,
    device,
    seed,
    correction_strength,
):
    generator = torch.Generator(device=device).manual_seed(seed)

    b, _, h, w = condition.shape

    residual_sample = torch.randn(
        (b, 1, h, w),
        generator=generator,
        device=device,
    )

    scheduler.set_timesteps(num_inference_steps)

    for t in scheduler.timesteps:
        model_input = torch.cat([residual_sample, condition], dim=1)
        noise_pred = model(model_input, t).sample
        residual_sample = scheduler.step(noise_pred, t, residual_sample).prev_sample

    # Undo residual scaling: residual_scaled = (target - condition) / 2
    predicted_residual = 2.0 * residual_sample.clamp(-1, 1)

    restored = condition + correction_strength * predicted_residual
    restored = restored.clamp(-1, 1)

    return restored


def main(
    model_dir,
    dataset_dir,
    metadata_file,
    split,
    output_dir,
    method_name,
    num_inference_steps,
    correction_strength,
    max_examples,
    seed,
):
    model_dir = Path(model_dir)
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    prediction_dir = output_dir / method_name
    prediction_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(metadata_file)
    if not metadata_path.is_absolute():
        metadata_path = dataset_dir / metadata_path

    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find metadata file: {metadata_path}")

    rows = read_metadata(metadata_path)

    if split is not None:
        if rows and "split" not in rows[0]:
            raise ValueError(
                f"Requested split='{split}', but metadata file has no 'split' column: {metadata_path}"
            )
        rows = [row for row in rows if row["split"] == split]

    if not rows:
        raise ValueError(f"No metadata rows found for split={split} in {metadata_path}")

    print(f"Sampling {len(rows)} examples from metadata: {metadata_path}")
    if split is not None:
        print(f"Using split: {split}")

    if max_examples is not None:
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:max_examples]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet2DModel.from_pretrained(model_dir / "unet").to(device)
    scheduler = DDPMScheduler.from_pretrained(model_dir / "scheduler")

    model.eval()

    manifest_rows = []

    for idx, row in enumerate(tqdm(rows, desc="Sampling conditional DDPM restorations")):
        condition = load_image(Path(row["input_path"])).to(device)

        start = time.time()

        restored = restore_image(
            model=model,
            scheduler=scheduler,
            condition=condition,
            num_inference_steps=num_inference_steps,
            device=device,
            seed=seed + idx,
            correction_strength=correction_strength
        )

        runtime_seconds = time.time() - start

        pred_path = prediction_dir / f"{row['example_id']}_{row['degradation_type']}_{method_name}_prediction.png"
        tensor_to_image(restored).save(pred_path)

        manifest_rows.append({
            "example_id": row["example_id"],
            "method": method_name,
            "degradation_type": row["degradation_type"],
            "input_path": row["input_path"],
            "target_path": row["target_path"],
            "mask_path": row["mask_path"],
            "prediction_path": str(pred_path),
            "num_inference_steps": num_inference_steps,
            "runtime_seconds": round(runtime_seconds, 6),
            "split": row.get("split", ""),
            "correction_strength": correction_strength,
        })

    manifest_path = output_dir / "prediction_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "example_id",
            "method",
            "degradation_type",
            "input_path",
            "target_path",
            "mask_path",
            "prediction_path",
            "num_inference_steps",
            "runtime_seconds",
            "split",
            "correction_strength",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote predictions to: {prediction_dir}")
    print(f"Wrote prediction manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method_name", default="conditional_ddpm_100_steps")
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    # validation
    parser.add_argument("--metadata_file", default="metadata.csv")
    parser.add_argument("--split", default=None)
    # residual training
    parser.add_argument("--correction_strength", type=float, default=1.0)

    args = parser.parse_args()

    main(
        model_dir=args.model_dir,
        dataset_dir=args.dataset_dir,
        metadata_file=args.metadata_file,
        split=args.split,
        output_dir=args.output_dir,
        method_name=args.method_name,
        num_inference_steps=args.num_inference_steps,
        max_examples=args.max_examples,
        seed=args.seed,
        correction_strength=args.correction_strength,
    )