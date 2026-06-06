from pathlib import Path
import argparse
import csv
import json
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from diffusers import UNet2DModel, DDPMScheduler


def load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("L")
    arr = np.asarray(img).astype(np.float32) / 255.0

    # Convert [0, 1] to [-1, 1].
    arr = arr * 2.0 - 1.0

    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor


class SyntheticRestorationDataset(Dataset):
    def __init__(self, metadata_path: Path, max_examples=None, seed=42):
        self.metadata_path = metadata_path

        with metadata_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        rng = random.Random(seed)
        rng.shuffle(rows)

        if max_examples is not None:
            rows = rows[:max_examples]

        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        target = load_image(Path(row["target_path"]))
        condition = load_image(Path(row["input_path"]))

        return {
            "target": target,
            "condition": condition,
            "example_id": row["example_id"],
            "degradation_type": row["degradation_type"],
        }


def build_model(image_size: int):
    model = UNet2DModel(
        sample_size=image_size,
        in_channels=2,   # noisy target + degraded condition
        out_channels=1,  # predicted noise for grayscale target
        layers_per_block=2,
        block_out_channels=(32, 64, 128, 128),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model


def main(
    dataset_dir,
    output_dir,
    image_size,
    batch_size,
    epochs,
    learning_rate,
    num_train_timesteps,
    max_examples,
    seed,
    save_every_epochs,
):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    metadata_path = dataset_dir / "metadata.csv"

    dataset = SyntheticRestorationDataset(
        metadata_path=metadata_path,
        max_examples=max_examples,
        seed=seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    model = build_model(image_size=image_size).to(device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    training_log_path = output_dir / "training_log.csv"

    with training_log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "step", "loss"])

    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")

        for batch in progress:
            clean = batch["target"].to(device)
            condition = batch["condition"].to(device)

            noise = torch.randn_like(clean)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (clean.shape[0],),
                device=device,
                dtype=torch.long,
            )

            noisy_clean = noise_scheduler.add_noise(clean, noise, timesteps)

            # Conditioning: concatenate noisy target and degraded input.
            model_input = torch.cat([noisy_clean, condition], dim=1)

            noise_pred = model(model_input, timesteps).sample

            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_value = float(loss.item())
            epoch_losses.append(loss_value)
            global_step += 1

            progress.set_postfix({"loss": f"{loss_value:.5f}"})

            with training_log_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, global_step, loss_value])

        mean_loss = float(np.mean(epoch_losses))
        print(f"Epoch {epoch} mean loss: {mean_loss:.6f}")

        if epoch % save_every_epochs == 0 or epoch == epochs:
            checkpoint_dir = output_dir / f"checkpoint_epoch_{epoch:04d}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            model.save_pretrained(checkpoint_dir / "unet")
            noise_scheduler.save_pretrained(checkpoint_dir / "scheduler")

            print(f"Saved checkpoint: {checkpoint_dir}")

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(final_dir / "unet")
    noise_scheduler.save_pretrained(final_dir / "scheduler")

    run_config = {
        "dataset_dir": str(dataset_dir),
        "metadata_path": str(metadata_path),
        "output_dir": str(output_dir),
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "num_train_timesteps": num_train_timesteps,
        "max_examples": max_examples,
        "seed": seed,
        "device": str(device),
        "model_type": "conditional_ddpm_unet2d",
        "conditioning": "channel_concat_noisy_target_plus_degraded_input",
    }

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"Saved final model to: {final_dir}")
    print(f"Saved training log to: {training_log_path}")
    print(f"Saved run config to: {output_dir / 'run_config.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every_epochs", type=int, default=10)

    args = parser.parse_args()

    main(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_train_timesteps=args.num_train_timesteps,
        max_examples=args.max_examples,
        seed=args.seed,
        save_every_epochs=args.save_every_epochs,
    )