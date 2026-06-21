from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def find_column(df, options):
    for option in options:
        if option in df.columns:
            return option
    return None


def load_training_log(training_log):
    df = pd.read_csv(training_log)

    epoch_col = find_column(df, ["epoch"])
    step_col = find_column(df, ["step", "global_step"])
    loss_col = find_column(df, ["loss", "train_loss", "train_loss_step"])

    if loss_col is None:
        raise ValueError(
            f"Could not find training loss column in {training_log}. "
            f"Columns: {list(df.columns)}"
        )

    if step_col is None:
        df["step"] = range(1, len(df) + 1)
        step_col = "step"

    return df, epoch_col, step_col, loss_col


def epoch_mean_training_loss(df, epoch_col, loss_col):
    if epoch_col is None:
        return None
    return (
        df.groupby(epoch_col, as_index=False)[loss_col]
        .mean()
        .rename(columns={loss_col: "train_loss_epoch_mean"})
    )


def main(run_dir, output_path, title, rolling_window):
    run_dir = Path(run_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    training_log = run_dir / "training_log.csv"
    validation_log = run_dir / "validation_log.csv"

    if not training_log.exists():
        raise FileNotFoundError(f"Could not find training log: {training_log}")

    train_df, epoch_col, step_col, loss_col = load_training_log(training_log)

    train_df["loss_rolling"] = train_df[loss_col].rolling(
        window=rolling_window,
        min_periods=1,
    ).mean()

    plt.figure(figsize=(10, 6))

    plt.plot(
        train_df[step_col],
        train_df[loss_col],
        alpha=0.25,
        label="train loss per step",
    )
    plt.plot(
        train_df[step_col],
        train_df["loss_rolling"],
        label=f"train rolling mean ({rolling_window} steps)",
    )

    train_epoch_df = epoch_mean_training_loss(train_df, epoch_col, loss_col)

    if train_epoch_df is not None:
        # Plot epoch means on a secondary x interpretation by mapping epoch end step.
        epoch_end_steps = train_df.groupby(epoch_col, as_index=False)[step_col].max()
        train_epoch_df = train_epoch_df.merge(epoch_end_steps, on=epoch_col, how="left")

        plt.plot(
            train_epoch_df[step_col],
            train_epoch_df["train_loss_epoch_mean"],
            marker="o",
            linewidth=1.5,
            label="train epoch mean",
        )

    if validation_log.exists():
        val_df = pd.read_csv(validation_log)
        val_epoch_col = find_column(val_df, ["epoch"])
        val_loss_col = find_column(val_df, ["val_loss", "loss", "validation_loss"])

        if val_epoch_col is not None and val_loss_col is not None and train_epoch_df is not None:
            val_plot_df = val_df.merge(
                train_df.groupby(epoch_col, as_index=False)[step_col].max(),
                left_on=val_epoch_col,
                right_on=epoch_col,
                how="left",
            )

            plt.plot(
                val_plot_df[step_col],
                val_plot_df[val_loss_col],
                marker="s",
                linewidth=2,
                label="validation loss",
            )
        elif val_epoch_col is not None and val_loss_col is not None:
            plt.plot(
                val_df[val_epoch_col],
                val_df[val_loss_col],
                marker="s",
                linewidth=2,
                label="validation loss by epoch",
            )
        else:
            plt.text(
                0.02,
                0.95,
                "Validation log found, but columns were not recognised.",
                transform=plt.gca().transAxes,
                va="top",
            )
    else:
        plt.text(
            0.02,
            0.95,
            "Validation loss unavailable for this run.",
            transform=plt.gca().transAxes,
            va="top",
        )

    plt.title(title)
    plt.xlabel("training step")
    plt.ylabel("diffusion denoising loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Wrote loss curve to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--title", default="Training curve")
    parser.add_argument("--rolling_window", type=int, default=20)

    args = parser.parse_args()

    main(
        run_dir=args.run_dir,
        output_path=args.output_path,
        title=args.title,
        rolling_window=args.rolling_window,
    )