# scripts/plot_model_calibration_barchart.py

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SUMMARY_CSV = Path("evaluations/human_review/model_calibration_v1_summary_by_method.csv")
OUTPUT_PATH = Path("outputs/figures/model_calibration_v1_human_scores.png")


def normalise_method_name(name: str) -> str:
    """Make long method names readable in the chart."""
    replacements = {
        "conditional_ddpm_v1_100_steps": "DDPM v1\n100 steps",
        "conditional_ddpm_v1_50_steps": "DDPM v1\n50 steps",
        "residual_ddpm_v1_full_test_50_steps": "Residual DDPM\nfull",
        "conditional_ddpm_v3_splitaware_50_steps": "DDPM v3\nsplit-aware",
        "residual_ddpm_v1_conservative_test_50_steps": "Residual DDPM\nconservative",
    }
    return replacements.get(name, name.replace("_", " "))


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Find the first matching column, with a helpful error if none are present."""
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not find any of the expected columns: "
        f"{candidates}. Available columns are: {list(df.columns)}"
    )


def main() -> None:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Could not find summary CSV: {SUMMARY_CSV}")

    df = pd.read_csv(SUMMARY_CSV)

    method_col = pick_column(
        df,
        ["true_method", "method", "method_name"],
    )
    overall_col = pick_column(
        df,
        ["overall_mean", "overall_1_5_mean", "mean_overall"],
    )
    yes_maybe_col = pick_column(
        df,
        [
            "choose_yes_or_maybe_rate",
            "yes_maybe_rate",
            "choose_for_restoration_yes_or_maybe_rate",
        ],
    )

    plot_df = df[[method_col, overall_col, yes_maybe_col]].copy()
    plot_df["method_label"] = plot_df[method_col].apply(normalise_method_name)
    plot_df = plot_df.sort_values(overall_col, ascending=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.barh(plot_df["method_label"], plot_df[overall_col])

    ax.set_title("Blinded human review: model calibration v1")
    ax.set_xlabel("Mean overall restoration score (1–5)")
    ax.set_xlim(0, 5)

    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        overall = row[overall_col]
        yes_maybe = row[yes_maybe_col]

        # Handle either 0-1 rates or 0-100 percentages.
        if yes_maybe <= 1:
            yes_maybe_pct = yes_maybe * 100
        else:
            yes_maybe_pct = yes_maybe

        ax.text(
            overall + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{overall:.2f} | yes/maybe {yes_maybe_pct:.1f}%",
            va="center",
            fontsize=9,
        )

    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200)
    print(f"Saved chart to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()