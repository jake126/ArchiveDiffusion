from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


RATINGS_CSV = Path("evaluations/human_review/model_calibration_v1_human_review_ratings.csv")
ANSWER_KEY_CSV = Path("evaluations/human_review/model_calibration_v1_answer_key.csv")
MERGED_CSV = Path("evaluations/human_review/model_calibration_v1_merged_ratings.csv")

OUTPUT_DIR = Path("docs/assets/writeup_examples")


METHOD_ORDER = [
    "conditional_ddpm_v1_100_steps",
    "conditional_ddpm_v1_50_steps",
    "residual_ddpm_v1_full_test_50_steps",
    "conditional_ddpm_v3_splitaware_50_steps",
    "residual_ddpm_v1_conservative_test_50_steps",
]


def find_col(df, candidates, required=True):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(
            f"Could not find any of {candidates}. Available columns: {list(df.columns)}"
        )
    return None


def load_results():
    if MERGED_CSV.exists():
        return pd.read_csv(MERGED_CSV)

    ratings = pd.read_csv(RATINGS_CSV)
    answer_key = pd.read_csv(ANSWER_KEY_CSV)

    # Most versions of the review tool use review_id as the merge key.
    if "review_id" in ratings.columns and "review_id" in answer_key.columns:
        return ratings.merge(answer_key, on="review_id", how="left", suffixes=("", "_key"))

    # Fallback: blind_label may also be usable.
    if "blind_label" in ratings.columns and "blind_label" in answer_key.columns:
        return ratings.merge(answer_key, on="blind_label", how="left", suffixes=("", "_key"))

    raise ValueError("Could not merge ratings and answer key. Need review_id or blind_label.")


def resolve_path(path_value):
    if pd.isna(path_value):
        return None

    p = Path(str(path_value))

    # If the CSV stores repo-relative paths, resolve from project root.
    if p.exists():
        return p

    candidate = Path.cwd() / p
    if candidate.exists():
        return candidate

    # Sometimes paths are stored with forward slashes from repo root.
    candidate = Path(str(path_value).replace("/", "\\"))
    if candidate.exists():
        return candidate

    return None


def save_triptych(row, output_path, title):
    input_col = find_col(
        row.to_frame().T,
        ["input_path", "degraded_path", "condition_path", "source_path"],
        required=False,
    )
    target_col = find_col(
        row.to_frame().T,
        ["target_path", "clean_path", "original_path"],
        required=False,
    )
    output_col = find_col(
        row.to_frame().T,
        ["prediction_path", "output_path", "restored_path", "image_path"],
        required=False,
    )

    if output_col is None:
        raise ValueError(
            "Could not find output image path column. Check the merged CSV columns."
        )

    image_specs = []

    if input_col is not None:
        image_specs.append(("Input", resolve_path(row[input_col])))

    if target_col is not None:
        image_specs.append(("Target", resolve_path(row[target_col])))

    image_specs.append(("Output", resolve_path(row[output_col])))

    image_specs = [(label, path) for label, path in image_specs if path is not None]

    if not image_specs:
        raise ValueError("No valid image paths found for selected row.")

    fig, axes = plt.subplots(1, len(image_specs), figsize=(4 * len(image_specs), 4))

    if len(image_specs) == 1:
        axes = [axes]

    for ax, (label, path) in zip(axes, image_specs):
        img = Image.open(path).convert("L")
        ax.imshow(img, cmap="gray")
        ax.set_title(label)
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_results()

    method_col = find_col(df, ["true_method", "method", "method_name"])
    overall_col = find_col(df, ["overall_1_5", "overall"])
    choice_col = find_col(df, ["choose_for_restoration"])
    degradation_col = find_col(df, ["degradation_type"])
    artifact_col = find_col(df, ["artifact_removal_1_5"], required=False)
    over_smoothing_col = find_col(df, ["over_smoothing_1_5"], required=False)
    hallucination_col = find_col(df, ["hallucination_risk_1_5"], required=False)

    selected = []

    # 1. Strong DDPM success: high overall, yes/maybe, preferably v1 100 or v1 50.
    ddpm_success = (
        df[
            df[method_col].isin(
                ["conditional_ddpm_v1_100_steps", "conditional_ddpm_v1_50_steps"]
            )
            & df[choice_col].isin(["yes", "maybe"])
        ]
        .sort_values(overall_col, ascending=False)
        .head(1)
    )

    if len(ddpm_success):
        selected.append(
            (
                "ddpm_success_triptych.png",
                ddpm_success.iloc[0],
                "Strong DDPM restoration example",
            )
        )

    # 2. Strong residual full example.
    residual_full = (
        df[
            (df[method_col] == "residual_ddpm_v1_full_test_50_steps")
            & df[choice_col].isin(["yes", "maybe"])
        ]
        .sort_values(overall_col, ascending=False)
        .head(1)
    )

    if len(residual_full):
        selected.append(
            (
                "residual_full_example_triptych.png",
                residual_full.iloc[0],
                "Residual DDPM full-correction example",
            )
        )

    # 3. Conservative residual under-correction:
    # low artifact removal / low overall, but possibly low risk ratings.
    conservative = df[df[method_col] == "residual_ddpm_v1_conservative_test_50_steps"].copy()

    if artifact_col is not None:
        conservative = conservative.sort_values(
            [overall_col, artifact_col],
            ascending=[True, True],
        )
    else:
        conservative = conservative.sort_values(overall_col, ascending=True)

    if len(conservative):
        selected.append(
            (
                "residual_conservative_undercorrection_triptych.png",
                conservative.iloc[0],
                "Conservative residual DDPM under-correction example",
            )
        )

    # 4. Scratch/dust failure case, if available.
    scratch = df[df[degradation_col] == "scratch_dust"].copy()
    if len(scratch):
        scratch = scratch.sort_values(overall_col, ascending=True).head(1)
        selected.append(
            (
                "scratch_dust_failure_triptych.png",
                scratch.iloc[0],
                "Scratch/dust failure case",
            )
        )

    print("Selected examples:")
    for filename, row, title in selected:
        output_path = OUTPUT_DIR / filename
        save_triptych(row, output_path, title)

        print()
        print(f"{filename}")
        print(f"  title: {title}")
        print(f"  method: {row[method_col]}")
        print(f"  example_id: {row.get('example_id', 'unknown')}")
        print(f"  degradation: {row.get(degradation_col, 'unknown')}")
        print(f"  overall: {row[overall_col]}")
        print(f"  choice: {row[choice_col]}")
        print(f"  saved: {output_path}")


if __name__ == "__main__":
    main()