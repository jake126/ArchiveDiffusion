from pathlib import Path
import argparse
import csv
import json

import pandas as pd


NUMERIC_RATING_COLUMNS = [
    "artifact_removal_1_5",
    "detail_preservation_1_5",
    "texture_authenticity_1_5",
    "over_smoothing_1_5",
    "hallucination_risk_1_5",
    "overall_1_5",
]


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def choose_rate(series, label):
    return int((series.fillna("").astype(str).str.lower() == label).sum())


def yes_or_maybe_rate(series):
    s = series.fillna("").astype(str).str.lower()
    return int(((s == "yes") | (s == "maybe")).sum())


def add_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    A simple diagnostic composite. Higher is better.

    This is not intended to replace the individual rubric scores. It is useful
    for quick sorting while preserving the original columns.
    """
    df = df.copy()

    required = [
        "artifact_removal_1_5",
        "detail_preservation_1_5",
        "texture_authenticity_1_5",
        "over_smoothing_1_5",
        "hallucination_risk_1_5",
        "overall_1_5",
    ]

    for col in required:
        if col not in df.columns:
            df[col] = pd.NA

    df["human_composite_score"] = (
        df["overall_1_5"]
        + 0.5 * df["artifact_removal_1_5"]
        + 0.5 * df["detail_preservation_1_5"]
        + 0.5 * df["texture_authenticity_1_5"]
        - 0.5 * df["over_smoothing_1_5"]
        - 0.5 * df["hallucination_risk_1_5"]
    )

    return df


def summarize_group(df: pd.DataFrame, group_cols):
    grouped = df.groupby(group_cols, dropna=False)

    summary = grouped.agg(
        n_items=("review_id", "count"),
        n_rated=("overall_1_5", lambda s: int(s.notna().sum())),
        artifact_removal_mean=("artifact_removal_1_5", "mean"),
        detail_preservation_mean=("detail_preservation_1_5", "mean"),
        texture_authenticity_mean=("texture_authenticity_1_5", "mean"),
        over_smoothing_mean=("over_smoothing_1_5", "mean"),
        hallucination_risk_mean=("hallucination_risk_1_5", "mean"),
        overall_mean=("overall_1_5", "mean"),
        human_composite_mean=("human_composite_score", "mean"),
        choose_yes=("choose_for_restoration", lambda s: choose_rate(s, "yes")),
        choose_maybe=("choose_for_restoration", lambda s: choose_rate(s, "maybe")),
        choose_no=("choose_for_restoration", lambda s: choose_rate(s, "no")),
        choose_yes_or_maybe=("choose_for_restoration", yes_or_maybe_rate),
    ).reset_index()

    summary["choose_yes_rate"] = summary["choose_yes"] / summary["n_items"]
    summary["choose_yes_or_maybe_rate"] = (
        summary["choose_yes_or_maybe"] / summary["n_items"]
    )

    sort_cols = []
    if "overall_mean" in summary.columns:
        sort_cols.append("overall_mean")
    if "choose_yes_or_maybe_rate" in summary.columns:
        sort_cols.append("choose_yes_or_maybe_rate")

    if sort_cols:
        summary = summary.sort_values(sort_cols, ascending=False)

    return summary


def write_markdown_report(
    output_path: Path,
    method_summary: pd.DataFrame,
    degradation_summary: pd.DataFrame,
    missing_ratings: pd.DataFrame,
    ratings_path: Path,
    answer_key_path: Path,
):
    ensure_parent(output_path)

    lines = []
    lines.append("# Human evaluation summary")
    lines.append("")
    lines.append(f"Ratings file: `{ratings_path}`")
    lines.append(f"Answer key: `{answer_key_path}`")
    lines.append("")
    lines.append("## Summary by method")
    lines.append("")
    lines.append(method_summary.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Summary by method and degradation type")
    lines.append("")
    lines.append(degradation_summary.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    if len(missing_ratings) > 0:
        lines.append("## Missing or incomplete ratings")
        lines.append("")
        keep_cols = [
            "review_id",
            "example_id",
            "degradation_type",
            "split",
            "true_method",
            "blind_label",
        ]
        available_cols = [c for c in keep_cols if c in missing_ratings.columns]
        lines.append(missing_ratings[available_cols].to_markdown(index=False))
        lines.append("")
    else:
        lines.append("## Missing or incomplete ratings")
        lines.append("")
        lines.append("No missing `overall_1_5` ratings found.")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main(
    ratings_csv,
    answer_key_csv,
    output_dir,
    output_prefix,
):
    ratings_csv = Path(ratings_csv)
    answer_key_csv = Path(answer_key_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ratings = pd.read_csv(ratings_csv)
    answer_key = pd.read_csv(answer_key_csv)

    if "review_id" not in ratings.columns:
        raise ValueError(f"`review_id` missing from ratings file: {ratings_csv}")

    if "review_id" not in answer_key.columns:
        raise ValueError(f"`review_id` missing from answer key: {answer_key_csv}")

    required_key_cols = ["review_id", "true_method"]
    missing_key_cols = [c for c in required_key_cols if c not in answer_key.columns]
    if missing_key_cols:
        raise ValueError(f"Missing columns in answer key: {missing_key_cols}")

    key_cols = [
        "review_id",
        "true_method",
        "prediction_path",
        "manifest_path",
    ]
    key_cols = [c for c in key_cols if c in answer_key.columns]

    df = ratings.merge(
        answer_key[key_cols],
        on="review_id",
        how="left",
        validate="one_to_one",
    )

    if df["true_method"].isna().any():
        missing = df[df["true_method"].isna()]["review_id"].tolist()
        raise ValueError(
            "Some ratings did not match the answer key. Missing review_ids: "
            + ", ".join(missing[:10])
        )

    for col in NUMERIC_RATING_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.NA

    if "choose_for_restoration" not in df.columns:
        df["choose_for_restoration"] = ""

    df["choose_for_restoration"] = (
        df["choose_for_restoration"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df = add_composite_score(df)

    method_summary = summarize_group(df, ["true_method"])
    degradation_summary = summarize_group(df, ["degradation_type", "true_method"])
    split_summary = summarize_group(df, ["split", "true_method"])

    missing_ratings = df[df["overall_1_5"].isna()].copy()

    merged_path = output_dir / f"{output_prefix}_merged_ratings.csv"
    method_path = output_dir / f"{output_prefix}_summary_by_method.csv"
    degradation_path = output_dir / f"{output_prefix}_summary_by_degradation.csv"
    split_path = output_dir / f"{output_prefix}_summary_by_split.csv"
    missing_path = output_dir / f"{output_prefix}_missing_ratings.csv"
    report_path = output_dir / f"{output_prefix}_report.md"
    json_path = output_dir / f"{output_prefix}_summary.json"

    df.to_csv(merged_path, index=False)
    method_summary.to_csv(method_path, index=False)
    degradation_summary.to_csv(degradation_path, index=False)
    split_summary.to_csv(split_path, index=False)
    missing_ratings.to_csv(missing_path, index=False)

    write_markdown_report(
        output_path=report_path,
        method_summary=method_summary,
        degradation_summary=degradation_summary,
        missing_ratings=missing_ratings,
        ratings_path=ratings_csv,
        answer_key_path=answer_key_csv,
    )

    summary_payload = {
        "ratings_csv": str(ratings_csv),
        "answer_key_csv": str(answer_key_csv),
        "n_review_items": int(len(df)),
        "n_missing_overall_ratings": int(df["overall_1_5"].isna().sum()),
        "outputs": {
            "merged_ratings": str(merged_path),
            "summary_by_method": str(method_path),
            "summary_by_degradation": str(degradation_path),
            "summary_by_split": str(split_path),
            "missing_ratings": str(missing_path),
            "markdown_report": str(report_path),
        },
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"Wrote merged ratings: {merged_path}")
    print(f"Wrote method summary: {method_path}")
    print(f"Wrote degradation summary: {degradation_path}")
    print(f"Wrote split summary: {split_path}")
    print(f"Wrote missing ratings: {missing_path}")
    print(f"Wrote markdown report: {report_path}")
    print(f"Wrote JSON summary: {json_path}")

    print("")
    print("Top methods by overall_mean:")
    print(method_summary[["true_method", "n_items", "n_rated", "overall_mean", "choose_yes_or_maybe_rate"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings_csv", required=True)
    parser.add_argument("--answer_key_csv", required=True)
    parser.add_argument("--output_dir", default="evaluations/human_review")
    parser.add_argument("--output_prefix", default="human_evaluation")

    args = parser.parse_args()

    main(
        ratings_csv=args.ratings_csv,
        answer_key_csv=args.answer_key_csv,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
    )