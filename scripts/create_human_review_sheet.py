from pathlib import Path
import argparse
import csv
import html
import json
import random
import shutil
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont


def read_csv(path):
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_id(text):
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in ["-", "_"]:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def load_image(path, size):
    img = Image.open(path).convert("L")
    img.thumbnail((size, size))
    canvas = Image.new("L", (size, size), color=255)
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas.convert("RGB")


def add_label(img, label, label_height=36):
    w, h = img.size
    out = Image.new("RGB", (w, h + label_height), color=(255, 255, 255))
    out.paste(img, (0, label_height))

    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    draw.rectangle([0, 0, w, label_height], fill=(245, 245, 245))
    draw.text((8, 10), label, fill=(0, 0, 0), font=font)

    return out


def make_triptych(input_path, target_path, candidate_path, output_path, panel_size):
    input_img = add_label(load_image(input_path, panel_size), "Input")
    target_img = add_label(load_image(target_path, panel_size), "Ground truth target")
    candidate_img = add_label(load_image(candidate_path, panel_size), "Anonymous output")

    gap = 12
    w = input_img.width * 3 + gap * 2
    h = input_img.height

    out = Image.new("RGB", (w, h), color=(255, 255, 255))
    out.paste(input_img, (0, 0))
    out.paste(target_img, (input_img.width + gap, 0))
    out.paste(candidate_img, ((input_img.width + gap) * 2, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)


def normalise_path(path):
    return str(Path(path))


def collect_predictions(prediction_manifests):
    """
    Returns:
      predictions_by_example[example_id] = list of prediction rows
    """
    predictions_by_example = defaultdict(list)

    for manifest_path in prediction_manifests:
        manifest_path = Path(manifest_path)
        rows = read_csv(manifest_path)

        for row in rows:
            example_id = row["example_id"]
            method = row.get("method", manifest_path.stem)
            prediction_path = row["prediction_path"]

            predictions_by_example[example_id].append({
                "example_id": example_id,
                "method": method,
                "prediction_path": prediction_path,
                "manifest_path": str(manifest_path),
                "degradation_type": row.get("degradation_type", ""),
                "split": row.get("split", ""),
                "input_path": row.get("input_path", ""),
                "target_path": row.get("target_path", ""),
                "mask_path": row.get("mask_path", ""),
            })

    return predictions_by_example


def build_review_items(
    metadata_rows,
    predictions_by_example,
    split,
    max_examples,
    seed,
    output_dir,
    panel_size,
):
    rng = random.Random(seed)

    if split is not None:
        metadata_rows = [row for row in metadata_rows if row.get("split", "") == split]

    if not metadata_rows:
        raise ValueError(f"No metadata rows found for split={split}")

    # Shuffle examples, then optionally limit.
    rng.shuffle(metadata_rows)
    if max_examples is not None:
        metadata_rows = metadata_rows[:max_examples]

    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    review_items = []
    answer_key_rows = []
    blinded_rows = []

    review_counter = 0

    for metadata_row in metadata_rows:
        example_id = metadata_row["example_id"]
        candidates = predictions_by_example.get(example_id, [])

        if not candidates:
            print(f"Warning: no predictions found for example_id={example_id}; skipping")
            continue

        # Shuffle candidate order so model presentation order is random.
        rng.shuffle(candidates)

        input_path = metadata_row.get("input_path", candidates[0].get("input_path", ""))
        target_path = metadata_row.get("target_path", candidates[0].get("target_path", ""))

        if not input_path or not target_path:
            print(f"Warning: missing input/target path for example_id={example_id}; skipping")
            continue

        if not Path(input_path).exists():
            print(f"Warning: input path does not exist: {input_path}; skipping")
            continue

        if not Path(target_path).exists():
            print(f"Warning: target path does not exist: {target_path}; skipping")
            continue

        for candidate in candidates:
            candidate_path = candidate["prediction_path"]

            if not Path(candidate_path).exists():
                print(f"Warning: prediction path does not exist: {candidate_path}; skipping")
                continue

            review_counter += 1
            review_id = f"review_{review_counter:05d}"
            blind_label = f"Output {review_counter:05d}"

            image_name = f"{review_id}_{safe_id(example_id)}.png"
            triptych_path = assets_dir / image_name

            make_triptych(
                input_path=input_path,
                target_path=target_path,
                candidate_path=candidate_path,
                output_path=triptych_path,
                panel_size=panel_size,
            )

            degradation_type = metadata_row.get("degradation_type", candidate.get("degradation_type", ""))
            row_split = metadata_row.get("split", candidate.get("split", ""))

            review_items.append({
                "review_id": review_id,
                "example_id": example_id,
                "degradation_type": degradation_type,
                "split": row_split,
                "blind_label": blind_label,
                "image_path": f"assets/{image_name}",
            })

            blinded_rows.append({
                "review_id": review_id,
                "example_id": example_id,
                "degradation_type": degradation_type,
                "split": row_split,
                "blind_label": blind_label,
                "image_path": f"assets/{image_name}",
            })

            answer_key_rows.append({
                "review_id": review_id,
                "example_id": example_id,
                "degradation_type": degradation_type,
                "split": row_split,
                "blind_label": blind_label,
                "true_method": candidate["method"],
                "prediction_path": candidate["prediction_path"],
                "manifest_path": candidate["manifest_path"],
            })

    return review_items, blinded_rows, answer_key_rows


def make_html(review_items, output_html_path, title):
    # Embed items directly so there is no browser CSV-loading problem.
    review_items_json = json.dumps(review_items, indent=2)

    page_title = html.escape(title)

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{page_title}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      background: #fafafa;
      color: #222;
    }}
    .topbar {{
      position: sticky;
      top: 0;
      background: #fafafa;
      padding: 12px 0;
      border-bottom: 1px solid #ddd;
      z-index: 10;
    }}
    .card {{
      background: white;
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 16px;
      margin: 18px 0;
      box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }}
    img {{
      max-width: 100%;
      border: 1px solid #ccc;
      background: white;
    }}
    .meta {{
      color: #555;
      font-size: 14px;
      margin-bottom: 10px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 240px 1fr;
      gap: 8px 14px;
      align-items: center;
      margin-top: 12px;
    }}
    label {{
      font-weight: bold;
    }}
    select, textarea {{
      width: 100%;
      padding: 6px;
      font-size: 14px;
    }}
    textarea {{
      height: 60px;
    }}
    button {{
      padding: 8px 12px;
      margin-right: 8px;
      cursor: pointer;
    }}
    .small {{
      font-size: 13px;
      color: #555;
    }}
    .warning {{
      background: #fff8d6;
      border: 1px solid #e0c85a;
      padding: 10px;
      border-radius: 6px;
      margin: 12px 0;
    }}
  </style>
</head>
<body>
  <div class="topbar">
    <h1>{page_title}</h1>
    <button onclick="exportCSV()">Export CSV</button>
    <button onclick="saveAll()">Save progress</button>
    <button onclick="clearAll()">Clear saved ratings</button>
    <span id="progress" class="small"></span>
    <div class="warning">
      You are reviewing anonymous model outputs. Do not open the answer key until scoring is complete.
      Ratings are saved in this browser's local storage. Use Export CSV to save a file.
    </div>
  </div>

  <div id="review-container"></div>

<script>
const reviewItems = {review_items_json};
const storageKey = "ArchiveDiffusionHumanReview::{page_title}";

function blankRating(item) {{
  return {{
    review_id: item.review_id,
    example_id: item.example_id,
    degradation_type: item.degradation_type,
    split: item.split,
    blind_label: item.blind_label,
    artifact_removal_1_5: "",
    detail_preservation_1_5: "",
    texture_authenticity_1_5: "",
    over_smoothing_1_5: "",
    hallucination_risk_1_5: "",
    overall_1_5: "",
    choose_for_restoration: "",
    notes: ""
  }};
}}

function loadRatings() {{
  const raw = localStorage.getItem(storageKey);
  if (!raw) {{
    const initial = {{}};
    for (const item of reviewItems) {{
      initial[item.review_id] = blankRating(item);
    }}
    return initial;
  }}
  const loaded = JSON.parse(raw);
  for (const item of reviewItems) {{
    if (!loaded[item.review_id]) {{
      loaded[item.review_id] = blankRating(item);
    }}
  }}
  return loaded;
}}

let ratings = loadRatings();

function scoreOptions(allowBlank=true) {{
  let html = allowBlank ? '<option value="">Not rated</option>' : '';
  for (let i = 1; i <= 5; i++) {{
    html += `<option value="${{i}}">${{i}}</option>`;
  }}
  return html;
}}

function yesNoOptions() {{
  return `
    <option value="">Not selected</option>
    <option value="yes">Yes — I would choose this restoration</option>
    <option value="maybe">Maybe / borderline</option>
    <option value="no">No</option>
  `;
}}

function render() {{
  const container = document.getElementById("review-container");
  container.innerHTML = "";

  for (const item of reviewItems) {{
    const r = ratings[item.review_id] || blankRating(item);

    const div = document.createElement("div");
    div.className = "card";
    div.innerHTML = `
      <h2>${{item.blind_label}}</h2>
      <div class="meta">
        Review ID: ${{item.review_id}} |
        Example: ${{item.example_id}} |
        Degradation: ${{item.degradation_type}} |
        Split: ${{item.split}}
      </div>
      <img src="${{item.image_path}}" alt="${{item.blind_label}}">
      <div class="grid">
        <label>Artifact removal, 1–5<br><span class="small">1 none, 5 strong removal</span></label>
        <select data-id="${{item.review_id}}" data-field="artifact_removal_1_5">${{scoreOptions()}}</select>

        <label>Detail preservation, 1–5<br><span class="small">1 poor, 5 strong</span></label>
        <select data-id="${{item.review_id}}" data-field="detail_preservation_1_5">${{scoreOptions()}}</select>

        <label>Texture authenticity, 1–5<br><span class="small">1 artificial, 5 authentic archival texture</span></label>
        <select data-id="${{item.review_id}}" data-field="texture_authenticity_1_5">${{scoreOptions()}}</select>

        <label>Over-smoothing, 1–5<br><span class="small">1 no over-smoothing, 5 severe</span></label>
        <select data-id="${{item.review_id}}" data-field="over_smoothing_1_5">${{scoreOptions()}}</select>

        <label>Hallucination risk, 1–5<br><span class="small">1 none, 5 clear invented/changed content</span></label>
        <select data-id="${{item.review_id}}" data-field="hallucination_risk_1_5">${{scoreOptions()}}</select>

        <label>Overall restoration quality, 1–5<br><span class="small">1 worse than input, 5 excellent</span></label>
        <select data-id="${{item.review_id}}" data-field="overall_1_5">${{scoreOptions()}}</select>

        <label>Would you choose this restoration?</label>
        <select data-id="${{item.review_id}}" data-field="choose_for_restoration">${{yesNoOptions()}}</select>

        <label>Notes</label>
        <textarea data-id="${{item.review_id}}" data-field="notes"></textarea>
      </div>
    `;

    container.appendChild(div);

    for (const field of [
      "artifact_removal_1_5",
      "detail_preservation_1_5",
      "texture_authenticity_1_5",
      "over_smoothing_1_5",
      "hallucination_risk_1_5",
      "overall_1_5",
      "choose_for_restoration",
      "notes"
    ]) {{
      const el = div.querySelector(`[data-id="${{item.review_id}}"][data-field="${{field}}"]`);
      if (el) el.value = r[field] || "";
    }}
  }}

  document.querySelectorAll("select, textarea").forEach(el => {{
    el.addEventListener("change", updateRating);
    el.addEventListener("keyup", updateRating);
  }});

  updateProgress();
}}

function updateRating(e) {{
  const id = e.target.dataset.id;
  const field = e.target.dataset.field;
  ratings[id][field] = e.target.value;
  saveAll(false);
  updateProgress();
}}

function saveAll(showAlert=true) {{
  localStorage.setItem(storageKey, JSON.stringify(ratings));
  if (showAlert) {{
    alert("Progress saved in browser local storage.");
  }}
}}

function clearAll() {{
  if (!confirm("Clear all saved ratings for this review page?")) return;
  localStorage.removeItem(storageKey);
  ratings = loadRatings();
  render();
}}

function updateProgress() {{
  let rated = 0;
  for (const item of reviewItems) {{
    const r = ratings[item.review_id];
    if (r && r.overall_1_5) rated += 1;
  }}
  document.getElementById("progress").textContent =
    `Rated ${{rated}} / ${{reviewItems.length}} outputs`;
}}

function csvEscape(value) {{
  if (value === null || value === undefined) return "";
  const s = String(value);
  if (s.includes('"') || s.includes(",") || s.includes("\\n")) {{
    return '"' + s.replaceAll('"', '""') + '"';
  }}
  return s;
}}

function exportCSV() {{
  saveAll(false);

  const fields = [
    "review_id",
    "example_id",
    "degradation_type",
    "split",
    "blind_label",
    "artifact_removal_1_5",
    "detail_preservation_1_5",
    "texture_authenticity_1_5",
    "over_smoothing_1_5",
    "hallucination_risk_1_5",
    "overall_1_5",
    "choose_for_restoration",
    "notes"
  ];

  const lines = [fields.join(",")];

  for (const item of reviewItems) {{
    const r = ratings[item.review_id] || blankRating(item);
    lines.push(fields.map(f => csvEscape(r[f])).join(","));
  }}

  const blob = new Blob([lines.join("\\n")], {{ type: "text/csv" }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "human_review_ratings.csv";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}}

render();
</script>
</body>
</html>
"""

    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    output_html_path.write_text(html_text, encoding="utf-8")


def main(
    metadata_file,
    prediction_manifests,
    output_dir,
    output_csv,
    answer_key_csv,
    split,
    max_examples,
    seed,
    panel_size,
    title,
):
    metadata_file = Path(metadata_file)
    output_dir = Path(output_dir)
    output_csv = Path(output_csv)
    answer_key_csv = Path(answer_key_csv)

    metadata_rows = read_csv(metadata_file)
    predictions_by_example = collect_predictions(prediction_manifests)

    review_items, blinded_rows, answer_key_rows = build_review_items(
        metadata_rows=metadata_rows,
        predictions_by_example=predictions_by_example,
        split=split,
        max_examples=max_examples,
        seed=seed,
        output_dir=output_dir,
        panel_size=panel_size,
    )

    if not review_items:
        raise ValueError("No review items were created. Check split, metadata, and prediction manifests.")

    blinded_fieldnames = [
        "review_id",
        "example_id",
        "degradation_type",
        "split",
        "blind_label",
        "image_path",
    ]

    answer_key_fieldnames = [
        "review_id",
        "example_id",
        "degradation_type",
        "split",
        "blind_label",
        "true_method",
        "prediction_path",
        "manifest_path",
    ]

    write_csv(output_csv, blinded_rows, blinded_fieldnames)
    write_csv(answer_key_csv, answer_key_rows, answer_key_fieldnames)

    html_path = output_dir / "index.html"
    make_html(review_items, html_path, title=title)

    print(f"Wrote browser review widget: {html_path}")
    print(f"Wrote blinded review items: {output_csv}")
    print(f"Wrote answer key: {answer_key_csv}")
    print("")
    print("Open this in your browser:")
    print(html_path.resolve())
    print("")
    print("Do not open the answer key until after scoring is complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_file", required=True)
    parser.add_argument(
        "--prediction_manifest",
        action="append",
        required=True,
        help="Can be passed multiple times.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--answer_key_csv", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_examples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panel_size", type=int, default=256)
    parser.add_argument("--title", default="ArchiveDiffusion blinded human review")

    args = parser.parse_args()

    main(
        metadata_file=args.metadata_file,
        prediction_manifests=args.prediction_manifest,
        output_dir=args.output_dir,
        output_csv=args.output_csv,
        answer_key_csv=args.answer_key_csv,
        split=args.split,
        max_examples=args.max_examples,
        seed=args.seed,
        panel_size=args.panel_size,
        title=args.title,
    )