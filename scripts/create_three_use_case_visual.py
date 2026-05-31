from pathlib import Path
import argparse

from PIL import Image, ImageDraw, ImageFont, ImageOps


def load_and_prepare(path: Path, panel_width: int, panel_height: int) -> Image.Image:
    img = Image.open(path).convert("L")
    img = ImageOps.autocontrast(img)
    img.thumbnail((panel_width, panel_height), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (panel_width, panel_height), color=245)
    x = (panel_width - img.width) // 2
    y = (panel_height - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def first_png(directory: Path) -> Path:
    paths = sorted(directory.glob("*.png"))
    if not paths:
        raise ValueError(f"No PNG files found in {directory}")
    return paths[0]


def make_visual(
    synthetic_input_path: Path,
    target_path: Path,
    natural_degraded_path: Path,
    output_path: Path,
):
    panel_w = 360
    panel_h = 280
    label_h = 120
    margin = 30
    title_h = 80

    width = panel_w * 3 + margin * 4
    height = title_h + panel_h + label_h + margin

    out = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(out)

    try:
        title_font = ImageFont.truetype("arial.ttf", 28)
        header_font = ImageFont.truetype("arial.ttf", 18)
        body_font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        title_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    title = "ArchiveDiffusion: controlled restoration and real archival stress testing"
    draw.text((margin, 24), title, fill=(0, 0, 0), font=title_font)

    panels = [
        {
            "path": synthetic_input_path,
            "header": "1. Synthetic restoration input",
            "body": "Artificial scratches, dust, and grain are added to a clean-ish archive frame.",
        },
        {
            "path": target_path,
            "header": "2. Known restoration target",
            "body": "The original clean-ish frame provides a proxy target for fidelity metrics.",
        },
        {
            "path": natural_degraded_path,
            "header": "3. Real archival stress test",
            "body": "Naturally degraded frames test practical restoration without ground truth.",
        },
    ]

    for i, panel in enumerate(panels):
        x = margin + i * (panel_w + margin)
        y = title_h

        img = load_and_prepare(panel["path"], panel_w, panel_h).convert("RGB")
        out.paste(img, (x, y))

        # Border
        draw.rectangle((x, y, x + panel_w, y + panel_h), outline=(0, 0, 0), width=1)

        text_y = y + panel_h + 12
        draw.text((x, text_y), panel["header"], fill=(0, 0, 0), font=header_font)

        # Simple wrapped text.
        words = panel["body"].split()
        lines = []
        current = ""
        for word in words:
            test = f"{current} {word}".strip()
            bbox = draw.textbbox((0, 0), test, font=body_font)
            if bbox[2] - bbox[0] <= panel_w:
                current = test
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)

        for j, line in enumerate(lines[:4]):
            draw.text((x, text_y + 28 + j * 18), line, fill=(40, 40, 40), font=body_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)
    print(f"Wrote figure: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_dir", required=True)
    parser.add_argument("--natural_degraded_dir", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    synthetic_dir = Path(args.synthetic_dir)
    natural_degraded_dir = Path(args.natural_degraded_dir)

    synthetic_input_path = first_png(synthetic_dir / "degraded")
    target_path = first_png(synthetic_dir / "original")
    # use specific example of visual blemishes
    natural_degraded_dir = Path("data/curated/nosferatu_degraded_candidates")
    natural_degraded_path = natural_degraded_dir / "nosferatu_000385.png"

    make_visual(
        synthetic_input_path=synthetic_input_path,
        target_path=target_path,
        natural_degraded_path=natural_degraded_path,
        output_path=Path(args.output_path),
    )