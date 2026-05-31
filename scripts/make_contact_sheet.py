from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import math
import argparse


def make_contact_sheet(input_dir, output_path, every=20, max_images=100, thumb_width=160):
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_dir.glob("*.png"))[::every][:max_images]

    if not image_paths:
        raise ValueError(f"No PNG files found in {input_dir}")

    thumbs = []
    for path in image_paths:
        img = Image.open(path).convert("L")
        img = ImageOps.autocontrast(img)

        aspect = img.height / img.width
        thumb_height = int(thumb_width * aspect)
        img = img.resize((thumb_width, thumb_height))

        canvas = Image.new("L", (thumb_width, thumb_height + 20), color=255)
        canvas.paste(img, (0, 0))

        draw = ImageDraw.Draw(canvas)
        draw.text((4, thumb_height + 4), path.stem, fill=0)

        thumbs.append(canvas)

    cols = 5
    rows = math.ceil(len(thumbs) / cols)
    cell_w = thumb_width
    cell_h = max(t.height for t in thumbs)

    sheet = Image.new("L", (cols * cell_w, rows * cell_h), color=255)

    for i, thumb in enumerate(thumbs):
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h
        sheet.paste(thumb, (x, y))

    sheet.save(output_path)
    print(f"Saved contact sheet to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--every", type=int, default=20)
    parser.add_argument("--max_images", type=int, default=100)
    args = parser.parse_args()

    make_contact_sheet(
        input_dir=args.input_dir,
        output_path=args.output_path,
        every=args.every,
        max_images=args.max_images,
    )