from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw

from .models import BoundingBox


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def decode_image_payload(payload: str) -> Image.Image:
    if payload.startswith("data:"):
        _, encoded = payload.split(",", 1)
    else:
        encoded = payload
    image_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def encode_png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    ensure_rgb(image).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_rgb(image).save(path, format="PNG")


def fit_image_inside_canvas(image: Image.Image, canvas_size: tuple[int, int], fill_color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    image = ensure_rgb(image)
    canvas_width, canvas_height = canvas_size
    source_width, source_height = image.size
    scale = min(canvas_width / max(1, source_width), canvas_height / max(1, source_height))
    resized_width = max(1, int(source_width * scale))
    resized_height = max(1, int(source_height * scale))
    resized = image.resize((resized_width, resized_height))
    canvas = Image.new("RGB", canvas_size, fill_color)
    left = (canvas_width - resized_width) // 2
    top = (canvas_height - resized_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def expand_box(box: BoundingBox, padding: int, image_size: tuple[int, int]) -> BoundingBox:
    width, height = image_size
    return BoundingBox(
        left=max(0, box.left - padding),
        top=max(0, box.top - padding),
        right=min(width, box.right + padding),
        bottom=min(height, box.bottom + padding),
    )


def center_box(image_size: tuple[int, int], scale: float = 0.38) -> BoundingBox:
    width, height = image_size
    box_width = max(32, int(width * scale))
    box_height = max(32, int(height * scale))
    left = (width - box_width) // 2
    top = (height - box_height) // 2
    return BoundingBox(left=left, top=top, right=left + box_width, bottom=top + box_height)


def region_mean_difference(before: Image.Image, after: Image.Image, box: BoundingBox) -> float:
    return normalized_mean_difference(before, after, box=box, outside=False)


def assess_preview_framing(source: Image.Image, preview: Image.Image, border_fraction: float = 0.08) -> dict:
    source = ensure_rgb(source)
    preview = fit_image_inside_canvas(preview, source.size)
    width, height = source.size
    border_width = max(4, int(width * border_fraction))
    border_height = max(4, int(height * border_fraction))
    left_box = BoundingBox(0, 0, border_width, height)
    right_box = BoundingBox(width - border_width, 0, width, height)
    top_box = BoundingBox(0, 0, width, border_height)
    bottom_box = BoundingBox(0, height - border_height, width, height)
    left = region_mean_difference(source, preview, left_box)
    right = region_mean_difference(source, preview, right_box)
    top = region_mean_difference(source, preview, top_box)
    bottom = region_mean_difference(source, preview, bottom_box)
    return {
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "average": (left + right + top + bottom) / 4.0,
        "preview": preview,
    }


def crop_box(image: Image.Image, box: BoundingBox) -> Image.Image:
    return ensure_rgb(image).crop(box.as_tuple())


def paste_crop(base_image: Image.Image, crop: Image.Image, box: BoundingBox) -> Image.Image:
    composite = ensure_rgb(base_image).copy()
    composite.paste(ensure_rgb(crop).resize((box.width, box.height)), box.as_tuple())
    return composite


def draw_bbox_overlay(image: Image.Image, box: BoundingBox, label: str = "") -> Image.Image:
    canvas = ensure_rgb(image).copy().convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    fill = ImageColor.getrgb("#F59E0B") + (48,)
    outline = ImageColor.getrgb("#B45309") + (255,)
    draw.rectangle(box.as_tuple(), fill=fill, outline=outline, width=4)
    if label:
        text_box = (box.left + 8, max(4, box.top - 28), box.left + 8 + min(240, len(label) * 9), max(28, box.top - 4))
        draw.rounded_rectangle(text_box, radius=10, fill=(19, 42, 47, 220))
        draw.text((text_box[0] + 10, text_box[1] + 7), label[:28], fill=(255, 255, 255, 255))
    return Image.alpha_composite(canvas, overlay).convert("RGB")


def normalized_mean_difference(
    before: Image.Image,
    after: Image.Image,
    *,
    box: BoundingBox | None = None,
    outside: bool = False,
) -> float:
    before = ensure_rgb(before)
    after = ensure_rgb(after).resize(before.size)
    width, height = before.size
    stride = max(1, min(width, height) // 128)
    before_pixels = before.load()
    after_pixels = after.load()
    total = 0
    count = 0

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            inside_box = False
            if box is not None:
                inside_box = box.left <= x < box.right and box.top <= y < box.bottom
                if outside and inside_box:
                    continue
                if not outside and not inside_box:
                    continue
            r1, g1, b1 = before_pixels[x, y]
            r2, g2, b2 = after_pixels[x, y]
            total += abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)
            count += 3

    if count == 0:
        return 0.0
    return total / (count * 255.0)
