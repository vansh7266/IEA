from __future__ import annotations

from pathlib import Path

from PIL import Image

from .models import BoundingBox
from .vision_old import (
    assess_preview_framing,
    center_box,
    crop_box,
    decode_image_payload,
    draw_bbox_overlay,
    encode_png_data_url,
    ensure_rgb,
    expand_box,
    fit_image_inside_canvas,
    normalized_mean_difference,
    region_mean_difference,
    save_png,
)


# ---------------------------------------------------------------------------
# Multi-band Laplacian Pyramid Blending  (Burt & Adelson, 1983)
# ---------------------------------------------------------------------------
# This is the "Gaussian blending" technique referenced in the Agent Banana
# paper (Section 2.4, Image Layer Decomposition).  It blends at multiple
# frequency bands so that:
#   - Low-frequency differences (colour / lighting) are smoothed over a WIDE
#     area, eliminating visible colour seams.
#   - High-frequency details (edges / textures) are blended with SHARP
#     boundaries, preventing ghosting or translucency.
# ---------------------------------------------------------------------------


def _build_gaussian_pyramid(img, levels):
    """Build a Gaussian pyramid by repeatedly downsampling."""
    import cv2
    pyramid = [img.astype("float32")]
    for _ in range(levels):
        img = cv2.pyrDown(img)
        pyramid.append(img.astype("float32"))
    return pyramid


def _build_laplacian_pyramid(gauss_pyr):
    """Build a Laplacian pyramid from a Gaussian pyramid."""
    import cv2
    lap_pyr = []
    for i in range(len(gauss_pyr) - 1):
        h, w = gauss_pyr[i].shape[:2]
        expanded = cv2.pyrUp(gauss_pyr[i + 1], dstsize=(w, h))
        lap = gauss_pyr[i] - expanded
        lap_pyr.append(lap)
    lap_pyr.append(gauss_pyr[-1])  # lowest-res residual
    return lap_pyr


def _reconstruct_from_laplacian(lap_pyr):
    """Reconstruct an image from its Laplacian pyramid."""
    import cv2
    img = lap_pyr[-1]
    for i in range(len(lap_pyr) - 2, -1, -1):
        h, w = lap_pyr[i].shape[:2]
        img = cv2.pyrUp(img, dstsize=(w, h)) + lap_pyr[i]
    return img


def _laplacian_blend(source_region, patch, mask_float, levels=4):
    """Blend patch into source_region using multi-band Laplacian pyramids.

    Args:
        source_region: (H, W, 3) float32 — the region of the base image
        patch:         (H, W, 3) float32 — the edited crop, same size
        mask_float:    (H, W, 1) float32 in [0, 1] — soft blend mask
        levels:        number of pyramid levels (more = wider low-freq blend)

    Returns:
        (H, W, 3) float32 blended result
    """
    # Clamp levels to what the image dimensions can support
    min_dim = min(source_region.shape[0], source_region.shape[1])
    max_levels = max(1, int(min_dim).bit_length() - 3)
    levels = min(levels, max_levels)

    gp_src = _build_gaussian_pyramid(source_region, levels)
    gp_patch = _build_gaussian_pyramid(patch, levels)
    gp_mask = _build_gaussian_pyramid(mask_float, levels)

    lp_src = _build_laplacian_pyramid(gp_src)
    lp_patch = _build_laplacian_pyramid(gp_patch)

    # Blend each frequency band using the corresponding mask level
    lp_blended = []
    for la, lb, gm in zip(lp_src, lp_patch, gp_mask):
        # Ensure mask broadcasts to 3-channel
        if gm.ndim == 2:
            gm = gm[:, :, None]
        elif gm.shape[2] == 1:
            pass  # already (H,W,1)
        blended = la * (1.0 - gm) + lb * gm
        lp_blended.append(blended)

    return _reconstruct_from_laplacian(lp_blended)


def paste_crop(base_image: Image.Image, crop: Image.Image, box: BoundingBox) -> Image.Image:
    """Paste an edited crop back onto the base image using Laplacian pyramid
    blending — the same 'Gaussian blending' technique described in the Agent
    Banana paper (Section 2.4).

    Low-frequency colour/lighting differences are smoothed over a wide radius
    while high-frequency edges and textures remain crisp at the boundary.
    """
    import numpy as np

    base = ensure_rgb(base_image)
    patch = ensure_rgb(crop).resize((box.width, box.height))

    base_np = np.array(base, dtype=np.float32)
    patch_np = np.array(patch, dtype=np.float32)

    # Extract the region of the base that the patch will replace
    source_region = base_np[box.top:box.bottom, box.left:box.right].copy()

    # Build a soft mask: 1.0 = fully patch, 0.0 = fully source
    # Solid interior with a narrow cosine taper at the edges
    h, w = box.height, box.width
    taper = 2  # minimal taper: ILD local crop already matches context

    mask = np.ones((h, w, 1), dtype=np.float32)

    # Horizontal taper
    for i in range(taper):
        alpha = i / taper
        mask[:, i, 0] = alpha
        mask[:, w - 1 - i, 0] = alpha
    # Vertical taper
    for i in range(taper):
        alpha = i / taper
        mask[i, :, 0] = np.minimum(mask[i, :, 0], alpha)
        mask[h - 1 - i, :, 0] = np.minimum(mask[h - 1 - i, :, 0], alpha)

    try:
        # Pyramid levels: ~4-6 depending on patch size
        levels = max(2, min(6, int(np.log2(min(w, h))) - 2))
        blended = _laplacian_blend(source_region, patch_np, mask, levels=levels)
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        print(f"[agent-banana] paste_crop: Laplacian pyramid blend ({levels} levels, {taper}px taper)")
    except Exception as exc:
        # Fallback: simple alpha composite
        print(f"[agent-banana] Laplacian blend failed ({exc}), using alpha fallback")
        blended = (source_region * (1 - mask) + patch_np * mask)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

    # Write the blended patch back into the full image
    result = np.array(base, dtype=np.uint8).copy()
    result[box.top:box.bottom, box.left:box.right] = blended
    return Image.fromarray(result)


__all__ = [
    "assess_preview_framing",
    "center_box",
    "crop_box",
    "decode_image_payload",
    "draw_bbox_overlay",
    "encode_png_data_url",
    "ensure_rgb",
    "expand_box",
    "fit_image_inside_canvas",
    "normalized_mean_difference",
    "paste_crop",
    "region_mean_difference",
    "save_png",
]
