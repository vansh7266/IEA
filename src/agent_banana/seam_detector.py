"""Seam Detection — Boundary Gradient Discontinuity (BGD) + Cross-Boundary Color Shift (CBCS).

Detects visible seams at the bounding box boundary after blending an edited
patch back into the original image.  Used by the quality gate to accept,
warn (retry with wider taper), or reject a composition.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .models import BoundingBox


def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Compute Sobel gradient magnitude on a single-channel image."""
    # Horizontal kernel
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    # Vertical kernel
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    from scipy.signal import convolve2d
    gx = convolve2d(gray, kx, mode="same", boundary="symm")
    gy = convolve2d(gray, ky, mode="same", boundary="symm")
    return np.sqrt(gx ** 2 + gy ** 2)


def _boundary_strip(h: int, w: int, box: BoundingBox, width: int = 1) -> np.ndarray:
    """Create a boolean mask of pixels within ±width of the bbox edge."""
    mask = np.zeros((h, w), dtype=bool)
    t, b, l, r = box.top, box.bottom, box.left, box.right

    # Clamp to image bounds
    t, b = max(0, t), min(h, b)
    l, r = max(0, l), min(w, r)

    # Top edge
    y0, y1 = max(0, t - width), min(h, t + width + 1)
    mask[y0:y1, l:r] = True
    # Bottom edge
    y0, y1 = max(0, b - width), min(h, b + width + 1)
    mask[y0:y1, l:r] = True
    # Left edge
    x0, x1 = max(0, l - width), min(w, l + width + 1)
    mask[t:b, x0:x1] = True
    # Right edge
    x0, x1 = max(0, r - width), min(w, r + width + 1)
    mask[t:b, x0:x1] = True

    return mask


def _context_strip(h: int, w: int, box: BoundingBox, inner: int = 5, outer: int = 15) -> np.ndarray:
    """Create a boolean mask of pixels 'inner' to 'outer' px from the bbox edge.
    This captures the local gradient baseline to compare against."""
    mask_wide = _boundary_strip(h, w, box, width=outer)
    mask_narrow = _boundary_strip(h, w, box, width=inner)
    return mask_wide & ~mask_narrow


def boundary_gradient_discontinuity(image: Image.Image, box: BoundingBox) -> float:
    """Compute BGD score: ratio of gradient at bbox boundary vs local context.

    Returns:
        float in [0, 1] where 0 = no seam, 1 = severe seam.
    """
    arr = np.array(image.convert("L"), dtype=np.float32)
    h, w = arr.shape
    grad = _sobel_magnitude(arr)

    boundary = _boundary_strip(h, w, box, width=1)
    context = _context_strip(h, w, box, inner=5, outer=15)

    eps = 1e-6
    boundary_mean = grad[boundary].mean() if boundary.any() else 0.0
    context_mean = grad[context].mean() if context.any() else eps

    ratio = boundary_mean / (context_mean + eps)
    # Normalize: ratio of 1.0 = no seam, 4.0+ = full penalty
    k = 3.0
    penalty = float(np.clip((ratio - 1.0) / k, 0.0, 1.0))
    return penalty


def cross_boundary_color_shift(image: Image.Image, box: BoundingBox) -> float:
    """Compute CBCS: color difference between strips just inside and outside the bbox.

    Returns:
        float in [0, 1] where 0 = colors match, 1 = severe color shift.
    """
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape

    # Inner strip: 2-6px inside the bbox edge
    inner_box = BoundingBox(
        left=min(box.left + 2, box.right),
        top=min(box.top + 2, box.bottom),
        right=max(box.right - 2, box.left),
        bottom=max(box.bottom - 2, box.top),
    )
    inner_mask = _boundary_strip(h, w, inner_box, width=4)

    # Outer strip: 2-6px outside the bbox edge
    outer_box = BoundingBox(
        left=max(box.left - 2, 0),
        top=max(box.top - 2, 0),
        right=min(box.right + 2, w),
        bottom=min(box.bottom + 2, h),
    )
    outer_mask = _boundary_strip(h, w, outer_box, width=4)
    # Exclude pixels that are inside the bbox
    outer_mask = outer_mask & ~_boundary_strip(h, w, box, width=0)

    if not inner_mask.any() or not outer_mask.any():
        return 0.0

    inner_mean = arr[inner_mask].reshape(-1, 3).mean(axis=0)
    outer_mean = arr[outer_mask].reshape(-1, 3).mean(axis=0)

    # L2 distance normalized by max possible (sqrt(3) * 255)
    shift = float(np.linalg.norm(inner_mean - outer_mean) / (255.0 * np.sqrt(3)))
    return min(shift, 1.0)


def boundary_penalty(image: Image.Image, box: BoundingBox) -> dict:
    """Combined seam detection metric.

    Returns:
        dict with keys: bgd, cbcs, penalty, verdict
        - penalty: weighted combination (0.7*BGD + 0.3*CBCS)
        - verdict: "accept" | "warn" | "reject"
    """
    bgd = boundary_gradient_discontinuity(image, box)
    cbcs = cross_boundary_color_shift(image, box)
    penalty = 0.7 * bgd + 0.3 * cbcs

    if penalty < 0.3:
        verdict = "accept"
    elif penalty < 0.6:
        verdict = "warn"
    else:
        verdict = "reject"

    return {
        "bgd": round(bgd, 4),
        "cbcs": round(cbcs, 4),
        "penalty": round(penalty, 4),
        "verdict": verdict,
    }
