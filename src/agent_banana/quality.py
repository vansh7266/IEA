from __future__ import annotations

from PIL import Image

from .models import BoundingBox, QualityMetrics
from .seam_detector import boundary_penalty
from .targeting import classify_target, ideal_change_range, max_bbox_area_ratio
from .vision import normalized_mean_difference


class QualityJudge:
    def evaluate(
        self,
        before: Image.Image,
        after: Image.Image,
        bbox: BoundingBox,
        *,
        preview: Image.Image | None = None,
        target: str = "",
        verb: str = "",
    ) -> QualityMetrics:
        profile = classify_target(target, verb)
        inside_change = normalized_mean_difference(before, after, box=bbox, outside=False)
        outside_change = normalized_mean_difference(before, after, box=bbox, outside=True)
        preview_alignment = 0.5
        if preview is not None:
            preview_alignment = 1.0 - normalized_mean_difference(preview, after, box=bbox, outside=False)
        image_area = max(1, before.size[0] * before.size[1])
        area_ratio = bbox.area / image_area
        max_area_ratio = max_bbox_area_ratio(profile)
        min_change, max_change = ideal_change_range(profile)
        size_fit = 1.0
        if area_ratio > max_area_ratio:
            overflow = min(1.0, (area_ratio - max_area_ratio) / max_area_ratio)
            size_fit = max(0.0, 1.0 - overflow)
        change_fit = 1.0
        if inside_change < min_change:
            change_fit = max(0.0, inside_change / max(min_change, 1e-6))
        elif inside_change > max_change:
            change_fit = max(0.0, 1.0 - min(1.0, (inside_change - max_change) / max(max_change, 1e-6)))

        locality = max(0.0, 1.0 - min(1.0, outside_change / 0.12))
        edit_strength = min(1.0, inside_change / 0.16)
        preview_fit = max(0.0, min(1.0, preview_alignment))
        score = 0.34 * edit_strength + 0.26 * locality + 0.16 * preview_fit + 0.14 * size_fit + 0.10 * change_fit

        notes = []
        if inside_change < 0.02:
            notes.append("The edited region barely changed.")
        if outside_change > 0.10:
            notes.append("Too much drift leaked outside the target region.")
        if preview_alignment < 0.35:
            notes.append("The final crop diverged from the preview localization cue.")
        if area_ratio > max_area_ratio:
            notes.append("The inferred edit region is too large for the requested target.")
        if inside_change > max_change:
            notes.append("The crop edit changed more structure than this target should require.")

        # Seam detection
        seam = boundary_penalty(after, bbox)
        seam_score = seam["penalty"]
        seam_verdict = seam["verdict"]
        if seam_verdict == "warn":
            notes.append(f"Minor seam detected (score={seam_score:.3f}). Consider widening taper.")
        elif seam_verdict == "reject":
            notes.append(f"Severe seam detected (score={seam_score:.3f}). Re-blend required.")

        accepted = bool(
            score >= 0.56
            and inside_change >= 0.02
            and outside_change <= 0.14
            and area_ratio <= max_area_ratio * 1.15
            and inside_change <= max_change * 1.15
            and seam_verdict != "reject"
        )
        if accepted:
            notes.append("Edit passed the local quality gate.")

        return QualityMetrics(
            score=score,
            accepted=accepted,
            inside_change=inside_change,
            outside_change=outside_change,
            preview_alignment=preview_alignment,
            seam_score=seam_score,
            seam_verdict=seam_verdict,
            notes=notes,
        )
