from __future__ import annotations

import re

from .models import BoundingBox, GroundingCandidate

FACE_ACCESSORY_KEYWORDS = {"glasses", "eyeglasses", "spectacles", "sunglasses", "goggles", "frames", "eyewear"}
HEAD_ACCESSORY_KEYWORDS = {"hat", "cap", "helmet", "headband", "tiara", "veil"}
SMALL_ACCESSORY_KEYWORDS = {"earring", "earrings", "ring", "bracelet", "watch", "necklace", "pendant", "brooch"}
GLOBAL_KEYWORDS = {"background", "scene", "lighting", "style", "mood", "whole image", "entire image"}
WATERMARK_KEYWORDS = {"logo", "watermark", "badge", "stamp", "overlay", "text overlay", "brand", "copyright", "credit", "label"}


def classify_target(target: str, verb: str = "") -> str:
    lowered = f"{verb} {target}".lower()
    if any(keyword in lowered for keyword in FACE_ACCESSORY_KEYWORDS):
        return "face_accessory"
    if any(keyword in lowered for keyword in HEAD_ACCESSORY_KEYWORDS):
        return "head_accessory"
    if any(keyword in lowered for keyword in SMALL_ACCESSORY_KEYWORDS):
        return "small_accessory"
    if any(keyword in lowered for keyword in GLOBAL_KEYWORDS):
        return "global_region"
    if any(keyword in lowered for keyword in WATERMARK_KEYWORDS):
        return "watermark"
    return "generic_local"


def _corner_from_modifiers(modifiers: list[str], instruction: str = "") -> str | None:
    """Extract corner hint from spatial modifiers or instruction text."""
    text = " ".join(modifiers + [instruction]).lower()
    bottom = "bottom" in text or "lower" in text
    top = "top" in text or "upper" in text
    left = "left" in text
    right = "right" in text
    if bottom and left:
        return "bottom_left"
    if bottom and right:
        return "bottom_right"
    if top and left:
        return "top_left"
    if top and right:
        return "top_right"
    if bottom:
        return "bottom_center"
    if top:
        return "top_center"
    return None


def _clean_phrase(phrase: str) -> str:
    """Strip location/relational suffixes that confuse Florence-2 grounding.

    Examples:
        "spectacles worn by the grandmother" -> "spectacles"
        "glasses from the woman's face"      -> "glasses"
        "hat on the man's head"              -> "hat"
    """
    # Remove relational clauses: "worn by ...", "on the ...", "from the ...", "near ...", etc.
    phrase = re.sub(
        r"\b(?:worn by|on the|on a|from the|from a|near the|near a|belonging to|attached to|of the|of a)\b.*",
        "",
        phrase,
        flags=re.IGNORECASE,
    ).strip()
    # Remove possessive tails: "... 's face", "... 's head"
    phrase = re.sub(r"\s+'s\b.*", "", phrase, flags=re.IGNORECASE).strip()
    # Remove trailing stop-words left over
    phrase = re.sub(r"\s+\b(?:the|a|an|of|on|in|at|by|for|with)\s*$", "", phrase, flags=re.IGNORECASE).strip()
    return phrase


def grounding_phrases_for_target(target: str, modifiers: list[str], verb: str) -> list[str]:
    phrases: list[str] = []
    lowered_target = target.lower().strip()

    # Always lead with the cleaned, minimal phrase so Florence-2 gets a tight query
    cleaned = _clean_phrase(lowered_target)
    if cleaned:
        phrases.append(cleaned)

    # Also add the raw target as a fallback (after the clean version)
    if lowered_target and lowered_target != cleaned:
        phrases.append(lowered_target)

    if any(keyword in lowered_target for keyword in FACE_ACCESSORY_KEYWORDS):
        accessory_terms = [keyword for keyword in FACE_ACCESSORY_KEYWORDS if keyword in lowered_target]
        phrases.extend(accessory_terms)
        phrases.append("eyewear")

    if any(keyword in lowered_target for keyword in HEAD_ACCESSORY_KEYWORDS):
        phrases.extend(keyword for keyword in HEAD_ACCESSORY_KEYWORDS if keyword in lowered_target)

    if any(keyword in lowered_target for keyword in SMALL_ACCESSORY_KEYWORDS):
        phrases.extend(keyword for keyword in SMALL_ACCESSORY_KEYWORDS if keyword in lowered_target)

    if verb == "replace":
        cleaned_target = re.sub(r"\b(?:worn by|on|from|near)\b.*", "", lowered_target).strip()
        if cleaned_target and cleaned_target != cleaned:
            phrases.append(cleaned_target)

    for modifier in modifiers:
        if verb == "replace" and modifier.lower().startswith("with "):
            continue
        cleaned_mod = _clean_phrase(modifier.lower().strip())
        if cleaned_mod:
            phrases.append(cleaned_mod)

    deduped = []
    seen = set()
    for phrase in phrases:
        normalized = " ".join(phrase.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def max_bbox_area_ratio(profile: str) -> float:
    return {
        "face_accessory": 0.07,
        "head_accessory": 0.14,
        "small_accessory": 0.05,
        "global_region": 1.0,
        "generic_local": 0.42,
    }.get(profile, 0.42)


def ideal_change_range(profile: str) -> tuple[float, float]:
    return {
        "face_accessory": (0.03, 0.24),
        "head_accessory": (0.04, 0.34),
        "small_accessory": (0.02, 0.22),
        "global_region": (0.05, 0.9),
        "generic_local": (0.03, 0.55),
    }.get(profile, (0.03, 0.55))


def fallback_box_for_profile(image_size: tuple[int, int], profile: str,
                             corner: str | None = None) -> BoundingBox:
    width, height = image_size
    if profile == "face_accessory":
        box_width = max(56, int(width * 0.22))
        box_height = max(28, int(height * 0.11))
        center_x = int(width * 0.38)
        center_y = int(height * 0.20)
        return box_from_center(center_x, center_y, box_width, box_height, image_size)
    if profile == "head_accessory":
        box_width = max(72, int(width * 0.30))
        box_height = max(48, int(height * 0.17))
        center_x = int(width * 0.40)
        center_y = int(height * 0.13)
        return box_from_center(center_x, center_y, box_width, box_height, image_size)
    if profile == "small_accessory":
        box_width = max(44, int(width * 0.16))
        box_height = max(44, int(height * 0.16))
        center_x = width // 2
        center_y = int(height * 0.42)
        return box_from_center(center_x, center_y, box_width, box_height, image_size)
    if profile == "watermark":
        bw = max(48, int(width * 0.18))
        bh = max(32, int(height * 0.12))
        corner_map = {
            "bottom_left":   (int(bw * 0.55), height - int(bh * 0.55)),
            "bottom_right":  (width - int(bw * 0.55), height - int(bh * 0.55)),
            "top_left":      (int(bw * 0.55), int(bh * 0.55)),
            "top_right":     (width - int(bw * 0.55), int(bh * 0.55)),
            "bottom_center": (width // 2, height - int(bh * 0.55)),
            "top_center":    (width // 2, int(bh * 0.55)),
        }
        cx, cy = corner_map.get(corner or "bottom_left", corner_map["bottom_left"])
        return box_from_center(cx, cy, bw, bh, image_size)
    return box_from_center(width // 2, height // 2, max(64, int(width * 0.38)), max(64, int(height * 0.38)), image_size)


def rank_grounding_candidates(
    candidates: list[GroundingCandidate],
    image_size: tuple[int, int],
    profile: str,
) -> list[GroundingCandidate]:
    width, height = image_size
    image_area = max(1, width * height)

    def candidate_score(candidate: GroundingCandidate) -> float:
        area_ratio = candidate.bbox.area / image_area
        max_ratio = max_bbox_area_ratio(profile)
        size_score = 1.0 if area_ratio <= max_ratio else max(0.0, 1.0 - min(1.0, (area_ratio - max_ratio) / max_ratio))
        center_x = (candidate.bbox.left + candidate.bbox.right) / 2.0
        center_y = (candidate.bbox.top + candidate.bbox.bottom) / 2.0
        vertical_score = 1.0
        if profile in {"face_accessory", "head_accessory"}:
            # Only penalise boxes that are implausibly in the very bottom strip of the image.
            # Do NOT assume a specific vertical band — subjects can be anywhere in frame.
            relative_y = center_y / max(1, height)
            vertical_score = 1.0 if relative_y < 0.85 else max(0.0, 1.0 - (relative_y - 0.85) / 0.15)
        horizontal_score = max(0.0, 1.0 - abs((center_x / max(1, width)) - 0.40) / 0.6)
        phrase_bonus = 0.1 if candidate.source == "phrase-grounding" else 0.0
        return 0.52 * candidate.score + 0.24 * size_score + 0.14 * vertical_score + 0.10 * horizontal_score + phrase_bonus

    return sorted(candidates, key=candidate_score, reverse=True)


def refine_bbox_for_profile(
    candidate: BoundingBox | None,
    image_size: tuple[int, int],
    profile: str,
) -> BoundingBox:
    """Size-constrain the Florence-2 bbox for the given profile.

    IMPORTANT: only the *size* of the box is clamped here.  The center is
    taken directly from the grounding result so we never drag the box away
    from where Florence-2 detected the object.
    """
    width, height = image_size
    if candidate is None:
        return fallback_box_for_profile(image_size, profile)

    if profile == "face_accessory":
        max_width = max(56, int(width * 0.28))
        max_height = max(28, int(height * 0.14))
        center_x = (candidate.left + candidate.right) // 2
        center_y = (candidate.top + candidate.bottom) // 2  # trust Florence-2
        refined_width = min(max_width, max(max_width // 2, candidate.width))
        refined_height = min(max_height, max(max_height // 2, candidate.height))
        return box_from_center(center_x, center_y, refined_width, refined_height, image_size)

    if profile == "head_accessory":
        max_width = max(72, int(width * 0.34))
        max_height = max(48, int(height * 0.20))
        center_x = (candidate.left + candidate.right) // 2
        center_y = (candidate.top + candidate.bottom) // 2  # trust Florence-2
        refined_width = min(max_width, max(max_width // 2, candidate.width))
        refined_height = min(max_height, max(max_height // 2, candidate.height))
        return box_from_center(center_x, center_y, refined_width, refined_height, image_size)

    if profile == "small_accessory":
        max_width = max(44, int(width * 0.18))
        max_height = max(44, int(height * 0.18))
        center_x = (candidate.left + candidate.right) // 2
        center_y = (candidate.top + candidate.bottom) // 2
        refined_width = min(max_width, max(max_width // 2, candidate.width))
        refined_height = min(max_height, max(max_height // 2, candidate.height))
        return box_from_center(center_x, center_y, refined_width, refined_height, image_size)

    return candidate


def box_from_center(center_x: int, center_y: int, width: int, height: int, image_size: tuple[int, int]) -> BoundingBox:
    image_width, image_height = image_size
    half_width = max(1, width // 2)
    half_height = max(1, height // 2)
    left = max(0, center_x - half_width)
    top = max(0, center_y - half_height)
    right = min(image_width, center_x + half_width)
    bottom = min(image_height, center_y + half_height)
    return BoundingBox(left=left, top=top, right=right, bottom=bottom)


def bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    ix1 = max(a.left, b.left)
    iy1 = max(a.top, b.top)
    ix2 = min(a.right, b.right)
    iy2 = min(a.bottom, b.bottom)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def rerank_with_llm_guidance(
    candidates: list[GroundingCandidate],
    guidance_bbox: BoundingBox,
    image_size: tuple[int, int],
    profile: str,
) -> list[GroundingCandidate]:
    """Re-rank Florence-2 candidates by spatial agreement with the LLM hint.

    Candidates that overlap with the LLM's predicted region get a significant
    score boost, pushing them above spurious detections elsewhere in the image.
    """
    if not candidates:
        return candidates

    def boosted_score(candidate: GroundingCandidate) -> float:
        iou = bbox_iou(candidate.bbox, guidance_bbox)
        # Also compute center distance as a softer signal
        cx = (candidate.bbox.left + candidate.bbox.right) / 2
        cy = (candidate.bbox.top + candidate.bbox.bottom) / 2
        gx = (guidance_bbox.left + guidance_bbox.right) / 2
        gy = (guidance_bbox.top + guidance_bbox.bottom) / 2
        w, h = max(1, image_size[0]), max(1, image_size[1])
        center_dist = ((cx - gx) / w) ** 2 + ((cy - gy) / h) ** 2
        proximity = max(0.0, 1.0 - center_dist * 4)  # decays with distance

        spatial_agreement = 0.6 * iou + 0.4 * proximity
        return candidate.score + 0.5 * spatial_agreement  # boost up to +0.5

    return sorted(candidates, key=boosted_score, reverse=True)

