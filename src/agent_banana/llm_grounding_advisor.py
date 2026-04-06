"""LLM Grounding Advisor — uses Gemini to reason about WHERE a target object
is located before handing off to Florence-2 for pixel-level grounding.

The advisor analyses the *source* image together with the edit instruction and
returns spatially-aware grounding phrases plus an approximate expected region.
Florence-2 candidates that agree with the LLM's spatial prediction are boosted
during re-ranking.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional
from urllib import error, parse, request

from PIL import Image

from .models import BoundingBox

# Use a text model for reasoning (cheaper + faster than image-gen model)
DEFAULT_REASONING_MODEL = "gemini-2.5-flash"
DEFAULT_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

_ADVISOR_PROMPT = """\
You are a precise visual grounding assistant.  You are given an image and an
editing instruction that mentions a **target object**.  Your job is to reason
about WHERE in the image that target is located so that a downstream object
detector (Florence-2) can find it accurately.

## Instruction
{instruction}

## Target object
{target} (verb: {verb})

## CRITICAL — Context-Based Disambiguation
Pay VERY close attention to **spatial context** in the instruction.  Many
common words are ambiguous:
  - "glasses FROM THE TABLE" = drinking glasses / cups / tumblers on a table,
    NOT eyeglasses on a person's face.
  - "glasses ON her face" = eyeglasses / spectacles.
  - "bat in the garden" = cricket / baseball bat, NOT an animal.
The phrase after the object name (e.g. "from the table", "on her face") tells
you WHERE to look and WHICH meaning is intended.  Always honour this.

## Task
1. **Disambiguate** — decide which meaning of the target word is intended
   based on the full instruction context.
2. Describe the target object as it appears in this specific image (colour,
   size, texture, spatial relationship to other objects).
3. Provide 2-4 **refined grounding phrases** — short noun phrases that would
   help a detector find the correct instance.  Be precise and context-aware.
   Example: if the instruction says "remove glasses from the table", use
   phrases like "drinking glass on table", "clear tumbler on wooden surface",
   NOT "eyeglasses on woman's face".
4. Estimate the **approximate bounding region** as normalised coordinates
   (0-1 range, origin top-left): [x_min, y_min, x_max, y_max].
5. State your **confidence** (0.0 - 1.0) that you identified the right object.

Reply with ONLY a JSON object:
```json
{{
  "object_description": "...",
  "refined_phrases": ["phrase1", "phrase2"],
  "expected_region": [x_min, y_min, x_max, y_max],
  "confidence": 0.85
}}
```
"""


@dataclass
class GroundingGuidance:
    """Structured output from the LLM advisor."""
    refined_phrases: List[str]
    expected_bbox_hint: Optional[BoundingBox]
    object_description: str
    confidence: float
    raw_response: str = ""


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _parse_guidance(raw: str, image_size: tuple[int, int]) -> GroundingGuidance:
    """Extract structured guidance from the LLM's JSON response."""
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find a JSON object in the response
        match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            return GroundingGuidance(
                refined_phrases=[],
                expected_bbox_hint=None,
                object_description="",
                confidence=0.0,
                raw_response=raw,
            )

    phrases = data.get("refined_phrases", [])
    description = str(data.get("object_description", ""))
    confidence = float(data.get("confidence", 0.5))

    bbox_hint = None
    region = data.get("expected_region")
    if region and len(region) == 4:
        w, h = image_size
        try:
            x_min, y_min, x_max, y_max = [float(v) for v in region]
            # Clamp to [0, 1]
            x_min = max(0.0, min(1.0, x_min))
            y_min = max(0.0, min(1.0, y_min))
            x_max = max(0.0, min(1.0, x_max))
            y_max = max(0.0, min(1.0, y_max))
            if x_max > x_min and y_max > y_min:
                bbox_hint = BoundingBox(
                    left=int(x_min * w),
                    top=int(y_min * h),
                    right=int(x_max * w),
                    bottom=int(y_max * h),
                )
        except (ValueError, TypeError):
            pass

    return GroundingGuidance(
        refined_phrases=phrases if isinstance(phrases, list) else [],
        expected_bbox_hint=bbox_hint,
        object_description=description,
        confidence=confidence,
        raw_response=raw,
    )


def _call_gemini_text(
    prompt: str,
    image: Image.Image,
    *,
    api_key: str,
    model: str = DEFAULT_REASONING_MODEL,
    api_base: str = DEFAULT_API_BASE,
    timeout: int = 30,
) -> str:
    """Call Gemini for text-only reasoning (no image generation)."""
    url = f"{api_base}/{parse.quote(model, safe='')}"
    url += ":generateContent"

    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": base64.b64encode(_image_to_png_bytes(image)).decode("ascii"),
                    }
                },
            ],
        }],
        "generationConfig": {
            "responseMimeType": "text/plain",
            "temperature": 0.2,
            "maxOutputTokens": 512,
        },
    }

    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")

    response_data = json.loads(raw)
    texts = []
    for candidate in response_data.get("candidates", []):
        for part in (candidate.get("content") or {}).get("parts", []):
            if part.get("text"):
                texts.append(part["text"].strip())
    return "\n".join(texts)


class GroundingAdvisor:
    """LLM-powered advisor that reasons about target location before grounding."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        self.model = model or os.getenv("AGENT_BANANA_REASONING_MODEL") or DEFAULT_REASONING_MODEL

    def advise(
        self,
        source_image: Image.Image,
        instruction: str,
        target: str,
        verb: str,
        profile: str,
    ) -> GroundingGuidance:
        """Analyse the source image and return grounding guidance."""
        if not self.api_key:
            print("[agent-banana] No API key for grounding advisor, using passthrough")
            return _passthrough_guidance()

        prompt = _ADVISOR_PROMPT.format(
            instruction=instruction,
            target=target,
            verb=verb,
        )

        try:
            raw_text = _call_gemini_text(
                prompt,
                source_image,
                api_key=self.api_key,
                model=self.model,
            )
            guidance = _parse_guidance(raw_text, source_image.size)
            print(
                f"[agent-banana] LLM advisor: {len(guidance.refined_phrases)} phrases, "
                f"bbox_hint={'yes' if guidance.expected_bbox_hint else 'no'}, "
                f"confidence={guidance.confidence:.2f}"
            )
            return guidance
        except Exception as exc:
            print(f"[agent-banana] LLM advisor failed, using passthrough: {exc}")
            return _passthrough_guidance()


class MockGroundingAdvisor:
    """Passthrough advisor for testing — returns empty guidance."""

    def advise(
        self,
        source_image: Image.Image,
        instruction: str,
        target: str,
        verb: str,
        profile: str,
    ) -> GroundingGuidance:
        return _passthrough_guidance()


def _passthrough_guidance() -> GroundingGuidance:
    return GroundingGuidance(
        refined_phrases=[],
        expected_bbox_hint=None,
        object_description="",
        confidence=0.0,
    )


def build_grounding_advisor() -> GroundingAdvisor:
    return GroundingAdvisor()


__all__ = [
    "GroundingAdvisor",
    "GroundingGuidance",
    "MockGroundingAdvisor",
    "build_grounding_advisor",
]
