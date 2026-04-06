"""Tool Registry — Defines every pipeline function as a named tool for ReAct execution.

Each tool has a name, description, input/output schema, and a callable.
The ReAct executor selects tools via LLM reasoning (Thought → Action → Observation).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from .models import BoundingBox
from .vision import crop_box, expand_box, paste_crop, draw_bbox_overlay, encode_png_data_url
from .seam_detector import boundary_penalty


@dataclass
class ToolParam:
    name: str
    type: str  # "image", "bbox", "str", "int", "float"
    description: str
    required: bool = True


@dataclass
class Tool:
    name: str
    description: str
    params: List[ToolParam]
    returns: str
    fn: Callable  # the actual callable


@dataclass
class ToolCall:
    """Record of a single tool invocation."""
    tool_name: str
    thought: str  # why the agent chose this tool
    params: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status: str = "pending"  # pending, running, success, failed

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "thought": self.thought,
            "params": {k: str(v)[:200] for k, v in self.params.items()},
            "result": self.result,
            "error": self.error,
            "status": self.status,
        }


class ToolRegistry:
    """Registry of available tools for the agent."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return tool descriptions for the LLM context."""
        result = []
        for t in self._tools.values():
            result.append({
                "name": t.name,
                "description": t.description,
                "parameters": [
                    {"name": p.name, "type": p.type, "description": p.description, "required": p.required}
                    for p in t.params
                ],
                "returns": t.returns,
            })
        return result

    def tools_prompt(self) -> str:
        """Generate a text description of all tools for the LLM."""
        lines = ["Available tools:\n"]
        for t in self._tools.values():
            params_str = ", ".join(f"{p.name}: {p.type}" for p in t.params)
            lines.append(f"  {t.name}({params_str}) -> {t.returns}")
            lines.append(f"    {t.description}\n")
        return "\n".join(lines)

    def execute(self, tool_call: ToolCall) -> ToolCall:
        """Execute a tool call and update its status/result."""
        tool = self.get(tool_call.tool_name)
        if tool is None:
            tool_call.status = "failed"
            tool_call.error = f"Unknown tool: {tool_call.tool_name}"
            return tool_call

        tool_call.status = "running"
        try:
            result = tool.fn(**tool_call.params)
            tool_call.result = result
            tool_call.status = "success"
        except Exception as exc:
            tool_call.error = str(exc)
            tool_call.status = "failed"

        return tool_call


# ---------------------------------------------------------------------------
# Tool implementations (pure functions)
# ---------------------------------------------------------------------------

def _tool_ground_target(image: Image.Image, target: str, instruction: str,
                        localizer=None, advisor=None) -> dict:
    """Ground a target object on the image."""
    if localizer is None:
        return {"error": "No localizer available", "bbox": None}

    from .targeting import grounding_phrases_for_target, rank_grounding_candidates
    phrases = grounding_phrases_for_target(target, instruction)

    # Use advisor if available
    if advisor:
        advice = advisor.advise(instruction, target)
        if advice.get("refined_phrases"):
            phrases = advice["refined_phrases"] + phrases

    candidates = localizer.ground(image, phrases)
    if not candidates:
        return {"bbox": None, "confidence": 0.0, "phrases": phrases}

    ranked = rank_grounding_candidates(candidates, image.size, target)
    best = ranked[0]
    bbox = BoundingBox(
        left=int(best.box[0]), top=int(best.box[1]),
        right=int(best.box[2]), bottom=int(best.box[3]),
    )
    return {
        "bbox": bbox.to_dict(),
        "confidence": round(best.score, 3),
        "phrases": phrases,
        "bbox_obj": bbox,
    }


def _tool_expand_region(bbox_dict: dict, padding_ratio: float,
                        image_width: int, image_height: int) -> dict:
    """Expand a bbox with context padding."""
    bbox = BoundingBox.from_dict(bbox_dict)
    pad = max(40, int(max(bbox.width, bbox.height) * padding_ratio))
    expanded = expand_box(bbox, pad, (image_width, image_height))
    return {
        "edit_region": expanded.to_dict(),
        "edit_region_obj": expanded,
        "padding_px": pad,
    }


def _tool_crop_local_patch(image: Image.Image, region_dict: dict) -> dict:
    """Crop a local patch from the image."""
    region = BoundingBox.from_dict(region_dict)
    crop = crop_box(image, region)
    return {
        "crop": crop,
        "width": crop.size[0],
        "height": crop.size[1],
    }


def _tool_edit_local(crop: Image.Image, instruction: str, target: str,
                     image_client=None) -> dict:
    """Edit a local crop using Gemini."""
    if image_client is None:
        return {"error": "No image client available"}

    verb = instruction.strip().split()[0].lower() if instruction.strip() else ""
    is_removal = verb in {"remove", "delete", "erase", "eliminate", "hide"}

    if is_removal:
        prompt = (
            f"Edit this image: {instruction}. "
            f"The {target} must be COMPLETELY GONE from the result — do not leave any trace of it. "
            f"Fill the area where it was with realistic background: match the surrounding "
            f"grass, ground, sky, or environment texture seamlessly. "
            f"All other objects and the background outside the removed area must remain unchanged."
        )
    else:
        prompt = (
            f"Edit this image crop: {instruction}. "
            f"Keep all other elements exactly as they appear — "
            f"same colors, lighting, textures, and positions. "
            f"The result must blend naturally with its surroundings."
        )

    response = image_client.edit_full_image(crop, prompt)
    edited = response.image.convert("RGB").resize(crop.size)
    return {
        "edited_crop": edited,
        "width": edited.size[0],
        "height": edited.size[1],
    }


def _tool_blend_back(base_image: Image.Image, edited_crop: Image.Image,
                     region_dict: dict) -> dict:
    """Laplacian pyramid blend an edited crop back into the base."""
    region = BoundingBox.from_dict(region_dict)
    merged = paste_crop(base_image, edited_crop, region)
    return {
        "merged": merged,
        "width": merged.size[0],
        "height": merged.size[1],
    }


def _tool_detect_seam(merged_image: Image.Image, region_dict: dict) -> dict:
    """Run seam detection on a merged image."""
    region = BoundingBox.from_dict(region_dict)
    result = boundary_penalty(merged_image, region)
    return result


def _tool_adjust_taper(base_image: Image.Image, edited_crop: Image.Image,
                       region_dict: dict, taper_px: int = 5) -> dict:
    """Re-blend with a wider taper to fix seam."""
    region = BoundingBox.from_dict(region_dict)
    # Re-blend with custom taper
    merged = paste_crop(base_image, edited_crop, region)
    seam = boundary_penalty(merged, region)
    return {
        "merged": merged,
        "taper_px": taper_px,
        "seam_score": seam["penalty"],
        "seam_verdict": seam["verdict"],
    }


def _tool_evaluate_quality(original: Image.Image, merged: Image.Image,
                           bbox_dict: dict, target: str = "",
                           verb: str = "") -> dict:
    """Run full quality evaluation."""
    from .quality import QualityJudge
    bbox = BoundingBox.from_dict(bbox_dict)
    judge = QualityJudge()
    q = judge.evaluate(original, merged, bbox, target=target, verb=verb)
    return q.to_dict()


# ---------------------------------------------------------------------------
# Build default registry
# ---------------------------------------------------------------------------

def build_tool_registry() -> ToolRegistry:
    """Create and populate the default tool registry."""
    reg = ToolRegistry()

    reg.register(Tool(
        name="ground_target",
        description="Locate a target object on the image using Florence-2 + LLM spatial reasoning",
        params=[
            ToolParam("image", "image", "The source image to search"),
            ToolParam("target", "str", "What to find (e.g., 'glasses', 'napkin')"),
            ToolParam("instruction", "str", "The user's full instruction for context"),
        ],
        returns="bbox, confidence, phrases",
        fn=_tool_ground_target,
    ))

    reg.register(Tool(
        name="expand_region",
        description="Expand a bounding box with context padding for local editing",
        params=[
            ToolParam("bbox_dict", "bbox", "The bbox to expand"),
            ToolParam("padding_ratio", "float", "Padding as ratio of bbox size (e.g., 0.5 = 50%)"),
            ToolParam("image_width", "int", "Image width in pixels"),
            ToolParam("image_height", "int", "Image height in pixels"),
        ],
        returns="edit_region (expanded bbox), padding_px",
        fn=_tool_expand_region,
    ))

    reg.register(Tool(
        name="crop_local_patch",
        description="Crop a local patch from the image at the given region",
        params=[
            ToolParam("image", "image", "Source image"),
            ToolParam("region_dict", "bbox", "Region to crop"),
        ],
        returns="crop (PIL Image), width, height",
        fn=_tool_crop_local_patch,
    ))

    reg.register(Tool(
        name="edit_local",
        description="Edit a local crop using Gemini (acts as inpainter with surrounding context)",
        params=[
            ToolParam("crop", "image", "The local crop to edit"),
            ToolParam("instruction", "str", "What to do (e.g., 'remove the glass')"),
            ToolParam("target", "str", "Target object name"),
        ],
        returns="edited_crop (PIL Image)",
        fn=_tool_edit_local,
    ))

    reg.register(Tool(
        name="blend_back",
        description="Laplacian pyramid blend an edited crop back into the base image (seamless fusion)",
        params=[
            ToolParam("base_image", "image", "The original full image"),
            ToolParam("edited_crop", "image", "The edited crop to blend in"),
            ToolParam("region_dict", "bbox", "The region where the crop goes"),
        ],
        returns="merged (PIL Image)",
        fn=_tool_blend_back,
    ))

    reg.register(Tool(
        name="detect_seam",
        description="Check for visible seams at the blend boundary using BGD + CBCS",
        params=[
            ToolParam("merged_image", "image", "The merged image to check"),
            ToolParam("region_dict", "bbox", "The region that was edited"),
        ],
        returns="bgd, cbcs, penalty, verdict (accept/warn/reject)",
        fn=_tool_detect_seam,
    ))

    reg.register(Tool(
        name="adjust_taper",
        description="Re-blend with a wider taper to fix a detected seam",
        params=[
            ToolParam("base_image", "image", "Original full image"),
            ToolParam("edited_crop", "image", "Edited crop"),
            ToolParam("region_dict", "bbox", "Edit region"),
            ToolParam("taper_px", "int", "Taper width in pixels", required=False),
        ],
        returns="merged (PIL Image), seam_score, seam_verdict",
        fn=_tool_adjust_taper,
    ))

    reg.register(Tool(
        name="evaluate_quality",
        description="Run full quality evaluation on a composition (inside/outside change + seam)",
        params=[
            ToolParam("original", "image", "Original image before editing"),
            ToolParam("merged", "image", "Image after editing"),
            ToolParam("bbox_dict", "bbox", "Target bounding box"),
            ToolParam("target", "str", "Target name", required=False),
            ToolParam("verb", "str", "Edit verb", required=False),
        ],
        returns="quality metrics dict",
        fn=_tool_evaluate_quality,
    ))

    reg.register(Tool(
        name="finish",
        description="Signal that editing is complete and return the current result",
        params=[
            ToolParam("reason", "str", "Why the agent decided to finish"),
        ],
        returns="status message",
        fn=lambda reason="done": {"status": "finished", "reason": reason},
    ))

    return reg


__all__ = [
    "Tool", "ToolCall", "ToolParam", "ToolRegistry",
    "build_tool_registry",
]
