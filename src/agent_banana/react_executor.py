"""ReAct Agent Executor — True VLM-driven Thought → Action → Observation loop.

The executor uses Gemini as a VLM orchestrator that SEES the current image
and bounding box, then DECIDES which tool to call at each step:

1. THINK: VLM analyzes the current image + bbox and decides the next action
2. ACT: Execute the selected tool from the ToolRegistry
3. OBSERVE: Feed the result back to the VLM (with updated image)
4. REPEAT until VLM calls 'finish' or max steps reached

Key design decisions:
- Each attempt CHAINS on the previous result (not the original)
- VLM Critic compares the UNTOUCHED original vs the current state
- The VLM sees the current image with bbox overlay + scratchpad context
- On failure to parse VLM output, falls back to a deterministic sequence
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib import error, parse, request

from PIL import Image

from .models import BoundingBox, QualityMetrics
from .quality import QualityJudge
from .seam_detector import boundary_penalty
from .tool_registry import ToolRegistry, build_tool_registry
from .vlm_critic import VLMCritic, CriticVerdict
from .vision import crop_box, expand_box, paste_crop, encode_png_data_url, center_box


# ─── Default orchestrator model ───
DEFAULT_ORCHESTRATOR_MODEL = "gemini-2.5-flash"
DEFAULT_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


@dataclass
class AgentStep:
    """A single step in the agent's reasoning trace."""
    step_num: int
    thought: str
    action: str
    params: Dict[str, Any]
    observation: str
    critic_verdict: Optional[CriticVerdict] = None
    image_url: str = ""
    duration_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "step_num": self.step_num,
            "thought": self.thought,
            "action": self.action,
            "params": self.params,
            "observation": self.observation,
            "critic_verdict": self.critic_verdict.to_dict() if self.critic_verdict else None,
            "image_url": self.image_url,
            "duration_ms": self.duration_ms,
        }


@dataclass
class AgentResult:
    """Complete result of an agent execution."""
    success: bool
    final_image: Optional[Image.Image]
    final_image_url: str
    quality: Optional[QualityMetrics]
    steps: List[AgentStep]
    total_attempts: int
    total_duration_ms: int

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "final_image_url": self.final_image_url,
            "quality": self.quality.to_dict() if self.quality else None,
            "steps": [s.to_dict() for s in self.steps],
            "total_attempts": self.total_attempts,
            "total_duration_ms": self.total_duration_ms,
        }


# ─── VLM Orchestrator Prompt ───

_SYSTEM_PROMPT = """\
You are a ReAct image editing agent with VISION. You can SEE the current state
of the image being edited. The attached image shows the current working image
with a RED bounding box drawn around the target region.

Analyze the image carefully before deciding your next action.

## State
- Original image size: {img_w}x{img_h}
- Target: "{target}" (action: {verb})
- Bounding box: left={bbox_left}, top={bbox_top}, right={bbox_right}, bottom={bbox_bottom}
- Instruction: "{instruction}"
- Attempt: {attempt}/{max_iterations}

## Available Tools
{tools_desc}

## Rules — FOLLOW THIS EXACT SEQUENCE, NO SKIPPING
1. LOOK at the attached image — the RED rectangle marks the bounding box region
2. ALWAYS call expand_region FIRST
3. ALWAYS call crop_local_patch SECOND
4. ALWAYS call edit_local THIRD — this is where the actual edit happens
5. ALWAYS call blend_back FOURTH — merge the edit into the full image
6. ALWAYS call evaluate_quality FIFTH
7. Only call finish AFTER blend_back and evaluate_quality are done
8. Do NOT call finish, verify_semantic, or ground_target before completing steps 2-6
9. If quality is poor, use detect_seam and adjust_taper, then finish

## Previous Steps
{scratchpad}

## Response Format
Respond with ONLY a JSON object:
```json
{{
  "thought": "your reasoning about what to do next based on what you see",
  "action": "tool_name",
  "action_input": {{
    "param1": "value1"
  }}
}}
```
"""


def _image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL Image to base64-encoded string."""
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        img.convert("RGB").save(buf, format="JPEG", quality=70)
    else:
        img.save(buf, format=fmt)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _downscale_for_vlm(img: Image.Image, max_dim: int = 512) -> Image.Image:
    """Downscale image so the longest side is at most max_dim pixels.

    This dramatically reduces the base64 payload size sent to the VLM,
    cutting API latency from 30s+ down to a few seconds.
    """
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    scale = max_dim / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def _draw_bbox_on_image(img: Image.Image, bbox: BoundingBox) -> Image.Image:
    """Draw a red bounding box rectangle on a copy of the image."""
    from PIL import ImageDraw
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    # Draw a 3-pixel-wide red rectangle around the bbox
    for offset in range(3):
        draw.rectangle(
            [
                bbox.left - offset, bbox.top - offset,
                bbox.right + offset, bbox.bottom + offset,
            ],
            outline="red",
        )
    return annotated


def _call_orchestrator_vlm(
    prompt: str,
    image: Image.Image,
    bbox: BoundingBox,
    *,
    api_key: str,
    model: str = DEFAULT_ORCHESTRATOR_MODEL,
    temperature: float = 0.2,
) -> str:
    """Call Gemini as a VLM with the current image + bbox overlay for orchestration."""
    url = f"{DEFAULT_API_BASE}/{parse.quote(model, safe='')}:generateContent"

    # Downscale image for faster VLM processing (keeps bbox proportional)
    orig_w, orig_h = image.size
    small_image = _downscale_for_vlm(image, max_dim=512)
    scale_x = small_image.size[0] / orig_w
    scale_y = small_image.size[1] / orig_h

    # Scale bbox to match the downscaled image
    scaled_bbox = BoundingBox(
        left=int(bbox.left * scale_x),
        top=int(bbox.top * scale_y),
        right=int(bbox.right * scale_x),
        bottom=int(bbox.bottom * scale_y),
    )

    # Draw the bounding box on the downscaled image
    annotated_image = _draw_bbox_on_image(small_image, scaled_bbox)
    # Use JPEG for much smaller payload (~50KB vs ~1MB+ for PNG)
    image_b64 = _image_to_base64(annotated_image, fmt="JPEG")

    # Build multimodal parts: text prompt + annotated image
    parts = [
        {"text": prompt},
        {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": image_b64,
            }
        },
    ]

    # Force structured JSON output so the VLM returns parseable responses
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "thought": {"type": "STRING"},
            "action": {"type": "STRING"},
            "action_input": {"type": "OBJECT"},
        },
        "required": ["thought", "action", "action_input"],
    }

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema,
            "temperature": temperature,
            "maxOutputTokens": 1024,
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

    with request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")

    data = json.loads(raw)
    texts = []
    for candidate in data.get("candidates", []):
        for part in (candidate.get("content") or {}).get("parts", []):
            if part.get("text"):
                texts.append(part["text"].strip())
    return "\n".join(texts)


def _parse_llm_action(text: str) -> Optional[Dict[str, Any]]:
    """Extract action JSON from LLM response text."""
    # Try direct JSON parse
    try:
        data = json.loads(text.strip())
        if "action" in data:
            return data
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code block
    json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if "action" in data:
                return data
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "action" in data:
                return data
        except json.JSONDecodeError:
            pass

    return None


class ReActExecutor:
    """ReAct-style agent executor for image editing.

    Uses a VLM (Gemini with vision) to dynamically decide which tool to call
    at each step, seeing the current image state with bounding box overlay.
    Forms a true Thought → Action → Observation loop. Falls back to a
    deterministic sequence if the VLM fails to produce valid actions.

    Critical design decisions:
    - Each attempt CHAINS on the previous result (not the original)
    - VLM Critic compares the UNTOUCHED original vs the current state
    - On retry: wider bbox to catch missed targets, stronger prompt
    - Tracks best result across all attempts
    """

    def __init__(
        self,
        image_client,
        quality_judge: QualityJudge,
        vlm_critic: Optional[VLMCritic] = None,
        max_iterations: int = 3,
        max_steps_per_attempt: int = 10,
    ):
        self.image_client = image_client
        self.quality_judge = quality_judge
        self.vlm_critic = vlm_critic
        self.max_iterations = max_iterations
        self.max_steps_per_attempt = max_steps_per_attempt
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.tool_registry = build_tool_registry()

    def execute_edit(
        self,
        original_image: Image.Image,
        instruction: str,
        target: str,
        verb: str,
        bbox: BoundingBox,
        *,
        target_profile: str = "",
        modifiers: list = None,
        step_callback=None,
    ) -> AgentResult:
        """Run the full ReAct agent loop for a single edit step.

        Args:
            step_callback: Optional callable(AgentStep) invoked after each
                step completes, enabling real-time streaming to the UI.
        """
        start_time = time.time()
        all_steps: List[AgentStep] = []
        step_num = 0
        _emit = step_callback or (lambda s: None)

        # UNTOUCHED original — never modified, used for VLM comparison
        untouched_original = original_image.copy()
        # Working image — updated after each attempt (chains results)
        working_image = original_image.copy()
        current_bbox = bbox
        best_result = None
        best_score = -1.0

        for attempt in range(1, self.max_iterations + 1):
            # ─── Try VLM-driven ReAct loop for this attempt ───
            attempt_steps, composed, quality, success = self._run_react_attempt(
                working_image=working_image,
                untouched_original=untouched_original,
                instruction=instruction,
                target=target,
                verb=verb,
                bbox=current_bbox,
                attempt=attempt,
                step_num_offset=step_num,
                step_callback=step_callback,
            )

            step_num += len(attempt_steps)
            all_steps.extend(attempt_steps)

            if composed is None:
                composed = working_image

            if quality is None:
                quality = self.quality_judge.evaluate(
                    untouched_original, composed, current_bbox,
                    preview=composed, target=target, verb=verb,
                )

            # ─── VLM Critic: compare UNTOUCHED original vs composed ───
            critic_verdict = None
            if self.vlm_critic:
                step_num += 1
                t0 = time.time()

                try:
                    critic_verdict = self.vlm_critic.verify_edit(
                        original=untouched_original,
                        result=composed,
                        instruction=instruction,
                        target=target,
                        verb=verb,
                    )
                except Exception as exc:
                    critic_verdict = CriticVerdict(
                        fulfilled=False, confidence=0.0, semantic_score=0.3,
                        reasoning=f"Critic error: {exc}",
                        residual_objects=[], suggestions=["VLM Critic failed, retry"],
                    )

                quality.semantic_score = critic_verdict.semantic_score
                quality.semantic_fulfilled = critic_verdict.fulfilled
                quality.semantic_reasoning = critic_verdict.reasoning

                critic_step = AgentStep(
                    step_num=step_num,
                    thought=(
                        f"VLM Critic comparing ORIGINAL (untouched) vs "
                        f"RESULT (after {attempt} attempt(s)). "
                        f"Looking for missed instances of '{target}'."
                    ),
                    action="verify_semantic",
                    params={"semantic_score": round(critic_verdict.semantic_score, 2),
                            "fulfilled": critic_verdict.fulfilled,
                            "attempt": attempt},
                    observation=critic_verdict.reasoning,
                    critic_verdict=critic_verdict,
                    duration_ms=int((time.time() - t0) * 1000),
                )
                all_steps.append(critic_step)
                _emit(critic_step)

                if not critic_verdict.fulfilled:
                    quality.accepted = False
                    quality.notes.append(
                        f"[Attempt {attempt}] VLM REJECTED: {critic_verdict.reasoning}"
                    )
                    if critic_verdict.residual_objects:
                        quality.notes.append(
                            f"Still visible: {', '.join(critic_verdict.residual_objects)}"
                        )
                    print(f"[react] Attempt {attempt} REJECTED "
                          f"(semantic={critic_verdict.semantic_score:.2f})")
                else:
                    quality.notes.append(
                        f"[Attempt {attempt}] VLM APPROVED "
                        f"(semantic={critic_verdict.semantic_score:.2f})"
                    )
                    print(f"[react] Attempt {attempt} APPROVED "
                          f"(semantic={critic_verdict.semantic_score:.2f})")

            # ─── Track best result ───
            combined_score = quality.score * quality.semantic_score
            if combined_score > best_score:
                best_score = combined_score
                best_result = (composed, quality)

            # ─── DECIDE: Continue or stop? ───
            is_approved = (
                critic_verdict is not None
                and critic_verdict.fulfilled
                and critic_verdict.semantic_score >= 0.7
                and quality.accepted
            )
            if is_approved or success:
                total_ms = int((time.time() - start_time) * 1000)
                return AgentResult(
                    success=True,
                    final_image=composed,
                    final_image_url=encode_png_data_url(composed),
                    quality=quality,
                    steps=all_steps,
                    total_attempts=attempt,
                    total_duration_ms=total_ms,
                )

            # ─── CHAIN: Use this result as starting point for next attempt ───
            working_image = composed

            # ─── Log strategy adjustment for next attempt ───
            if attempt < self.max_iterations:
                step_num += 1
                suggestion = ""
                if critic_verdict and critic_verdict.suggestions:
                    suggestion = critic_verdict.suggestions[0]
                residuals = ""
                if critic_verdict and critic_verdict.residual_objects:
                    residuals = ", ".join(critic_verdict.residual_objects)

                strategy_step = AgentStep(
                    step_num=step_num,
                    thought=(
                        f"Chaining result: using attempt {attempt}'s output as "
                        f"new starting image. On next attempt I'll use a wider "
                        f"edit region to catch ALL missed instances."
                        + (f" Residuals to fix: {residuals}" if residuals else "")
                        + (f" Suggestion: {suggestion}" if suggestion else "")
                    ),
                    action="adjust_strategy",
                    params={"chained": True,
                            "residuals": residuals,
                            "suggestion": suggestion[:100]},
                    observation=(
                        f"Next attempt will edit from the current result image "
                        f"with wider region"
                    ),
                )
                all_steps.append(strategy_step)
                _emit(strategy_step)

        # ─── Max attempts reached ───
        total_ms = int((time.time() - start_time) * 1000)
        best_image, best_quality = best_result if best_result else (working_image, None)

        step_num += 1
        final_step = AgentStep(
            step_num=step_num,
            thought=f"Reached max attempts ({self.max_iterations}). "
                    f"Returning best result (score={best_score:.3f}).",
            action="return_best",
            params={"best_score": round(best_score, 3),
                    "attempts": self.max_iterations},
            observation=f"Returning best of {self.max_iterations} attempts",
        )
        all_steps.append(final_step)
        _emit(final_step)

        return AgentResult(
            success=False,
            final_image=best_image,
            final_image_url=encode_png_data_url(best_image),
            quality=best_quality,
            steps=all_steps,
            total_attempts=self.max_iterations,
            total_duration_ms=total_ms,
        )

    def _run_react_attempt(
        self,
        working_image: Image.Image,
        untouched_original: Image.Image,
        instruction: str,
        target: str,
        verb: str,
        bbox: BoundingBox,
        attempt: int,
        step_num_offset: int,
        step_callback=None,
    ) -> tuple:
        """Run a single ReAct attempt with VLM-driven tool selection.

        Returns: (steps, composed_image, quality, success)
        """
        # Try VLM-driven approach first, fall back to deterministic
        if self.api_key:
            try:
                return self._vlm_driven_attempt(
                    working_image, untouched_original, instruction,
                    target, verb, bbox, attempt, step_num_offset,
                    step_callback=step_callback,
                )
            except Exception as exc:
                print(f"[react] VLM-driven attempt failed ({exc}), falling back to deterministic")

        return self._deterministic_attempt(
            working_image, untouched_original, instruction,
            target, verb, bbox, attempt, step_num_offset,
            step_callback=step_callback,
        )

    def _vlm_driven_attempt(
        self,
        working_image: Image.Image,
        untouched_original: Image.Image,
        instruction: str,
        target: str,
        verb: str,
        bbox: BoundingBox,
        attempt: int,
        step_num_offset: int,
        step_callback=None,
    ) -> tuple:
        """VLM-driven ReAct loop: See Image → Think → Act → Observe."""
        steps: List[AgentStep] = []
        step_num = step_num_offset
        scratchpad_entries: List[str] = []
        _emit = step_callback or (lambda s: None)

        # State that tools can read and write
        state = {
            "working_image": working_image,
            "original_image": untouched_original,
            "bbox": bbox,
            "edit_region": None,
            "local_crop": None,
            "edited_crop": None,
            "composed": None,
            "quality": None,
        }

        # Only show tools the executor actually handles (exclude ground_target)
        _executor_tools = {
            "expand_region", "crop_local_patch", "edit_local",
            "blend_back", "detect_seam", "adjust_taper",
            "evaluate_quality", "finish",
        }
        all_tools = self.tool_registry.list_tools()
        filtered_lines = ["Available tools:\n"]
        for t in all_tools:
            if t["name"] in _executor_tools:
                params_str = ", ".join(f"{p['name']}: {p['type']}" for p in t["parameters"])
                filtered_lines.append(f"  {t['name']}({params_str}) -> {t['returns']}")
                filtered_lines.append(f"    {t['description']}\n")
        tools_desc = "\n".join(filtered_lines)

        for loop_step in range(self.max_steps_per_attempt):
            step_num += 1
            t0 = time.time()

            # Build scratchpad from previous steps
            scratchpad = "\n".join(scratchpad_entries) if scratchpad_entries else "(No previous steps)"

            # ─── THINK: Ask VLM what to do next (with current image + bbox) ───
            # Use the current working image (which may have been updated by
            # previous steps like blend_back)
            current_image = state.get("composed") or working_image

            system_prompt = _SYSTEM_PROMPT.format(
                img_w=current_image.size[0],
                img_h=current_image.size[1],
                target=target,
                verb=verb,
                bbox_left=bbox.left,
                bbox_top=bbox.top,
                bbox_right=bbox.right,
                bbox_bottom=bbox.bottom,
                instruction=instruction,
                attempt=attempt,
                max_iterations=self.max_iterations,
                tools_desc=tools_desc,
                scratchpad=scratchpad,
            )

            vlm_response = _call_orchestrator_vlm(
                system_prompt,
                image=current_image,
                bbox=bbox,
                api_key=self.api_key,
            )

            # ─── Parse the VLM's decision ───
            parsed = _parse_llm_action(vlm_response)
            if not parsed:
                print(f"[react] VLM response unparseable at step {loop_step + 1}, "
                      f"falling back to deterministic")
                # Fall back to deterministic for remaining steps
                det_steps, composed, quality, success = self._deterministic_attempt(
                    state["composed"] or working_image,
                    untouched_original, instruction, target, verb,
                    bbox, attempt, step_num,
                )
                steps.extend(det_steps)
                return steps, composed, quality, success

            thought = parsed.get("thought", "")
            action_name = parsed.get("action", "")
            action_input = parsed.get("action_input", {})

            # ─── Handle 'finish' action ───
            if action_name == "finish":
                # If no edit was performed yet, force the pipeline first
                if state.get("composed") is None:
                    forced = self._force_remaining_pipeline(
                        state, working_image, untouched_original,
                        instruction, target, verb, attempt, step_num,
                        _emit,
                    )
                    steps.extend(forced[0])
                    return steps, forced[1], forced[2], False
                composed = state.get("composed") or working_image
                quality = state.get("quality")
                steps.append(AgentStep(
                    step_num=step_num,
                    thought=thought,
                    action="finish",
                    params=action_input,
                    observation="Agent decided to finish — returning result.",
                    image_url=encode_png_data_url(composed) if composed else "",
                    duration_ms=int((time.time() - t0) * 1000),
                ))
                return steps, composed, quality, True

            # ─── ACT: Execute the tool ───
            observation, result_data = self._execute_react_tool(
                action_name, action_input, state, instruction, target, verb, attempt,
            )

            elapsed_ms = int((time.time() - t0) * 1000)

            # Build image_url for the step if we have a composed image
            image_url = ""
            if action_name == "blend_back" and state.get("composed"):
                image_url = encode_png_data_url(state["composed"])
            elif action_name == "edit_local" and state.get("edited_crop"):
                image_url = encode_png_data_url(state["edited_crop"])

            step_obj = AgentStep(
                step_num=step_num,
                thought=thought,
                action=action_name,
                params={k: str(v)[:200] if not isinstance(v, (int, float, bool)) else v
                        for k, v in action_input.items()},
                observation=observation,
                image_url=image_url,
                duration_ms=elapsed_ms,
            )
            steps.append(step_obj)
            _emit(step_obj)

            # Update scratchpad
            scratchpad_entries.append(
                f"Step {loop_step + 1}: Thought: {thought[:150]}\n"
                f"  Action: {action_name}\n"
                f"  Observation: {observation[:200]}"
            )

        # Max steps reached — if the edit pipeline never completed, force it now
        if state.get("composed") is None:
            forced = self._force_remaining_pipeline(
                state, working_image, untouched_original,
                instruction, target, verb, attempt, step_num,
                _emit,
            )
            steps.extend(forced[0])
            return steps, forced[1], forced[2], False

        composed = state.get("composed") or working_image
        quality = state.get("quality")
        return steps, composed, quality, False

    def _execute_react_tool(
        self,
        action_name: str,
        action_input: dict,
        state: dict,
        instruction: str,
        target: str,
        verb: str,
        attempt: int,
    ) -> tuple:
        """Execute a tool and update the shared state. Returns (observation, result_data)."""

        working_image = state["working_image"]
        bbox = state["bbox"]

        if action_name == "expand_region":
            padding_ratio = float(action_input.get("padding_ratio", 0.5 + (attempt - 1) * 0.25))
            w, h = working_image.size

            if attempt >= 2:
                # On retries, use near-global region
                margin_x = int(w * 0.05)
                margin_y = int(h * 0.05)
                edit_region = BoundingBox(
                    left=margin_x, top=margin_y,
                    right=w - margin_x, bottom=h - margin_y,
                )
                strategy = "global-retry"
            else:
                pad = max(40, int(max(bbox.width, bbox.height) * padding_ratio))
                edit_region = expand_box(bbox, pad, working_image.size)
                strategy = "local"

            state["edit_region"] = edit_region
            observation = (
                f"Expanded region: {edit_region.width}x{edit_region.height} "
                f"({strategy}, padding={padding_ratio:.0%})"
            )
            return observation, {"edit_region": edit_region.to_dict()}

        elif action_name == "crop_local_patch":
            edit_region = state.get("edit_region")
            if not edit_region:
                return "Error: No edit region set. Call expand_region first.", {}

            local_crop = crop_box(working_image, edit_region)
            state["local_crop"] = local_crop
            observation = f"Cropped {local_crop.size[0]}x{local_crop.size[1]} from working image"
            return observation, {"size": local_crop.size}

        elif action_name == "edit_local":
            local_crop = state.get("local_crop")
            if not local_crop:
                return "Error: No local crop available. Call crop_local_patch first.", {}

            prompt = self._build_prompt(instruction, target, verb, attempt)
            custom_prompt = action_input.get("prompt")
            if custom_prompt:
                prompt = custom_prompt

            try:
                edited_response = self.image_client.edit_full_image(local_crop, prompt)
                edited_crop = edited_response.image.convert("RGB").resize(local_crop.size)
                state["edited_crop"] = edited_crop
                observation = (
                    f"Gemini returned {edited_crop.size[0]}x{edited_crop.size[1]} edit "
                    f"(attempt {attempt})"
                )
                return observation, {"size": edited_crop.size}
            except Exception as exc:
                observation = f"Edit failed: {exc}"
                return observation, {"error": str(exc)}

        elif action_name == "blend_back":
            edited_crop = state.get("edited_crop")
            edit_region = state.get("edit_region")
            if not edited_crop or not edit_region:
                return "Error: Need edited_crop and edit_region. Call edit_local first.", {}

            composed = paste_crop(working_image, edited_crop, edit_region)
            state["composed"] = composed
            state["working_image"] = composed  # Chain for subsequent operations
            observation = (
                f"Blended {edit_region.width}x{edit_region.height} region "
                f"(attempt {attempt})"
            )
            return observation, {"region": edit_region.to_dict()}

        elif action_name == "detect_seam":
            composed = state.get("composed")
            if not composed:
                return "Error: No composed image. Call blend_back first.", {}

            seam = boundary_penalty(composed, bbox)
            observation = (
                f"BGD={seam['bgd']:.3f}, CBCS={seam['cbcs']:.3f}, "
                f"penalty={seam['penalty']:.3f}, verdict={seam['verdict']}"
            )
            return observation, seam

        elif action_name == "adjust_taper":
            edited_crop = state.get("edited_crop")
            edit_region = state.get("edit_region")
            if not edited_crop or not edit_region:
                return "Error: Need edited_crop and edit_region.", {}

            composed = paste_crop(working_image, edited_crop, edit_region)
            state["composed"] = composed
            seam = boundary_penalty(composed, bbox)
            observation = (
                f"Re-blended with taper. seam_score={seam['penalty']:.3f}, "
                f"verdict={seam['verdict']}"
            )
            return observation, {"seam": seam}

        elif action_name == "evaluate_quality":
            composed = state.get("composed")
            if not composed:
                return "Error: No composed image to evaluate.", {}

            quality = self.quality_judge.evaluate(
                state["original_image"], composed, bbox,
                preview=composed, target=target, verb=verb,
            )
            state["quality"] = quality
            observation = (
                f"Score={quality.score:.3f}, seam={quality.seam_verdict}, "
                f"inside_Δ={quality.inside_change:.3f}, "
                f"outside_Δ={quality.outside_change:.3f}, "
                f"accepted={quality.accepted}"
            )
            return observation, quality.to_dict()

        else:
            return f"Unknown tool: {action_name}. Use one of the available tools.", {}

    def _force_remaining_pipeline(
        self,
        state: dict,
        working_image: Image.Image,
        untouched_original: Image.Image,
        instruction: str,
        target: str,
        verb: str,
        attempt: int,
        step_num: int,
        emit,
    ) -> tuple:
        """Force the mandatory crop→edit→blend→evaluate sequence when the VLM skipped it."""
        steps = []
        bbox = state["bbox"]

        # Ensure we have an edit_region
        if state.get("edit_region") is None:
            padding_ratio = 0.5 + (attempt - 1) * 0.25
            pad = max(40, int(max(bbox.width, bbox.height) * padding_ratio))
            state["edit_region"] = expand_box(bbox, pad, working_image.size)

        edit_region = state["edit_region"]

        # Crop
        step_num += 1
        local_crop = crop_box(working_image, edit_region)
        state["local_crop"] = local_crop
        s = AgentStep(step_num=step_num, thought="Forcing crop (VLM skipped pipeline).",
                      action="crop_local_patch", params={},
                      observation=f"Cropped {local_crop.size[0]}x{local_crop.size[1]} from working image")
        steps.append(s); emit(s)

        # Edit
        step_num += 1
        prompt = self._build_prompt(instruction, target, verb, attempt)
        try:
            edited_response = self.image_client.edit_full_image(local_crop, prompt)
            edited_crop = edited_response.image.convert("RGB").resize(local_crop.size)
            state["edited_crop"] = edited_crop
            obs = f"Gemini returned {edited_crop.size[0]}x{edited_crop.size[1]} edit (attempt {attempt})"
            img_url = encode_png_data_url(edited_crop)
        except Exception as exc:
            edited_crop = local_crop
            obs = f"Edit failed: {exc}"
            img_url = ""
        s = AgentStep(step_num=step_num, thought="Applying edit.", action="edit_local",
                      params={}, observation=obs, image_url=img_url)
        steps.append(s); emit(s)

        # Blend
        step_num += 1
        composed = paste_crop(working_image, edited_crop, edit_region)
        state["composed"] = composed
        img_url = encode_png_data_url(composed)
        s = AgentStep(step_num=step_num, thought="Blending edited crop back.", action="blend_back",
                      params={}, observation=f"Blended {edit_region.width}x{edit_region.height} region",
                      image_url=img_url)
        steps.append(s); emit(s)

        # Evaluate
        step_num += 1
        quality = self.quality_judge.evaluate(
            untouched_original, composed, bbox,
            preview=composed, target=target, verb=verb,
        )
        state["quality"] = quality
        s = AgentStep(step_num=step_num, thought="Evaluating quality.", action="evaluate_quality",
                      params={}, observation=(
                          f"Score={quality.score:.3f}, seam={quality.seam_verdict}, "
                          f"inside_Δ={quality.inside_change:.3f}, outside_Δ={quality.outside_change:.3f}"))
        steps.append(s); emit(s)

        return steps, composed, quality

    def _deterministic_attempt(
        self,
        working_image: Image.Image,
        untouched_original: Image.Image,
        instruction: str,
        target: str,
        verb: str,
        bbox: BoundingBox,
        attempt: int,
        step_num_offset: int,
        step_callback=None,
    ) -> tuple:
        """Deterministic fallback: expand → crop → edit → blend → evaluate."""
        steps: List[AgentStep] = []
        step_num = step_num_offset
        padding_ratio = 0.5 + (attempt - 1) * 0.25
        _emit = step_callback or (lambda s: None)

        # ─── Step 1: Expand region ───
        step_num += 1
        t0 = time.time()
        w, h = working_image.size

        if attempt >= 2:
            margin_x = int(w * 0.05)
            margin_y = int(h * 0.05)
            edit_region = BoundingBox(
                left=margin_x, top=margin_y,
                right=w - margin_x, bottom=h - margin_y,
            )
            strategy = "global-retry"
        else:
            pad = max(40, int(max(bbox.width, bbox.height) * padding_ratio))
            edit_region = expand_box(bbox, pad, working_image.size)
            strategy = "local"

        s1 = AgentStep(
            step_num=step_num,
            thought=(
                f"Attempt {attempt}/{self.max_iterations} (deterministic fallback). "
                f"Expanding bbox by {int(padding_ratio*100)}% for context."
            ),
            action="expand_region",
            params={"bbox": bbox.to_dict(), "padding": padding_ratio,
                    "edit_region": edit_region.to_dict(), "strategy": strategy},
            observation=f"Edit region: {edit_region.width}x{edit_region.height} ({strategy})",
            duration_ms=int((time.time() - t0) * 1000),
        )
        steps.append(s1)
        _emit(s1)

        # ─── Step 2: Crop ───
        step_num += 1
        t0 = time.time()
        local_crop = crop_box(working_image, edit_region)

        s2 = AgentStep(
            step_num=step_num,
            thought="Cropping the expanded region from the working image.",
            action="crop_local_patch",
            params={"crop_size": f"{local_crop.size[0]}x{local_crop.size[1]}"},
            observation=f"Cropped {local_crop.size[0]}x{local_crop.size[1]}",
            duration_ms=int((time.time() - t0) * 1000),
        )
        steps.append(s2)
        _emit(s2)

        # ─── Step 3: Edit ───
        step_num += 1
        t0 = time.time()
        prompt = self._build_prompt(instruction, target, verb, attempt)

        try:
            edited_response = self.image_client.edit_full_image(local_crop, prompt)
            edited_crop = edited_response.image.convert("RGB").resize(local_crop.size)
            edit_observation = (
                f"Gemini returned {edited_crop.size[0]}x{edited_crop.size[1]} edit"
            )
        except Exception as exc:
            edit_observation = f"Edit failed: {exc}"
            s3_err = AgentStep(
                step_num=step_num,
                thought=f"Editing local crop with Gemini",
                action="edit_local",
                params={},
                observation=edit_observation,
                duration_ms=int((time.time() - t0) * 1000),
            )
            steps.append(s3_err)
            _emit(s3_err)
            return steps, working_image, None, False

        s3 = AgentStep(
            step_num=step_num,
            thought=f"Editing local crop with Gemini (attempt {attempt})",
            action="edit_local",
            params={"prompt_preview": prompt[:150]},
            observation=edit_observation,
            duration_ms=int((time.time() - t0) * 1000),
        )
        steps.append(s3)
        _emit(s3)

        # ─── Step 4: Blend ───
        step_num += 1
        t0 = time.time()
        composed = paste_crop(working_image, edited_crop, edit_region)

        s4 = AgentStep(
            step_num=step_num,
            thought="Blending edited crop back into working image.",
            action="blend_back",
            params={"region": edit_region.to_dict(), "attempt": attempt},
            observation=f"Blended {edit_region.width}x{edit_region.height} region",
            image_url=encode_png_data_url(composed),
            duration_ms=int((time.time() - t0) * 1000),
        )
        steps.append(s4)
        _emit(s4)

        # ─── Step 5: Evaluate quality ───
        step_num += 1
        t0 = time.time()
        quality = self.quality_judge.evaluate(
            untouched_original, composed, bbox,
            preview=composed, target=target, verb=verb,
        )

        s5 = AgentStep(
            step_num=step_num,
            thought="Running quality check (comparing against untouched original).",
            action="evaluate_quality",
            params={"score": round(quality.score, 3), "seam": quality.seam_verdict},
            observation=f"Score={quality.score:.3f}, seam={quality.seam_verdict}, "
                        f"inside_Δ={quality.inside_change:.3f}",
            duration_ms=int((time.time() - t0) * 1000),
        )
        steps.append(s5)
        _emit(s5)

        return steps, composed, quality, False

    def _build_prompt(self, instruction: str, target: str, verb: str,
                      attempt: int) -> str:
        """Build edit prompt with escalating strength per attempt."""
        base = (
            f"This is a cropped region from a larger image. "
            f"{instruction} "
            f"Keep everything else EXACTLY the same — "
            f"same colors, lighting, textures, and positions."
        )

        if attempt >= 2:
            base += (
                f" CRITICAL: Make sure EVERY SINGLE instance of '{target}' "
                f"in this crop is affected. Do not miss any. "
                f"Check the entire image carefully for ALL occurrences."
            )

        if attempt >= 3:
            base += (
                f" ABSOLUTELY ENSURE that ALL instances of '{target}' "
                f"have been edited — left side, right side, foreground, "
                f"background, every single one. Previous attempts missed some."
            )

        if attempt > 1:
            base += (
                f" NOTE: A previous editing pass may have already changed "
                f"some instances. Look for any that are STILL unchanged and "
                f"edit those too, while preserving the already-edited ones."
            )

        return base


__all__ = ["ReActExecutor", "AgentResult", "AgentStep"]
