from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PIL import Image

from .config import load_dotenv
from .memory import ContextFolder, SessionStore
from .models import PipelineResult, PlanStep, StepResult, TurnRecord
from .nano_banana import MockNanoBananaClient, NanoBananaClient, build_image_client
from .planning import RLPlanner, RLValueStore
from .quality import QualityJudge
from .llm_grounding_advisor import GroundingAdvisor, MockGroundingAdvisor, build_grounding_advisor
from .targeting import (
    bbox_iou,
    classify_target,
    fallback_box_for_profile,
    grounding_phrases_for_target,
    rank_grounding_candidates,
    refine_bbox_for_profile,
    rerank_with_llm_guidance,
    _corner_from_modifiers,
)
from .vision import (
    center_box,
    crop_box,
    draw_bbox_overlay,
    encode_png_data_url,
    expand_box,
    paste_crop,
)
from .vlm_localizer import GroundingResult, MockVlmLocalizer, VlmLocalizer, build_localizer
from .vlm_critic import VLMCritic, OllamaVLMCritic, HuggingFaceVLMCritic, CriticVerdict


class AgentBananaApp:
    def __init__(
        self,
        *,
        root: Path,
        image_client: NanoBananaClient | None = None,
        localizer: VlmLocalizer | None = None,
        grounding_advisor: GroundingAdvisor | MockGroundingAdvisor | None = None,
        max_iterations: int = 2,
    ):
        self.root = root
        artifacts_root = self.root / "artifacts" / "agent_banana"
        self.image_client = image_client or build_image_client()
        self.fallback_image_client = MockNanoBananaClient()
        self.localizer = localizer or build_localizer()
        self.fallback_localizer = MockVlmLocalizer()
        self.grounding_advisor = grounding_advisor or build_grounding_advisor()
        self.context_folder = ContextFolder()
        self.session_store = SessionStore(artifacts_root / "sessions")
        self.planner = RLPlanner(RLValueStore(artifacts_root / "planner_values.json"))
        self.quality_judge = QualityJudge()
        self.max_iterations = max_iterations
        # VLM Critic for semantic verification
        import os
        critic_provider = os.environ.get("CRITIC_PROVIDER", "gemini").lower()
        
        if critic_provider == "huggingface":
            hf_model = os.environ.get("HF_CRITIC_MODEL", "meta-llama/Llama-3.2-11B-Vision-Instruct")
            hf_token = os.environ.get("HF_API_TOKEN", "")
            print(f"[pipeline] Booting Cloud VLM Critic via Hugging Face ({hf_model})...")
            self.vlm_critic = HuggingFaceVLMCritic(api_token=hf_token, model=hf_model) if hf_token else None
            if not hf_token:
                print("[pipeline] WARNING: HF_API_TOKEN not found! VLM Critic disabled.")
        elif critic_provider == "ollama":
            ollama_model = os.environ.get("OLLAMA_CRITIC_MODEL", "llama3.2-vision")
            print(f"[pipeline] Booting local VLM Critic via Ollama ({ollama_model})...")
            self.vlm_critic = OllamaVLMCritic(model=ollama_model)
        else:
            api_key = os.environ.get("GEMINI_API_KEY", "")
            self.vlm_critic = VLMCritic(api_key=api_key) if api_key else None

    @classmethod
    def from_env(cls, root: Path | None = None) -> "AgentBananaApp":
        root = root or Path(__file__).resolve().parents[2]
        load_dotenv(root / ".env")
        return cls(root=root)

    # ──────────────────────────────────────────────────────────────────
    # Ground-only: parse instruction, locate targets, return bbox info
    # ──────────────────────────────────────────────────────────────────
    def ground_targets(self, image: Image.Image, instruction: str) -> dict:
        """Parse instruction and ground each target, returning bboxes + overlays.

        Used by Manual Mode: frontend shows these for user confirmation/adjustment
        before actually running the edit pipeline.
        """
        current_image = image.convert("RGB")
        folded_context = self.context_folder.fold([])
        parsed_edits = self.planner.parse_instruction(instruction, folded_context)
        edits_by_id = {edit.edit_id: edit for edit in parsed_edits}
        candidate_plans = self.planner.plan(parsed_edits, folded_context)
        selected_plan = candidate_plans[0]

        grounding_mode = self.localizer.mode_label()
        targets = []

        for step in selected_plan.steps:
            edit = edits_by_id[step.edit_id]
            target_profile = classify_target(step.target, step.verb)

            grounding_result = GroundingResult(phrases=[], candidates=[])
            ranked_candidates = []
            localizer_mode = grounding_mode
            guidance = None

            if step.scope == "global":
                bbox = center_box(current_image.size, scale=0.82)
            else:
                guidance = self.grounding_advisor.advise(
                    source_image=current_image,
                    instruction=instruction,
                    target=step.target,
                    verb=step.verb,
                    profile=target_profile,
                )

                grounding_phrases = grounding_phrases_for_target(step.target, edit.modifiers, step.verb)
                if guidance.refined_phrases:
                    seen = set(p.lower() for p in guidance.refined_phrases)
                    merged = list(guidance.refined_phrases)
                    for p in grounding_phrases:
                        if p.lower() not in seen:
                            merged.append(p)
                            seen.add(p.lower())
                    grounding_phrases = merged

                grounding_result, localizer_mode = self._safe_localize(current_image, grounding_phrases, target_profile)
                if localizer_mode != grounding_mode:
                    grounding_mode = localizer_mode
                ranked_candidates = rank_grounding_candidates(grounding_result.candidates, current_image.size, target_profile)

                if guidance.expected_bbox_hint and guidance.confidence >= 0.4:
                    ranked_candidates = rerank_with_llm_guidance(
                        ranked_candidates, guidance.expected_bbox_hint, current_image.size, target_profile
                    )

                if ranked_candidates:
                    raw_bbox = ranked_candidates[0].bbox
                    bbox = refine_bbox_for_profile(raw_bbox, current_image.size, target_profile)
                    if bbox_iou(raw_bbox, bbox) < 0.15:
                        bbox = raw_bbox
                elif guidance.expected_bbox_hint and guidance.confidence >= 0.6:
                    bbox = guidance.expected_bbox_hint
                else:
                    corner = _corner_from_modifiers(list(edit.modifiers or []), instruction)
                    bbox = fallback_box_for_profile(current_image.size, target_profile, corner=corner)

            overlay_image = draw_bbox_overlay(current_image, bbox, step.target)

            targets.append({
                "step_index": step.order,
                "target": step.target,
                "verb": step.verb,
                "prompt": step.prompt,
                "scope": step.scope,
                "bbox": bbox.to_dict(),
                "overlay_data_url": encode_png_data_url(overlay_image),
                "grounding_phrases": list(grounding_result.phrases or []),
                "llm_confidence": guidance.confidence if guidance else 0.0,
                "llm_description": guidance.object_description if guidance else "",
                "image_width": current_image.size[0],
                "image_height": current_image.size[1],
            })

        return {
            "instruction": instruction,
            "targets": targets,
            "image_width": current_image.size[0],
            "image_height": current_image.size[1],
        }

    # ──────────────────────────────────────────────────────────────────
    # Run with user-confirmed bboxes (skips re-grounding)
    # ──────────────────────────────────────────────────────────────────
    def run_with_bboxes(self, image: Image.Image, instruction: str,
                        confirmed_bboxes: list[dict],
                        session_id: str | None = None,
                        step_callback=None) -> "PipelineResult":
        """Run the edit pipeline using user-confirmed bboxes (manual mode).

        Args:
            confirmed_bboxes: List of bbox dicts [{left, top, right, bottom}, ...]
                              one per parsed edit step, in order.
        """
        from .models import BoundingBox as BB

        session = self.session_store.load_or_create(session_id)
        current_image = image.convert("RGB")
        folded_context = self.context_folder.fold(session.turns)
        parsed_edits = self.planner.parse_instruction(instruction, folded_context)
        edits_by_id = {edit.edit_id: edit for edit in parsed_edits}
        candidate_plans = self.planner.plan(parsed_edits, folded_context)
        selected_plan = candidate_plans[0]

        runtime_mode = self.image_client.mode_label()
        grounding_mode = self.localizer.mode_label()
        step_results = []
        reward_components = []
        bboxes_out = []

        for i, step in enumerate(selected_plan.steps):
            edit = edits_by_id[step.edit_id]
            target_profile = classify_target(step.target, step.verb)

            # Use user-confirmed bbox (fall back to center if missing)
            if i < len(confirmed_bboxes) and confirmed_bboxes[i]:
                bd = confirmed_bboxes[i]
                bbox = BB(left=int(bd["left"]), top=int(bd["top"]),
                          right=int(bd["right"]), bottom=int(bd["bottom"]))
            else:
                bbox = center_box(current_image.size, scale=0.82)

            from .react_executor import ReActExecutor
            executor = ReActExecutor(
                image_client=self.image_client,
                quality_judge=self.quality_judge,
                vlm_critic=self.vlm_critic,
                max_iterations=self.max_iterations + 1,
            )

            agent_result = executor.execute_edit(
                original_image=current_image,
                instruction=instruction,
                target=step.target,
                verb=step.verb,
                bbox=bbox,
                target_profile=target_profile,
                step_callback=step_callback,
            )

            composed_image = agent_result.final_image or current_image
            quality = agent_result.quality or self.quality_judge.evaluate(
                current_image, composed_image, bbox,
                preview=composed_image, target=step.target, verb=step.verb,
            )

            print(f"[react] Completed in {agent_result.total_attempts} attempt(s), "
                  f"{len(agent_result.steps)} steps, "
                  f"{'SUCCESS' if agent_result.success else 'BEST-EFFORT'}")

            overlay_image = draw_bbox_overlay(current_image, bbox, step.target)
            step_results.append(
                StepResult(
                    step=step,
                    bbox=bbox,
                    quality=quality,
                    preview_data_url=encode_png_data_url(
                        crop_box(current_image, expand_box(bbox, max(40, max(bbox.width, bbox.height)//2), current_image.size))
                    ),
                    overlay_data_url=encode_png_data_url(overlay_image),
                    edited_data_url=agent_result.final_image_url,
                    attempts=agent_result.total_attempts,
                    grounding_phrases=[],
                    grounding_candidates=[],
                    localizer_mode="manual",
                    llm_object_description="",
                    llm_refined_phrases=[],
                    llm_bbox_hint=bbox.to_dict(),
                    llm_confidence=1.0,
                    image_width=current_image.size[0],
                    image_height=current_image.size[1],
                    agent_steps=[s.to_dict() for s in agent_result.steps],
                )
            )
            semantic_multiplier = quality.semantic_score if quality.semantic_score < 0.7 else 1.0
            reward_components.append(quality.score * semantic_multiplier)
            bboxes_out.append(bbox)
            current_image = composed_image

        reward = 0.0 if not reward_components else sum(reward_components) / len(reward_components)
        self.planner.record_feedback(selected_plan, reward)
        session.turns.append(
            TurnRecord(
                instruction=instruction,
                parsed_edits=parsed_edits,
                selected_plan=selected_plan,
                reward=reward,
                bboxes=bboxes_out,
            )
        )
        session.folded_context = self.context_folder.fold(session.turns)
        self.session_store.save(session)

        return PipelineResult(
            session_id=session.session_id,
            mode=runtime_mode,
            grounding_mode="manual",
            instruction=instruction,
            folded_context=session.folded_context,
            parsed_edits=parsed_edits,
            candidate_plans=candidate_plans,
            selected_plan=selected_plan,
            source_image=encode_png_data_url(image),
            final_image=encode_png_data_url(current_image),
            step_results=step_results,
            reward=reward,
        )

    def run(self, image: Image.Image, instruction: str, session_id: str | None = None,
            step_callback=None) -> PipelineResult:
        session = self.session_store.load_or_create(session_id)
        current_image = image.convert("RGB")
        folded_context = self.context_folder.fold(session.turns)
        parsed_edits = self.planner.parse_instruction(instruction, folded_context)
        edits_by_id = {edit.edit_id: edit for edit in parsed_edits}
        candidate_plans = self.planner.plan(parsed_edits, folded_context)
        selected_plan = candidate_plans[0]

        runtime_mode = self.image_client.mode_label()
        grounding_mode = self.localizer.mode_label()
        step_results = []
        reward_components = []
        bboxes = []

        for step in selected_plan.steps:
            edit = edits_by_id[step.edit_id]
            target_profile = classify_target(step.target, step.verb)

            # ==============================================================
            # ILD Step 1: GROUND FIRST — locate target on ORIGINAL image
            # ==============================================================
            grounding_result = GroundingResult(phrases=[], candidates=[])
            ranked_candidates = []
            localizer_mode = grounding_mode
            guidance = None

            if step.scope == "global":
                bbox = center_box(current_image.size, scale=0.82)
            else:
                # LLM advisor reasons about WHERE the target is
                guidance = self.grounding_advisor.advise(
                    source_image=current_image,
                    instruction=instruction,
                    target=step.target,
                    verb=step.verb,
                    profile=target_profile,
                )

                # Build grounding phrases: LLM-refined + rule-based
                grounding_phrases = grounding_phrases_for_target(step.target, edit.modifiers, step.verb)
                if guidance.refined_phrases:
                    seen = set(p.lower() for p in guidance.refined_phrases)
                    merged = list(guidance.refined_phrases)
                    for p in grounding_phrases:
                        if p.lower() not in seen:
                            merged.append(p)
                            seen.add(p.lower())
                    grounding_phrases = merged

                # Florence-2 runs on the ORIGINAL image (not a preview)
                grounding_result, localizer_mode = self._safe_localize(current_image, grounding_phrases, target_profile)
                if localizer_mode != grounding_mode:
                    grounding_mode = localizer_mode
                ranked_candidates = rank_grounding_candidates(grounding_result.candidates, current_image.size, target_profile)

                # Re-rank with LLM spatial guidance
                if guidance.expected_bbox_hint and guidance.confidence >= 0.4:
                    ranked_candidates = rerank_with_llm_guidance(
                        ranked_candidates, guidance.expected_bbox_hint, current_image.size, target_profile
                    )

                if ranked_candidates:
                    raw_bbox = ranked_candidates[0].bbox
                    bbox = refine_bbox_for_profile(raw_bbox, current_image.size, target_profile)
                    if bbox_iou(raw_bbox, bbox) < 0.15:
                        bbox = raw_bbox
                elif guidance.expected_bbox_hint and guidance.confidence >= 0.6:
                    bbox = guidance.expected_bbox_hint
                    print(f"[agent-banana] Using LLM bbox hint as fallback (confidence={guidance.confidence:.2f})")
                else:
                    corner = _corner_from_modifiers(list(edit.modifiers or []), instruction)
                    bbox = fallback_box_for_profile(current_image.size, target_profile, corner=corner)

            # ==============================================================
            # ILD Steps 2-5: ReAct Agent Loop
            # (expand → crop → edit → blend → critic → retry if needed)
            # ==============================================================
            from .react_executor import ReActExecutor
            executor = ReActExecutor(
                image_client=self.image_client,
                quality_judge=self.quality_judge,
                vlm_critic=self.vlm_critic,
                max_iterations=self.max_iterations + 1,  # +1 because first attempt isn't a "retry"
            )

            agent_result = executor.execute_edit(
                original_image=current_image,
                instruction=instruction,
                target=step.target,
                verb=step.verb,
                bbox=bbox,
                target_profile=target_profile,
                step_callback=step_callback,
            )

            composed_image = agent_result.final_image or current_image
            quality = agent_result.quality or self.quality_judge.evaluate(
                current_image, composed_image, bbox,
                preview=composed_image, target=step.target, verb=step.verb,
            )

            print(f"[react] Completed in {agent_result.total_attempts} attempt(s), "
                  f"{len(agent_result.steps)} steps, "
                  f"{'SUCCESS' if agent_result.success else 'BEST-EFFORT'}")

            overlay_image = draw_bbox_overlay(current_image, bbox, step.target)
            step_results.append(
                StepResult(
                    step=step,
                    bbox=bbox,
                    quality=quality,
                    preview_data_url=encode_png_data_url(
                        crop_box(current_image, expand_box(bbox, max(40, max(bbox.width, bbox.height)//2), current_image.size))
                    ),
                    overlay_data_url=encode_png_data_url(overlay_image),
                    edited_data_url=agent_result.final_image_url,
                    attempts=agent_result.total_attempts,
                    grounding_phrases=list(grounding_result.phrases),
                    grounding_candidates=ranked_candidates[:5],
                    localizer_mode=localizer_mode,
                    llm_object_description=guidance.object_description if guidance else "",
                    llm_refined_phrases=list(guidance.refined_phrases) if guidance else [],
                    llm_bbox_hint=guidance.expected_bbox_hint.to_dict() if guidance and guidance.expected_bbox_hint else None,
                    llm_confidence=guidance.confidence if guidance else 0.0,
                    image_width=current_image.size[0],
                    image_height=current_image.size[1],
                    agent_steps=[s.to_dict() for s in agent_result.steps],
                )
            )
            # Reward: combine quality score with semantic score
            semantic_multiplier = quality.semantic_score if quality.semantic_score < 0.7 else 1.0
            reward_components.append(quality.score * semantic_multiplier)
            bboxes.append(bbox)
            current_image = composed_image

        reward = 0.0 if not reward_components else sum(reward_components) / len(reward_components)
        self.planner.record_feedback(selected_plan, reward)
        session.turns.append(
            TurnRecord(
                instruction=instruction,
                parsed_edits=parsed_edits,
                selected_plan=selected_plan,
                reward=reward,
                bboxes=bboxes,
            )
        )
        session.folded_context = self.context_folder.fold(session.turns)
        self.session_store.save(session)

        return PipelineResult(
            session_id=session.session_id,
            mode=runtime_mode,
            grounding_mode=grounding_mode,
            instruction=instruction,
            folded_context=session.folded_context,
            parsed_edits=parsed_edits,
            candidate_plans=candidate_plans,
            selected_plan=selected_plan,
            source_image=encode_png_data_url(image),
            final_image=encode_png_data_url(current_image),
            step_results=step_results,
            reward=reward,
        )

    def _safe_full_image_edit(self, image: Image.Image, prompt: str):
        try:
            return self.image_client.edit_full_image(image, prompt), self.image_client.mode_label()
        except Exception as exc:
            print(f"[agent-banana] image_client failed, falling back to mock: {exc}")
            return self.fallback_image_client.edit_full_image(image, prompt), "mock-fallback"

    def _safe_localize(self, image: Image.Image, phrases: list[str], profile: str):
        try:
            return self.localizer.localize(image, phrases, profile=profile), self.localizer.mode_label()
        except Exception:
            return self.fallback_localizer.localize(image, phrases, profile=profile), "mock-vlm-fallback"

    def _local_edit_prompt(self, instruction: str, step: PlanStep, target_profile: str, modifiers: list[str] | None = None) -> str:
        """Prompt for editing a LOCAL CROP (not full image).

        The crop already contains the target object + surrounding context.
        The model should edit only what's described and preserve everything else
        in the crop so that the result blends seamlessly back into the original.
        """
        base = (
            f"This is a cropped region from a larger image. "
            f"Edit ONLY what is described: {step.prompt}. "
            f"Keep everything else in this crop EXACTLY the same — "
            f"same colors, lighting, textures, and positions. "
            f"The edited region must blend naturally with its surroundings."
        )
        if target_profile == "face_accessory":
            modifier_text = ""
            if modifiers:
                modifier_text = " " + " ".join(modifiers) + "."
            base += f" Keep the face, pose, hair, and skin unchanged.{modifier_text}"
        return base

    def _full_image_prompt(self, instruction: str, step: PlanStep, context_summary: str, target_profile: str, modifiers: list[str] | None = None) -> str:
        base = (
            f"Context: {context_summary} "
            f"Step {step.order}: {step.prompt}. "
            f"Global instruction: {instruction}. "
            "Produce a full-image proposal that applies the requested change while preserving all non-target content."
        )
        if target_profile == "face_accessory":
            modifier_text = ""
            if modifiers:
                modifier_text = " " + " ".join(modifiers) + "."
            if step.verb in ("adjust", "replace"):
                base += f" Keep the face, pose, hair, and skin unchanged. Apply the requested modification to the eyewear only.{modifier_text}"
            else:
                base += f" Keep the face, pose, hair, and skin unchanged. Only change the eyewear.{modifier_text}"
        return base

    def _merge_step(
        self,
        *,
        current_image: Image.Image,
        proposal_image: Image.Image,
        bbox,
        target: str,
        verb: str,
    ) -> Tuple[Image.Image, object, object, int]:
        proposal_crop = crop_box(proposal_image, bbox)
        composed = paste_crop(current_image, proposal_crop, bbox)
        quality = self.quality_judge.evaluate(
            current_image,
            composed,
            bbox,
            preview=proposal_image,
            target=target,
            verb=verb,
        )
        return composed, bbox, quality, 1

    def recompose(
        self,
        source_image: Image.Image,
        preview_image: Image.Image,
        bbox_dict: dict,
        target: str = "object",
        verb: str = "edit",
        custom_instruction: str = "",
    ) -> dict:
        """Re-run composition with a manually adjusted bounding box."""
        from .models import BoundingBox as BB
        bbox = BB(
            left=int(bbox_dict["left"]),
            top=int(bbox_dict["top"]),
            right=int(bbox_dict["right"]),
            bottom=int(bbox_dict["bottom"]),
        )
        src = source_image.convert("RGB")
        # ILD: expand bbox, crop local patch, edit, blend
        pad = max(40, max(bbox.width, bbox.height) // 2)
        edit_region = expand_box(bbox, pad, src.size)
        local_crop = crop_box(src, edit_region)
        if custom_instruction:
            local_prompt = (
                f"This is a cropped region from a larger image. "
                f"{custom_instruction} "
                f"Keep everything else EXACTLY the same — "
                f"same colors, lighting, textures, and positions. "
                f"The result must blend naturally with its surroundings."
            )
        else:
            local_prompt = (
                f"This is a cropped region from a larger image. "
                f"{verb} the {target} in this crop. "
                f"Keep everything else EXACTLY the same — "
                f"same colors, lighting, textures, and positions. "
                f"The result must blend naturally with its surroundings."
            )
        print(f"[agent-banana] recompose ILD: bbox {bbox.width}x{bbox.height} -> "
              f"edit_region {edit_region.width}x{edit_region.height} (pad={pad})")
        edited_response, _ = self._safe_full_image_edit(local_crop, local_prompt)
        edited_crop = edited_response.image.convert("RGB").resize(local_crop.size)
        composed = paste_crop(src, edited_crop, edit_region)
        quality = self.quality_judge.evaluate(
            src, composed, bbox,
            preview=composed, target=target, verb=verb,
        )
        overlay = draw_bbox_overlay(src, bbox, target)
        return {
            "final_image": encode_png_data_url(composed),
            "overlay_image": encode_png_data_url(overlay),
            "bbox": bbox.to_dict(),
            "quality": quality.to_dict(),
        }


__all__ = ["AgentBananaApp"]
