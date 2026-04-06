from __future__ import annotations

import json
import math
import re
from itertools import permutations, product
from pathlib import Path
from typing import Dict, Iterable, List

from .models import FoldedContext, ParsedEdit, PlanCandidate, PlanStep
from .targeting import classify_target

VERB_ALIASES = {
    "add": ("add", "insert", "place", "put", "include"),
    "remove": ("remove", "delete", "erase", "take away", "clear"),
    "replace": ("replace", "swap", "change out", "turn into"),
    "restyle": ("restyle", "stylize", "make", "render", "give"),
    "adjust": ("adjust", "change", "modify", "brighten", "darken", "recolor", "move", "resize"),
}
GLOBAL_HINTS = {
    "background",
    "entire image",
    "whole image",
    "overall",
    "global",
    "lighting",
    "mood",
    "style",
    "scene",
}
MODE_CONFIG = {
    "preview_tight": {"padding": 10, "risk": 0.12},
    "preview_local": {"padding": 22, "risk": 0.18},
    "preview_expand": {"padding": 42, "risk": 0.29},
    "global_preview": {"padding": 12, "risk": 0.52},
}


class RLValueStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"action_values": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def value(self, signature: str) -> float:
        action_values = self._data.get("action_values", {})
        payload = action_values.get(signature)
        if not payload:
            return 0.5
        return float(payload.get("value", 0.5))

    def average_value(self, signatures: Iterable[str]) -> float:
        signatures = list(signatures)
        if not signatures:
            return 0.5
        return sum(self.value(signature) for signature in signatures) / len(signatures)

    def update(self, signatures: Iterable[str], reward: float) -> None:
        action_values = self._data.setdefault("action_values", {})
        for signature in signatures:
            payload = action_values.setdefault(signature, {"value": 0.5, "visits": 0})
            visits = int(payload.get("visits", 0)) + 1
            old_value = float(payload.get("value", 0.5))
            payload["visits"] = visits
            payload["value"] = old_value + (reward - old_value) / visits
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")


class EditParser:
    def parse(self, instruction: str, context: FoldedContext | None = None) -> List[ParsedEdit]:
        segments = self._split_instruction(instruction)
        edits: List[ParsedEdit] = []
        chained_order = any(token in instruction.lower() for token in ("then", "after", "finally"))

        for index, segment in enumerate(segments):
            verb = self._detect_verb(segment)
            target, modifiers = self._extract_target_and_modifiers(segment, verb)
            scope = self._detect_scope(segment, target, context)
            dependencies = []
            if chained_order and index > 0:
                dependencies.append(edits[-1].edit_id)
            edits.append(
                ParsedEdit(
                    edit_id=f"edit-{index + 1}",
                    original_text=segment,
                    verb=verb,
                    target=target,
                    scope=scope,
                    priority=index,
                    dependencies=dependencies,
                    modifiers=modifiers,
                )
            )

        if edits:
            return edits

        return [
            ParsedEdit(
                edit_id="edit-1",
                original_text=instruction.strip(),
                verb="adjust",
                target="main subject",
                scope="local",
                priority=0,
                dependencies=[],
                modifiers=[],
            )
        ]

    def _split_instruction(self, instruction: str) -> List[str]:
        raw_parts = re.split(r"\s*(?:;|,|\band then\b|\bthen\b|\balso\b|\bplus\b|\nafter\b|\n)\s*", instruction, flags=re.IGNORECASE)
        parts: List[str] = []
        for raw_part in raw_parts:
            part = raw_part.strip(" .")
            if not part:
                continue
            if " and " in part.lower() and self._count_verb_hits(part) >= 2:
                parts.extend(sub.strip(" .") for sub in re.split(r"\s+\band\b\s+", part, flags=re.IGNORECASE) if sub.strip(" ."))
            else:
                parts.append(part)
        return parts

    def _count_verb_hits(self, text: str) -> int:
        lowered = text.lower()
        hits = 0
        for aliases in VERB_ALIASES.values():
            if any(alias in lowered for alias in aliases):
                hits += 1
        return hits

    def _detect_verb(self, segment: str) -> str:
        lowered = segment.lower()
        for canonical, aliases in VERB_ALIASES.items():
            if any(re.search(rf"\b{re.escape(alias)}\b", lowered) for alias in aliases):
                return canonical
        return "adjust"

    def _extract_target_and_modifiers(self, segment: str, verb: str) -> tuple[str, List[str]]:
        lowered = segment.lower()
        modifiers: List[str] = []
        match_end = 0
        for alias in VERB_ALIASES[verb]:
            match = re.search(rf"\b{re.escape(alias)}\b", lowered)
            if match:
                match_end = match.end()
                break
        remainder = segment[match_end:].strip() if match_end else segment.strip()
        remainder = re.sub(r"^(the|a|an)\s+", "", remainder, flags=re.IGNORECASE).strip()

        if verb == "replace" and re.search(r"\bwith\b", remainder, flags=re.IGNORECASE):
            target, replacement = re.split(r"\bwith\b", remainder, maxsplit=1, flags=re.IGNORECASE)
            modifiers.append(f"with {replacement.strip()}")
            remainder = target.strip()
        elif verb == "add" and re.search(r"\bto\b", remainder, flags=re.IGNORECASE):
            target, location = re.split(r"\bto\b", remainder, maxsplit=1, flags=re.IGNORECASE)
            modifiers.append(f"to {location.strip()}")
            remainder = target.strip()
        elif re.search(r"\b(?:in|on|at|into|from|near)\b", remainder, flags=re.IGNORECASE):
            target, location = re.split(r"\b(?:in|on|at|into|from|near)\b", remainder, maxsplit=1, flags=re.IGNORECASE)
            modifiers.append(location.strip())
            remainder = target.strip()

        if not remainder:
            remainder = "main subject"
        return remainder, modifiers

    def _detect_scope(self, segment: str, target: str, context: FoldedContext | None) -> str:
        lowered = f"{segment} {target}".lower()
        if any(hint in lowered for hint in GLOBAL_HINTS):
            return "global"
        if context and target.lower() in {entity.lower() for entity in context.active_entities}:
            return "local"
        return "local"


class RLPlanner:
    def __init__(self, value_store: RLValueStore, *, top_k: int = 8, max_enumeration: int = 4096, beam_width: int = 24):
        self.value_store = value_store
        self.parser = EditParser()
        self.top_k = top_k
        self.max_enumeration = max_enumeration
        self.beam_width = beam_width

    def parse_instruction(self, instruction: str, context: FoldedContext | None = None) -> List[ParsedEdit]:
        return self.parser.parse(instruction, context)

    def plan(self, edits: List[ParsedEdit], context: FoldedContext) -> List[PlanCandidate]:
        option_map = {edit.edit_id: self._step_options(edit) for edit in edits}
        estimated_paths = math.factorial(max(1, len(edits)))
        for options in option_map.values():
            estimated_paths *= max(1, len(options))

        if estimated_paths <= self.max_enumeration:
            candidates = self._enumerate_all_paths(edits, context, option_map)
        else:
            candidates = self._beam_search(edits, context, option_map)

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[: self.top_k]

    def record_feedback(self, plan: PlanCandidate, reward: float) -> None:
        self.value_store.update((step.signature() for step in plan.steps), reward)

    def _enumerate_all_paths(
        self,
        edits: List[ParsedEdit],
        context: FoldedContext,
        option_map: Dict[str, List[PlanStep]],
    ) -> List[PlanCandidate]:
        candidates: List[PlanCandidate] = []
        for ordering in permutations(edits):
            if not self._dependencies_satisfied(ordering):
                continue
            option_lists = [option_map[edit.edit_id] for edit in ordering]
            for choice_index, choice_tuple in enumerate(product(*option_lists), start=1):
                steps = [
                    self._materialize_step(option, order=order_index)
                    for order_index, option in enumerate(choice_tuple, start=1)
                ]
                candidate = self._make_candidate(steps, context, len(candidates) + 1)
                candidates.append(candidate)
        return candidates

    def _beam_search(
        self,
        edits: List[ParsedEdit],
        context: FoldedContext,
        option_map: Dict[str, List[PlanStep]],
    ) -> List[PlanCandidate]:
        beams = [([], edits)]
        completed: List[PlanCandidate] = []

        while beams:
            next_beams = []
            for partial_steps, remaining in beams:
                if not remaining:
                    completed.append(self._make_candidate(partial_steps, context, len(completed) + 1))
                    continue
                seen_ids = {step.edit_id for step in partial_steps}
                for edit in remaining:
                    if any(dep not in seen_ids for dep in edit.dependencies):
                        continue
                    next_remaining = [item for item in remaining if item.edit_id != edit.edit_id]
                    for option in option_map[edit.edit_id]:
                        step = self._materialize_step(option, order=len(partial_steps) + 1)
                        optimistic_steps = partial_steps + [step]
                        optimistic_score = self._score_candidate(optimistic_steps, context)["total"]
                        next_beams.append((optimistic_score, optimistic_steps, next_remaining))

            next_beams.sort(key=lambda item: item[0], reverse=True)
            beams = [(steps, remaining) for _, steps, remaining in next_beams[: self.beam_width]]

        return completed

    def _step_options(self, edit: ParsedEdit) -> List[PlanStep]:
        profile = classify_target(edit.target, edit.verb)
        if edit.scope == "global":
            modes = ("global_preview", "preview_expand")
        elif profile in {"face_accessory", "small_accessory"} and edit.verb in {"remove", "replace", "adjust"}:
            modes = ("preview_tight", "preview_local", "preview_expand")
        elif edit.verb in {"replace", "remove"}:
            modes = ("preview_expand", "preview_local", "global_preview")
        else:
            modes = ("preview_local", "preview_expand", "global_preview")

        options = []
        for mode in modes:
            config = MODE_CONFIG[mode]
            prompt = self._build_step_prompt(edit, mode)
            options.append(
                PlanStep(
                    step_id=f"{edit.edit_id}-{mode}",
                    edit_id=edit.edit_id,
                    order=edit.priority + 1,
                    verb=edit.verb,
                    target=edit.target,
                    scope=edit.scope,
                    mode=mode,
                    prompt=prompt,
                    padding=int(config["padding"]),
                    risk=float(config["risk"]),
                )
            )
        return options

    def _build_step_prompt(self, edit: ParsedEdit, mode: str) -> str:
        instruction = edit.original_text.rstrip(".")
        guidance = {
            "preview_tight": "Treat this as a very tight local edit centered on the target object only.",
            "preview_local": "Treat this as a tight local edit after preview localization.",
            "preview_expand": "Assume the object extends slightly beyond the first bbox estimate and preserve surrounding structure.",
            "global_preview": "Allow a broad edit if necessary, but keep the composition coherent after preview localization.",
        }[mode]
        modifier_text = f" Modifiers: {', '.join(edit.modifiers)}." if edit.modifiers else ""
        return f"{instruction}. Target: {edit.target}.{modifier_text} {guidance}"

    def _dependencies_satisfied(self, ordering: Iterable[ParsedEdit]) -> bool:
        seen = set()
        for edit in ordering:
            if any(dependency not in seen for dependency in edit.dependencies):
                return False
            seen.add(edit.edit_id)
        return True

    def _materialize_step(self, option: PlanStep, *, order: int) -> PlanStep:
        return PlanStep(
            step_id=option.step_id,
            edit_id=option.edit_id,
            order=order,
            verb=option.verb,
            target=option.target,
            scope=option.scope,
            mode=option.mode,
            prompt=option.prompt,
            padding=option.padding,
            risk=option.risk,
        )

    def _make_candidate(self, steps: List[PlanStep], context: FoldedContext, plan_number: int) -> PlanCandidate:
        breakdown = self._score_candidate(steps, context)
        return PlanCandidate(
            plan_id=f"plan-{plan_number:03d}",
            steps=steps,
            score=breakdown["total"],
            score_breakdown=breakdown,
        )

    def _score_candidate(self, steps: List[PlanStep], context: FoldedContext) -> Dict[str, float]:
        if not steps:
            return {"total": 0.0}

        order_alignment_values = []
        locality_values = []
        risk_values = []
        entity_bonus_values = []
        mode_fit_values = []
        active_entities = {entity.lower() for entity in context.active_entities}

        for natural_order, step in enumerate(steps, start=1):
            order_alignment_values.append(1.0 - min(1.0, abs(step.order - natural_order) / max(1, len(steps) - 1 or 1)))
            if step.scope == "global":
                locality_values.append(1.0 if step.mode == "global_preview" else 0.72)
            else:
                locality_values.append(1.0 if step.mode != "global_preview" else 0.38)
            risk_values.append(step.risk)
            entity_bonus_values.append(1.0 if step.target.lower() in active_entities else 0.55)
            mode_fit_values.append(self._mode_fit(step))

        tool_switch_penalty = 0.0
        for previous, current in zip(steps, steps[1:]):
            if previous.mode != current.mode:
                tool_switch_penalty += 0.05

        learned_value = self.value_store.average_value(step.signature() for step in steps)
        order_alignment = sum(order_alignment_values) / len(order_alignment_values)
        locality = sum(locality_values) / len(locality_values)
        risk = sum(risk_values) / len(risk_values)
        entity_bonus = sum(entity_bonus_values) / len(entity_bonus_values)
        mode_fit = sum(mode_fit_values) / len(mode_fit_values)
        dependency_score = 1.0
        if len(steps) >= 2 and steps[0].scope == "global" and any(step.scope == "local" for step in steps[1:]):
            dependency_score = 0.65

        total = (
            0.24 * order_alignment
            + 0.20 * locality
            + 0.14 * dependency_score
            + 0.14 * entity_bonus
            + 0.14 * mode_fit
            + 0.18 * learned_value
            - risk
            - tool_switch_penalty
        )

        return {
            "order_alignment": order_alignment,
            "locality": locality,
            "dependency_score": dependency_score,
            "entity_bonus": entity_bonus,
            "mode_fit": mode_fit,
            "learned_value": learned_value,
            "risk_penalty": risk,
            "tool_switch_penalty": tool_switch_penalty,
            "total": total,
        }

    def _mode_fit(self, step: PlanStep) -> float:
        profile = classify_target(step.target, step.verb)
        if step.scope == "global":
            return 1.0 if step.mode == "global_preview" else 0.68
        if profile in {"face_accessory", "small_accessory"} and step.verb in {"remove", "replace", "adjust"}:
            if step.mode == "preview_tight":
                return 1.0
            if step.mode == "preview_local":
                return 0.84
            if step.mode == "preview_expand":
                return 0.52
            return 0.24
        if step.verb in {"replace", "remove"}:
            if step.mode == "preview_expand":
                return 1.0
            if step.mode == "preview_local":
                return 0.76
            return 0.42
        if step.mode == "preview_local":
            return 1.0
        if step.mode == "preview_expand":
            return 0.88
        return 0.4
