from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BoundingBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)

    def to_dict(self) -> dict:
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "BoundingBox":
        return cls(
            left=int(payload["left"]),
            top=int(payload["top"]),
            right=int(payload["right"]),
            bottom=int(payload["bottom"]),
        )


@dataclass
class FoldedContext:
    summary: str
    active_entities: List[str]
    constraints: List[str]
    turn_count: int

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "active_entities": list(self.active_entities),
            "constraints": list(self.constraints),
            "turn_count": self.turn_count,
        }


@dataclass
class ParsedEdit:
    edit_id: str
    original_text: str
    verb: str
    target: str
    scope: str
    priority: int
    dependencies: List[str] = field(default_factory=list)
    modifiers: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return f"{self.verb} {self.target}".strip()

    def to_dict(self) -> dict:
        return {
            "edit_id": self.edit_id,
            "original_text": self.original_text,
            "verb": self.verb,
            "target": self.target,
            "scope": self.scope,
            "priority": self.priority,
            "dependencies": list(self.dependencies),
            "modifiers": list(self.modifiers),
        }


@dataclass
class PlanStep:
    step_id: str
    edit_id: str
    order: int
    verb: str
    target: str
    scope: str
    mode: str
    prompt: str
    padding: int
    risk: float

    def signature(self) -> str:
        return "|".join((self.verb, self.scope, self.mode, self.target[:48]))

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "edit_id": self.edit_id,
            "order": self.order,
            "verb": self.verb,
            "target": self.target,
            "scope": self.scope,
            "mode": self.mode,
            "prompt": self.prompt,
            "padding": self.padding,
            "risk": round(self.risk, 4),
        }


@dataclass
class PlanCandidate:
    plan_id: str
    steps: List[PlanStep]
    score: float
    score_breakdown: Dict[str, float]

    def signature(self) -> str:
        return " -> ".join(step.signature() for step in self.steps)

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "score": round(self.score, 4),
            "score_breakdown": {key: round(value, 4) for key, value in self.score_breakdown.items()},
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass
class GroundingCandidate:
    phrase: str
    bbox: BoundingBox
    score: float
    source: str

    def to_dict(self) -> dict:
        return {
            "phrase": self.phrase,
            "bbox": self.bbox.to_dict(),
            "score": round(self.score, 4),
            "source": self.source,
        }


@dataclass
class QualityMetrics:
    score: float
    accepted: bool
    inside_change: float
    outside_change: float
    preview_alignment: float
    seam_score: float = 0.0
    seam_verdict: str = "accept"
    semantic_score: float = 1.0
    semantic_fulfilled: bool = True
    semantic_reasoning: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "accepted": self.accepted,
            "inside_change": round(self.inside_change, 4),
            "outside_change": round(self.outside_change, 4),
            "preview_alignment": round(self.preview_alignment, 4),
            "seam_score": round(self.seam_score, 4),
            "seam_verdict": self.seam_verdict,
            "semantic_score": round(self.semantic_score, 4),
            "semantic_fulfilled": self.semantic_fulfilled,
            "semantic_reasoning": self.semantic_reasoning,
            "notes": list(self.notes),
        }


@dataclass
class StepResult:
    step: PlanStep
    bbox: BoundingBox
    quality: QualityMetrics
    preview_data_url: str
    overlay_data_url: str
    edited_data_url: str
    attempts: int
    grounding_phrases: List[str] = field(default_factory=list)
    grounding_candidates: List[GroundingCandidate] = field(default_factory=list)
    localizer_mode: str = ""
    llm_object_description: str = ""
    llm_refined_phrases: List[str] = field(default_factory=list)
    llm_bbox_hint: Optional[Dict] = None
    llm_confidence: float = 0.0
    image_width: int = 0
    image_height: int = 0
    agent_steps: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "step": self.step.to_dict(),
            "bbox": self.bbox.to_dict(),
            "quality": self.quality.to_dict(),
            "preview_image": self.preview_data_url,
            "overlay_image": self.overlay_data_url,
            "edited_image": self.edited_data_url,
            "attempts": self.attempts,
            "grounding_phrases": list(self.grounding_phrases),
            "grounding_candidates": [candidate.to_dict() for candidate in self.grounding_candidates],
            "localizer_mode": self.localizer_mode,
            "llm_object_description": self.llm_object_description,
            "llm_refined_phrases": list(self.llm_refined_phrases),
            "llm_bbox_hint": self.llm_bbox_hint,
            "llm_confidence": round(self.llm_confidence, 4),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "agent_steps": list(self.agent_steps),
        }


@dataclass
class TurnRecord:
    instruction: str
    parsed_edits: List[ParsedEdit]
    selected_plan: PlanCandidate
    reward: float
    bboxes: List[BoundingBox]

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "parsed_edits": [edit.to_dict() for edit in self.parsed_edits],
            "selected_plan": self.selected_plan.to_dict(),
            "reward": round(self.reward, 4),
            "bboxes": [bbox.to_dict() for bbox in self.bboxes],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TurnRecord":
        edits = [ParsedEdit(**edit) for edit in payload.get("parsed_edits", [])]
        plan_steps = [PlanStep(**step) for step in payload["selected_plan"]["steps"]]
        plan = PlanCandidate(
            plan_id=payload["selected_plan"]["plan_id"],
            steps=plan_steps,
            score=float(payload["selected_plan"]["score"]),
            score_breakdown=dict(payload["selected_plan"].get("score_breakdown", {})),
        )
        bboxes = [BoundingBox.from_dict(item) for item in payload.get("bboxes", [])]
        return cls(
            instruction=str(payload.get("instruction", "")),
            parsed_edits=edits,
            selected_plan=plan,
            reward=float(payload.get("reward", 0.0)),
            bboxes=bboxes,
        )


@dataclass
class SessionState:
    session_id: str
    turns: List[TurnRecord] = field(default_factory=list)
    folded_context: Optional[FoldedContext] = None

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "turns": [turn.to_dict() for turn in self.turns],
            "folded_context": None if self.folded_context is None else self.folded_context.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SessionState":
        folded = payload.get("folded_context")
        context = None
        if folded:
            context = FoldedContext(
                summary=str(folded.get("summary", "")),
                active_entities=list(folded.get("active_entities", [])),
                constraints=list(folded.get("constraints", [])),
                turn_count=int(folded.get("turn_count", 0)),
            )
        return cls(
            session_id=str(payload["session_id"]),
            turns=[TurnRecord.from_dict(item) for item in payload.get("turns", [])],
            folded_context=context,
        )


@dataclass
class PipelineResult:
    session_id: str
    mode: str
    grounding_mode: str
    instruction: str
    folded_context: FoldedContext
    parsed_edits: List[ParsedEdit]
    candidate_plans: List[PlanCandidate]
    selected_plan: PlanCandidate
    source_image: str
    final_image: str
    step_results: List[StepResult]
    reward: float

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "grounding_mode": self.grounding_mode,
            "instruction": self.instruction,
            "folded_context": self.folded_context.to_dict(),
            "parsed_edits": [edit.to_dict() for edit in self.parsed_edits],
            "candidate_plans": [plan.to_dict() for plan in self.candidate_plans],
            "selected_plan": self.selected_plan.to_dict(),
            "source_image": self.source_image,
            "final_image": self.final_image,
            "step_results": [step.to_dict() for step in self.step_results],
            "reward": round(self.reward, 4),
        }
