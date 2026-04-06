from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image

from .models import BoundingBox, GroundingCandidate
from .targeting import classify_target, fallback_box_for_profile

DEFAULT_GROUNDING_MODEL = "florence-community/Florence-2-base"


class GroundingError(RuntimeError):
    pass


@dataclass
class GroundingResult:
    phrases: List[str]
    candidates: List[GroundingCandidate]


class VlmLocalizer:
    def mode_label(self) -> str:
        raise NotImplementedError

    def localize(
        self,
        image: Image.Image,
        phrases: List[str],
        *,
        profile: str,
    ) -> GroundingResult:
        raise NotImplementedError


class Florence2PhraseGrounder(VlmLocalizer):
    def __init__(self, model_name: str = DEFAULT_GROUNDING_MODEL):
        self.model_name = model_name
        self._torch = None
        self._processor = None
        self._model = None
        self._device = None
        self._dtype = None

    @classmethod
    def from_env(cls, model_name: Optional[str] = None) -> Optional["Florence2PhraseGrounder"]:
        if os.getenv("AGENT_BANANA_DISABLE_VLM") == "1":
            return None
        return cls(model_name=model_name or os.getenv("AGENT_BANANA_GROUNDING_MODEL") or DEFAULT_GROUNDING_MODEL)

    def mode_label(self) -> str:
        device = self._device if self._device is not None else "lazy"
        return f"{self.model_name} @ {device}"

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Florence2ForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - dependency availability
            raise GroundingError(
                "Florence-2 grounding requires the optional dependencies `transformers`, `torch`, and `torchvision`."
            ) from exc

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"

        dtype = torch.float32
        if device == "cuda":
            dtype = torch.float16
        elif device == "mps":
            dtype = torch.float16

        self._torch = torch
        self._device = device
        self._dtype = dtype
        self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True, use_fast=False)
        self._model = Florence2ForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        self._model.eval()

    def localize(
        self,
        image: Image.Image,
        phrases: List[str],
        *,
        profile: str,
    ) -> GroundingResult:
        self._ensure_loaded()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None

        candidates: List[GroundingCandidate] = []
        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        image = image.convert("RGB")

        for phrase in phrases:
            prompt = f"{task} {phrase}".strip()
            inputs = self._processor(text=prompt, images=image, return_tensors="pt")
            prepared = {}
            for key, value in inputs.items():
                if hasattr(value, "to"):
                    if getattr(value, "dtype", None) is not None and getattr(value, "dtype", None).is_floating_point:
                        prepared[key] = value.to(self._device, dtype=self._dtype)
                    else:
                        prepared[key] = value.to(self._device)
                else:
                    prepared[key] = value

            with self._torch.no_grad():
                generated_ids = self._model.generate(
                    input_ids=prepared["input_ids"],
                    pixel_values=prepared["pixel_values"],
                    max_new_tokens=256,
                    num_beams=3,
                    do_sample=False,
                )
            generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = self._processor.post_process_generation(
                generated_text,
                task=task,
                image_size=image.size,
            )
            task_payload = parsed.get(task, parsed)
            bboxes = task_payload.get("bboxes", []) or task_payload.get("boxes", [])
            labels = task_payload.get("labels", []) or []
            for index, raw_box in enumerate(bboxes):
                if len(raw_box) != 4:
                    continue
                bbox = BoundingBox(
                    left=max(0, int(raw_box[0])),
                    top=max(0, int(raw_box[1])),
                    right=min(image.size[0], int(raw_box[2])),
                    bottom=min(image.size[1], int(raw_box[3])),
                )
                if bbox.area <= 0:
                    continue
                label = labels[index] if index < len(labels) and labels[index] else phrase
                score = 0.9 if label.lower() == phrase.lower() else 0.76
                candidates.append(
                    GroundingCandidate(
                        phrase=label,
                        bbox=bbox,
                        score=score,
                        source="phrase-grounding",
                    )
                )

        return GroundingResult(phrases=phrases, candidates=candidates)


class MockVlmLocalizer(VlmLocalizer):
    def mode_label(self) -> str:
        return "mock-vlm-localizer"

    def localize(
        self,
        image: Image.Image,
        phrases: List[str],
        *,
        profile: str,
    ) -> GroundingResult:
        bbox = fallback_box_for_profile(image.size, profile)
        phrase = phrases[0] if phrases else profile
        return GroundingResult(
            phrases=phrases,
            candidates=[
                GroundingCandidate(
                    phrase=phrase,
                    bbox=bbox,
                    score=0.55,
                    source="mock-prior",
                )
            ],
        )


def build_localizer() -> VlmLocalizer:
    localizer = Florence2PhraseGrounder.from_env()
    if localizer is None:
        raise GroundingError(
            "Florence-2 is disabled via environment (AGENT_BANANA_DISABLE_VLM=1). "
            "Remove this flag to enable grounding."
        )
    return localizer
