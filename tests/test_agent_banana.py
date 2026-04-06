from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageChops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_banana.memory import ContextFolder  # noqa: E402
from agent_banana.models import BoundingBox, GroundingCandidate  # noqa: E402
from agent_banana.nano_banana import MockNanoBananaClient  # noqa: E402
from agent_banana.llm_grounding_advisor import MockGroundingAdvisor  # noqa: E402
from agent_banana.pipeline import AgentBananaApp  # noqa: E402
from agent_banana.planning import RLPlanner, RLValueStore  # noqa: E402
from agent_banana.quality import QualityJudge  # noqa: E402
from agent_banana.targeting import bbox_iou, classify_target, rank_grounding_candidates, refine_bbox_for_profile  # noqa: E402
from agent_banana.vision import assess_preview_framing, decode_image_payload, fit_image_inside_canvas  # noqa: E402
from agent_banana.vlm_localizer import GroundingResult, MockVlmLocalizer, VlmLocalizer  # noqa: E402


def make_test_image() -> Image.Image:
    image = Image.new("RGB", (220, 180), "#f7f2e8")
    draw = ImageDraw.Draw(image)
    draw.ellipse((72, 48, 146, 122), fill="#d97706", outline="#8c3b12", width=3)
    draw.rectangle((20, 132, 200, 164), fill="#d7e1c6")
    return image


class FakeVlmLocalizer(VlmLocalizer):
    def __init__(self, candidates: list[GroundingCandidate]):
        self._candidates = list(candidates)

    def mode_label(self) -> str:
        return "fake-vlm"

    def localize(self, image: Image.Image, phrases: list[str], *, profile: str) -> GroundingResult:
        return GroundingResult(phrases=list(phrases), candidates=list(self._candidates))


class AgentBananaPlannerTests(unittest.TestCase):
    def test_rl_planner_ranks_local_replacement_before_global_style(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            planner = RLPlanner(RLValueStore(Path(tmp_dir) / "planner.json"))
            context = ContextFolder().fold([])
            edits = planner.parse_instruction("Replace the bowl with a banana then warm the background lighting.", context)
            candidates = planner.plan(edits, context)

        self.assertGreater(len(candidates), 1)
        selected = candidates[0]
        self.assertEqual([step.edit_id for step in selected.steps], ["edit-1", "edit-2"])
        self.assertEqual(selected.steps[0].mode, "preview_expand")
        self.assertEqual(selected.steps[1].scope, "global")

    def test_glasses_removal_prefers_tight_local_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            planner = RLPlanner(RLValueStore(Path(tmp_dir) / "planner.json"))
            context = ContextFolder().fold([])
            edits = planner.parse_instruction("Remove the glasses worn by the lady from the image.", context)
            candidates = planner.plan(edits, context)

        self.assertEqual(classify_target(edits[0].target, edits[0].verb), "face_accessory")
        self.assertEqual(candidates[0].steps[0].mode, "preview_tight")

    def test_glasses_replacement_prefers_tight_local_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            planner = RLPlanner(RLValueStore(Path(tmp_dir) / "planner.json"))
            context = ContextFolder().fold([])
            edits = planner.parse_instruction("Replace the spectacles worn by the grandma with red spectacles.", context)
            candidates = planner.plan(edits, context)

        self.assertEqual(classify_target(edits[0].target, edits[0].verb), "face_accessory")
        self.assertEqual(candidates[0].steps[0].mode, "preview_tight")

    def test_spectacles_color_change_uses_tight_mode(self) -> None:
        """Verify that 'adjust' verb on spectacles still selects preview_tight."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            planner = RLPlanner(RLValueStore(Path(tmp_dir) / "planner.json"))
            context = ContextFolder().fold([])
            edits = planner.parse_instruction(
                "Change the colour of grandma's spectacles to red.", context
            )
            candidates = planner.plan(edits, context)

        self.assertEqual(classify_target(edits[0].target, edits[0].verb), "face_accessory")
        self.assertEqual(candidates[0].steps[0].mode, "preview_tight")


class AgentBananaVisionTests(unittest.TestCase):
    def test_face_accessory_bbox_is_shrunk_from_large_grounding_region(self) -> None:
        source = Image.new("RGB", (480, 640), "white")
        raw_bbox = BoundingBox(left=70, top=30, right=240, bottom=220)
        refined = refine_bbox_for_profile(raw_bbox, source.size, "face_accessory")

        self.assertLess(refined.area, raw_bbox.area)
        self.assertLessEqual(refined.width, int(source.size[0] * 0.28))
        self.assertLessEqual(refined.height, int(source.size[1] * 0.14))

    def test_face_accessory_refine_preserves_detected_vertical_position(self) -> None:
        candidate = BoundingBox(left=90, top=300, right=180, bottom=340)
        refined = refine_bbox_for_profile(candidate, (480, 640), "face_accessory")

        self.assertEqual(
            (refined.top + refined.bottom) // 2,
            (candidate.top + candidate.bottom) // 2,
        )

    def test_grounding_candidate_ranking_prefers_plausible_face_accessory_box(self) -> None:
        image_size = (480, 640)
        candidates = [
            GroundingCandidate(
                phrase="spectacles",
                bbox=BoundingBox(left=40, top=20, right=280, bottom=260),
                score=0.95,
                source="phrase-grounding",
            ),
            GroundingCandidate(
                phrase="spectacles",
                bbox=BoundingBox(left=88, top=82, right=180, bottom=132),
                score=0.82,
                source="phrase-grounding",
            ),
        ]

        ranked = rank_grounding_candidates(candidates, image_size, "face_accessory")

        self.assertEqual(ranked[0].bbox.as_tuple(), (88, 82, 180, 132))

    def test_grounding_candidate_ranking_keeps_mid_frame_detection_viable(self) -> None:
        image_size = (480, 640)
        candidates = [
            GroundingCandidate(
                phrase="spectacles",
                bbox=BoundingBox(left=100, top=110, right=180, bottom=150),
                score=0.60,
                source="phrase-grounding",
            ),
            GroundingCandidate(
                phrase="spectacles",
                bbox=BoundingBox(left=102, top=320, right=182, bottom=360),
                score=0.62,
                source="phrase-grounding",
            ),
        ]

        ranked = rank_grounding_candidates(candidates, image_size, "face_accessory")

        self.assertEqual(ranked[0].bbox.as_tuple(), (102, 320, 182, 360))

    def test_bbox_iou_zero_for_disjoint_boxes(self) -> None:
        a = BoundingBox(left=10, top=10, right=30, bottom=30)
        b = BoundingBox(left=40, top=40, right=60, bottom=60)

        self.assertEqual(bbox_iou(a, b), 0.0)

    def test_preview_framing_assessment_detects_reframed_preview(self) -> None:
        source = make_test_image().resize((480, 640))
        cropped = source.crop((60, 0, 420, 640))
        reframed = fit_image_inside_canvas(cropped, source.size)

        assessment = assess_preview_framing(source, reframed)

        self.assertGreater(assessment["average"], 0.03)


class AgentBananaQualityTests(unittest.TestCase):
    def test_quality_rejects_oversized_face_accessory_edit(self) -> None:
        before = Image.new("RGB", (240, 240), "white")
        after = before.copy()
        draw = ImageDraw.Draw(after)
        draw.rectangle((40, 40, 180, 170), fill="black")
        judge = QualityJudge()

        quality = judge.evaluate(
            before,
            after,
            BoundingBox(left=40, top=40, right=180, bottom=170),
            preview=after,
            target="glasses",
            verb="remove",
        )

        self.assertFalse(quality.accepted)
        self.assertTrue(any("changed more structure" in note or "too large" in note for note in quality.notes))


class AgentBananaPipelineTests(unittest.TestCase):
    def test_mock_pipeline_runs_end_to_end_and_persists_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            localizer = FakeVlmLocalizer(
                [
                    GroundingCandidate(
                        phrase="fruit",
                        bbox=BoundingBox(left=72, top=48, right=146, bottom=122),
                        score=0.92,
                        source="phrase-grounding",
                    )
                ]
            )
            app = AgentBananaApp(root=root, image_client=MockNanoBananaClient(), localizer=localizer, grounding_advisor=MockGroundingAdvisor())
            image = make_test_image()

            result = app.run(image, "Replace the center fruit with a banana and warm the background.")

            self.assertEqual(result.mode, "mock")
            self.assertEqual(result.grounding_mode, "fake-vlm")
            self.assertEqual(len(result.step_results), 2)
            self.assertTrue(result.session_id)
            session_path = root / "artifacts" / "agent_banana" / "sessions" / f"{result.session_id}.json"
            self.assertTrue(session_path.exists())

            final_image = decode_image_payload(result.final_image)
            diff = ImageChops.difference(image, final_image)
            self.assertIsNotNone(diff.getbbox())
            self.assertGreater(result.reward, 0.0)

    def test_local_edit_uses_vlm_localization_instead_of_preview_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            localizer = FakeVlmLocalizer(
                [
                    GroundingCandidate(
                        phrase="glasses",
                        bbox=BoundingBox(left=110, top=90, right=182, bottom=132),
                        score=0.94,
                        source="phrase-grounding",
                    )
                ]
            )
            app = AgentBananaApp(root=root, image_client=MockNanoBananaClient(), localizer=localizer, grounding_advisor=MockGroundingAdvisor())
            image = make_test_image().resize((480, 640))
            result = app.run(image, "Remove the glasses from the woman.")

            self.assertEqual(result.step_results[0].localizer_mode, "fake-vlm")
            self.assertEqual(result.step_results[0].grounding_candidates[0].bbox.as_tuple(), (110, 90, 182, 132))

    def test_mock_localizer_is_available_without_transformers(self) -> None:
        localizer = MockVlmLocalizer()
        result = localizer.localize(make_test_image(), ["glasses"], profile="face_accessory")
        self.assertTrue(result.candidates)


if __name__ == "__main__":
    unittest.main()
