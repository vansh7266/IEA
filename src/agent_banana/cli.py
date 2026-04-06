from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .pipeline import AgentBananaApp
from .vision import decode_image_payload, save_png


def main() -> None:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run the Agent Banana editing pipeline on a local image.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the source image.")
    parser.add_argument("--instruction", required=True, help="Natural-language edit request.")
    parser.add_argument("--session-id", default=None, help="Optional session id for multi-turn editing.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_root / "artifacts" / "agent_banana" / "runs" / "latest",
        help="Directory where the artifact bundle will be written.",
    )
    args = parser.parse_args()

    app = AgentBananaApp.from_env(default_root)
    image = Image.open(args.image).convert("RGB")
    result = app.run(image, args.instruction, session_id=args.session_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_png(image, args.output_dir / "source.png")
    save_png(decode_image_payload(result.final_image), args.output_dir / "final.png")
    for index, step_result in enumerate(result.step_results, start=1):
        save_png(decode_image_payload(step_result.preview_data_url), args.output_dir / f"step-{index:02d}-preview.png")
        save_png(decode_image_payload(step_result.overlay_data_url), args.output_dir / f"step-{index:02d}-overlay.png")
        save_png(decode_image_payload(step_result.edited_data_url), args.output_dir / f"step-{index:02d}-edited.png")

    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"Session: {result.session_id}")
    print(f"Runtime mode: {result.mode}")
    print(f"Reward: {result.reward:.3f}")
    print(f"Artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
