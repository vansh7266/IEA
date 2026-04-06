from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from urllib import error, parse, request

from PIL import Image

DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image-preview"
DEFAULT_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiError(RuntimeError):
    pass


@dataclass
class GeminiResponse:
    image: Image.Image
    text: str


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _decode_image(encoded: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


def call_gemini(
    prompt: str,
    image: Image.Image,
    *,
    api_key: str | None = None,
    model: str = DEFAULT_IMAGE_MODEL,
    api_base: str = DEFAULT_API_BASE,
    timeout: int = 90,
) -> GeminiResponse:
    """Send a prompt + image to Gemini and return the response image and text."""
    api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise GeminiError("No API key found. Set GEMINI_API_KEY or pass api_key=")

    url = f"{api_base}/{parse.quote(model, safe='')}:generateContent"

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
            "responseModalities": ["TEXT", "IMAGE"],
            "temperature": 0.4,
            "topP": 0.9,
            "maxOutputTokens": 8192,
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

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise GeminiError(f"HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise GeminiError(f"Request failed: {exc.reason}") from exc

    data = json.loads(raw)
    texts, image_out = [], None

    for candidate in data.get("candidates", []):
        for part in (candidate.get("content") or {}).get("parts", []):
            if part.get("text"):
                texts.append(part["text"].strip())
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data") and image_out is None:
                image_out = _decode_image(inline["data"])

    if image_out is None:
        raise GeminiError(f"Response contained no image: {data}")

    return GeminiResponse(image=image_out, text="\n".join(t for t in texts if t))

class NanoBananaClient:
    def mode_label(self) -> str:
        raise NotImplementedError

    def edit_full_image(self, image: Image.Image, prompt: str) -> GeminiResponse:
        raise NotImplementedError


class MockNanoBananaClient(NanoBananaClient):
    def mode_label(self) -> str:
        return "mock"

    def edit_full_image(self, image: Image.Image, prompt: str) -> GeminiResponse:
        return GeminiResponse(image=image.convert("RGB").copy(), text="Mock full-image edit.")


@dataclass
class _GeminiClientShim(NanoBananaClient):
    api_key: str
    model: str = DEFAULT_IMAGE_MODEL
    api_base: str = DEFAULT_API_BASE
    timeout_seconds: int = 90

    def mode_label(self) -> str:
        return self.model

    def edit_full_image(self, image: Image.Image, prompt: str) -> GeminiResponse:
        return call_gemini(prompt, image, api_key=self.api_key, model=self.model,
                           api_base=self.api_base, timeout=self.timeout_seconds)


def build_image_client() -> NanoBananaClient:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        model = os.getenv("AGENT_BANANA_IMAGE_MODEL") or DEFAULT_IMAGE_MODEL
        return _GeminiClientShim(api_key=api_key, model=model)
    return MockNanoBananaClient()
