"""Analisis de imagenes mediante un modelo de vision en Ollama local."""

from __future__ import annotations

import base64
from io import BytesIO

import requests
from PIL import Image

OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
VISION_MODEL = "moondream"


def model_is_available() -> bool:
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=4)
        response.raise_for_status()
        return any(
            model.get("name", "").split(":")[0] == VISION_MODEL
            for model in response.json().get("models", [])
        )
    except requests.RequestException:
        return False


def analyze_image(image_bytes: bytes, question: str) -> str:
    """Analiza una imagen en local y devuelve observaciones en espanol."""
    with Image.open(BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
        image.thumbnail((1280, 1280))
        encoded_image = BytesIO()
        image.save(encoded_image, format="JPEG", quality=88, optimize=True)

    prompt = (
        "Describe only the clearly visible contents in two concise sentences. "
        "If the user asks for feedback, add one practical suggestion grounded in "
        "the visible image. Answer in English. User request: " + question.strip()
    )
    response = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": VISION_MODEL,
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [base64.b64encode(encoded_image.getvalue()).decode("ascii")],
            }],
            "stream": False,
            "keep_alive": "0",
            "options": {"num_ctx": 2048, "num_predict": 384, "temperature": 0.2},
        },
        timeout=(8, 300),
    )
    response.raise_for_status()
    answer = response.json().get("message", {}).get("content", "").strip()
    if not answer:
        raise RuntimeError("El modelo local de vision no devolvio una respuesta.")
    return _rewrite_in_spanish(answer, question)


def _rewrite_in_spanish(visual_draft: str, question: str) -> str:
    prompt = (
        "Translate the following visual analysis into natural Spanish faithfully. "
        "Do not add or infer details. Keep any suggestion as a suggestion. "
        "Analysis: " + visual_draft
    )
    response = requests.post(
        OLLAMA_CHAT_URL,
        json={
            'model': 'qwen3:1.7b',
            'messages': [{'role': 'user', 'content': prompt}],
            'stream': False,
            'think': False,
            'keep_alive': '10m',
            'options': {'num_ctx': 2048, 'num_predict': 220, 'temperature': 0.1},
        },
        timeout=(8, 90),
    )
    response.raise_for_status()
    rewritten = response.json().get('message', {}).get('content', '').strip()
    return rewritten or visual_draft
