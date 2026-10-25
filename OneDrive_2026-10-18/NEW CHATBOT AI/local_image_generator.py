"""Generacion de imagenes completamente local para Erika."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import streamlit as st

MODEL_ID = "segmind/tiny-sd"
MODEL_DIR = Path(__file__).parent / "models" / "tiny-sd"
OUTPUT_DIR = Path(__file__).parent / "generated_images"


def model_is_downloaded() -> bool:
    return (MODEL_DIR / "model_index.json").is_file()


def download_model() -> Path:
    """Descarga pesos publicos una vez; la inferencia posterior es local."""
    import truststore
    truststore.inject_into_ssl()
    from huggingface_hub import snapshot_download

    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=MODEL_ID, local_dir=str(MODEL_DIR), max_workers=2)
    return MODEL_DIR


def generate_image(prompt: str, seed: int | None = None) -> Path:
    """Genera una imagen 512x512 en CPU para evitar depender de CUDA."""
    if not model_is_downloaded():
        raise RuntimeError("Falta descargar el modelo local de imagenes.")

    import torch
    from diffusers import DiffusionPipeline

    pipeline = _get_pipeline()
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    result = pipeline(
        prompt=prompt.strip(),
        width=512,
        height=512,
        num_inference_steps=8,
        guidance_scale=7.5,
        generator=generator,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"erika_{datetime.now():%Y%m%d_%H%M%S}.png"
    result.images[0].save(output_path, format="PNG")
    return output_path


@st.cache_resource(show_spinner=False)
def _get_pipeline():
    import torch
    from diffusers import DiffusionPipeline

    return DiffusionPipeline.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.float32,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cpu")
