"""The editing and generation models: what they are, where they live, and how
they are downloaded.

Weights are never part of the repository. They are fetched into
`<models dir>/studio/` on request, and nothing but the model files themselves
is exchanged with the download hosts.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class Model:
    id: str
    label: str
    licence: str
    size_mb: int
    folder: str  # under <models dir>/studio
    files: tuple[str, ...]  # every file that must exist, relative to `folder`
    repo: str | None = None  # Hugging Face repository
    revision: str | None = None
    # File in the repository -> file in `folder`, when the names differ.
    rename: dict[str, str] = field(default_factory=dict)
    urls: dict[str, str] = field(default_factory=dict)  # file -> direct download address


_SDXL_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer_2/merges.txt",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.fp16.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
)

MODELS: dict[str, Model] = {
    m.id: m
    for m in (
        Model(
            "birefnet", "BiRefNet (background removal)", "MIT", 930, "birefnet",
            ("onnx/model.onnx",), repo="onnx-community/BiRefNet-ONNX",
        ),
        Model(
            "realesrgan", "Real-ESRGAN x4 (upscaling)", "BSD-3-Clause", 64, "upscale",
            ("RealESRGAN_x4plus.pth",),
            urls={"RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"},
        ),
        Model(
            "gfpgan", "GFPGAN v1.4 (face restoration)", "Apache-2.0", 333, "upscale",
            ("GFPGANv1.4.pth",),
            urls={"GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"},
        ),
        Model(
            "sdxl", "Stable Diffusion XL base 1.0", "CreativeML Open RAIL++-M", 6620, "sdxl-base",
            _SDXL_FILES, repo="stabilityai/stable-diffusion-xl-base-1.0",
        ),
        Model(
            "sdxl-vae", "SDXL VAE (half-precision fix)", "MIT", 320, "sdxl-vae-fp16-fix",
            ("config.json", "diffusion_pytorch_model.safetensors"), repo="madebyollin/sdxl-vae-fp16-fix",
        ),
        Model(
            "sdxl-inpaint", "SDXL inpainting 0.1", "CreativeML Open RAIL++-M", 4900, "sdxl-inpaint",
            ("unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors"),
            repo="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        ),
        Model(
            "controlnet-canny", "ControlNet Canny for SDXL", "CreativeML Open RAIL++-M", 2390, "controlnet-canny",
            ("config.json", "diffusion_pytorch_model.fp16.safetensors"), repo="diffusers/controlnet-canny-sdxl-1.0",
        ),
        Model(
            "instantid", "InstantID (likeness of a person)", "Apache-2.0", 4000, "instantid",
            ("ControlNetModel/config.json", "ControlNetModel/diffusion_pytorch_model.safetensors", "ip-adapter.bin"),
            repo="InstantX/InstantID",
        ),
        Model(
            "antelopev2", "InsightFace antelopev2 face embedding", "InsightFace, non-commercial research only", 250,
            "antelopev2", ("glintr100.onnx",), repo="DIAMONIK7777/antelopev2",
        ),
        Model(
            "lora-anime", "Anime style LoRA", "MIT", 8, "loras", ("anime.safetensors",),
            repo="ntc-ai/SDXL-LoRA-slider.anime",
        ),
        Model(
            "lora-3d", "3D render style LoRA", "Apache-2.0", 81, "loras", ("3d_render_style_xl.safetensors",),
            repo="goofyai/3d_render_style_xl",
        ),
        Model(
            "lora-watercolor", "Watercolour style LoRA", "Apache-2.0", 11, "loras", ("watercolor_v1_sdxl.safetensors",),
            repo="ostris/watercolor_style_lora_sdxl",
        ),
        Model(
            "lora-pixel", "Pixel art style LoRA", "CreativeML Open RAIL-M", 163, "loras", ("pixel-art-xl.safetensors",),
            repo="nerijs/pixel-art-xl",
        ),
        Model(
            "lora-clay", "Clay animation style LoRA", "CreativeML Open RAIL-M", 163, "loras", ("ClayAnimationRedm.safetensors",),
            repo="artificialguybr/ClayAnimationRedmond",
        ),
    )
}

_SDXL = ("sdxl", "sdxl-vae")
# The models each feature needs before it can run.
FEATURES: dict[str, tuple[str, ...]] = {
    "remove_background": ("birefnet",),
    "enhance": ("realesrgan", "gfpgan"),
    "restore": ("realesrgan", "gfpgan"),
    "replace_background": ("birefnet", *_SDXL, "sdxl-inpaint"),
    "inpaint": (*_SDXL, "sdxl-inpaint"),
    "extend": (*_SDXL, "sdxl-inpaint"),
    "stylize": (*_SDXL, "controlnet-canny"),
    "generate": _SDXL,
    "generate_person": (*_SDXL, "instantid", "antelopev2"),
}


def studio_dir(models_dir: Path) -> Path:
    return models_dir / "studio"


def model_dir(models_dir: Path, model_id: str) -> Path:
    return studio_dir(models_dir) / MODELS[model_id].folder


def is_installed(models_dir: Path, model_id: str) -> bool:
    folder = model_dir(models_dir, model_id)
    return all((folder / f).is_file() for f in MODELS[model_id].files)


def missing_for(models_dir: Path, feature: str, extra: tuple[str, ...] = ()) -> list[str]:
    """Ids of the models `feature` still needs."""
    return [m for m in (*FEATURES[feature], *extra) if not is_installed(models_dir, m)]


def _fetch_url(url: str, dest: Path, cancelled: Callable[[], bool]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "pixora"})
    with urllib.request.urlopen(request, timeout=60) as response, open(tmp, "wb") as out:
        while chunk := response.read(1024 * 1024):
            if cancelled():
                break
            out.write(chunk)
    if cancelled():
        tmp.unlink(missing_ok=True)
        return
    os.replace(tmp, dest)


def download(
    models_dir: Path,
    model_ids: list[str],
    progress: Callable[[float, str], None] = lambda fraction, message: None,
    cancelled: Callable[[], bool] = lambda: False,
) -> None:
    """Download the given models, skipping files that are already there."""
    wanted = [MODELS[m] for m in model_ids if not is_installed(models_dir, m)]
    total = sum(m.size_mb for m in wanted) or 1
    done = 0
    for model in wanted:
        if cancelled():
            return
        progress(done / total, f"Downloading {model.label} ({model.size_mb:,} MB)")
        folder = studio_dir(models_dir) / model.folder
        folder.mkdir(parents=True, exist_ok=True)
        for name in model.files:
            if cancelled():
                return
            if (folder / name).is_file():
                continue
            if name in model.urls:
                _fetch_url(model.urls[name], folder / name, cancelled)
            else:
                from huggingface_hub import hf_hub_download  # lazily: only downloads need it

                source = next((k for k, v in model.rename.items() if v == name), name)
                got = Path(hf_hub_download(model.repo, source, revision=model.revision, local_dir=folder))
                if source != name:
                    (folder / name).parent.mkdir(parents=True, exist_ok=True)
                    os.replace(got, folder / name)
        done += model.size_mb
    progress(1.0, "Models are ready")
