"""The editing and generation models, run locally with PyTorch.

Importing this module needs the optional `studio` dependencies. Everything is
loaded on first use and kept in ordinary memory; only the parts a step needs
sit on the graphics card while it runs.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from . import faces as face_engines
from . import identity, media, model_store
from .studio import DEFAULT_STEPS, Progress, StudioError, Style

log = logging.getLogger("pixora.studio")

GUIDANCE = 4.5
# The image model is told the picture comes from a larger original, and that poor
# pictures come from smaller ones. It answers with finer detail.
DETAIL_ORIGINAL, DETAIL_NEGATIVE = 1.5, 0.8
# After a picture is painted, each face in it is painted again on its own at full
# size, which is what gives clear eyes and skin. This is how far that second pass
# may move from the first (0 to 1), and how many faces it covers.
FACE_DETAIL_STRENGTH = 0.4
FACE_DETAIL_SIDE = 1024
FACE_DETAIL_MAX = 6
# How strongly a restyled photo follows the outlines of the original.
CANNY_SCALE = 0.6
# The outlines guide only the first part of the painting, which fixes the composition
# and then leaves the style free to simplify detail.
CANNY_END = 0.5
# Fine photographic texture is smoothed away before outlines are taken.
CANNY_BLUR = 2.0
# A picture of a known person is composed freely first; then the face in it is painted
# again as that person. This is how far the second painting may move from the first.
IDENTITY_STRENGTH = 0.8
IDENTITY_SIDE = 1016  # see the note on watermarks in `_posed_person`
# InstantID: how firmly the face is placed, and how much it looks like the person.
IDENTITY_POSE_SCALE = 0.8
IDENTITY_FACE_SCALE = 0.8
# How large the face is drawn, as the distance between the eyes over the picture's width.
IDENTITY_EYE_DISTANCE = 0.085
# Lower than for ordinary pictures: with a face to follow, more than this burns the colours.
IDENTITY_GUIDANCE = 3.5
# What following a face tends to bring with it, named so the model steers away from it.
IDENTITY_NEGATIVE = "stock photo watermark, logo, caption, signature, monochrome"

_MATTE_SIDE = 1024
_TILE, _TILE_PAD = 512, 16
# Where GFPGAN expects the eyes, nose and mouth corners in its 512 px input.
_FFHQ_512 = np.array(
    [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935], [201.26117, 371.41043], [313.08905, 371.15118]],
    dtype=np.float32,
)
# The optional parts: at most one of these groups is kept in memory at a time.
_OPTIONAL = ("inpaint_unet", "canny", "identity_net")


def _providers() -> list[str]:
    return ["DmlExecutionProvider", "CPUExecutionProvider"]


# The few faces a job looks at are quick on the processor, and a DirectML
# session would keep graphics memory that the image model needs.
_CPU = ["CPUExecutionProvider"]


class TorchEngine:
    def __init__(self, models_dir: Path):
        if not torch.cuda.is_available():
            raise StudioError("Editing needs an NVIDIA graphics card, and none was found", 503)
        self.models_dir = models_dir
        self.device = torch.device("cuda")
        self.dtype = torch.float16
        self._parts: dict[str, object] = {}
        self._matte: ort.InferenceSession | None = None
        self._faces: face_engines.InsightFaceEngine | None = None
        self._identity: ort.InferenceSession | None = None
        self._attributes: ort.InferenceSession | None = None
        self._expander = None
        self._ip_processors: dict | None = None

    # ---- loading -----------------------------------------------------------

    def _dir(self, model_id: str) -> Path:
        return model_store.model_dir(self.models_dir, model_id)

    def _part(self, name: str):
        """A model part, loaded into ordinary memory on first use."""
        if name in self._parts:
            return self._parts[name]
        from diffusers import AutoencoderKL, ControlNetModel, DPMSolverMultistepScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

        if name in _OPTIONAL:
            for other in _OPTIONAL:
                if other != name:
                    self._drop(other)
        base = self._dir("sdxl")
        half = {"torch_dtype": self.dtype, "variant": "fp16", "use_safetensors": True}
        if name == "unet":
            part = UNet2DConditionModel.from_pretrained(base / "unet", **half)
        elif name == "inpaint_unet":
            part = UNet2DConditionModel.from_pretrained(self._dir("sdxl-inpaint") / "unet", **half)
        elif name == "vae":
            part = AutoencoderKL.from_pretrained(self._dir("sdxl-vae"), torch_dtype=self.dtype, use_safetensors=True)
        elif name == "text_encoder":
            part = CLIPTextModel.from_pretrained(base / "text_encoder", **half)
        elif name == "text_encoder_2":
            part = CLIPTextModelWithProjection.from_pretrained(base / "text_encoder_2", **half)
        elif name == "tokenizer":
            part = CLIPTokenizer.from_pretrained(base / "tokenizer")
        elif name == "tokenizer_2":
            part = CLIPTokenizer.from_pretrained(base / "tokenizer_2")
        elif name == "scheduler":
            # DPM++ 2M SDE Karras: the fresh noise it adds at every step gives finer texture.
            part = DPMSolverMultistepScheduler.from_pretrained(
                base / "scheduler", algorithm_type="sde-dpmsolver++", use_karras_sigmas=True, euler_at_final=True
            )
        elif name == "canny":
            part = ControlNetModel.from_pretrained(self._dir("controlnet-canny"), **half)
        elif name == "identity_net":
            part = ControlNetModel.from_pretrained(
                self._dir("instantid") / "ControlNetModel", torch_dtype=self.dtype, use_safetensors=True
            )
        else:
            raise KeyError(name)
        self._parts[name] = part
        return part

    def _drop(self, name: str) -> None:
        part = self._parts.pop(name, None)
        if part is not None:
            if name == "identity_net" and self._ip_processors is not None:
                self._plain_attention()
                self._ip_processors = None
                self._parts.pop("image_proj", None)
            del part
            gc.collect()
            torch.cuda.empty_cache()

    def _on_gpu(self, *names: str) -> None:
        """Put exactly these parts on the graphics card; everything else leaves it."""
        for name, part in self._parts.items():
            if isinstance(part, torch.nn.Module):
                part.to(self.device if name in names else "cpu")
        torch.cuda.empty_cache()

    def rest(self) -> None:
        """Between jobs: give the graphics card back, keep the models in memory."""
        self._on_gpu()

    def unload(self) -> None:
        self._parts.clear()
        self._ip_processors = None
        self._upscaler = self._restorer = None
        self._matte = self._identity = self._faces = self._attributes = self._expander = None
        gc.collect()
        torch.cuda.empty_cache()

    def _components(self, unet: str = "unet") -> dict:
        return {
            "vae": self._part("vae"),
            "text_encoder": self._part("text_encoder"),
            "text_encoder_2": self._part("text_encoder_2"),
            "tokenizer": self._part("tokenizer"),
            "tokenizer_2": self._part("tokenizer_2"),
            "unet": self._part(unet),
            "scheduler": self._part("scheduler"),
        }

    # ---- shared steps ------------------------------------------------------

    def _encode(self, pipe, prompt: str, negative: str) -> dict:
        """Turn the prompts into what the image model reads, then free the text encoders."""
        self._on_gpu("text_encoder", "text_encoder_2")
        embeds, negative_embeds, pooled, negative_pooled = pipe.encode_prompt(
            prompt, device=self.device, do_classifier_free_guidance=True, negative_prompt=negative
        )
        return {
            "prompt_embeds": embeds,
            "negative_prompt_embeds": negative_embeds,
            "pooled_prompt_embeds": pooled,
            "negative_pooled_prompt_embeds": negative_pooled,
        }

    def _run(self, pipe, progress: Progress, seed: int, steps: int = DEFAULT_STEPS, label: str = "Painting", **kwargs) -> Image.Image:
        # Img2img and inpainting run only part of the schedule.
        total = max(1, round(steps * float(kwargs.get("strength", 1.0))))

        def on_step(_pipe, step, _timestep, tensors):
            progress.report(0.1 + 0.85 * (step + 1) / total)
            return tensors

        progress.report(0.1, label)
        with torch.inference_mode():
            out = pipe(
                num_inference_steps=steps,
                generator=torch.Generator("cpu").manual_seed(seed),
                callback_on_step_end=on_step,
                **kwargs,
            )
        return out.images[0].convert("RGB")

    @staticmethod
    def _sizes(width: int, height: int) -> dict:
        return {
            "original_size": (round(height * DETAIL_ORIGINAL), round(width * DETAIL_ORIGINAL)),
            "target_size": (height, width),
            "negative_original_size": (round(height * DETAIL_NEGATIVE), round(width * DETAIL_NEGATIVE)),
            "negative_target_size": (height, width),
        }

    @staticmethod
    def _face_square(points: np.ndarray, width: int, height: int) -> tuple[int, int, int, float]:
        """The square around a face that is worked on: left, top, side, and the face's size."""
        eyes = (points[0] + points[1]) / 2
        mouth = (points[3] + points[4]) / 2
        # A face turned aside has its eyes close together, so the height counts too.
        size = max(float(np.linalg.norm(points[0] - points[1])), float(np.linalg.norm(eyes - mouth)))
        side = int(min(size * 5, width, height))
        cx, cy = points.mean(axis=0)
        x = int(min(max(0, cx - side / 2), width - side))
        y = int(min(max(0, cy - side / 2), height - side))
        return x, y, side, size

    @staticmethod
    def _set_back(
        out: np.ndarray, fixed: Image.Image, x: int, y: int, side: int, head: tuple[float, float, float] | None = None
    ) -> None:
        """Put a repainted square back, fading in from its edge so no seam shows.

        With `head` (centre x, centre y, face size, in the square) only the head is
        taken, so a background that came out a different shade does not show as a patch.
        """
        new = np.asarray(fixed.convert("RGB").resize((side, side), Image.Resampling.LANCZOS), dtype=np.float32)
        margin = max(2, side // 8)
        alpha = np.zeros((side, side), dtype=np.float32)
        alpha[margin:-margin, margin:-margin] = 1.0
        alpha = cv2.GaussianBlur(alpha, (0, 0), margin / 2)
        if head is not None:
            cx, cy, size = head
            oval = np.zeros((side, side), dtype=np.float32)
            # The points are eyes, nose and mouth, so their middle sits low in the head.
            centre, axes = (round(cx), round(cy - size * 0.3)), (round(size * 1.6), round(size * 2.0))
            cv2.ellipse(oval, centre, axes, 0, 0, 360, 1.0, -1)
            alpha *= cv2.GaussianBlur(oval, (0, 0), max(1.0, size * 0.3))
        alpha = alpha[:, :, None]
        out[y : y + side, x : x + side] = out[y : y + side, x : x + side] * (1 - alpha) + new * alpha

    def _with_style(self, pipe, style: Style | None, extras: tuple[tuple[str, float], ...] = ()):
        """Load the look's LoRA, and any others, for one run. Returns what to call afterwards."""
        wanted = [(style.model, style.weight)] if style is not None and style.model else []
        wanted += list(extras)
        if not wanted:
            return lambda: None
        names = []
        for n, (model_id, _) in enumerate(wanted):
            names.append(f"lora{n}")
            file = model_store.MODELS[model_id].files[0]
            pipe.load_lora_weights(str(self._dir(model_id)), weight_name=file, adapter_name=names[-1])
        pipe.set_adapters(names, adapter_weights=[weight for _, weight in wanted])
        return pipe.unload_lora_weights

    def expand(self, prompt: str, seed: int) -> str:
        if self._expander is None:
            from .expansion import PromptExpander

            self._expander = PromptExpander(self._dir("prompt-expansion"))
        return self._expander.expand(prompt, seed)

    def _detail_faces(
        self, image: Image.Image, embeds: dict, seed: int, progress: Progress, steps: int = DEFAULT_STEPS,
        skip: np.ndarray | None = None,
    ) -> Image.Image:
        """Paint every face again, enlarged, and set it back into the picture.

        A face that takes up a small part of a picture gets too few pixels for
        clear eyes. Uses whichever model parts are already on the graphics card.
        `skip` is a face (its five points) to leave alone.
        """
        from diffusers import StableDiffusionXLImg2ImgPipeline

        rgb = np.asarray(image.convert("RGB"))
        h, w = rgb.shape[:2]
        crops = []
        for points in self._landmarks(np.ascontiguousarray(rgb[:, :, ::-1])):
            x, y, side, size = self._face_square(points, w, h)
            if size < 6 or side < 48 or side > FACE_DETAIL_SIDE * 0.9:
                continue  # too small to be a real face, or already painted at full size
            if skip is not None and np.linalg.norm(points.mean(axis=0) - skip.mean(axis=0)) < size:
                continue
            crops.append((x, y, side))
            if len(crops) == FACE_DETAIL_MAX:
                break
        if not crops:
            return image
        pipe = StableDiffusionXLImg2ImgPipeline(**self._components())
        pipe.set_progress_bar_config(disable=True)
        out = rgb.astype(np.float32)
        full = (FACE_DETAIL_SIDE, FACE_DETAIL_SIDE)
        for n, (x, y, side) in enumerate(crops):
            face = Image.fromarray(rgb[y : y + side, x : x + side]).resize(full, Image.Resampling.LANCZOS)
            fixed = self._run(
                pipe, progress.part(n / len(crops), (n + 1) / len(crops)), seed, steps=steps, label="Refining faces",
                image=face, strength=FACE_DETAIL_STRENGTH, guidance_scale=GUIDANCE, **self._sizes(*full), **embeds,
            )
            self._set_back(out, fixed, x, y, side)
        return Image.fromarray(out.round().clip(0, 255).astype(np.uint8))

    def _plain_attention(self) -> None:
        unet = self._parts.get("unet")
        if unet is not None:
            from diffusers.models.attention_processor import AttnProcessor2_0

            # Set outright: diffusers' own reset refuses the InstantID processors.
            unet.set_attn_processor(AttnProcessor2_0())

    # ---- background --------------------------------------------------------

    def matte(self, image: Image.Image, progress: Progress) -> Image.Image:
        progress.report(0.0, "Finding the subject")
        if self._matte is None:
            path = self._dir("birefnet") / "onnx" / "model.onnx"
            try:
                self._matte = ort.InferenceSession(str(path), providers=_providers())
            except Exception:
                log.warning("BiRefNet could not start on the graphics card; using the processor", exc_info=True)
                self._matte = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        small = image.convert("RGB").resize((_MATTE_SIDE, _MATTE_SIDE), Image.Resampling.BILINEAR)
        x = (np.asarray(small, dtype=np.float32) / 255.0 - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
        x = x.transpose(2, 0, 1)[None].astype(np.float32)
        session = self._matte
        try:
            logits = session.run(None, {session.get_inputs()[0].name: x})[-1]
        except Exception:
            if session.get_providers() == ["CPUExecutionProvider"]:
                raise
            log.warning("BiRefNet failed on the graphics card; using the processor", exc_info=True)
            path = self._dir("birefnet") / "onnx" / "model.onnx"
            session = self._matte = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            logits = session.run(None, {session.get_inputs()[0].name: x})[-1]
        # DirectML keeps several gigabytes for this model; give them back at once.
        self._matte = session = None
        gc.collect()
        alpha = 1.0 / (1.0 + np.exp(-logits.reshape(_MATTE_SIDE, _MATTE_SIDE).astype(np.float32)))
        mask = Image.fromarray((alpha * 255).astype(np.uint8), "L")
        progress.report(1.0)
        return mask.resize(image.size, Image.Resampling.BILINEAR)

    # ---- quality -----------------------------------------------------------

    _upscaler = None
    _restorer = None

    def _spandrel(self, model_id: str):
        from spandrel import ModelLoader

        path = self._dir(model_id) / model_store.MODELS[model_id].files[0]
        return ModelLoader(device="cpu").load_from_file(str(path)).eval()

    def upscale(self, image: Image.Image, scale: float, progress: Progress) -> Image.Image:
        progress.report(0.0, "Sharpening")
        self._on_gpu()
        if self._upscaler is None:
            self._upscaler = self._spandrel("realesrgan")
        model = self._upscaler.to(self.device)
        src = np.asarray(image.convert("RGB"))
        h, w = src.shape[:2]
        out = np.zeros((round(h * scale), round(w * scale), 3), dtype=np.uint8)
        tiles = [(x, y) for y in range(0, h, _TILE) for x in range(0, w, _TILE)]
        try:
            for n, (x, y) in enumerate(tiles):
                # Each tile is run with a margin of its neighbours, which is then cut off,
                # so no seams show where tiles meet.
                x1, y1 = max(0, x - _TILE_PAD), max(0, y - _TILE_PAD)
                x2, y2 = min(w, x + _TILE + _TILE_PAD), min(h, y + _TILE + _TILE_PAD)
                tile = torch.from_numpy(src[y1:y2, x1:x2].copy()).to(self.device).permute(2, 0, 1)[None].float() / 255
                with torch.inference_mode():
                    big = model(tile).clamp(0, 1)
                factor = big.shape[-1] / tile.shape[-1]
                tx2, ty2 = min(w, x + _TILE), min(h, y + _TILE)
                big = big[
                    ..., round((y - y1) * factor) : round((ty2 - y1) * factor),
                    round((x - x1) * factor) : round((tx2 - x1) * factor),
                ]
                piece = (big[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)
                ox1, oy1, ox2, oy2 = round(x * scale), round(y * scale), round(tx2 * scale), round(ty2 * scale)
                if piece.shape[:2] != (oy2 - oy1, ox2 - ox1):
                    piece = cv2.resize(piece, (ox2 - ox1, oy2 - oy1), interpolation=cv2.INTER_AREA)
                out[oy1:oy2, ox1:ox2] = piece
                progress.report((n + 1) / len(tiles))
        finally:
            self._upscaler.to("cpu")
            torch.cuda.empty_cache()
        return Image.fromarray(out)

    def _face_engine(self) -> face_engines.InsightFaceEngine | None:
        if self._faces is None:
            try:
                self._faces = face_engines.InsightFaceEngine(self.models_dir, _CPU)
            except face_engines.ModelsMissing:
                return None
        return self._faces

    def _landmarks(self, bgr: np.ndarray) -> list[np.ndarray]:
        """Five points for every face in a BGR image, largest face first."""
        engine = self._face_engine()
        if engine is None:
            return []
        work, scale = face_engines.working_image(bgr)
        boxes, _, points = engine._detect(work)
        offset = 0
        if len(boxes) == 0:
            # The detector misses a face that fills its whole view. Look again from further back.
            offset = max(bgr.shape[:2]) // 2
            wide = cv2.copyMakeBorder(bgr, offset, offset, offset, offset, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            work, scale = face_engines.working_image(wide)
            boxes, _, points = engine._detect(work)
        order = np.argsort([-(b[2] - b[0]) * (b[3] - b[1]) for b in boxes])
        return [(points[i] / scale - offset).astype(np.float32) for i in order]

    def restore_faces(self, image: Image.Image, strength: float, progress: Progress) -> Image.Image:
        progress.report(0.0, "Restoring faces")
        rgb = np.asarray(image.convert("RGB"))
        found = self._landmarks(np.ascontiguousarray(rgb[:, :, ::-1]))
        if not found:
            return image
        self._on_gpu()
        if self._restorer is None:
            self._restorer = self._spandrel("gfpgan")
        model = self._restorer.to(self.device)
        out = rgb.astype(np.float32)
        h, w = rgb.shape[:2]
        # The middle of the 512 px face crop, fading out towards its edge.
        inner = np.zeros((512, 512), dtype=np.float32)
        cv2.ellipse(inner, (256, 270), (170, 210), 0, 0, 360, 1.0, -1)
        inner = cv2.GaussianBlur(inner, (0, 0), 24)
        try:
            for n, points in enumerate(found[:20]):
                matrix, _ = cv2.estimateAffinePartial2D(points, _FFHQ_512, method=cv2.LMEDS)
                if matrix is None:
                    continue
                face = cv2.warpAffine(rgb, matrix, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                # spandrel takes and returns values from 0 to 1 for every model it wraps.
                x = torch.from_numpy(face).to(self.device).permute(2, 0, 1)[None].float() / 255
                with torch.inference_mode():
                    y = model(x)
                fixed = (y[0].clamp(0, 1) * 255).permute(1, 2, 0).float().cpu().numpy()
                back = cv2.invertAffineTransform(matrix)
                fixed_full = cv2.warpAffine(fixed, back, (w, h), flags=cv2.INTER_LINEAR)
                alpha = cv2.warpAffine(inner, back, (w, h), flags=cv2.INTER_LINEAR)[:, :, None] * strength
                out = out * (1 - alpha) + fixed_full * alpha
                progress.report((n + 1) / min(len(found), 20))
        finally:
            self._restorer.to("cpu")
            torch.cuda.empty_cache()
        return Image.fromarray(out.round().clip(0, 255).astype(np.uint8))

    # ---- painting ----------------------------------------------------------

    def inpaint(
        self, image: Image.Image, mask: Image.Image, prompt: str, negative: str, seed: int, progress: Progress,
        steps: int = DEFAULT_STEPS,
    ) -> Image.Image:
        from diffusers import StableDiffusionXLInpaintPipeline

        progress.report(0.0, "Loading the painting model")
        pipe = StableDiffusionXLInpaintPipeline(**self._components("inpaint_unet"))
        pipe.set_progress_bar_config(disable=True)
        embeds = self._encode(pipe, prompt, negative)
        self._on_gpu("inpaint_unet", "vae")
        return self._run(
            pipe, progress, seed, steps=steps, image=image, mask_image=mask.convert("L"), width=image.width, height=image.height,
            strength=0.99, guidance_scale=7.5, **embeds,
        )

    def restyle(
        self, image: Image.Image, prompt: str, negative: str, style: Style, strength: float, seed: int,
        progress: Progress, steps: int = DEFAULT_STEPS,
    ) -> Image.Image:
        from diffusers import StableDiffusionXLControlNetImg2ImgPipeline

        progress.report(0.0, "Loading the style")
        self._plain_attention()
        pipe = StableDiffusionXLControlNetImg2ImgPipeline(controlnet=self._part("canny"), **self._components())
        pipe.set_progress_bar_config(disable=True)
        undo = self._with_style(pipe, style)
        try:
            embeds = self._encode(pipe, prompt, negative)
            grey = cv2.GaussianBlur(np.asarray(image.convert("L")), (0, 0), CANNY_BLUR) if CANNY_BLUR else np.asarray(image.convert("L"))
            edges = cv2.Canny(grey, 60, 150)
            outline = Image.fromarray(np.stack([edges] * 3, axis=-1))
            self._on_gpu("unet", "vae", "canny")
            paint = progress.part(0.0, 0.8) if style.face_detail else progress
            out = self._run(
                pipe, paint, seed, steps=steps, image=image, control_image=outline, width=image.width, height=image.height,
                strength=strength, controlnet_conditioning_scale=CANNY_SCALE, control_guidance_end=CANNY_END,
                guidance_scale=GUIDANCE, **embeds,
            )
            if not style.face_detail:
                return out
            return self._detail_faces(out, embeds, seed, progress.part(0.8, 1.0), steps=steps)
        finally:
            undo()

    def generate(
        self, prompt: str, negative: str, width: int, height: int, seed: int, style: Style | None,
        references: list[identity.Reference], progress: Progress, steps: int = DEFAULT_STEPS,
        extras: tuple[tuple[str, float], ...] = (),
    ) -> Image.Image:
        if references:
            return self._person(prompt, negative, width, height, seed, style, references, progress, steps, extras)
        from diffusers import StableDiffusionXLPipeline

        progress.report(0.0, "Loading the image model")
        self._plain_attention()
        pipe = StableDiffusionXLPipeline(**self._components())
        pipe.set_progress_bar_config(disable=True)
        undo = self._with_style(pipe, style, extras)
        try:
            embeds = self._encode(pipe, prompt, negative)
            self._on_gpu("unet", "vae")
            out = self._run(
                pipe, progress.part(0.0, 0.8), seed, steps=steps, width=width, height=height, guidance_scale=GUIDANCE,
                **self._sizes(width, height), **embeds,
            )
            return self._detail_faces(out, embeds, seed, progress.part(0.8, 1.0), steps=steps)
        finally:
            undo()

    # ---- a known person ----------------------------------------------------

    def _gender_age(self, bgr: np.ndarray, box: tuple[float, float, float, float]) -> tuple[bool, float] | None:
        """Whether a face looks male, and its apparent age. None when that model is not installed."""
        if self._attributes is None:
            path = self.models_dir / "buffalo_l" / "genderage.onnx"
            if not path.is_file():
                return None
            self._attributes = ort.InferenceSession(str(path), providers=_CPU)
        x1, y1, x2, y2 = box
        scale = 96 / (max(x2 - x1, y2 - y1) * 1.5)
        matrix = np.array(
            [[scale, 0, 48 - (x1 + x2) / 2 * scale], [0, scale, 48 - (y1 + y2) / 2 * scale]], dtype=np.float32
        )
        blob = cv2.dnn.blobFromImage(cv2.warpAffine(bgr, matrix, (96, 96), borderValue=0), 1.0, (96, 96), swapRB=True)
        out = self._attributes.run(None, {self._attributes.get_inputs()[0].name: blob})[0][0]
        return bool(out[1] > out[0]), float(out[2]) * 100

    @staticmethod
    def _in_words(looks: list[tuple[bool, float]]) -> str:
        """'a 25 year old man': without it the scene is composed around a stranger of any kind."""
        if not looks:
            return identity.PERSON
        male = sum(m for m, _ in looks) * 2 >= len(looks)
        age = float(np.median([a for _, a in looks]))
        if age < 13:
            return "a young boy" if male else "a young girl"
        if age < 20:
            return "a teenage boy" if male else "a teenage girl"
        return f"a {max(20, round(age / 5) * 5)} year old {'man' if male else 'woman'}"

    def _identity_embedding(self, references: list[identity.Reference], progress: Progress) -> tuple[np.ndarray, str]:
        """What InstantID needs to know about a face, averaged over the reference photos,
        and the person in a few words."""
        engine = self._face_engine()
        if engine is None:
            raise StudioError("The face models are not installed")
        if self._identity is None:
            self._identity = ort.InferenceSession(
                str(self._dir("antelopev2") / "glintr100.onnx"), providers=_CPU
            )
        session, found, looks = self._identity, [], []
        for n, ref in enumerate(references):
            progress.report(n / len(references), "Studying the reference photos")
            if ref.path is None:
                continue
            try:
                full = media.load_bgr(ref.path)
            except Exception:
                continue
            h, w = full.shape[:2]
            x1, y1, x2, y2 = ref.box[0] * w, ref.box[1] * h, ref.box[2] * w, ref.box[3] * h
            # Look around the stored face only. The detector misses a face that fills its
            # whole view, so the crop is a few times the face's size, and of the faces
            # found in it the one where the stored face was is taken.
            mx, my = (x2 - x1) * 1.5, (y2 - y1) * 1.5
            cx1, cy1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
            crop = np.ascontiguousarray(full[cy1 : int(min(h, y2 + my)), cx1 : int(min(w, x2 + mx))])
            centre = np.array([(x1 + x2) / 2 - cx1, (y1 + y2) / 2 - cy1], dtype=np.float32)
            points = [
                pts for pts in self._landmarks(crop)
                if np.linalg.norm(pts.mean(axis=0) - centre) < 0.5 * max(x2 - x1, y2 - y1)
            ]
            if not points:
                continue
            matrix, _ = cv2.estimateAffinePartial2D(points[0], face_engines._ARCFACE_DST, method=cv2.LMEDS)
            if matrix is None:
                continue
            aligned = cv2.warpAffine(crop, matrix, (112, 112), borderValue=0)
            blob = cv2.dnn.blobFromImage(aligned, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
            found.append(session.run(None, {session.get_inputs()[0].name: blob})[0].reshape(-1))
            look = self._gender_age(full, (x1, y1, x2, y2))
            if look is not None:
                looks.append(look)
        if not found:
            raise StudioError("None of this person's photos could be read to learn their face")
        stack = np.stack(found)
        norms = np.linalg.norm(stack, axis=1, keepdims=True)
        mean = (stack / norms).mean(axis=0)
        # The model was trained on raw embeddings, so give the average a typical length.
        return (mean / np.linalg.norm(mean) * float(norms.mean())).astype(np.float32), self._in_words(looks)

    @staticmethod
    def _face_placement(width: int, height: int) -> np.ndarray:
        """Where the face goes: upper middle, looking at the camera."""
        scale = IDENTITY_EYE_DISTANCE * width / 35.2  # 35.2: eye distance in the 112 px template
        centre = np.array([width / 2, height * 0.3], dtype=np.float32)
        return (face_engines._ARCFACE_DST - np.array([56.0, 72.0], dtype=np.float32)) * scale + centre

    def _identity_pipe(self):
        """The InstantID pipeline, with the likeness adapter on the image model."""
        from .vendor.instantid_pipeline import StableDiffusionXLInstantIDPipeline

        pipe = StableDiffusionXLInstantIDPipeline(controlnet=self._part("identity_net"), **self._components())
        pipe.set_progress_bar_config(disable=True)
        unet = self._part("unet")
        if self._ip_processors is None:
            pipe.load_ip_adapter_instantid(str(self._dir("instantid") / "ip-adapter.bin"), scale=IDENTITY_FACE_SCALE)
            self._ip_processors = dict(unet.attn_processors)
            self._parts["image_proj"] = pipe.image_proj_model
        else:
            unet.set_attn_processor(dict(self._ip_processors))
            pipe.image_proj_model = self._parts["image_proj"]
            pipe.image_proj_model_in_features = 512
            pipe.set_ip_adapter_scale(IDENTITY_FACE_SCALE)
        return pipe

    def _person(
        self, prompt: str, negative: str, width: int, height: int, seed: int, style: Style | None,
        references: list[identity.Reference], progress: Progress, steps: int = DEFAULT_STEPS,
        extras: tuple[tuple[str, float], ...] = (),
    ) -> Image.Image:
        """A picture of a known person.

        The scene is painted from the words alone, so the pose and composition are
        whatever suits it. Then the main face is painted again, enlarged, as the person.
        """
        from diffusers import StableDiffusionXLPipeline

        from .vendor.instantid_pipeline import draw_kps

        embedding, who = self._identity_embedding(references, progress.part(0.0, 0.05))
        # A scene with the person seen from behind leaves no face to make theirs.
        prompt = prompt.replace(identity.PERSON, who, 1) + ", face clearly visible"
        negative = ", ".join(x for x in (IDENTITY_NEGATIVE, negative) if x)
        progress.report(0.05, "Loading the image model")
        self._plain_attention()
        pipe = StableDiffusionXLPipeline(**self._components())
        pipe.set_progress_bar_config(disable=True)
        undo = self._with_style(pipe, style, extras)
        try:
            found = []
            for attempt in range(2):
                # If the first composition shows no face, the second is asked for one outright.
                asked = prompt if attempt == 0 else f"{prompt}, seen from the front, facing the camera"
                embeds = self._encode(pipe, asked, negative)
                self._on_gpu("unet", "vae")
                scene = self._run(
                    pipe, progress.part(0.05 if attempt == 0 else 0.3, 0.5), seed + attempt * 7919, steps=steps,
                    width=width, height=height, guidance_scale=GUIDANCE, **self._sizes(width, height), **embeds,
                )
                found = self._landmarks(np.ascontiguousarray(np.asarray(scene)[:, :, ::-1]))
                if found:
                    break
            if not found:
                # Nobody to give the face to: fall back to placing the face ourselves.
                return self._posed_person(embedding, embeds, width, height, seed, progress.part(0.5, 1.0), steps)
            main = found[0]  # the largest face is the person asked for
            scene = self._detail_faces(scene, embeds, seed, progress.part(0.5, 0.6), steps, skip=main)

            x, y, side, size = self._face_square(main, width, height)
            rgb = np.asarray(scene)
            full = (IDENTITY_SIDE, IDENTITY_SIDE)
            face = Image.fromarray(rgb[y : y + side, x : x + side]).resize(full, Image.Resampling.LANCZOS)
            pose = draw_kps(Image.new("RGB", full), (main - np.array([x, y], dtype=np.float32)) * IDENTITY_SIDE / side)
            likeness = self._identity_pipe()
            self._on_gpu("unet", "vae", "identity_net", "image_proj")
            fixed = self._run(
                likeness, progress.part(0.6, 1.0), seed, steps=steps, label="Painting the face", image=pose,
                init_image=face, strength=IDENTITY_STRENGTH, image_embeds=torch.from_numpy(embedding)[None],
                width=IDENTITY_SIDE, height=IDENTITY_SIDE, controlnet_conditioning_scale=IDENTITY_POSE_SCALE,
                guidance_scale=IDENTITY_GUIDANCE, **self._sizes(*full), **embeds,
            )
            out = rgb.astype(np.float32)
            cx, cy = main.mean(axis=0)
            self._set_back(out, fixed, x, y, side, head=(cx - x, cy - y, size))
            return Image.fromarray(out.round().clip(0, 255).astype(np.uint8))
        finally:
            undo()
            self._plain_attention()

    def _posed_person(
        self, embedding: np.ndarray, embeds: dict, width: int, height: int, seed: int, progress: Progress,
        steps: int,
    ) -> Image.Image:
        """The whole picture painted around a face placed upper middle, looking at the camera."""
        from .vendor.instantid_pipeline import draw_kps

        # The likeness model learnt from stock photos and stamps their watermark on pictures
        # of exactly the usual sizes. A size slightly off them avoids it.
        width, height = width - 8, height - 8
        pipe = self._identity_pipe()
        pose = draw_kps(Image.new("RGB", (width, height)), self._face_placement(width, height))
        self._on_gpu("unet", "vae", "identity_net", "image_proj")
        return self._run(
            pipe, progress, seed, steps=steps, image=pose, image_embeds=torch.from_numpy(embedding)[None],
            width=width, height=height, controlnet_conditioning_scale=IDENTITY_POSE_SCALE,
            guidance_scale=IDENTITY_GUIDANCE, **embeds,
        )
