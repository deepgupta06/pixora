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
from .studio import Progress, StudioError, Style

log = logging.getLogger("pixora.studio")

STEPS = 28
GUIDANCE = 6.0
# How strongly a restyled photo follows the outlines of the original.
CANNY_SCALE = 0.6
# The outlines guide only the first part of the painting, which fixes the composition
# and then leaves the style free to simplify detail.
CANNY_END = 0.5
# Fine photographic texture is smoothed away before outlines are taken.
CANNY_BLUR = 2.0
# InstantID: how firmly the face is placed, and how much it looks like the person.
IDENTITY_POSE_SCALE = 0.8
IDENTITY_FACE_SCALE = 0.7
# How large the face is drawn, as the distance between the eyes over the picture's width.
IDENTITY_EYE_DISTANCE = 0.085
# Lower than for ordinary pictures: with a face to follow, more than this burns the colours.
IDENTITY_GUIDANCE = 4.0

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
        self._ip_processors: dict | None = None

    # ---- loading -----------------------------------------------------------

    def _dir(self, model_id: str) -> Path:
        return model_store.model_dir(self.models_dir, model_id)

    def _part(self, name: str):
        """A model part, loaded into ordinary memory on first use."""
        if name in self._parts:
            return self._parts[name]
        from diffusers import AutoencoderKL, ControlNetModel, EulerDiscreteScheduler, UNet2DConditionModel
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
            part = EulerDiscreteScheduler.from_pretrained(base / "scheduler")
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
        self._matte = self._identity = self._faces = None
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

    def _run(self, pipe, progress: Progress, seed: int, steps: int = STEPS, **kwargs) -> Image.Image:
        # Img2img and inpainting run only part of the schedule.
        total = max(1, round(steps * float(kwargs.get("strength", 1.0))))

        def on_step(_pipe, step, _timestep, tensors):
            progress.report(0.1 + 0.85 * (step + 1) / total)
            return tensors

        progress.report(0.1, "Painting")
        with torch.inference_mode():
            out = pipe(
                num_inference_steps=steps,
                generator=torch.Generator("cpu").manual_seed(seed),
                callback_on_step_end=on_step,
                **kwargs,
            )
        return out.images[0].convert("RGB")

    def _with_style(self, pipe, style: Style | None):
        """Load the look's LoRA for one run. Returns what to call afterwards."""
        if style is None or style.model is None:
            return lambda: None
        model = model_store.MODELS[style.model]
        pipe.load_lora_weights(str(self._dir(style.model)), weight_name=model.files[0], adapter_name="style")
        pipe.set_adapters(["style"], adapter_weights=[style.weight])
        return pipe.unload_lora_weights

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
        order = np.argsort([-(b[2] - b[0]) * (b[3] - b[1]) for b in boxes])
        return [(points[i] / scale).astype(np.float32) for i in order]

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
        self, image: Image.Image, mask: Image.Image, prompt: str, negative: str, seed: int, progress: Progress
    ) -> Image.Image:
        from diffusers import StableDiffusionXLInpaintPipeline

        progress.report(0.0, "Loading the painting model")
        pipe = StableDiffusionXLInpaintPipeline(**self._components("inpaint_unet"))
        pipe.set_progress_bar_config(disable=True)
        embeds = self._encode(pipe, prompt, negative)
        self._on_gpu("inpaint_unet", "vae")
        return self._run(
            pipe, progress, seed, image=image, mask_image=mask.convert("L"), width=image.width, height=image.height,
            strength=0.99, guidance_scale=7.5, **embeds,
        )

    def restyle(
        self, image: Image.Image, prompt: str, negative: str, style: Style, strength: float, seed: int,
        progress: Progress,
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
            return self._run(
                pipe, progress, seed, image=image, control_image=outline, width=image.width, height=image.height,
                strength=strength, controlnet_conditioning_scale=CANNY_SCALE, control_guidance_end=CANNY_END,
                guidance_scale=GUIDANCE, **embeds,
            )
        finally:
            undo()

    def generate(
        self, prompt: str, negative: str, width: int, height: int, seed: int, style: Style | None,
        references: list[identity.Reference], progress: Progress,
    ) -> Image.Image:
        if references:
            return self._generate_person(prompt, negative, width, height, seed, style, references, progress)
        from diffusers import StableDiffusionXLPipeline

        progress.report(0.0, "Loading the image model")
        self._plain_attention()
        pipe = StableDiffusionXLPipeline(**self._components())
        pipe.set_progress_bar_config(disable=True)
        undo = self._with_style(pipe, style)
        try:
            embeds = self._encode(pipe, prompt, negative)
            self._on_gpu("unet", "vae")
            return self._run(pipe, progress, seed, width=width, height=height, guidance_scale=GUIDANCE, **embeds)
        finally:
            undo()

    # ---- a known person ----------------------------------------------------

    def _identity_embedding(self, references: list[identity.Reference], progress: Progress) -> np.ndarray:
        """What InstantID needs to know about a face, averaged over the reference photos."""
        engine = self._face_engine()
        if engine is None:
            raise StudioError("The face models are not installed")
        if self._identity is None:
            self._identity = ort.InferenceSession(
                str(self._dir("antelopev2") / "glintr100.onnx"), providers=_CPU
            )
        session, found = self._identity, []
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
        if not found:
            raise StudioError("None of this person's photos could be read to learn their face")
        stack = np.stack(found)
        norms = np.linalg.norm(stack, axis=1, keepdims=True)
        mean = (stack / norms).mean(axis=0)
        # The model was trained on raw embeddings, so give the average a typical length.
        return (mean / np.linalg.norm(mean) * float(norms.mean())).astype(np.float32)

    @staticmethod
    def _face_placement(width: int, height: int) -> np.ndarray:
        """Where the face goes: upper middle, looking at the camera."""
        scale = IDENTITY_EYE_DISTANCE * width / 35.2  # 35.2: eye distance in the 112 px template
        centre = np.array([width / 2, height * 0.3], dtype=np.float32)
        return (face_engines._ARCFACE_DST - np.array([56.0, 72.0], dtype=np.float32)) * scale + centre

    def _generate_person(
        self, prompt: str, negative: str, width: int, height: int, seed: int, style: Style | None,
        references: list[identity.Reference], progress: Progress,
    ) -> Image.Image:
        from .vendor.instantid_pipeline import StableDiffusionXLInstantIDPipeline, draw_kps

        embedding = self._identity_embedding(references, progress.part(0.0, 0.08))
        progress.report(0.08, "Loading the image model")
        pipe = StableDiffusionXLInstantIDPipeline(controlnet=self._part("identity_net"), **self._components())
        pipe.set_progress_bar_config(disable=True)
        adapter = str(self._dir("instantid") / "ip-adapter.bin")
        unet = self._part("unet")
        if self._ip_processors is None:
            pipe.load_ip_adapter_instantid(adapter, scale=IDENTITY_FACE_SCALE)
            self._ip_processors = dict(unet.attn_processors)
            self._parts["image_proj"] = pipe.image_proj_model
        else:
            unet.set_attn_processor(dict(self._ip_processors))
            pipe.image_proj_model = self._parts["image_proj"]
            pipe.image_proj_model_in_features = 512
            pipe.set_ip_adapter_scale(IDENTITY_FACE_SCALE)
        undo = self._with_style(pipe, style)
        try:
            embeds = self._encode(pipe, prompt, negative)
            pose = draw_kps(Image.new("RGB", (width, height)), self._face_placement(width, height))
            self._on_gpu("unet", "vae", "identity_net", "image_proj")
            return self._run(
                pipe, progress.part(0.08, 1.0), seed, steps=30, image=pose,
                image_embeds=torch.from_numpy(embedding)[None], width=width, height=height,
                controlnet_conditioning_scale=IDENTITY_POSE_SCALE, guidance_scale=IDENTITY_GUIDANCE, **embeds,
            )
        finally:
            undo()
            self._plain_attention()
