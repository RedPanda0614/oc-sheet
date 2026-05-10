from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw


DEFAULT_EMOTIONS = ("neutral", "happy", "sad", "angry", "surprised", "crying")

EMOTION_PROMPTS = {
    "neutral": (
        "anime portrait, neutral calm face, "
        "half-lidded eyes, relaxed eyebrows, lips closed, white background"
    ),
    "happy": (
        "anime portrait, laughing happily, "
        "eyes curved into crescents upward, wide open smile showing teeth, rosy cheeks, white background"
    ),
    "sad": (
        "anime portrait, sad grieving face, "
        "eyes drooping downward at corners, eyebrows slanted inward, pursed trembling lips, white background"
    ),
    "angry": (
        "anime portrait, furious rage expression, "
        "eyes narrowed to slits, thick eyebrows sharply angled down toward nose, jaw clenched, white background"
    ),
    "surprised": (
        "anime portrait, utterly shocked face, "
        "eyes perfectly round and wide open, eyebrows raised as high as possible, mouth dropped open, white background"
    ),
    "crying": (
        "anime portrait, weeping face, "
        "eyes tightly shut with tears flowing down cheeks, large teardrops, eyebrows raised and scrunched, white background"
    ),
}

EMOTION_SEEDS = {
    "neutral": 42,
    "happy": 123,
    "sad": 456,
    "angry": 789,
    "surprised": 1024,
    "crying": 2048,
}

EMOTION_COLORS = {
    "neutral": (160, 160, 160),
    "happy": (80, 200, 80),
    "sad": (80, 120, 220),
    "angry": (220, 60, 60),
    "surprised": (220, 180, 40),
    "crying": (100, 160, 240),
}

NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, worst quality, blurry, deformed, extra fingers"


@dataclass(frozen=True)
class GenerationConfig:
    checkpoint: str | None = "checkpoints/p4_on_p1_3epochs"
    sd_path: str = "models/sd-v1-5"
    ip_repo: str = "models/ip-adapter"
    ip_weight: str = "ip-adapter-plus_sd15.bin"
    scale: float = 0.7
    steps: int = 30
    guidance: float = 7.5
    size: int = 512


@dataclass(frozen=True)
class GenerationResult:
    output_dir: Path
    reference_path: Path
    sheet_path: Path
    image_paths: dict[str, Path]
    metadata_path: Path


class ExpressionSheetGenerator:
    def __init__(self, config: GenerationConfig | None = None) -> None:
        self.config = config or GenerationConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = self._load_pipeline()

    def generate(
        self,
        reference_path: str | Path,
        output_dir: str | Path,
        emotions: list[str] | tuple[str, ...] = DEFAULT_EMOTIONS,
        seed: int | None = None,
    ) -> GenerationResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        requested_emotions = self._validate_emotions(emotions)
        reference_image = Image.open(reference_path).convert("RGB")
        saved_reference_path = output_dir / "reference.jpg"
        reference_image.save(saved_reference_path, quality=95)

        generated: dict[str, Image.Image] = {}
        image_paths: dict[str, Path] = {}

        for emotion in requested_emotions:
            generator = torch.Generator(device=self.device).manual_seed(
                self._seed_for_emotion(emotion=emotion, seed=seed)
            )
            image = self.pipe(
                prompt=EMOTION_PROMPTS[emotion],
                negative_prompt=NEGATIVE_PROMPT,
                ip_adapter_image=[reference_image],
                num_inference_steps=self.config.steps,
                guidance_scale=self.config.guidance,
                generator=generator,
                height=self.config.size,
                width=self.config.size,
            ).images[0]

            generated[emotion] = image
            image_path = output_dir / f"{emotion}.jpg"
            image.save(image_path, quality=95)
            image_paths[emotion] = image_path

        sheet = make_sheet(generated, reference_image, size=self.config.size)
        sheet_path = output_dir / "sheet.jpg"
        sheet.save(sheet_path, quality=95)

        metadata_path = output_dir / "metadata.json"
        metadata = {
            "emotions": requested_emotions,
            "seed": seed,
            "device": self.device,
            "checkpoint": self.config.checkpoint,
            "sd_path": self.config.sd_path,
            "ip_repo": self.config.ip_repo,
            "scale": self.config.scale,
            "steps": self.config.steps,
            "guidance": self.config.guidance,
            "size": self.config.size,
            "reference_path": str(saved_reference_path.resolve()),
            "sheet_path": str(sheet_path.resolve()),
            "image_paths": {
                emotion: str(path.resolve())
                for emotion, path in image_paths.items()
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return GenerationResult(
            output_dir=output_dir,
            reference_path=saved_reference_path,
            sheet_path=sheet_path,
            image_paths=image_paths,
            metadata_path=metadata_path,
        )

    def _load_pipeline(self):
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            self.config.sd_path,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(self.device)

        pipe.load_ip_adapter(
            self.config.ip_repo,
            subfolder="models",
            weight_name=self.config.ip_weight,
        )
        pipe.set_ip_adapter_scale(self.config.scale)

        if self.config.checkpoint:
            self._load_finetuned_weights(pipe)

        return pipe

    def _load_finetuned_weights(self, pipe) -> None:
        checkpoint = Path(self.config.checkpoint or "")
        image_proj_path = checkpoint / "image_proj_model.pt"
        ip_attn_path = checkpoint / "ip_attn_procs.pt"
        missing = [path for path in (image_proj_path, ip_attn_path) if not path.exists()]
        if missing:
            missing_text = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Missing checkpoint file(s): {missing_text}")

        target_dtype = torch.float16 if self.device == "cuda" else torch.float32
        pipe.unet.encoder_hid_proj.load_state_dict(
            torch.load(image_proj_path, map_location=self.device),
            strict=True,
        )

        attn_state = torch.load(ip_attn_path, map_location="cpu")
        matched = 0
        for name, proc_state in attn_state.items():
            if name not in pipe.unet.attn_processors:
                continue
            pipe.unet.attn_processors[name].load_state_dict(
                {
                    key: value.to(device=self.device, dtype=target_dtype)
                    for key, value in proc_state.items()
                }
            )
            matched += 1

        if matched == 0:
            raise RuntimeError("No IP-Adapter attention processor weights matched the pipeline.")

    def _seed_for_emotion(self, emotion: str, seed: int | None) -> int:
        if seed is None:
            return EMOTION_SEEDS[emotion]
        emotion_offset = DEFAULT_EMOTIONS.index(emotion) if emotion in DEFAULT_EMOTIONS else 0
        return seed + emotion_offset * 1000

    def _validate_emotions(self, emotions: list[str] | tuple[str, ...]) -> list[str]:
        cleaned = [emotion.strip().lower() for emotion in emotions if emotion.strip()]
        if not cleaned:
            raise ValueError("At least one emotion is required.")

        unknown = [emotion for emotion in cleaned if emotion not in EMOTION_PROMPTS]
        if unknown:
            valid = ", ".join(EMOTION_PROMPTS)
            raise ValueError(f"Unsupported emotion(s): {', '.join(unknown)}. Valid emotions: {valid}")

        return cleaned


def make_sheet(images: dict[str, Image.Image], ref_img: Image.Image, size: int = 512) -> Image.Image:
    emotions = list(images.keys())
    total_cells = len(emotions) + 1
    cols = min(4, total_cells)
    rows = (total_cells + cols - 1) // cols
    bar_h = 24
    pad = 6

    cell_w = size
    cell_h = size + bar_h
    canvas_w = cols * (cell_w + pad) + pad
    canvas_h = rows * (cell_h + pad) + pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    def paste_cell(img: Image.Image, label: str, color: tuple[int, int, int], idx: int) -> None:
        col = idx % cols
        row = idx // cols
        x = pad + col * (cell_w + pad)
        y = pad + row * (cell_h + pad)
        canvas.paste(img.resize((size, size), Image.LANCZOS), (x, y))
        draw.rectangle([x, y + size, x + cell_w, y + size + bar_h], fill=color)
        bbox = draw.textbbox((0, 0), label)
        text_w = bbox[2] - bbox[0]
        draw.text((x + (cell_w - text_w) // 2, y + size + 4), label, fill=(255, 255, 255))

    paste_cell(ref_img, "REFERENCE", (60, 60, 60), 0)
    for idx, emotion in enumerate(emotions, start=1):
        paste_cell(images[emotion], emotion, EMOTION_COLORS[emotion], idx)

    return canvas
