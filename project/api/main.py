from __future__ import annotations

import asyncio
import os
import shutil
import sys
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.expression_service import (  # noqa: E402
    DEFAULT_EMOTIONS,
    EMOTION_PROMPTS,
    ExpressionSheetGenerator,
    GenerationConfig,
)


OUTPUT_ROOT = Path(os.environ.get("OC_API_OUTPUT_ROOT", PROJECT_ROOT / "results" / "api_jobs"))

app = FastAPI(title="OC Expression Sheet API")
generation_lock = asyncio.Lock()


@app.on_event("startup")
def startup() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    app.state.generator = ExpressionSheetGenerator(
        GenerationConfig(
            checkpoint=os.environ.get("OC_CHECKPOINT", str(PROJECT_ROOT / "checkpoints" / "p4_on_p1_3epochs")),
            sd_path=os.environ.get("OC_SD_PATH", str(PROJECT_ROOT / "models" / "sd-v1-5")),
            ip_repo=os.environ.get("OC_IP_REPO", str(PROJECT_ROOT / "models" / "ip-adapter")),
            scale=float(os.environ.get("OC_IP_SCALE", "0.7")),
            steps=int(os.environ.get("OC_STEPS", "30")),
            guidance=float(os.environ.get("OC_GUIDANCE", "7.5")),
            size=int(os.environ.get("OC_IMAGE_SIZE", "512")),
        )
    )


@app.get("/api/health")
def health() -> dict:
    generator = getattr(app.state, "generator", None)
    return {
        "status": "ok",
        "model_loaded": generator is not None,
        "device": generator.device if generator else None,
        "emotions": list(EMOTION_PROMPTS),
    }


@app.post("/api/generate")
async def generate(
    reference: UploadFile = File(...),
    emotions: str = Form(",".join(DEFAULT_EMOTIONS)),
    seed: int | None = Form(None),
) -> dict:
    parsed_emotions = parse_emotions(emotions)
    job_id = uuid.uuid4().hex
    job_dir = OUTPUT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=False)

    reference_path = await save_upload(reference)
    try:
        async with generation_lock:
            result = await asyncio.to_thread(
                app.state.generator.generate,
                reference_path,
                job_dir,
                parsed_emotions,
                seed,
            )
    except ValueError as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc
    finally:
        reference_path.unlink(missing_ok=True)

    return {
        "job_id": job_id,
        "status": "completed",
        "sheet_url": f"/api/jobs/{job_id}/sheet.jpg",
        "metadata_url": f"/api/jobs/{job_id}/metadata.json",
        "images": {
            emotion: f"/api/jobs/{job_id}/{path.name}"
            for emotion, path in result.image_paths.items()
        },
    }


@app.get("/api/jobs/{job_id}/{filename}")
def get_job_file(job_id: str, filename: str):
    if not is_safe_name(job_id) or not is_safe_name(filename):
        raise HTTPException(status_code=400, detail="Invalid path.")

    path = OUTPUT_ROOT / job_id / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(path)


def parse_emotions(value: str) -> list[str]:
    emotions = [item.strip().lower() for item in value.split(",") if item.strip()]
    return emotions or list(DEFAULT_EMOTIONS)


async def save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"

    with NamedTemporaryFile(delete=False, suffix=suffix, dir=OUTPUT_ROOT) as tmp:
        while chunk := await upload.read(1024 * 1024):
            tmp.write(chunk)
        return Path(tmp.name)


def is_safe_name(value: str) -> bool:
    return bool(value) and "/" not in value and "\\" not in value and value not in {".", ".."}
