"""
FastAPI server exposing the Storifai improved model for inference.

Endpoints:
    GET  /         -> status + model metadata
    GET  /healthz  -> liveness/readiness for Cloud Run
    POST /generate -> takes 5 image uploads, returns 5-sentence story
"""

import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference import StoryGenerator, NUM_PHOTOS


generator = StoryGenerator()
startup_error: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global startup_error
    try:
        generator.load()
        print("[startup] model ready")
    except Exception as exc:
        startup_error = f"{type(exc).__name__}: {exc}"
        print(f"[startup] FAILED: {startup_error}")
    yield


app = FastAPI(
    title="Storifai Inference API",
    description="Visual storytelling from 5 photos using the improved CLIP + Cross-Attention + Transformer model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "storifai-inference",
        "model": "improved (CLIP ViT-B/32 + Cross-Attention + Transformer Decoder)",
        "ready": generator.model is not None,
        "startup_error": startup_error or None,
    }


@app.get("/healthz")
def healthz():
    if generator.model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "loading", "error": startup_error or None},
        )
    return {"status": "ok"}


@app.post("/generate")
async def generate(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    image3: UploadFile = File(...),
    image4: UploadFile = File(...),
    image5: UploadFile = File(...),
):
    if generator.model is None:
        raise HTTPException(status_code=503, detail=f"model not ready: {startup_error}")

    uploads = [image1, image2, image3, image4, image5]
    image_bytes_list: List[bytes] = []
    for i, upload in enumerate(uploads, start=1):
        if upload.content_type and not upload.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"image{i} has unsupported content_type: {upload.content_type}",
            )
        data = await upload.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"image{i} is empty")
        image_bytes_list.append(data)

    if len(image_bytes_list) != NUM_PHOTOS:
        raise HTTPException(status_code=400, detail=f"expected {NUM_PHOTOS} images")

    t0 = time.time()
    try:
        stories = generator.generate_stories(image_bytes_list, num_versions=3)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"inference failed: {exc}")
    elapsed = time.time() - t0

    return {
        "stories": stories,
        "model": "improved",
        "inference_seconds": round(elapsed, 2),
    }
