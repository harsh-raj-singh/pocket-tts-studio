"""
Pocket TTS Studio — FastAPI Server
===================================
A production-grade web interface for local text-to-speech generation.
Supports direct text input, URL extraction, multiple voices, and
automatic chunking for long-form content.
"""

import time
import uuid
import logging
from pathlib import Path
from typing import Optional

import torch
import scipy.io.wavfile
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from pocket_tts import TTSModel
from utils import VOICES, split_into_sentences, extract_text_from_url

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pocket-tts-server")

# ---------------------------------------------------------------------------
# Model loading (singleton at startup)
# ---------------------------------------------------------------------------
log.info("Loading Pocket TTS model...")
tts_model: TTSModel = TTSModel.load_model()
log.info("Model loaded")

_voice_cache: dict[str, dict] = {}


def get_voice_state(voice_name: str) -> dict:
    if voice_name not in _voice_cache:
        log.info(f"Loading voice: {voice_name}")
        _voice_cache[voice_name] = tts_model.get_state_for_audio_prompt(voice_name)
    return _voice_cache[voice_name]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Pocket TTS Studio")

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


class GenerateRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None
    voice: str = "alba"


class GenerateResponse(BaseModel):
    filename: str
    duration_seconds: float
    generation_time_seconds: float
    num_chunks: int
    char_count: int
    download_url: str


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/voices")
async def list_voices():
    return {"voices": VOICES}


@app.post("/generate", response_model=GenerateResponse)
async def generate_audio(req: GenerateRequest):
    if req.url:
        try:
            text = extract_text_from_url(req.url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to extract text from URL: {e}")
    elif req.text:
        text = req.text
    else:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'url'.")

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text content found.")

    voice = req.voice.lower()
    if voice not in VOICES:
        raise HTTPException(status_code=400, detail=f"Unknown voice '{voice}'. Choose from: {VOICES}")

    voice_state = get_voice_state(voice)
    chunks = split_into_sentences(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Text resulted in 0 chunks after splitting.")

    log.info(f"Generating: {len(chunks)} chunk(s), {len(text)} chars, voice='{voice}'")

    t0 = time.perf_counter()
    audio_parts: list[torch.Tensor] = []
    for i, chunk in enumerate(chunks):
        log.info(f"  Chunk {i + 1}/{len(chunks)}: {len(chunk)} chars")
        audio_parts.append(tts_model.generate_audio(voice_state, chunk))

    full_audio = torch.cat(audio_parts, dim=0)
    generation_time = time.perf_counter() - t0
    duration = full_audio.shape[0] / tts_model.sample_rate

    filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
    output_path = OUTPUT_DIR / filename
    scipy.io.wavfile.write(str(output_path), tts_model.sample_rate, full_audio.numpy())

    log.info(f"Done: {duration:.1f}s audio in {generation_time:.1f}s ({duration / generation_time:.1f}x real-time)")

    return GenerateResponse(
        filename=filename,
        duration_seconds=round(duration, 2),
        generation_time_seconds=round(generation_time, 2),
        num_chunks=len(chunks),
        char_count=len(text),
        download_url=f"/download/{filename}",
    )


@app.get("/download/{filename}")
async def download(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(str(path), media_type="audio/wav", filename=filename)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
