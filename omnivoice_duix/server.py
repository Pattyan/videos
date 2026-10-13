"""Servidor TTS compatible con Duix.Avatar, impulsado por OmniVoice.

Sustituye al servicio ``guiji2025/fish-speech-ziming`` (puerto 18180) implementando
los dos endpoints que llama el cliente de Duix.Avatar (src/main/api/tts.js):

- ``POST /v1/preprocess_and_tran``: "entrena" una voz a partir de un audio.
- ``POST /v1/invoke``: sintetiza texto con esa voz y devuelve un WAV.

Además expone ``POST /v1/omnivoice`` con las opciones propias de OmniVoice
(diseño de voz, idioma, velocidad).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from . import config
from .tts import OmniVoiceEngine

log = logging.getLogger("omnivoice_duix")

# Prefijo con el que el contenedor original de Duix ve la carpeta de voz.
CONTAINER_PREFIX = "/code/data/"
PROMPTS_DIR = "omnivoice_voices"


class PreprocessRequest(BaseModel):
    reference_audio: str
    format: Optional[str] = None
    lang: Optional[str] = None
    reference_text: Optional[str] = None


class InvokeRequest(BaseModel):
    # Duix envía además topP, temperature, etc.; OmniVoice no los usa.
    model_config = {"extra": "ignore"}

    text: str
    speaker: Optional[str] = None
    reference_audio: Optional[str] = None
    reference_text: Optional[str] = None
    format: str = "wav"


class OmniVoiceRequest(BaseModel):
    text: str
    reference_audio: Optional[str] = None
    reference_text: Optional[str] = None
    instruct: Optional[str] = None
    language: Optional[str] = None
    speed: Optional[float] = None


def resolve_data_path(data_root: Path, raw: str) -> Path:
    """Convierte la ruta que envía Duix en una ruta dentro de ``data_root``."""
    raw = raw.replace("\\", "/")
    if raw.startswith(CONTAINER_PREFIX):
        raw = raw[len(CONTAINER_PREFIX):]
    root = data_root.resolve()
    path = Path(raw)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"ruta fuera de la carpeta de datos: {raw}")
    if not path.is_file():
        raise FileNotFoundError(f"no existe el archivo: {raw}")
    return path


def create_app(engine: OmniVoiceEngine, data_root: Path = config.VOICE_DATA) -> FastAPI:
    app = FastAPI(title="OmniVoice para Duix.Avatar")

    def wav_response(audio) -> Response:
        return Response(content=engine.to_wav_bytes(audio), media_type="audio/wav")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": engine.model_name, "data_root": str(data_root)}

    @app.post("/v1/preprocess_and_tran")
    def preprocess_and_tran(req: PreprocessRequest):
        try:
            audio_path = resolve_data_path(data_root, req.reference_audio)
            prompt_path, text = engine.create_voice(audio_path, req.reference_text, data_root / PROMPTS_DIR)
        except (ValueError, FileNotFoundError) as e:
            return {"code": 1, "msg": str(e)}
        except Exception as e:  # noqa: BLE001 - Duix solo entiende code/msg
            log.exception("fallo al crear la voz")
            return {"code": 1, "msg": f"error de OmniVoice: {e}"}
        return {
            "code": 0,
            "msg": "ok",
            # Duix guarda este valor y lo devuelve como reference_audio en /v1/invoke.
            "asr_format_audio_url": prompt_path.relative_to(data_root.resolve()).as_posix(),
            "reference_audio_text": text,
        }

    @app.post("/v1/invoke")
    def invoke(req: InvokeRequest):
        if req.format.lower() != "wav":
            return JSONResponse({"code": 1, "msg": "solo se admite format=wav"}, status_code=400)
        try:
            ref = resolve_data_path(data_root, req.reference_audio) if req.reference_audio else None
        except (ValueError, FileNotFoundError) as e:
            return JSONResponse({"code": 1, "msg": str(e)}, status_code=400)
        audio = engine.synthesize(req.text, ref_audio=ref, ref_text=req.reference_text)
        return wav_response(audio)

    @app.post("/v1/omnivoice")
    def omnivoice(req: OmniVoiceRequest):
        try:
            ref = resolve_data_path(data_root, req.reference_audio) if req.reference_audio else None
        except (ValueError, FileNotFoundError) as e:
            return JSONResponse({"code": 1, "msg": str(e)}, status_code=400)
        audio = engine.synthesize(
            req.text,
            ref_audio=ref,
            ref_text=req.reference_text,
            instruct=req.instruct,
            language=req.language,
            speed=req.speed,
        )
        return wav_response(audio)

    return app


def main(argv: Optional[list[str]] = None) -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18180, help="18180 es el puerto que usa Duix.Avatar")
    parser.add_argument("--data-root", type=Path, default=config.VOICE_DATA)
    parser.add_argument("--preload", action="store_true", help="cargar el modelo al arrancar")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    args.data_root.mkdir(parents=True, exist_ok=True)
    engine = OmniVoiceEngine()
    if args.preload:
        _ = engine.model
    uvicorn.run(create_app(engine, args.data_root), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
