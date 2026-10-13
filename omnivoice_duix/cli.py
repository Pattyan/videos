"""Texto → voz con OmniVoice → video con lip-sync en Duix.Avatar.

Ejemplo:
    omnivoice-duix-video --text "Hola a todos" --avatar presentadora.mp4 \\
        --voice muestra_voz.wav --language es -o saludo.mp4
"""

from __future__ import annotations

import argparse
import shutil
import uuid
from pathlib import Path
from typing import Optional

import soundfile as sf

from . import config
from .duix_client import DuixVideoClient
from .tts import OmniVoiceEngine


def make_video(
    text: str,
    avatar: Path,
    output: Path,
    engine: OmniVoiceEngine,
    client: DuixVideoClient,
    voice: Optional[Path] = None,
    voice_text: Optional[str] = None,
    instruct: Optional[str] = None,
    language: Optional[str] = None,
    speed: Optional[float] = None,
    timeout: Optional[float] = None,
) -> Path:
    audio = engine.synthesize(
        text, ref_audio=voice, ref_text=voice_text, instruct=instruct, language=language, speed=speed
    )
    client.temp_dir.mkdir(parents=True, exist_ok=True)
    audio_name = f"omnivoice_{uuid.uuid4().hex}.wav"
    sf.write(client.temp_dir / audio_name, audio, engine.sampling_rate)
    print(f"Audio generado: {client.temp_dir / audio_name}")

    code = client.submit(audio_name, client.stage(avatar))
    print(f"Tarea enviada a Duix.Avatar: {code}")

    def progress(data: dict) -> None:
        if data.get("progress") is not None:
            print(f"  progreso: {data['progress']}% {data.get('msg', '')}".rstrip())

    result = client.wait(code, timeout=timeout, on_progress=progress)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(result, output)
    return output


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Genera un video de avatar Duix con voz de OmniVoice.")
    text = p.add_mutually_exclusive_group(required=True)
    text.add_argument("--text", help="texto a decir")
    text.add_argument("--text-file", type=Path, help="archivo con el texto")
    p.add_argument("--avatar", type=Path, required=True, help="video del avatar (persona hablando a cámara)")
    p.add_argument("-o", "--output", type=Path, default=Path("output/video.mp4"))
    voice = p.add_mutually_exclusive_group()
    voice.add_argument("--voice", type=Path, help="audio de 3-10 s para clonar la voz (o un .omnivoice.pt)")
    voice.add_argument("--instruct", help='diseño de voz, p. ej. "female, low pitch"')
    p.add_argument("--voice-text", help="transcripción del audio de --voice (si no, usa Whisper)")
    p.add_argument("--language", default=config.OMNIVOICE_LANGUAGE, help="idioma, p. ej. es, en")
    p.add_argument("--speed", type=float, help="velocidad (>1 más rápido)")
    p.add_argument("--duix-url", default=config.DUIX_VIDEO_URL)
    p.add_argument("--duix-temp", type=Path, default=config.FACE2FACE_TEMP)
    p.add_argument("--timeout", type=float, help="segundos máximos de espera")
    args = p.parse_args(argv)

    out = make_video(
        text=args.text if args.text else args.text_file.read_text(encoding="utf-8"),
        avatar=args.avatar,
        output=args.output,
        engine=OmniVoiceEngine(),
        client=DuixVideoClient(base_url=args.duix_url, temp_dir=args.duix_temp),
        voice=args.voice,
        voice_text=args.voice_text,
        instruct=args.instruct,
        language=args.language,
        speed=args.speed,
        timeout=args.timeout,
    )
    print(f"Video listo: {out}")


if __name__ == "__main__":
    main()
