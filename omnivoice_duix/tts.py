"""Envoltorio de OmniVoice: carga perezosa del modelo y caché de voces clonadas."""

from __future__ import annotations

import hashlib
import io
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from . import config

PROMPT_SUFFIX = ".omnivoice.pt"


class OmniVoiceEngine:
    """Genera audio con OmniVoice. El modelo se carga en la primera petición."""

    def __init__(
        self,
        model_name: str = config.OMNIVOICE_MODEL,
        device: Optional[str] = config.OMNIVOICE_DEVICE,
        language: Optional[str] = config.OMNIVOICE_LANGUAGE,
        num_step: int = config.OMNIVOICE_NUM_STEP,
    ):
        self.model_name = model_name
        self.device = device
        self.language = language
        self.num_step = num_step
        self._model = None
        self._prompts: dict[str, object] = {}
        # OmniVoice no es seguro para llamadas concurrentes sobre la misma GPU.
        self._lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            import torch
            from omnivoice import OmniVoice

            device = self.device
            if device is None:
                if torch.cuda.is_available():
                    device = "cuda:0"
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            dtype = torch.float32 if device == "cpu" else torch.float16
            self._model = OmniVoice.from_pretrained(self.model_name, device_map=device, dtype=dtype)
        return self._model

    @property
    def sampling_rate(self) -> int:
        return self.model.sampling_rate

    def create_voice(self, ref_audio: Path, ref_text: Optional[str], out_dir: Path) -> tuple[Path, str]:
        """Codifica una voz de referencia y la guarda como prompt reutilizable.

        Si ``ref_text`` es None, OmniVoice transcribe el audio con Whisper.
        Devuelve la ruta del prompt guardado y la transcripción usada.
        """
        digest = hashlib.sha1(Path(ref_audio).read_bytes()).hexdigest()[:16]
        out_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = out_dir / f"{digest}{PROMPT_SUFFIX}"
        with self._lock:
            prompt = self.model.create_voice_clone_prompt(ref_audio=str(ref_audio), ref_text=ref_text)
            prompt.save(str(prompt_path))
        self._prompts[str(prompt_path)] = prompt
        return prompt_path, prompt.ref_text

    def _load_prompt(self, prompt_path: Path):
        key = str(prompt_path)
        if key not in self._prompts:
            from omnivoice import VoiceClonePrompt

            self._prompts[key] = VoiceClonePrompt.load(key)
        return self._prompts[key]

    def synthesize(
        self,
        text: str,
        ref_audio: Optional[Path] = None,
        ref_text: Optional[str] = None,
        instruct: Optional[str] = None,
        language: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> np.ndarray:
        """Sintetiza ``text``.

        - ``ref_audio`` terminado en ``.omnivoice.pt``: usa la voz guardada.
        - ``ref_audio`` de audio: clonación de voz (``ref_text`` opcional).
        - ``instruct``: diseño de voz (p. ej. "female, low pitch").
        - Ninguno: OmniVoice elige una voz.
        """
        kwargs: dict = {"text": text, "language": language or self.language, "num_step": self.num_step}
        if speed:
            kwargs["speed"] = speed
        if ref_audio is not None:
            if str(ref_audio).endswith(PROMPT_SUFFIX):
                kwargs["voice_clone_prompt"] = self._load_prompt(Path(ref_audio))
            else:
                kwargs["ref_audio"] = str(ref_audio)
                kwargs["ref_text"] = ref_text or None
        elif instruct:
            kwargs["instruct"] = instruct
        with self._lock:
            return self.model.generate(**kwargs)[0]

    def to_wav_bytes(self, audio: np.ndarray) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, audio, self.sampling_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()
