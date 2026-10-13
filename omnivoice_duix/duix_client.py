"""Cliente del servicio de video de Duix.Avatar (``guiji2025/duix.avatar``, puerto 8383).

Contrato tomado de Duix.Avatar (src/main/api/f2f.js y src/main/service/video.js):
``audio_url`` y ``video_url`` son nombres de archivo relativos a
``~/duix_avatar_data/face2face/temp`` y el resultado también se deja ahí.
"""

from __future__ import annotations

import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import requests

from . import config

CODE_OK = 10000
CODES_FAILED = {9999, 10002, 10003}
STATUS_RUNNING, STATUS_DONE, STATUS_FAILED = 1, 2, 3


class DuixError(RuntimeError):
    pass


@dataclass
class DuixVideoClient:
    base_url: str = config.DUIX_VIDEO_URL
    temp_dir: Path = config.FACE2FACE_TEMP
    timeout: float = 30.0

    def stage(self, src: Path) -> str:
        """Copia ``src`` a la carpeta temp de Duix si hace falta y devuelve su nombre relativo."""
        src = Path(src).resolve()
        temp = self.temp_dir.resolve()
        if src.is_relative_to(temp):
            return src.relative_to(temp).as_posix()
        temp.mkdir(parents=True, exist_ok=True)
        name = f"{uuid.uuid4().hex}{src.suffix}"
        shutil.copyfile(src, temp / name)
        return name

    def submit(self, audio_name: str, video_name: str, code: Optional[str] = None) -> str:
        code = code or str(uuid.uuid4())
        payload = {
            "audio_url": audio_name,
            "video_url": video_name,
            "code": code,
            "chaofen": 0,
            "watermark_switch": 0,
            "pn": 1,
        }
        res = requests.post(f"{self.base_url}/submit", json=payload, timeout=self.timeout).json()
        if res.get("code") != CODE_OK:
            raise DuixError(f"Duix rechazó la tarea: {res.get('msg', res)}")
        return code

    def query(self, code: str) -> dict:
        return requests.get(f"{self.base_url}/query", params={"code": code}, timeout=self.timeout).json()

    def wait(
        self,
        code: str,
        poll_seconds: float = 2.0,
        timeout: Optional[float] = None,
        on_progress: Optional[Callable[[dict], None]] = None,
    ) -> Path:
        """Espera a que termine la tarea y devuelve la ruta del video generado."""
        start = time.monotonic()
        while True:
            res = self.query(code)
            if res.get("code") in CODES_FAILED:
                raise DuixError(res.get("msg", "fallo en Duix"))
            data = res.get("data") or {}
            if res.get("code") == CODE_OK:
                if on_progress:
                    on_progress(data)
                if data.get("status") == STATUS_DONE:
                    return self.temp_dir / data["result"].lstrip("/\\")
                if data.get("status") == STATUS_FAILED:
                    raise DuixError(data.get("msg", "fallo en Duix"))
            if timeout is not None and time.monotonic() - start > timeout:
                raise DuixError(f"tiempo de espera agotado para la tarea {code}")
            time.sleep(poll_seconds)
