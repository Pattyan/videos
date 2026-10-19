"""Rutas y URLs compartidas, configurables por variables de entorno.

Siguen las convenciones de Duix.Avatar (src/main/config/config.js):
- El servicio TTS ve ``~/duix_avatar_data/voice/data`` montado en ``/code/data``.
- El servicio de video ve ``~/duix_avatar_data/face2face`` montado en ``/code/data``
  y recibe rutas de audio/video relativas a ``face2face/temp``.
"""

import os
from pathlib import Path

# En Windows, Duix.Avatar guarda sus datos en D:\duix_avatar_data.
_DEFAULT_DUIX_DATA = Path("D:/duix_avatar_data") if os.name == "nt" else Path.home() / "duix_avatar_data"
DUIX_DATA = Path(os.environ.get("DUIX_DATA", _DEFAULT_DUIX_DATA))

# Raíz de datos del servicio TTS (lo que Duix monta en /code/data del contenedor TTS).
VOICE_DATA = Path(os.environ.get("OMNIVOICE_DATA", DUIX_DATA / "voice" / "data"))

# Carpeta donde Duix espera los audios/videos a sintetizar y deja los resultados.
FACE2FACE_TEMP = Path(os.environ.get("DUIX_FACE2FACE_TEMP", DUIX_DATA / "face2face" / "temp"))

DUIX_VIDEO_URL = os.environ.get("DUIX_VIDEO_URL", "http://127.0.0.1:8383/easy")

OMNIVOICE_MODEL = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
OMNIVOICE_DEVICE = os.environ.get("OMNIVOICE_DEVICE")  # p. ej. "cuda:0", "mps", "cpu"
OMNIVOICE_LANGUAGE = os.environ.get("OMNIVOICE_LANGUAGE") or None  # p. ej. "es"
OMNIVOICE_NUM_STEP = int(os.environ.get("OMNIVOICE_NUM_STEP", "32"))
