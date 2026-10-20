# OmniVoice + Duix.Avatar

Conecta [OmniVoice](https://github.com/k2-fsa/OmniVoice) (TTS con clonación de voz, más de 600 idiomas) con [Duix.Avatar](https://github.com/duixcom/Duix-Avatar) (avatares con lip-sync, de duix.com). Escribes un texto, OmniVoice lo convierte en voz y Duix.Avatar genera el video del avatar diciéndolo.

```
texto ──► OmniVoice (voz clonada / diseñada) ──► WAV ──► Duix.Avatar :8383 ──► video.mp4
```

Hay dos formas de usarlo:

1. **Reemplazar el TTS de Duix.Avatar.** `omnivoice-duix-server` implementa la misma API que el servicio `fish-speech` de Duix (`/v1/preprocess_and_tran` y `/v1/invoke` en el puerto 18180). La app de escritorio de Duix.Avatar usa OmniVoice sin ningún cambio.
2. **Generar videos desde la línea de comandos.** `omnivoice-duix-video` sintetiza el audio con OmniVoice, lo envía a Duix.Avatar (`/easy/submit`), espera el resultado (`/easy/query`) y copia el video.

## Requisitos

- Duix.Avatar desplegado con Docker (sigue [su README](https://github.com/duixcom/Duix-Avatar#3-how-to-run-locally)); necesita una GPU NVIDIA.
- Python 3.10 o superior y PyTorch ([instrucciones de OmniVoice](https://github.com/k2-fsa/OmniVoice#installation)).

```bash
pip install -e ".[tts]"
```

## Opción 1: OmniVoice como voz de Duix.Avatar

Con Docker, sustituyendo el contenedor `duix-avatar-tts`:

```bash
docker build -t omnivoice-duix .
docker compose -f /ruta/Duix-Avatar/deploy/docker-compose-linux.yml \
               -f deploy/docker-compose.omnivoice.yml up -d
```

En Windows (PowerShell), con Duix.Avatar clonado en tu carpeta de usuario:

```powershell
docker build -t omnivoice-duix .
$env:OMNIVOICE_LANGUAGE = "es"
docker compose -f $HOME\Duix-Avatar\deploy\docker-compose.yml `
               -f deploy\docker-compose.omnivoice.windows.yml up -d
```

O sin Docker. Primero detén `duix-avatar-tts` para liberar el puerto 18180:

```bash
docker stop duix-avatar-tts
OMNIVOICE_LANGUAGE=es omnivoice-duix-server --port 18180 \
    --data-root ~/duix_avatar_data/voice/data --preload
```

Al crear un modelo en Duix.Avatar, el audio de su video se codifica como voz de OmniVoice y se guarda en `voice/data/omnivoice_voices/`. Whisper transcribe el audio de forma automática. Cada video que sintetices en la app usará esa voz.

Endpoint adicional con las opciones propias de OmniVoice:

```bash
curl -X POST localhost:18180/v1/omnivoice -H 'Content-Type: application/json' \
  -d '{"text": "Hola, soy tu avatar.", "instruct": "female, low pitch", "language": "es"}' \
  -o hola.wav
```

## Opción 2: video desde la terminal

```bash
# Clonando una voz (audio de referencia de 3 a 10 segundos)
omnivoice-duix-video --text "Bienvenidos al canal" --avatar presentadora.mp4 \
    --voice muestra.wav --language es -o output/bienvenida.mp4

# Con una voz diseñada y el texto en un archivo
omnivoice-duix-video --text-file guion.txt --avatar presentador.mp4 \
    --instruct "male, middle-aged, low pitch" --language es -o output/guion.mp4
```

`--voice` también acepta un `.omnivoice.pt` guardado por el servidor, así reutilizas la voz sin volver a codificarla.

## Solo voz, sin GPU

Si tu PC no tiene tarjeta NVIDIA, Duix.Avatar no funciona, pero OmniVoice sí, aunque más lento:

```bash
omnivoice-tts --text "Hola, esta es mi voz." --language es -o output/hola.wav
omnivoice-tts --text "Hola" --voice muestra.wav --voice-text "lo que dice la muestra" --language es -o output/clon.wav
omnivoice-tts --text "Hola" --instruct "female, low pitch" --language es --steps 16 -o output/disenada.wav
```

## Configuración

| Variable | Valor por defecto | Uso |
| --- | --- | --- |
| `DUIX_DATA` | `~/duix_avatar_data` | Carpeta de datos de Duix.Avatar (en Windows: `D:\duix_avatar_data`) |
| `DUIX_VIDEO_URL` | `http://127.0.0.1:8383/easy` | Servicio de video de Duix |
| `OMNIVOICE_MODEL` | `k2-fsa/OmniVoice` | Modelo de Hugging Face o ruta local |
| `OMNIVOICE_DEVICE` | automático | `cuda:0`, `mps` o `cpu` |
| `OMNIVOICE_LANGUAGE` | ninguno | Idioma por defecto, por ejemplo `es` |
| `OMNIVOICE_NUM_STEP` | `32` | Pasos de difusión (`16` es más rápido) |

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Los tests usan un motor falso y simulan el servicio de Duix, así que no hace falta GPU para correrlos.
