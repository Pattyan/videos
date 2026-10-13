# Servidor OmniVoice compatible con Duix.Avatar (sustituye a guiji2025/fish-speech-ziming).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY omnivoice_duix ./omnivoice_duix
RUN pip install --no-cache-dir ".[tts]"

ENV OMNIVOICE_DATA=/code/data \
    HF_HOME=/models
EXPOSE 8080
CMD ["omnivoice-duix-server", "--port", "8080", "--data-root", "/code/data", "--preload"]
