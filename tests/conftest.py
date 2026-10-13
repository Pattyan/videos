from pathlib import Path

import numpy as np
import pytest


class FakeEngine:
    """Sustituye a OmniVoiceEngine sin cargar el modelo."""

    model_name = "fake"
    sampling_rate = 24000

    def __init__(self):
        self.calls = []

    def create_voice(self, ref_audio, ref_text, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "abc.omnivoice.pt"
        path.write_bytes(b"prompt")
        return path, ref_text or "texto transcrito"

    def synthesize(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return np.zeros(2400, dtype=np.float32)

    def to_wav_bytes(self, audio):
        import io

        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, audio, self.sampling_rate, format="WAV")
        return buf.getvalue()


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def data_root(tmp_path) -> Path:
    root = tmp_path / "voice" / "data"
    (root / "origin_audio").mkdir(parents=True)
    (root / "origin_audio" / "ref.wav").write_bytes(b"RIFF")
    return root
