import pytest

from omnivoice_duix import duix_client
from omnivoice_duix.cli import make_video
from omnivoice_duix.duix_client import DuixError, DuixVideoClient


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


@pytest.fixture
def fake_duix(monkeypatch, tmp_path):
    temp = tmp_path / "face2face" / "temp"
    state = {"submitted": None, "queries": 0}

    def post(url, json, timeout):
        state["submitted"] = json
        return FakeResponse({"code": 10000, "msg": "ok"})

    def get(url, params, timeout):
        state["queries"] += 1
        if state["queries"] < 2:
            return FakeResponse({"code": 10000, "data": {"status": 1, "progress": 50}})
        (temp / "result.mp4").write_bytes(b"video")
        return FakeResponse({"code": 10000, "data": {"status": 2, "progress": 100, "result": "/result.mp4"}})

    monkeypatch.setattr(duix_client.requests, "post", post)
    monkeypatch.setattr(duix_client.requests, "get", get)
    monkeypatch.setattr(duix_client.time, "sleep", lambda s: None)
    return DuixVideoClient(base_url="http://duix/easy", temp_dir=temp), state


def test_make_video_extremo_a_extremo(fake_duix, engine, tmp_path):
    client, state = fake_duix
    avatar = tmp_path / "avatar.mp4"
    avatar.write_bytes(b"mp4")

    out = make_video("Hola", avatar, tmp_path / "out.mp4", engine, client, instruct="male", language="es")

    assert out.read_bytes() == b"video"
    sent = state["submitted"]
    assert (client.temp_dir / sent["audio_url"]).is_file()
    assert (client.temp_dir / sent["video_url"]).read_bytes() == b"mp4"
    assert sent["chaofen"] == 0 and sent["pn"] == 1


def test_stage_no_copia_si_ya_esta_en_temp(tmp_path):
    client = DuixVideoClient(temp_dir=tmp_path)
    f = tmp_path / "sub" / "a.mp4"
    f.parent.mkdir()
    f.write_bytes(b"x")
    assert client.stage(f) == "sub/a.mp4"


def test_error_de_duix(monkeypatch, tmp_path):
    monkeypatch.setattr(duix_client.requests, "get", lambda *a, **k: FakeResponse({"code": 10002, "msg": "sin rostro"}))
    with pytest.raises(DuixError, match="sin rostro"):
        DuixVideoClient(temp_dir=tmp_path).wait("c")


def test_tts_main_solo_voz(monkeypatch, engine, tmp_path):
    from omnivoice_duix import cli

    monkeypatch.setattr(cli, "OmniVoiceEngine", lambda **kw: engine)
    out = tmp_path / "voz.wav"
    cli.tts_main(["--text", "Hola", "--instruct", "female", "--language", "es", "-o", str(out)])
    assert out.read_bytes()[:4] == b"RIFF"
    assert engine.calls[-1][1]["instruct"] == "female"
