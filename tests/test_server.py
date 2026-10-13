from fastapi.testclient import TestClient

from omnivoice_duix.server import create_app


def test_flujo_duix_preprocess_e_invoke(engine, data_root):
    client = TestClient(create_app(engine, data_root))

    # Lo que envía Duix.Avatar al "entrenar" una voz.
    res = client.post(
        "/v1/preprocess_and_tran",
        json={"format": "wav", "reference_audio": "origin_audio/ref.wav", "lang": "zh"},
    ).json()
    assert res["code"] == 0
    assert res["asr_format_audio_url"] == "omnivoice_voices/abc.omnivoice.pt"
    assert res["reference_audio_text"] == "texto transcrito"

    # Lo que envía Duix.Avatar al sintetizar (con sus parámetros fijos de fish-speech).
    res = client.post(
        "/v1/invoke",
        json={
            "speaker": "uuid",
            "text": "Hola",
            "format": "wav",
            "topP": 0.7,
            "max_new_tokens": 1024,
            "reference_audio": res["asr_format_audio_url"],
            "reference_text": res["reference_audio_text"],
        },
    )
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert res.content[:4] == b"RIFF"
    text, kwargs = engine.calls[-1]
    assert text == "Hola"
    assert kwargs["ref_audio"] == (data_root / "omnivoice_voices/abc.omnivoice.pt").resolve()


def test_acepta_prefijo_del_contenedor(engine, data_root):
    client = TestClient(create_app(engine, data_root))
    res = client.post("/v1/preprocess_and_tran", json={"reference_audio": "/code/data/origin_audio/ref.wav"})
    assert res.json()["code"] == 0


def test_rechaza_rutas_fuera_de_datos(engine, data_root):
    client = TestClient(create_app(engine, data_root))
    res = client.post("/v1/preprocess_and_tran", json={"reference_audio": "../../etc/passwd"})
    assert res.json()["code"] == 1
    res = client.post("/v1/invoke", json={"text": "x", "reference_audio": "/etc/passwd"})
    assert res.status_code == 400


def test_endpoint_omnivoice_diseno_de_voz(engine, data_root):
    client = TestClient(create_app(engine, data_root))
    res = client.post("/v1/omnivoice", json={"text": "Hola", "instruct": "female", "language": "es"})
    assert res.status_code == 200
    _, kwargs = engine.calls[-1]
    assert kwargs["instruct"] == "female" and kwargs["language"] == "es"
