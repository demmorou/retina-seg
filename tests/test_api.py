import pytest
from fastapi.testclient import TestClient

from src.config import CONFIG

pytestmark = pytest.mark.skipif(
    not (CONFIG.models_dir / "checkpoint_best.pt").is_file(),
    reason="checkpoint models/checkpoint_best.pt não encontrado (rode o treino, etapa 4)",
)


@pytest.fixture(scope="module")
def client():
    from src.api.main import app

    with TestClient(app) as test_client:
        yield test_client


def _sample_image_path():
    from src.preprocess import list_pairs

    pairs = list_pairs("test")
    pair = next(p for p in pairs if p.masks["EX"] is not None)
    return pair.image_path


def test_analyze_returns_contract_shape(client):
    image_path = _sample_image_path()
    with open(image_path, "rb") as f:
        response = client.post("/analyze", files={"file": (image_path.name, f, "image/jpeg")})

    assert response.status_code == 200
    body = response.json()

    assert set(body.keys()) == {"image", "grade", "lesions", "summary", "report"}
    assert body["image"]["width"] > 0 and body["image"]["height"] > 0
    assert body["grade"]["value"] in (0, 1, 2, 3)
    assert isinstance(body["grade"]["label"], str) and body["grade"]["label"]
    assert set(body["summary"].keys()) == set(CONFIG.lesion_types)

    for lesion in body["lesions"]:
        assert lesion["type"] in CONFIG.lesion_types
        assert {"type", "cx", "cy", "r", "area", "score", "quadrant"} <= set(lesion.keys())

    assert "NÃO substitui avaliação" in body["report"]


def test_analyze_rejects_invalid_image(client):
    response = client.post("/analyze", files={"file": ("not_an_image.jpg", b"lixo", "image/jpeg")})
    assert response.status_code == 400
