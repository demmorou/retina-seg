"""FastAPI: `POST /analyze` recebe uma imagem de retina e devolve o contrato JSON do CLAUDE.md
(grau, lesões, contagens, laudo). Ver docs/ROADMAP.md etapa 6.

Uso:
    uv run uvicorn src.api.main:app --reload
"""

from collections import Counter
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.api.report import build_report
from src.api.schemas import AnalyzeResponse, GradeInfo, ImageInfo, LesionOut
from src.config import CONFIG
from src.infer import load_model, predict
from src.postprocess import compute_grade, masks_to_lesions
from src.train import get_device

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint = CONFIG.models_dir / "checkpoint_best.pt"
    if not checkpoint.is_file():
        raise RuntimeError(
            f"Checkpoint não encontrado em {checkpoint} — rode o treino (`uv run python -m src.train`, "
            "etapa 4) antes de subir a API."
        )
    device = get_device()
    _state["device"] = device
    _state["model"] = load_model(checkpoint, device)
    yield
    _state.clear()


app = FastAPI(title="retina-seg API", lifespan=lifespan)


def _decode_image(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    image_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Não foi possível decodificar a imagem enviada.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
    data = await file.read()
    image = _decode_image(data)

    result = predict(_state["model"], image, _state["device"])
    lesions = masks_to_lesions(result.prob_masks)
    grade = compute_grade(lesions)

    summary = {lesion_type: 0 for lesion_type in CONFIG.lesion_types}
    summary.update(Counter(lesion.type for lesion in lesions))

    height, width = image.shape[:2]
    return AnalyzeResponse(
        image=ImageInfo(width=width, height=height),
        grade=GradeInfo(value=grade.value, label=grade.label),
        lesions=[
            LesionOut(type=l.type, cx=l.cx, cy=l.cy, r=l.r, area=l.area, score=l.score, quadrant=l.quadrant)
            for l in lesions
        ],
        summary=summary,
        report=build_report(grade.label, summary),
    )
