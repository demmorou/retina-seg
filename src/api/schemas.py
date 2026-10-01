"""Contrato JSON do `POST /analyze` — ver CLAUDE.md."""

from pydantic import BaseModel


class ImageInfo(BaseModel):
    width: int
    height: int


class GradeInfo(BaseModel):
    value: int
    label: str


class LesionOut(BaseModel):
    type: str
    cx: float
    cy: float
    r: float
    area: float
    score: float
    quadrant: str


class AnalyzeResponse(BaseModel):
    image: ImageInfo
    grade: GradeInfo
    lesions: list[LesionOut]
    summary: dict[str, int]
    report: str
