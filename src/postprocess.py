"""Máscara de probabilidade -> lesões individuais (componentes conectados via OpenCV) + regras de
grau inspiradas na escala ICDR.

Quadrante é geométrico (relativo ao centro da imagem: superior/inferior x esquerdo/direito), não
clínico — a lateralidade do olho (nasal/temporal) não dá pra inferir só da imagem.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from src.config import CONFIG

Quadrant = str


@dataclass(frozen=True)
class Lesion:
    type: str
    cx: float
    cy: float
    r: float
    area: float
    score: float
    quadrant: Quadrant
    bbox: tuple[int, int, int, int]  # x_min, y_min, x_max, y_max


@dataclass(frozen=True)
class Grade:
    value: int
    label: str


GRADE_LABELS = {
    0: "Sem sinais de retinopatia diabética",
    1: "Retinopatia diabética não proliferativa leve",
    2: "Retinopatia diabética não proliferativa moderada",
    3: "Retinopatia diabética não proliferativa grave",
}


def _quadrant(cx: float, cy: float, width: int, height: int) -> Quadrant:
    vertical = "superior" if cy < height / 2 else "inferior"
    horizontal = "esquerdo" if cx < width / 2 else "direito"
    return f"{vertical}-{horizontal}"


def mask_to_lesions(
    prob_mask: np.ndarray,
    lesion_type: str,
    threshold: float = CONFIG.lesion_prob_threshold,
    min_area: int = CONFIG.lesion_min_area,
) -> list[Lesion]:
    """`prob_mask`: probabilidades (H, W) float 0-1 de um único tipo de lesão, em resolução
    original (a mesma da imagem de entrada)."""
    height, width = prob_mask.shape
    binary = (prob_mask >= threshold).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    lesions = []
    for label in range(1, n_labels):  # label 0 = fundo
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label]
        score = float(prob_mask[labels == label].mean())
        lesions.append(
            Lesion(
                type=lesion_type,
                cx=float(cx),
                cy=float(cy),
                r=float(np.sqrt(area / np.pi)),
                area=float(area),
                score=score,
                quadrant=_quadrant(cx, cy, width, height),
                bbox=(x, y, x + w, y + h),
            )
        )
    return lesions


def masks_to_lesions(prob_masks: dict[str, np.ndarray]) -> list[Lesion]:
    """`prob_masks`: um mapa tipo de lesão -> probabilidades (H, W), todas do mesmo tamanho."""
    lesions = []
    for lesion_type, prob_mask in prob_masks.items():
        lesions.extend(mask_to_lesions(prob_mask, lesion_type))
    return lesions


def compute_grade(lesions: list[Lesion]) -> Grade:
    """Regras heurísticas inspiradas na escala ICDR, derivadas só da contagem/distribuição das
    lesões detectadas (MA/HE/EX/SE). NÃO é uma classificação ICDR completa: sinais como
    neovascularização, IRMA e rosário venoso não são detectados por este modelo, então o grau
    máximo derivável aqui é 3 (RDNP grave) — nunca 4 (proliferativa). Regra 4-2-1 aproximada pelo
    número de quadrantes com hemorragia. Uso educacional, nunca substitui avaliação oftalmológica."""
    if not lesions:
        return Grade(0, GRADE_LABELS[0])

    if all(lesion.type == "MA" for lesion in lesions):
        return Grade(1, GRADE_LABELS[1])

    he_lesions = [lesion for lesion in lesions if lesion.type == "HE"]
    he_quadrants = {lesion.quadrant for lesion in he_lesions}
    if len(he_quadrants) >= CONFIG.severe_he_quadrants or len(he_lesions) >= CONFIG.severe_he_count:
        return Grade(3, GRADE_LABELS[3])

    return Grade(2, GRADE_LABELS[2])
