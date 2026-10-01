"""Testes com máscaras de probabilidade sintéticas — não dependem do dataset IDRiD nem de um
modelo treinado."""

import numpy as np

from src.config import CONFIG
from src.postprocess import Lesion, compute_grade, mask_to_lesions, masks_to_lesions


def _blob_prob_mask(size=100, center=(30, 40), radius=5, prob=0.9):
    mask = np.zeros((size, size), dtype=np.float32)
    y, x = np.ogrid[:size, :size]
    cy, cx = center
    circle = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
    mask[circle] = prob
    return mask


def test_mask_to_lesions_finds_single_blob_with_expected_centroid_and_area():
    mask = _blob_prob_mask(size=100, center=(30, 40), radius=5, prob=0.9)
    lesions = mask_to_lesions(mask, "MA", threshold=0.5, min_area=1)

    assert len(lesions) == 1
    lesion = lesions[0]
    assert lesion.type == "MA"
    assert abs(lesion.cx - 40) < 1.0
    assert abs(lesion.cy - 30) < 1.0
    assert lesion.area > 0
    assert lesion.score > 0.5
    x_min, y_min, x_max, y_max = lesion.bbox
    assert x_min <= 40 <= x_max
    assert y_min <= 30 <= y_max


def test_mask_to_lesions_discards_components_below_min_area():
    mask = _blob_prob_mask(size=100, center=(30, 40), radius=1, prob=0.9)  # blob bem pequeno
    lesions = mask_to_lesions(mask, "MA", threshold=0.5, min_area=1000)
    assert lesions == []


def test_mask_to_lesions_respects_threshold():
    mask = _blob_prob_mask(size=100, center=(30, 40), radius=5, prob=0.3)  # abaixo do threshold padrão
    lesions = mask_to_lesions(mask, "MA", threshold=0.5, min_area=1)
    assert lesions == []


def test_mask_to_lesions_quadrant_matches_position():
    size = 100
    top_left = _blob_prob_mask(size, center=(10, 10), radius=3)
    bottom_right = _blob_prob_mask(size, center=(90, 90), radius=3)

    lesion_tl = mask_to_lesions(top_left, "EX", threshold=0.5, min_area=1)[0]
    lesion_br = mask_to_lesions(bottom_right, "EX", threshold=0.5, min_area=1)[0]

    assert lesion_tl.quadrant == "superior-esquerdo"
    assert lesion_br.quadrant == "inferior-direito"


def test_masks_to_lesions_aggregates_across_types():
    prob_masks = {
        "MA": _blob_prob_mask(center=(10, 10)),
        "HE": _blob_prob_mask(center=(80, 80)),
    }
    lesions = masks_to_lesions(prob_masks)
    types = {lesion.type for lesion in lesions}
    assert types == {"MA", "HE"}


def test_compute_grade_zero_when_no_lesions():
    assert compute_grade([]).value == 0


def test_compute_grade_one_when_only_microaneurysms():
    lesions = [Lesion("MA", 1, 1, 1, 1, 0.9, "superior-esquerdo", (0, 0, 1, 1))]
    assert compute_grade(lesions).value == 1


def test_compute_grade_two_when_he_in_few_quadrants():
    lesions = [
        Lesion("MA", 1, 1, 1, 1, 0.9, "superior-esquerdo", (0, 0, 1, 1)),
        Lesion("HE", 1, 1, 1, 1, 0.9, "superior-esquerdo", (0, 0, 1, 1)),
    ]
    assert compute_grade(lesions).value == 2


def test_compute_grade_three_when_he_spread_across_quadrants():
    quadrants = ["superior-esquerdo", "superior-direito", "inferior-esquerdo"]
    lesions = [Lesion("HE", 1, 1, 1, 1, 0.9, q, (0, 0, 1, 1)) for q in quadrants]
    assert compute_grade(lesions).value == 3


def test_compute_grade_three_when_he_count_is_high_even_in_one_quadrant():
    lesions = [
        Lesion("HE", 1, 1, 1, 1, 0.9, "superior-esquerdo", (0, 0, 1, 1))
        for _ in range(CONFIG.severe_he_count)
    ]
    assert compute_grade(lesions).value == 3
