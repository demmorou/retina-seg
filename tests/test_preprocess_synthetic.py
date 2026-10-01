"""Testes com dados sintéticos — não dependem do dataset IDRiD estar baixado."""

import numpy as np

from src.preprocess import ImagePair, apply_clahe, crop_to_bbox, find_retina_bbox, split_train_val, tile_image_and_masks


def _synthetic_retina(height=200, width=300, square=(40, 160, 60, 220)):
    """Canvas preto com um quadrado claro no meio, simulando o círculo da retina cercado de fundo preto."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    row_min, row_max, col_min, col_max = square
    image[row_min:row_max, col_min:col_max] = 200
    return image, square


def test_find_retina_bbox_matches_known_square():
    image, (row_min, row_max, col_min, col_max) = _synthetic_retina()
    bbox = find_retina_bbox(image, threshold=10)
    assert bbox == (row_min, row_max - 1, col_min, col_max - 1)


def test_crop_to_bbox_removes_black_border():
    image, square = _synthetic_retina()
    bbox = find_retina_bbox(image, threshold=10)
    cropped = crop_to_bbox(image, bbox)
    assert cropped.shape[:2] == (square[1] - square[0], square[3] - square[2])
    assert (cropped > 0).all()


def test_apply_clahe_preserves_shape_and_dtype():
    image, square = _synthetic_retina()
    cropped = crop_to_bbox(image, find_retina_bbox(image, threshold=10))
    result = apply_clahe(cropped)
    assert result.shape == cropped.shape
    assert result.dtype == cropped.dtype


def test_tile_image_and_masks_aligns_image_and_mask_tiles():
    image = np.full((600, 600, 3), 150, dtype=np.uint8)
    mask = np.zeros((600, 600), dtype=np.uint8)
    mask[100:110, 100:110] = 1  # lesão isolada num canto

    tiles = tile_image_and_masks(image, {"MA": mask}, tile_size=256, stride=200, min_retina_frac=0.0)

    assert len(tiles) > 1
    for tile in tiles:
        assert tile.image.shape[:2] == (256, 256)
        assert tile.masks["MA"].shape == (256, 256)
        # a máscara do tile tem que corresponder exatamente ao recorte da máscara original
        expected = mask[tile.row : tile.row + 256, tile.col : tile.col + 256]
        assert np.array_equal(tile.masks["MA"], expected)


def test_tile_image_and_masks_discards_low_content_tiles():
    image = np.zeros((600, 600, 3), dtype=np.uint8)
    image[:, :50] = 200  # só uma faixa fina com conteúdo, resto preto
    mask = np.zeros((600, 600), dtype=np.uint8)

    tiles = tile_image_and_masks(image, {"MA": mask}, tile_size=256, stride=200, min_retina_frac=0.5)
    assert tiles == []


def test_split_train_val_is_deterministic_and_disjoint():
    pairs = [ImagePair(image_id=f"img_{i}", image_path=None, masks={}) for i in range(20)]

    train_a, val_a = split_train_val(pairs, n_val=5, seed=42)
    train_b, val_b = split_train_val(pairs, n_val=5, seed=42)

    assert [p.image_id for p in train_a] == [p.image_id for p in train_b]
    assert [p.image_id for p in val_a] == [p.image_id for p in val_b]
    assert len(val_a) == 5
    assert len(train_a) == 15
    assert set(p.image_id for p in train_a).isdisjoint(p.image_id for p in val_a)
