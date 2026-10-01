import pytest

from src.config import CONFIG
from src.preprocess import list_pairs, load_mask

pytestmark = pytest.mark.skipif(
    not CONFIG.raw_dir.is_dir(),
    reason="dataset IDRiD não encontrado em data/raw/idrid/ (rode download-kaggle.py)",
)


def test_train_split_has_54_images_with_existing_paths():
    pairs = list_pairs("train")
    assert len(pairs) == 54
    for pair in pairs:
        assert pair.image_path.is_file()


def test_test_split_has_27_images():
    pairs = list_pairs("test")
    assert len(pairs) == 27


def test_masks_present_are_real_files_and_absent_are_none():
    pairs = list_pairs("train")
    for pair in pairs:
        for lesion_type in CONFIG.lesion_types:
            mask_path = pair.masks[lesion_type]
            assert mask_path is None or mask_path.is_file()


def test_not_every_image_has_every_lesion_mask():
    # Soft Exudates (SE) não existe para todas as imagens de treino — confirmado em disco.
    pairs = list_pairs("train")
    se_present = sum(1 for pair in pairs if pair.masks["SE"] is not None)
    assert 0 < se_present < len(pairs)


def test_load_mask_handles_rgba_tif():
    # IDRiD_81_EX.tif (Testing Set) é um TIFF RGBA em vez de escala de cinza — caso real do dataset.
    pairs = list_pairs("test")
    pair = next(p for p in pairs if p.image_id == "IDRiD_81")
    mask = load_mask(pair.masks["EX"], shape=(2848, 4288))
    assert mask.shape == (2848, 4288)
    assert set(mask.tolist()[0]) <= {0, 1}
    assert mask.any()
