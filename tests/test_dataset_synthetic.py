"""Testes com tiles sintéticos gravados em disco — não dependem do dataset IDRiD nem de
`src/preprocess.py` já ter rodado."""

import cv2
import numpy as np
import pytest
import torch

from src.config import CONFIG
from src.dataset import RetinaTileDataset, build_train_transform, build_val_transform, list_tile_ids


def _write_fake_split(tmp_path, n_tiles=3, size=64):
    split_dir = tmp_path / "train"
    images_dir = split_dir / "images"
    masks_dir = split_dir / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    for i in range(n_tiles):
        tile_id = f"IDRiD_01_r0_c{i}"
        image = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / f"{tile_id}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        mask = np.zeros((size, size, len(CONFIG.lesion_types)), dtype=np.uint8)
        mask[10:20, 10:20, 0] = 255  # lesão sintética só no canal MA
        np.save(masks_dir / f"{tile_id}.npy", mask)

    return split_dir


def test_list_tile_ids_finds_saved_tiles(tmp_path):
    split_dir = _write_fake_split(tmp_path, n_tiles=3)
    assert list_tile_ids(split_dir) == [f"IDRiD_01_r0_c{i}" for i in range(3)]


def test_dataset_item_shapes_and_dtypes(tmp_path):
    split_dir = _write_fake_split(tmp_path, n_tiles=2, size=64)
    dataset = RetinaTileDataset("train", processed_dir=tmp_path)

    assert len(dataset) == 2
    item = dataset[0]
    assert item["image"].shape == (3, 64, 64)
    assert item["mask"].shape == (len(CONFIG.lesion_types), 64, 64)
    assert item["image"].dtype == torch.float32
    assert item["mask"].dtype == torch.float32
    assert set(item["mask"].unique().tolist()) <= {0.0, 1.0}


def test_dataset_raises_when_split_empty(tmp_path):
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "train" / "masks").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        RetinaTileDataset("train", processed_dir=tmp_path)


def test_dataloader_iterates_with_matching_batch_shapes(tmp_path):
    split_dir = _write_fake_split(tmp_path, n_tiles=6, size=64)
    dataset = RetinaTileDataset("train", transform=build_val_transform(), processed_dir=tmp_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    batch = next(iter(loader))
    assert batch["image"].shape == (4, 3, 64, 64)
    assert batch["mask"].shape == (4, len(CONFIG.lesion_types), 64, 64)


def test_train_transform_flip_keeps_image_and_mask_aligned(tmp_path):
    # Máscara com lesão só no canto superior-esquerdo: um flip horizontal determinístico
    # (p=1) deve mover a lesão pro canto superior-direito, igual na imagem e na máscara.
    size = 64
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[10:20, 10:20] = 255  # "lesão" visível na imagem também, pra checar alinhamento
    mask = np.zeros((size, size, len(CONFIG.lesion_types)), dtype=np.float32)
    mask[10:20, 10:20, 0] = 1.0

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose(
        [A.HorizontalFlip(p=1.0), A.Normalize(mean=CONFIG.imagenet_mean, std=CONFIG.imagenet_std), ToTensorV2(transpose_mask=True)]
    )
    out = transform(image=image, mask=mask)

    mask_ch0 = out["mask"][0]
    lesion_rows, lesion_cols = torch.where(mask_ch0 > 0)
    assert lesion_cols.min().item() == size - 20
    assert lesion_cols.max().item() == size - 11
