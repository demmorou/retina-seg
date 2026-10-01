"""Dataset PyTorch para os tiles pré-processados (etapa 2, `src/preprocess.py`) + augmentations
via albumentations.

Lê `data/processed/<split>/images/<tile_id>.png` (3ch RGB) e
`data/processed/<split>/masks/<tile_id>.npy` (uint8, shape (H, W, 4), valores 0/255, uma
fatia por tipo de lesão na ordem `CONFIG.lesion_types`).
"""

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from src.config import CONFIG


def list_tile_ids(split_dir: Path) -> list[str]:
    return sorted(p.stem for p in (split_dir / "images").glob("*.png"))


def build_train_transform(
    mean: tuple[float, ...] = CONFIG.imagenet_mean, std: tuple[float, ...] = CONFIG.imagenet_std
) -> A.Compose:
    """Augmentations geométricas (seguras para lesões pequenas — sem crop/distorção) + normalização
    ImageNet. Brilho/contraste leve simula variação de exposição entre exames."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(transpose_mask=True),
        ]
    )


def build_val_transform(
    mean: tuple[float, ...] = CONFIG.imagenet_mean, std: tuple[float, ...] = CONFIG.imagenet_std
) -> A.Compose:
    """Sem augmentation — só normalização, pra validação/teste."""
    return A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2(transpose_mask=True)])


class RetinaTileDataset(Dataset):
    """Tiles processados de um split (`train`, `val` ou `test`): imagem 3ch + máscara 4ch (uma
    por lesão, ordem `CONFIG.lesion_types`), ambas 512x512."""

    def __init__(
        self,
        split: str,
        transform: A.Compose | None = None,
        processed_dir: Path = CONFIG.processed_dir,
    ):
        self.split_dir = processed_dir / split
        self.tile_ids = list_tile_ids(self.split_dir)
        if not self.tile_ids:
            raise FileNotFoundError(
                f"Nenhum tile encontrado em {self.split_dir} (rodou `uv run python -m src.preprocess`?)"
            )
        self.transform = transform if transform is not None else build_val_transform()

    def __len__(self) -> int:
        return len(self.tile_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        tile_id = self.tile_ids[idx]

        image_bgr = cv2.imread(str(self.split_dir / "images" / f"{tile_id}.png"), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Falha ao ler tile de imagem: {tile_id}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask = np.load(self.split_dir / "masks" / f"{tile_id}.npy").astype(np.float32) / 255.0

        transformed = self.transform(image=image, mask=mask)
        return {"image": transformed["image"], "mask": transformed["mask"], "tile_id": tile_id}
