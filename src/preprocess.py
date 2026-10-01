"""Leitura da estrutura bruta do IDRiD e listagem dos pares imagem/máscara.

Estrutura real encontrada em `data/raw/idrid/A. Segmentation/` (ver docs/ROADMAP.md etapa 1):

    1. Original Images/{a. Training Set,b. Testing Set}/IDRiD_NN.jpg
    2. All Segmentation Groundtruths/{a. Training Set,b. Testing Set}/<N>. <Tipo>/IDRiD_NN_<SIGLA>.tif

Nem toda imagem tem máscara de todo tipo de lesão (ex.: Soft Exudates só existe para uma fração
das imagens) — a ausência do arquivo é esperada e tratada como "sem lesão desse tipo", não como erro.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from tqdm import tqdm

from src.config import CONFIG

BBox = tuple[int, int, int, int]  # row_min, row_max, col_min, col_max (inclusive)

Split = Literal["train", "test"]

_SPLIT_DIRS: dict[Split, str] = {
    "train": "a. Training Set",
    "test": "b. Testing Set",
}

_LESION_FOLDERS: dict[str, str] = {
    "MA": "1. Microaneurysms",
    "HE": "2. Haemorrhages",
    "EX": "3. Hard Exudates",
    "SE": "4. Soft Exudates",
}


@dataclass(frozen=True)
class ImagePair:
    image_id: str
    image_path: Path
    masks: dict[str, Path | None]  # chave = tipo de lesão (CONFIG.lesion_types), valor = caminho ou None


def list_pairs(split: Split, raw_dir: Path = CONFIG.raw_dir) -> list[ImagePair]:
    split_dir = _SPLIT_DIRS[split]
    images_dir = raw_dir / "1. Original Images" / split_dir
    gt_dir = raw_dir / "2. All Segmentation Groundtruths" / split_dir

    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Diretório de imagens não encontrado: {images_dir}. "
            "O dataset IDRiD foi baixado para data/raw/idrid/? (ver download-kaggle.py)"
        )

    pairs = []
    for image_path in sorted(images_dir.glob("*.jpg")):
        image_id = image_path.stem
        masks = {}
        for lesion_type in CONFIG.lesion_types:
            mask_path = gt_dir / _LESION_FOLDERS[lesion_type] / f"{image_id}_{lesion_type}.tif"
            masks[lesion_type] = mask_path if mask_path.is_file() else None
        pairs.append(ImagePair(image_id=image_id, image_path=image_path, masks=masks))

    return pairs


def load_image_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Falha ao ler imagem: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Carrega máscara binária (0/1) uint8. `shape` = (altura, largura) usado quando path é None
    (imagem sem lesão desse tipo -> máscara toda zero)."""
    if path is None:
        return np.zeros(shape, dtype=np.uint8)
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Falha ao ler máscara: {path}")
    if raw.ndim == 3:
        # Ao menos um arquivo do IDRiD (IDRiD_81_EX.tif) é salvo como TIFF RGBA em vez de
        # escala de cinza — descarta o alpha (sempre opaco) e reduz as cores via max.
        raw = raw[..., :3].max(axis=2)
    return (raw > 0).astype(np.uint8)


def find_retina_bbox(image_rgb: np.ndarray, threshold: int = CONFIG.retina_crop_threshold) -> BBox:
    """Bounding box (inclusive) do círculo da retina, descartando a moldura preta ao redor."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        raise ValueError("Nenhum pixel acima do threshold de recorte — imagem parece toda preta.")
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    return int(row_min), int(row_max), int(col_min), int(col_max)


def crop_to_bbox(image: np.ndarray, bbox: BBox) -> np.ndarray:
    row_min, row_max, col_min, col_max = bbox
    return image[row_min : row_max + 1, col_min : col_max + 1]


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


@dataclass(frozen=True)
class Tile:
    row: int
    col: int
    image: np.ndarray
    masks: dict[str, np.ndarray]


def tile_image_and_masks(
    image: np.ndarray,
    masks: dict[str, np.ndarray],
    tile_size: int = CONFIG.tile_size,
    stride: int = CONFIG.tile_stride,
    min_retina_frac: float = CONFIG.min_retina_frac,
) -> list[Tile]:
    """Corta imagem+máscaras em tiles quadrados sobrepostos, descartando tiles quase todo preto."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    tiles = []
    for row in tile_starts(height, tile_size, stride):
        for col in tile_starts(width, tile_size, stride):
            gray_tile = gray[row : row + tile_size, col : col + tile_size]
            retina_frac = float(np.mean(gray_tile > CONFIG.retina_crop_threshold))
            if retina_frac < min_retina_frac:
                continue
            image_tile = image[row : row + tile_size, col : col + tile_size]
            mask_tiles = {k: v[row : row + tile_size, col : col + tile_size] for k, v in masks.items()}
            tiles.append(Tile(row=row, col=col, image=image_tile, masks=mask_tiles))

    return tiles


def apply_clahe(image_rgb: np.ndarray) -> np.ndarray:
    """CLAHE no canal L (LAB) pra realçar contraste sem distorcer cor — ajuda a destacar
    microaneurismas pequenos sem estourar saturação."""
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CONFIG.clahe_clip_limit, tileGridSize=CONFIG.clahe_tile_grid_size)
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def split_train_val(
    pairs: list[ImagePair], n_val: int = CONFIG.n_val, seed: int = CONFIG.seed
) -> tuple[list[ImagePair], list[ImagePair]]:
    """Separação determinística (seed fixa) de `n_val` imagens de treino oficiais para validação."""
    shuffled = pairs.copy()
    random.Random(seed).shuffle(shuffled)
    return shuffled[n_val:], shuffled[:n_val]


def process_pair(pair: ImagePair) -> list[Tile]:
    """Recorte do círculo + CLAHE + tiling, pra uma imagem e suas máscaras."""
    image = load_image_rgb(pair.image_path)
    bbox = find_retina_bbox(image)
    cropped = crop_to_bbox(image, bbox)
    clahe_img = apply_clahe(cropped)
    masks = {
        lesion_type: crop_to_bbox(load_mask(pair.masks[lesion_type], shape=image.shape[:2]), bbox)
        for lesion_type in CONFIG.lesion_types
    }
    return tile_image_and_masks(clahe_img, masks)


def save_tiles(pair: ImagePair, tiles: list[Tile], out_dir: Path) -> None:
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for tile in tiles:
        tile_id = f"{pair.image_id}_r{tile.row}_c{tile.col}"
        cv2.imwrite(str(images_dir / f"{tile_id}.png"), cv2.cvtColor(tile.image, cv2.COLOR_RGB2BGR))
        mask_stack = np.stack([tile.masks[lt] for lt in CONFIG.lesion_types], axis=-1) * 255
        np.save(masks_dir / f"{tile_id}.npy", mask_stack.astype(np.uint8))


def process_split(out_name: str, pairs: list[ImagePair]) -> int:
    """Processa e salva os tiles de uma lista de pares em `data/processed/<out_name>/`. Retorna o
    total de tiles salvos."""
    out_dir = CONFIG.processed_dir / out_name
    total = 0
    for pair in tqdm(pairs, desc=out_name):
        tiles = process_pair(pair)
        save_tiles(pair, tiles, out_dir)
        total += len(tiles)
    return total


def main() -> None:
    train_pairs, val_pairs = split_train_val(list_pairs("train"))
    test_pairs = list_pairs("test")
    print(f"train: {len(train_pairs)} imagens, val: {len(val_pairs)} imagens, test: {len(test_pairs)} imagens")

    n_train = process_split("train", train_pairs)
    n_val = process_split("val", val_pairs)
    n_test = process_split("test", test_pairs)
    print(f"tiles salvos -> train: {n_train}, val: {n_val}, test: {n_test}")


if __name__ == "__main__":
    main()
