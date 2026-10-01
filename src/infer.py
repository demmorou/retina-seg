"""Inferência: janela deslizante sobre uma imagem em resolução original, remontagem da máscara de
probabilidades por lesão, e export do modelo pra ONNX.

Reusa o mesmo pré-processamento do treino (recorte do círculo da retina + CLAHE, `src/preprocess.py`
etapa 2) pra manter a imagem consistente com o que o modelo viu — nunca redimensiona a imagem
inteira, só tila em 512x512.

Uso:
    uv run python -m src.infer <caminho da imagem> [--checkpoint models/checkpoint_best.pt]
                                                     [--export-onnx models/model.onnx]
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.config import CONFIG
from src.dataset import build_val_transform
from src.preprocess import apply_clahe, crop_to_bbox, find_retina_bbox, load_image_rgb, tile_starts
from src.train import build_model, get_device


@dataclass(frozen=True)
class InferenceResult:
    image_rgb: np.ndarray  # imagem original completa (sem crop), pra desenhar overlay
    prob_masks: dict[str, np.ndarray]  # uma por lesão, mesmo shape de image_rgb[:2], valores 0-1


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = build_model(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _predict_tiles(
    model: torch.nn.Module,
    image: np.ndarray,  # RGB uint8, já recortado + CLAHE
    device: torch.device,
    tile_size: int = CONFIG.tile_size,
    stride: int = CONFIG.tile_stride,
) -> np.ndarray:
    """Retorna probabilidades (n_lesion_types, H, W) float32 — média nas sobreposições dos tiles."""
    height, width = image.shape[:2]
    n_types = len(CONFIG.lesion_types)
    prob_sum = np.zeros((n_types, height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    dummy_mask = np.zeros((tile_size, tile_size, n_types), dtype=np.float32)
    transform = build_val_transform()

    with torch.no_grad():
        for row in tile_starts(height, tile_size, stride):
            for col in tile_starts(width, tile_size, stride):
                tile = image[row : row + tile_size, col : col + tile_size]
                th, tw = tile.shape[:2]
                if th < tile_size or tw < tile_size:
                    # tile na borda pode sobrar menor que tile_size (imagem menor que o tile) — preenche com preto
                    padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                    padded[:th, :tw] = tile
                    tile = padded

                tensor = transform(image=tile, mask=dummy_mask)["image"].unsqueeze(0).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(tensor)
                probs = torch.sigmoid(logits).float().cpu().numpy()[0]  # (n_types, tile_size, tile_size)

                prob_sum[:, row : row + th, col : col + tw] += probs[:, :th, :tw]
                count[row : row + th, col : col + tw] += 1.0

    count = np.maximum(count, 1.0)
    return prob_sum / count[None, :, :]


def predict(model: torch.nn.Module, image: np.ndarray, device: torch.device) -> InferenceResult:
    """`image`: RGB uint8 em resolução original (já decodificada — de arquivo ou de upload)."""
    height, width = image.shape[:2]

    bbox = find_retina_bbox(image)
    cropped = crop_to_bbox(image, bbox)
    clahe_img = apply_clahe(cropped)

    probs_cropped = _predict_tiles(model, clahe_img, device)

    row_min, row_max, col_min, col_max = bbox
    prob_masks = {}
    for i, lesion_type in enumerate(CONFIG.lesion_types):
        full = np.zeros((height, width), dtype=np.float32)
        full[row_min : row_max + 1, col_min : col_max + 1] = probs_cropped[i]
        prob_masks[lesion_type] = full

    return InferenceResult(image_rgb=image, prob_masks=prob_masks)


def predict_image(model: torch.nn.Module, image_path: Path, device: torch.device) -> InferenceResult:
    return predict(model, load_image_rgb(image_path), device)


def export_onnx(model: torch.nn.Module, out_path: Path, tile_size: int = CONFIG.tile_size) -> None:
    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(1, 3, tile_size, tile_size, device=device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # exportador "dynamo" (padrão do torch 2.14) exige o pacote onnxscript — evita a dependência extra
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="caminho de uma imagem de retina")
    parser.add_argument("--checkpoint", type=Path, default=CONFIG.models_dir / "checkpoint_best.pt")
    parser.add_argument("--export-onnx", type=Path, default=None)
    args = parser.parse_args()

    device = get_device()
    model = load_model(args.checkpoint, device)

    result = predict_image(model, args.image, device)
    for lesion_type, prob in result.prob_masks.items():
        print(f"{lesion_type}: prob máxima {prob.max():.3f}, pixels acima de 0.5: {int((prob > 0.5).sum())}")

    if args.export_onnx:
        export_onnx(model, args.export_onnx)
        print(f"ONNX exportado em {args.export_onnx}")


if __name__ == "__main__":
    main()
