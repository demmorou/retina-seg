"""Configuração centralizada do projeto. Sem números mágicos espalhados pelo código."""

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Ordem fixa dos canais de lesão usados pelo modelo (CLAUDE.md: 4 canais, um por lesão).
LESION_TYPES = ("MA", "HE", "EX", "SE")


@dataclass(frozen=True)
class Config:
    raw_dir: Path = REPO_ROOT / "data" / "raw" / "idrid" / "A. Segmentation"
    processed_dir: Path = REPO_ROOT / "data" / "processed"
    lesion_types: tuple[str, ...] = field(default_factory=lambda: LESION_TYPES)
    seed: int = 42
    n_val: int = 8  # imagens de validação retiradas das 54 de treino oficiais

    # Recorte do círculo da retina: pixels de fundo ficam próximos de preto (mas não exatamente
    # 0 por causa de compressão JPEG), threshold baixo já separa retina do fundo.
    retina_crop_threshold: int = 10

    # CLAHE aplicado no canal L (LAB) — valores padrão comuns em literatura de fundo de olho.
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)

    # Tiling: 512x512 com stride 384 (25% de sobreposição) — evita cortar lesão na borda do tile.
    tile_size: int = 512
    tile_stride: int = 384
    # Descarta tiles quase todo preto (fora do círculo da retina) — sem sinal de treino.
    min_retina_frac: float = 0.1

    # Normalização ImageNet — exigida pelo encoder pré-treinado (efficientnet-b3, encoder_weights="imagenet").
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Treino
    encoder_name: str = "efficientnet-b3"
    models_dir: Path = REPO_ROOT / "models"
    batch_size: int = 8
    batch_size_fallback: int = 4  # usado com gradient accumulation se der OOM com batch_size
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    grad_clip_norm: float = 1.0

    # Inferência / pós-processamento
    lesion_prob_threshold: float = 0.5  # binariza a máscara de probabilidade antes dos componentes conectados
    lesion_min_area: int = 3  # descarta componentes menores que isso (ruído) — ponto de partida, ajustar com dados reais
    severe_he_quadrants: int = 3  # RDNP grave (aprox. regra 4-2-1 do ICDR): hemorragias em >= 3 dos 4 quadrantes
    severe_he_count: int = 20  # ou uma contagem total de hemorragias muito alta


CONFIG = Config()
