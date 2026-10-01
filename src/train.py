"""Loop de treino: `smp.Unet` (efficientnet-b3), loss Dice+BCE, AMP, checkpoints, métricas de
validação (Dice e AUPR por lesão). Ver docs/ROADMAP.md etapa 4.

Uso:
    uv run python -m src.train --epochs 2 --quick     # smoke test rápido (poucos tiles)
    uv run python -m src.train --epochs 30              # treino completo (só com confirmação)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.losses import DiceLoss, SoftBCEWithLogitsLoss
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Subset

from src.config import CONFIG
from src.dataset import RetinaTileDataset, build_train_transform, build_val_transform


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(device: torch.device) -> torch.nn.Module:
    model = smp.Unet(
        encoder_name=CONFIG.encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=len(CONFIG.lesion_types),
        activation=None,
    )
    return model.to(device)


def build_loaders(batch_size: int, quick: bool) -> tuple[DataLoader, DataLoader]:
    train_ds = RetinaTileDataset("train", transform=build_train_transform())
    val_ds = RetinaTileDataset("val", transform=build_val_transform())

    if quick:
        # smoke test: subconjunto pequeno, só pra validar que o pipeline roda sem erro
        rng = np.random.RandomState(CONFIG.seed)
        train_idx = rng.choice(len(train_ds), size=min(200, len(train_ds)), replace=False).tolist()
        val_idx = rng.choice(len(val_ds), size=min(64, len(val_ds)), replace=False).tolist()
        train_ds = Subset(train_ds, train_idx)
        val_ds = Subset(val_ds, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=CONFIG.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


class LesionLoss(torch.nn.Module):
    """Dice (multilabel — canais de lesão são independentes, não mutuamente exclusivos) + BCE."""

    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(mode="multilabel")
        self.bce = SoftBCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.dice(logits, mask) + self.bce(logits, mask)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loss_fn: torch.nn.Module,
    device: torch.device,
    accumulation_steps: int,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images)
            loss = loss_fn(logits, masks) / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, loss_fn: torch.nn.Module, device: torch.device) -> dict:
    """Dice (streaming, via tp/fp/fn) + AUPR (métrica oficial IDRiD) por tipo de lesão."""
    model.eval()
    total_loss = 0.0
    n_types = len(CONFIG.lesion_types)
    tp = torch.zeros(n_types)
    fp = torch.zeros(n_types)
    fn = torch.zeros(n_types)
    probs_by_type: list[list[np.ndarray]] = [[] for _ in range(n_types)]
    targets_by_type: list[list[np.ndarray]] = [[] for _ in range(n_types)]

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images)
            loss = loss_fn(logits, masks)
        total_loss += loss.item()

        probs = torch.sigmoid(logits).float().cpu()
        preds = (probs > 0.5).float()
        masks_cpu = masks.float().cpu()

        tp += (preds * masks_cpu).sum(dim=(0, 2, 3))
        fp += (preds * (1 - masks_cpu)).sum(dim=(0, 2, 3))
        fn += ((1 - preds) * masks_cpu).sum(dim=(0, 2, 3))

        for i in range(n_types):
            probs_by_type[i].append(probs[:, i].reshape(-1).numpy())
            targets_by_type[i].append(masks_cpu[:, i].reshape(-1).numpy())

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)

    aupr = []
    for i in range(n_types):
        y_true = np.concatenate(targets_by_type[i])
        y_score = np.concatenate(probs_by_type[i])
        aupr.append(float("nan") if y_true.sum() == 0 else average_precision_score(y_true, y_score))

    return {
        "loss": total_loss / len(loader),
        "dice": {lt: dice[i].item() for i, lt in enumerate(CONFIG.lesion_types)},
        "aupr": {lt: aupr[i] for i, lt in enumerate(CONFIG.lesion_types)},
    }


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_metric: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "lesion_types": CONFIG.lesion_types,
            "encoder_name": CONFIG.encoder_name,
        },
        path,
    )


def run_training(
    n_epochs: int,
    quick: bool = False,
    batch_size: int | None = None,
    resume: Path | None = None,
) -> None:
    device = get_device()
    print(f"device: {device}")

    batch_size = batch_size or CONFIG.batch_size
    accumulation_steps = 1

    model = build_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay)
    scaler = torch.amp.GradScaler(device=device.type, enabled=device.type == "cuda")
    loss_fn = LesionLoss()

    start_epoch = 0
    best_dice = -1.0
    if resume is not None:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_dice = ckpt["best_metric"]
        print(f"retomado de {resume} (epoch {start_epoch}, best_dice {best_dice:.4f})")

    train_loader, val_loader = build_loaders(batch_size, quick)

    for epoch in range(start_epoch, n_epochs):
        t0 = time.time()
        try:
            train_loss = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, device, accumulation_steps)
        except torch.cuda.OutOfMemoryError:
            if batch_size <= CONFIG.batch_size_fallback:
                raise
            torch.cuda.empty_cache()
            accumulation_steps = CONFIG.batch_size // CONFIG.batch_size_fallback
            batch_size = CONFIG.batch_size_fallback
            print(
                f"OOM na época {epoch} — caindo pro batch {batch_size} com "
                f"{accumulation_steps} passos de gradient accumulation (batch efetivo mantido em {CONFIG.batch_size})"
            )
            train_loader, val_loader = build_loaders(batch_size, quick)
            train_loss = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, device, accumulation_steps)

        metrics = evaluate(model, val_loader, loss_fn, device)
        dt = time.time() - t0

        dice_str = " ".join(f"{lt}={metrics['dice'][lt]:.3f}" for lt in CONFIG.lesion_types)
        aupr_str = " ".join(f"{lt}={metrics['aupr'][lt]:.3f}" for lt in CONFIG.lesion_types)
        print(
            f"epoch {epoch + 1}/{n_epochs} ({dt:.1f}s) "
            f"train_loss={train_loss:.4f} val_loss={metrics['loss']:.4f} "
            f"dice[{dice_str}] aupr[{aupr_str}]"
        )

        mean_dice = sum(metrics["dice"].values()) / len(metrics["dice"])
        save_checkpoint(CONFIG.models_dir / "checkpoint_last.pt", model, optimizer, epoch, best_dice)
        if mean_dice > best_dice:
            best_dice = mean_dice
            save_checkpoint(CONFIG.models_dir / "checkpoint_best.pt", model, optimizer, epoch, best_dice)
            print(f"  novo melhor dice médio: {best_dice:.4f} -> checkpoint_best.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, required=True, help="número de épocas")
    parser.add_argument("--quick", action="store_true", help="smoke test: subconjunto pequeno de tiles")
    parser.add_argument("--batch-size", type=int, default=None, help=f"padrão: CONFIG.batch_size ({CONFIG.batch_size})")
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint .pt pra retomar treino")
    args = parser.parse_args()

    run_training(n_epochs=args.epochs, quick=args.quick, batch_size=args.batch_size, resume=args.resume)


if __name__ == "__main__":
    main()
