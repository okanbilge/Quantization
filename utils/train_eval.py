"""Training and evaluation utilities (PyTorch)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class EarlyStopper:
    patience: int
    min_delta: float
    mode: str = "max"  # "max" for accuracy, "min" for loss
    best: Optional[float] = None
    bad_epochs: int = 0

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            self.bad_epochs = 0
            return False

        if self.mode == "max":
            improved = (value - self.best) > self.min_delta
        else:
            improved = (self.best - value) > self.min_delta

        if improved:
            self.best = value
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def set_determinism(seed: int, deterministic: bool = True) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, config: Dict) -> torch.optim.Optimizer:
    lr = float(config["training"]["learning_rate"])
    wd = float(config["training"]["weight_decay"])
    opt_name = str(config["training"].get("optimizer", "adam")).lower()

    params = [p for p in model.parameters() if p.requires_grad]
    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if opt_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    return torch.optim.Adam(params, lr=lr, weight_decay=wd)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    n = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        bs = labels.shape[0]
        running_loss += float(loss.detach().cpu()) * bs
        n += bs

    return running_loss / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        all_logits.append(logits.detach().cpu().float().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_metric: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_metric": best_metric,
        },
        str(path),
    )


def load_checkpoint(path: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
    ckpt = torch.load(str(path), map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt
