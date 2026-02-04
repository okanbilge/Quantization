"""Model factory utilities using timm."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import timm


_TORCHVISION_ALIAS_TO_TIMM: Dict[str, str] = {
    "ResNet18": "resnet18",
    "ResNet50": "resnet50",
    "DenseNet121": "densenet121",
    "EfficientNet-B0": "efficientnet_b0",
    "EfficientNet-B3": "efficientnet_b3",
    "MobileNetV2": "mobilenetv2_100",
}


def _to_timm_name(arch: str) -> str:
    arch = arch.strip()
    if arch.startswith("timm:"):
        return arch.split(":", 1)[1].strip()
    if arch in _TORCHVISION_ALIAS_TO_TIMM:
        return _TORCHVISION_ALIAS_TO_TIMM[arch]
    return arch


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_backbone(model: nn.Module) -> None:
    classifier = None
    if hasattr(model, "get_classifier"):
        classifier = model.get_classifier()

    for p in model.parameters():
        p.requires_grad = False

    if classifier is not None and hasattr(classifier, "parameters"):
        for p in classifier.parameters():
            p.requires_grad = True


def create_model(
    model_key: str,
    num_classes: int,
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
) -> nn.Module:
    mcfg = config["models"]["deep_learning"][model_key]
    arch = _to_timm_name(mcfg["architecture"])
    pretrained = bool(mcfg.get("pretrained", True))
    drop_rate = float(mcfg.get("dropout", 0.0))

    try:
        model = timm.create_model(
            arch,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
    except Exception:
        if pretrained:
            model = timm.create_model(
                arch,
                pretrained=False,
                num_classes=num_classes,
                drop_rate=drop_rate,
            )
        else:
            raise

    if mcfg.get("freeze_backbone", False):
        freeze_backbone(model)

    if device is not None:
        model = model.to(device)

    return model
