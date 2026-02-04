"""Quantization helpers.

INT8 PTQ runs on CPU using torch.ao quantization.

- CNN-like models: FX static quantization with calibration.
- Transformer-like models (ViT/Swin): dynamic quantization for Linear layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def is_transformer_like(model_key: str) -> bool:
    mk = model_key.lower()
    return ("vit" in mk) or ("swin" in mk)


@torch.no_grad()
def quantize_dynamic_int8(model: nn.Module) -> nn.Module:
    model = model.cpu().eval()
    qmodel = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return qmodel


@torch.no_grad()
def quantize_static_fx_int8(
    model: nn.Module,
    calib_loader: torch.utils.data.DataLoader,
    num_calib_batches: int = 16,
    backend: str = "fbgemm",
) -> nn.Module:
    model = model.cpu().eval()
    torch.backends.quantized.engine = backend

    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

    qconfig_mapping = get_default_qconfig_mapping(backend)

    example_inputs = None
    for images, _ in calib_loader:
        example_inputs = (images.cpu(),)
        break
    if example_inputs is None:
        raise RuntimeError("Calibration loader is empty")

    prepared = prepare_fx(model, qconfig_mapping, example_inputs)

    it = 0
    for images, _ in calib_loader:
        prepared(images.cpu())
        it += 1
        if it >= num_calib_batches:
            break

    converted = convert_fx(prepared)
    return converted


@torch.no_grad()
def quantize_int8_ptq(
    model: nn.Module,
    model_key: str,
    calib_loader: torch.utils.data.DataLoader,
    num_calib_batches: int = 16,
    backend: str = "fbgemm",
) -> nn.Module:
    if is_transformer_like(model_key):
        return quantize_dynamic_int8(model)
    return quantize_static_fx_int8(
        model,
        calib_loader,
        num_calib_batches=num_calib_batches,
        backend=backend,
    )
