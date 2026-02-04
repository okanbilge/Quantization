#!/usr/bin/env python3
"""
One-Click Inference Benchmark (CPU + GPU) with Fair Comparisons
==============================================================

This script benchmarks inference latency and throughput for multiple backbones
across multiple datasets, with consistent measurement settings.

What it measures
- GPU FP32 latency and throughput (if CUDA is available)
- GPU FP16 latency and throughput (if CUDA is available)
- CPU FP32 latency and throughput
- CPU BF16 latency and throughput (if supported by your CPU and PyTorch build)
- CPU INT8 PTQ latency and throughput (eager-mode static quantization)

Fair speedups
- GPU FP16 speedup is computed vs GPU FP32
- CPU BF16 speedup is computed vs CPU FP32
- CPU INT8 speedup is computed vs CPU FP32

Notes
- PyTorch eager-mode INT8 quantized models do not run on CUDA.
  If you need GPU INT8, use TensorRT or ONNX Runtime CUDA INT8.
- CPU "FP16" is usually not beneficial in PyTorch. BF16 is the common CPU reduced-precision path.
  This script uses BF16 for CPU reduced precision.

Typical usage
  python benchmark_inference_fair_v2.py
  python benchmark_inference_fair_v2.py --dataset chestxray
  python benchmark_inference_fair_v2.py --batch-sizes 1,32
  python benchmark_inference_fair_v2.py --quick
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from utils.model_factory import create_model, count_parameters
from utils.quantization_utils import quantize_int8_ptq

# ============================================================================
# Config
# ============================================================================
MODELS = [
    "resnet18",
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "mobilenet_v2",
    "convnext_tiny",
    "swin_tiny",
    "vit_base",
]

DATASETS_INFO = {
    "brainmri": {"num_classes": 4, "name": "Brain MRI"},
    "chestxray": {"num_classes": 4, "name": "Chest X-ray"},
    "skincancer": {"num_classes": 7, "name": "Dermoscopy (HAM10000)"},
}



INPUT_SIZE = (3, 224, 224)

# Default benchmark settings
DEFAULT_BATCH_SIZES = [1, 32]
NUM_WARMUP = 20
NUM_MEASURE = 100

# PTQ calibration settings
CALIB_BATCH_SIZE = 32
CALIB_NUM_BATCHES = 16
QUANT_ENGINE = "fbgemm"  # x86 CPUs. Use "qnnpack" for ARM.

# ============================================================================
# Helpers
# ============================================================================
def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    cfg_path = Path(config_path) if config_path else (REPO_ROOT / "configs" / "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _now() -> float:
    return time.perf_counter()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def get_model_size_naive_param_buffer_mb(model: nn.Module) -> float:
    """
    Naive estimate: parameters + buffers. Often wrong for quantized models due to packed weights.
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return float((param_size + buffer_size) / (1024 * 1024))


def get_state_dict_size_mb(model: nn.Module) -> float:
    """
    More reliable estimate: serialize state_dict and measure bytes.
    """
    sd = model.state_dict()
    sd_cpu: Dict[str, Any] = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            sd_cpu[k] = v.detach().cpu()
        else:
            sd_cpu[k] = v

    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save(sd_cpu, path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return float(size_mb)


def create_dummy_calib_loader(batch_size: int = CALIB_BATCH_SIZE, num_batches: int = CALIB_NUM_BATCHES):
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size: int):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return torch.randn(*INPUT_SIZE), 0

    dataset = DummyDataset(batch_size * num_batches)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ============================================================================
# Benchmark core
# ============================================================================
@torch.no_grad()
def benchmark_inference(
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    autocast_dtype: Optional[torch.dtype] = None,
    num_warmup: int = NUM_WARMUP,
    num_measure: int = NUM_MEASURE,
    input_size: Tuple[int, int, int] = INPUT_SIZE,
) -> Dict[str, Any]:
    """
    Measures model forward latency. Uses autocast if autocast_dtype is provided and supported.

    For CUDA: uses torch.cuda.amp.autocast(dtype=torch.float16) when autocast_dtype=torch.float16.
    For CPU: uses torch.autocast(device_type="cpu", dtype=torch.bfloat16) when autocast_dtype=torch.bfloat16.
    """
    model.eval()
    x = torch.randn(batch_size, *input_size, device=device)

    use_autocast = autocast_dtype is not None

    def _forward():
        if not use_autocast:
            _ = model(x)
            return

        if device.type == "cuda":
            # CUDA autocast supports float16 and bfloat16 depending on hardware and build.
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                _ = model(x)
            return

        # CPU autocast
        with torch.autocast(device_type="cpu", dtype=autocast_dtype):
            _ = model(x)

    # Warmup
    for _ in range(num_warmup):
        _sync(device)
        _forward()
        _sync(device)

    # Measure
    times: List[float] = []
    for _ in range(num_measure):
        _sync(device)
        t0 = _now()
        _forward()
        _sync(device)
        times.append(_now() - t0)

    arr = np.array(times, dtype=np.float64)
    total_samples = batch_size * num_measure
    total_time_sec = float(arr.sum())

    return {
        "batch_size": int(batch_size),
        "num_iterations": int(num_measure),
        "mean_ms": float(arr.mean() * 1000),
        "p50_ms": float(np.percentile(arr, 50) * 1000),
        "p90_ms": float(np.percentile(arr, 90) * 1000),
        "p99_ms": float(np.percentile(arr, 99) * 1000),
        "throughput_samples_per_sec": float(total_samples / total_time_sec) if total_time_sec > 0 else 0.0,
        "autocast_dtype": str(autocast_dtype) if autocast_dtype is not None else None,
        "device": str(device),
    }


def benchmark_one_model_one_dataset(
    model_key: str,
    dataset_key: str,
    config: Dict[str, Any],
    cpu_device: torch.device,
    gpu_device: Optional[torch.device],
    batch_sizes: List[int],
) -> Dict[str, Any]:
    num_classes = DATASETS_INFO[dataset_key]["num_classes"]

    out: Dict[str, Any] = {
        "dataset": dataset_key,
        "dataset_name": DATASETS_INFO[dataset_key]["name"],
        "model": model_key,
        "num_classes": int(num_classes),
        "batch_sizes": [int(b) for b in batch_sizes],
        "num_warmup": int(NUM_WARMUP),
        "num_measure": int(NUM_MEASURE),
        "sizes": {},
        "cpu_fp32": {},
        "cpu_bf16": {},
        "cpu_int8_ptq": {},
        "gpu_fp32": {},
        "gpu_fp16": {},
        "speedups": {},
        "errors": {},
        "meta": {},
    }

    # -------------------------
    # CPU FP32
    # -------------------------
    try:
        model_cpu = create_model(model_key, num_classes, config, cpu_device)
        model_cpu.eval()

        out["meta"]["num_params"] = int(count_parameters(model_cpu))
        out["sizes"]["cpu_fp32_naive_param_buffer_mb"] = get_model_size_naive_param_buffer_mb(model_cpu)
        out["sizes"]["cpu_fp32_state_dict_mb"] = get_state_dict_size_mb(model_cpu)

        for bs in batch_sizes:
            out["cpu_fp32"][str(bs)] = benchmark_inference(model_cpu, cpu_device, bs, autocast_dtype=None)
        del model_cpu
    except Exception as e:
        out["errors"]["cpu_fp32"] = str(e)

    clear_memory()

    # -------------------------
    # CPU BF16 (optional, may fail on some CPUs/builds)
    # -------------------------
    try:
        model_cpu_bf16 = create_model(model_key, num_classes, config, cpu_device)
        model_cpu_bf16.eval()

        # BF16 autocast on CPU
        for bs in batch_sizes:
            out["cpu_bf16"][str(bs)] = benchmark_inference(
                model_cpu_bf16, cpu_device, bs, autocast_dtype=torch.bfloat16
            )
        del model_cpu_bf16
    except Exception as e:
        out["errors"]["cpu_bf16"] = str(e)

    clear_memory()

    # -------------------------
    # CPU INT8 PTQ
    # -------------------------
    try:
        torch.backends.quantized.engine = QUANT_ENGINE

        model_for_quant = create_model(model_key, num_classes, config, cpu_device)
        model_for_quant.eval()

        calib_loader = create_dummy_calib_loader()

        arch = config["models"]["deep_learning"][model_key]["architecture"]
        t0 = _now()
        model_int8 = quantize_int8_ptq(model_for_quant, arch, calib_loader, num_calib_batches=CALIB_NUM_BATCHES)
        out["meta"]["ptq_time_sec"] = float(_now() - t0)

        out["sizes"]["cpu_int8_naive_param_buffer_mb"] = get_model_size_naive_param_buffer_mb(model_int8)
        out["sizes"]["cpu_int8_state_dict_mb"] = get_state_dict_size_mb(model_int8)

        for bs in batch_sizes:
            out["cpu_int8_ptq"][str(bs)] = benchmark_inference(model_int8, cpu_device, bs, autocast_dtype=None)

        del model_int8
    except Exception as e:
        out["errors"]["cpu_int8_ptq"] = str(e)

    clear_memory()

    # -------------------------
    # GPU FP32 + FP16 (if available)
    # -------------------------
    if gpu_device is not None and gpu_device.type == "cuda":
        try:
            model_gpu = create_model(model_key, num_classes, config, gpu_device)
            model_gpu.eval()

            out["sizes"]["gpu_fp32_naive_param_buffer_mb"] = get_model_size_naive_param_buffer_mb(model_gpu)
            out["sizes"]["gpu_fp32_state_dict_mb"] = get_state_dict_size_mb(model_gpu)

            for bs in batch_sizes:
                out["gpu_fp32"][str(bs)] = benchmark_inference(model_gpu, gpu_device, bs, autocast_dtype=None)
                out["gpu_fp16"][str(bs)] = benchmark_inference(model_gpu, gpu_device, bs, autocast_dtype=torch.float16)

            del model_gpu
        except Exception as e:
            out["errors"]["gpu"] = str(e)

    clear_memory()

    # -------------------------
    # Speedups (fair comparisons per batch size)
    # -------------------------
    out["speedups"] = {str(bs): {} for bs in batch_sizes}

    for bs in batch_sizes:
        bs = int(bs)
        bs_key = str(bs)

        # GPU FP16 over GPU FP32
        if bs_key in out.get("gpu_fp32", {}) and bs_key in out.get("gpu_fp16", {}):
            fp32_ms = out["gpu_fp32"][bs_key]["mean_ms"]
            fp16_ms = out["gpu_fp16"][bs_key]["mean_ms"]
            if fp16_ms > 0:
                out["speedups"][bs_key]["gpu_fp16_over_gpu_fp32"] = float(fp32_ms / fp16_ms)

        # CPU BF16 over CPU FP32
        if bs_key in out.get("cpu_fp32", {}) and bs_key in out.get("cpu_bf16", {}):
            cpu_fp32_ms = out["cpu_fp32"][bs_key]["mean_ms"]
            cpu_bf16_ms = out["cpu_bf16"][bs_key]["mean_ms"]
            if cpu_bf16_ms > 0:
                out["speedups"][bs_key]["cpu_bf16_over_cpu_fp32"] = float(cpu_fp32_ms / cpu_bf16_ms)

        # CPU INT8 over CPU FP32
        if bs_key in out.get("cpu_fp32", {}) and bs_key in out.get("cpu_int8_ptq", {}):
            cpu_fp32_ms = out["cpu_fp32"][bs_key]["mean_ms"]
            cpu_int8_ms = out["cpu_int8_ptq"][bs_key]["mean_ms"]
            if cpu_int8_ms > 0:
                out["speedups"][bs_key]["cpu_int8_over_cpu_fp32"] = float(cpu_fp32_ms / cpu_int8_ms)

    return out


def print_compact_line(r: Dict[str, Any], batch: int) -> None:
    model = r["model"]
    ds = r["dataset"]
    bkey = str(batch)

    def fmt_block(block: Dict[str, Any], k: str) -> str:
        if not block or bkey not in block:
            return "N/A"
        b = block[bkey]
        return f"{b['mean_ms']:.2f} ms | {b['throughput_samples_per_sec']:.1f} FPS"

    cpu_fp32 = fmt_block(r.get("cpu_fp32", {}), bkey)
    cpu_bf16 = fmt_block(r.get("cpu_bf16", {}), bkey)
    cpu_int8 = fmt_block(r.get("cpu_int8_ptq", {}), bkey)
    gpu_fp32 = fmt_block(r.get("gpu_fp32", {}), bkey)
    gpu_fp16 = fmt_block(r.get("gpu_fp16", {}), bkey)

    sp = r.get("speedups", {}).get(bkey, {})
    sp_gpu = sp.get("gpu_fp16_over_gpu_fp32", None)
    sp_bf16 = sp.get("cpu_bf16_over_cpu_fp32", None)
    sp_int8 = sp.get("cpu_int8_over_cpu_fp32", None)

    sp_gpu_s = f"{sp_gpu:.2f}x" if sp_gpu is not None else "N/A"
    sp_bf16_s = f"{sp_bf16:.2f}x" if sp_bf16 is not None else "N/A"
    sp_int8_s = f"{sp_int8:.2f}x" if sp_int8 is not None else "N/A"

    print(
        f"[{ds}] {model:14s} | batch={batch:2d} | "
        f"CPU FP32: {cpu_fp32:24s} | CPU BF16: {cpu_bf16:24s} | CPU INT8: {cpu_int8:24s} | "
        f"GPU FP32: {gpu_fp32:24s} | GPU FP16: {gpu_fp16:24s} | "
        f"Spd GPU(FP16): {sp_gpu_s:6s} | Spd CPU(BF16): {sp_bf16_s:6s} | Spd CPU(INT8): {sp_int8_s:6s}"
    )


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Fair benchmark (CPU+GPU) for all datasets/models")
    parser.add_argument("--config", type=str, default=None, help="Config file path (optional)")
    parser.add_argument("--dataset", type=str, default=None, choices=list(DATASETS_INFO.keys()), help="Run only one dataset (optional)")
    parser.add_argument("--out-dir", type=str, default="./benchmark_results_fair", help="Output directory")
    parser.add_argument("--batch-sizes", type=str, default="1,32", help="Comma-separated batch sizes, e.g., 1,32")
    parser.add_argument("--quick", action="store_true", help="Quick test: only resnet18, all datasets")
    args = parser.parse_args()
    torch.set_num_threads(8)

    torch.set_num_interop_threads(1)
    config = load_config(args.config)

    cpu_device = torch.device("cpu")

    gpu_device: Optional[torch.device] = None
    gpu_name: Optional[str] = None
    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)

    # Determine which datasets to run
    if args.dataset is None:
        dataset_keys = list(DATASETS_INFO.keys())
    else:
        dataset_keys = [args.dataset]

    # Determine which models to run
    models = ["resnet18"] if args.quick else MODELS

    # Filter models that exist in config
    cfg_models = config.get("models", {}).get("deep_learning", {})
    models = [m for m in models if m in cfg_models]

    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    if len(batch_sizes) == 0:
        batch_sizes = list(DEFAULT_BATCH_SIZES)

    # Run context
    run_context = {
        "batch_sizes": [int(b) for b in batch_sizes],
        "num_warmup": int(NUM_WARMUP),
        "num_measure": int(NUM_MEASURE),
        "calib_batch_size": int(CALIB_BATCH_SIZE),
        "calib_num_batches": int(CALIB_NUM_BATCHES),
        "quant_engine": QUANT_ENGINE,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": gpu_name,
        "cpu_num_threads": int(torch.get_num_threads()),
        "cpu_num_interop_threads": int(torch.get_num_interop_threads()),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
    }

    print("\n" + "#" * 90)
    print("Fair Inference Benchmark")
    print(f"Datasets: {dataset_keys}")
    print(f"Models: {models}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"CUDA: {run_context['cuda_available']} | GPU: {gpu_name}")
    print(f"CPU threads: {run_context['cpu_num_threads']} | inter-op: {run_context['cpu_num_interop_threads']}")
    print(f"Quant engine: {QUANT_ENGINE}")
    print("#" * 90 + "\n")

    all_results: List[Dict[str, Any]] = []

    for ds in dataset_keys:
        for m in models:
            print(f"Running: dataset={ds}, model={m}")
            r = benchmark_one_model_one_dataset(
                model_key=m,
                dataset_key=ds,
                config=config,
                cpu_device=cpu_device,
                gpu_device=gpu_device,
                batch_sizes=batch_sizes,
            )
            all_results.append(r)
            for bs in batch_sizes:
                print_compact_line(r, batch=int(bs))

            if r.get("errors"):
                for k, v in r["errors"].items():
                    print(f"  WARN: {k}: {v}")

    # Save outputs
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    payload = {
        "run_context": run_context,
        "datasets": dataset_keys,
        "models": models,
        "results": all_results,
    }

    json_path = out_dir / "benchmark_all.json"
    save_json(json_path, payload)
    print(f"\nSaved JSON: {json_path}")

    # Also save per-dataset JSON
    for ds in dataset_keys:
        ds_results = [r for r in all_results if r["dataset"] == ds]
        ds_path = out_dir / f"benchmark_{ds}.json"
        save_json(ds_path, {"run_context": run_context, "dataset": ds, "results": ds_results})
        print(f"Saved JSON: {ds_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
