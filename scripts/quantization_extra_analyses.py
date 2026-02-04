#!/usr/bin/env python3
"""
experiments/quantization_extra_analyses.py

Adds three paper-strengthening analyses on top of your existing CHIL quantization pipeline:

1) PTQ calibration-set size ablation
   - Actionable takeaway: "PTQ needs at least X images" to stabilize AUPRC/ECE/Brier.

2) Temperature scaling (post-hoc calibration) for INT8-PTQ
   - Answers the reviewer question: "Did you try a simple fix?"
   - Fits temperature on VAL, reports calibration on TEST (when available) to avoid leakage.

3) Paired bootstrap confidence intervals for delta metrics
   - Adds statistical rigor: CI and p-value for deltas (FP32 vs PTQ, PTQ vs QAT, etc.)

This script is designed to work inside the same repository as:
  - experiments/run_precision_experiment.py
  - experiments/run_qat_experiment.py
and reuses the same utils modules and split logic.

Typical usage (examples):

# (A) PTQ calibration-size ablation for one dataset/model/fold/backend
python experiments/quantization_extra_analyses.py calib_size_ablation \
  --dataset chestxray --model resnet18 --fold 0 --backend fbgemm \
  --calib_sizes 16,32,64,128,256,512,1024 \
  --device cuda --config configs/config.yaml \
  --bench_warmup 5 --bench_batches 20

# (B) Temperature scaling for a saved logits file pair (calibrate on val, evaluate on test)
python experiments/quantization_extra_analyses.py temperature_scaling \
  --calib_logits /path/to/int8_val_logits.npy --calib_labels /path/to/int8_val_labels.npy \
  --eval_logits  /path/to/int8_test_logits.npy --eval_labels  /path/to/int8_test_labels.npy \
  --num_classes 3 --out_json /tmp/ts_results.json

# (C) Paired bootstrap delta CI on AUPRC between two variants
python experiments/quantization_extra_analyses.py paired_bootstrap \
  --a_logits /path/to/fp32_test_logits.npy --b_logits /path/to/int8_test_logits.npy \
  --labels  /path/to/test_labels.npy \
  --metric auprc --num_classes 3 --n_boot 2000 --seed 42 --out_json /tmp/bootstrap.json

Notes:
- For chestxray/skincancer: the split builder uses the same holdout-test logic as your runner scripts.
- For brainmri: uses the separate Testing folder if available via utils.data_loader.load_test_dataset.
- Temperature scaling is fit on the calibration split you provide and evaluated on the eval split you provide.

Author: (generated with ChatGPT, reviewed and corrected)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# Repo wiring
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

# Data loader imports (version tolerant)
try:
    from utils.data_loader import (
        create_cv_splits,
        create_cv_splits_with_holdout,
        create_dataloaders,
        create_test_loader,
        load_brainmri_dataset,
        load_chestxray_dataset,
        load_skincancer_dataset,
        load_test_dataset,
    )
except Exception:
    from data_loader_helper import (
        create_cv_splits,
        create_cv_splits_with_holdout,
        create_dataloaders,
        create_test_loader,
        load_brainmri_dataset,
        load_chestxray_dataset,
        load_skincancer_dataset,
        load_test_dataset,
    )

from utils.metrics import (
    brier_score_multiclass,
    compute_classification_metrics,
    expected_calibration_error,
    softmax_np,
)
from utils.model_factory import create_model, count_parameters
from utils.quantization_utils import quantize_int8_ptq
from utils.train_eval import load_checkpoint, set_determinism

DEFAULT_ROOT = Path(os.environ.get("EXPERIMENT_ROOT", str(REPO_ROOT / "runs")))
# -----------------------------------------------------------------------------
# General helpers (mirrors runner scripts)
# -----------------------------------------------------------------------------
def set_local_caches(root: Path) -> None:
    cache_root = root / ".cache"
    (cache_root / "torch").mkdir(parents=True, exist_ok=True)
    (cache_root / "hf").mkdir(parents=True, exist_ok=True)
    (cache_root / "xdg").mkdir(parents=True, exist_ok=True)
    (cache_root / "timm").mkdir(parents=True, exist_ok=True)

    os.environ["TORCH_HOME"] = str(cache_root / "torch")
    os.environ["HF_HOME"] = str(cache_root / "hf")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_root / "hf" / "hub")
    os.environ["XDG_CACHE_HOME"] = str(cache_root / "xdg")
    os.environ["TIMM_HOME"] = str(cache_root / "timm")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    import csv

    ensure_dir(path.parent)
    if not rows:
        return
    keys: List[str] = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def save_npy(path: Path, arr: np.ndarray) -> None:
    ensure_dir(path.parent)
    np.save(str(path), arr)


def load_npy(path: Path) -> np.ndarray:
    return np.load(str(path))


def _now() -> float:
    return time.perf_counter()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def get_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    cfg_path = Path(config_path) if config_path else (REPO_ROOT / "configs" / "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _expand_and_resolve_path(p: str, base_dir: Path) -> str:
    v = os.path.expandvars(os.path.expanduser(p))
    pp = Path(v)
    if not pp.is_absolute():
        pp = (base_dir / pp).resolve()
    return str(pp)


def normalize_paths_in_config(config: Dict[str, Any], base_dir: Path) -> None:
    def _walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                if isinstance(v, str) and any(t in k.lower() for t in ["path", "dir", "root", "folder"]):
                    obj[k] = _expand_and_resolve_path(v, base_dir)
                else:
                    _walk(v)
        elif isinstance(obj, list):
            for it in obj:
                _walk(it)
        return obj

    _walk(config)


def resolve_results_dir(_config: Dict[str, Any]) -> Path:
    return DEFAULT_ROOT / "results_additional_analyses"


def get_env_metadata() -> Dict[str, Any]:
    def _safe_cwd() -> str:
        try:
            rel = os.path.relpath(os.getcwd(), str(REPO_ROOT))
            if rel.startswith(".."):
                return "outside_repo"
            return rel
        except Exception:
            return "redacted"

    md: Dict[str, Any] = {
        "hostname": os.environ.get("RUN_HOSTNAME", "redacted"),
        "fqdn": os.environ.get("RUN_FQDN", "redacted"),
        "platform": platform.platform(),
        "python": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_node_list": os.environ.get("SLURM_NODELIST"),
        "cwd": _safe_cwd(),
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT)
        ).decode().strip()
        md["git_commit"] = commit
    except Exception:
        md["git_commit"] = None
    return md


# -----------------------------------------------------------------------------
# Dataset loader wrappers (version tolerant, matches runner scripts)
# -----------------------------------------------------------------------------
def load_dataset(dataset: str, config: Dict[str, Any]) -> Tuple[list, list, list, list]:
    if dataset == "brainmri":
        result = load_brainmri_dataset(config)
    elif dataset == "chestxray":
        result = load_chestxray_dataset(config)
    elif dataset == "skincancer":
        result = load_skincancer_dataset(config)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if len(result) == 3:
        return result[0], result[1], [], []
    if len(result) == 4:
        return result
    raise ValueError(f"Unexpected dataset loader return length: {len(result)}")


def _create_splits(
    image_paths: list, labels: list, config: Dict[str, Any], dataset: Optional[str] = None
):
    try:
        return create_cv_splits(image_paths, labels, config, dataset=dataset)
    except TypeError:
        try:
            return create_cv_splits(image_paths, labels, config)
        except TypeError:
            try:
                return create_cv_splits(labels, config)
            except TypeError:
                return create_cv_splits(image_paths, labels)


def _create_splits_with_holdout(
    image_paths: list,
    labels: list,
    config: Dict[str, Any],
    test_ratio: float,
    dataset: Optional[str] = None,
):
    try:
        return create_cv_splits_with_holdout(
            image_paths, labels, config, test_ratio=test_ratio, dataset=dataset
        )
    except TypeError:
        try:
            return create_cv_splits_with_holdout(
                image_paths, labels, config, test_ratio=test_ratio
            )
        except TypeError:
            return create_cv_splits_with_holdout(image_paths, labels, config, test_ratio)


def _create_loaders(
    image_paths: list,
    labels: list,
    train_idx,
    val_idx,
    config: Dict[str, Any],
    dataset: str,
):
    try:
        return create_dataloaders(image_paths, labels, train_idx, val_idx, config, dataset)
    except TypeError:
        return create_dataloaders(image_paths, labels, train_idx, val_idx, config)


def build_test_loader(
    dataset: str,
    config: Dict[str, Any],
    image_paths: list,
    labels: list,
    test_indices: Optional[List[int]],
) -> Tuple[Optional[Any], Dict[str, Any]]:
    try:
        if dataset == "brainmri":
            test_paths, test_labels = load_test_dataset(dataset, config)
            loader = create_test_loader(test_paths, test_labels, config, dataset=dataset)
            return loader, {"test_type": "folder", "num_test": len(test_labels)}

        if test_indices is None:
            return None, {"test_type": "none", "reason": "no_test_indices"}

        test_paths, test_labels = load_test_dataset(
            dataset, config, image_paths=image_paths, labels=labels, test_indices=test_indices
        )
        loader = create_test_loader(test_paths, test_labels, config, dataset=dataset)
        return loader, {"test_type": "holdout", "num_test": len(test_labels)}
    except Exception as e:
        return None, {"test_type": "none", "reason": str(e)}


# -----------------------------------------------------------------------------
# Metric computation (mirrors runner scripts)
# -----------------------------------------------------------------------------
def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def per_class_from_confusion(cm: np.ndarray) -> Dict[str, Any]:
    eps = 1e-12
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": cm.sum(axis=1).astype(int).tolist(),
    }


def calibration_bins(labels: np.ndarray, probs: np.ndarray, n_bins: int) -> Dict[str, Any]:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    bin_acc = np.zeros(n_bins, dtype=np.float64)
    bin_conf = np.zeros(n_bins, dtype=np.float64)

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        cnt = int(mask.sum())
        bin_counts[b] = cnt
        if cnt > 0:
            bin_acc[b] = float(correct[mask].mean())
            bin_conf[b] = float(conf[mask].mean())

    return {
        "bin_edges": bin_edges.tolist(),
        "bin_counts": bin_counts.tolist(),
        "bin_acc": bin_acc.tolist(),
        "bin_conf": bin_conf.tolist(),
    }


def safe_sklearn_metrics(labels: np.ndarray, probs: np.ndarray, num_classes: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

        out["auc_roc_ovr_macro_sklearn"] = float(
            roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        )

        y_onehot = np.eye(num_classes, dtype=np.float32)[labels.astype(int)]
        out["auprc_ovr_macro_sklearn"] = float(
            average_precision_score(y_onehot, probs, average="macro")
        )

        out["log_loss_sklearn"] = float(
            log_loss(labels, probs, labels=list(range(num_classes)))
        )
    except Exception as e:
        out["sklearn_metrics_error"] = str(e)
    return out


def compute_all_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    n_bins: int,
) -> Dict[str, Any]:
    probs = softmax_np(logits)
    preds = np.argmax(probs, axis=1)

    m = compute_classification_metrics(labels, preds, probs, num_classes=num_classes)
    m["ece"] = expected_calibration_error(labels, probs, n_bins=n_bins)
    m["brier"] = brier_score_multiclass(labels, probs, num_classes=num_classes)

    cm = compute_confusion_matrix(labels, preds, num_classes=num_classes)
    m["confusion_matrix"] = cm.tolist()
    m["per_class"] = per_class_from_confusion(cm)
    m["calibration_bins"] = calibration_bins(labels, probs, n_bins=n_bins)

    m.update(safe_sklearn_metrics(labels, probs, num_classes=num_classes))
    return m


# -----------------------------------------------------------------------------
# Eval and benchmarking
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_logits_labels(
    model: torch.nn.Module, loader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    for batch in loader:
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError("Dataloader batch must be (inputs, labels, ...).")
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        logits_list.append(out.detach().cpu().numpy())
        labels_list.append(y.detach().cpu().numpy())

    return np.concatenate(logits_list, axis=0), np.concatenate(labels_list, axis=0)


@torch.no_grad()
def benchmark_inference(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_warmup_batches: int = 5,
    num_meas_batches: int = 20,
) -> Dict[str, Any]:
    model.eval()

    times: List[float] = []
    n_samples = 0
    n_batches = 0
    it = iter(loader)

    # Warmup
    for _ in range(num_warmup_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        x = batch[0].to(device, non_blocking=True)
        _sync(device)
        _ = model(x)
        _sync(device)

    # Measure
    for _ in range(num_meas_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        x = batch[0].to(device, non_blocking=True)
        bs = int(x.shape[0])
        _sync(device)
        t0 = _now()
        _ = model(x)
        _sync(device)
        dt = _now() - t0

        times.append(float(dt))
        n_samples += bs
        n_batches += 1

    if n_batches == 0:
        return {"error": "Not enough batches for benchmarking."}

    arr = np.array(times, dtype=np.float64)
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    throughput = float(n_samples / arr.sum())

    return {
        "device": str(device),
        "num_batches": int(n_batches),
        "num_samples": int(n_samples),
        "mean_batch_sec": float(arr.mean()),
        "p50_batch_sec": p50,
        "p90_batch_sec": p90,
        "throughput_samples_per_sec": throughput,
    }


# -----------------------------------------------------------------------------
# PTQ calibration loader helpers
# -----------------------------------------------------------------------------
class CalibLoaderWrapper:
    """Ensures calibration data is on CPU for PTQ calibration."""

    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for batch in self.loader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                raise ValueError("Dataloader batch must be (inputs, labels, ...).")
            x, y = batch[0], batch[1]
            # FIX: Ensure both x and y are on CPU
            x_cpu = x.cpu() if hasattr(x, "cpu") else x
            y_cpu = y.cpu() if hasattr(y, "cpu") else y
            yield (x_cpu, y_cpu)

    def __len__(self):
        try:
            return len(self.loader)
        except Exception:
            return 0


class LimitedSamplesLoader:
    """
    Wraps a dataloader and yields at most max_samples total items (across batches).
    The last batch is truncated if needed.
    """

    def __init__(self, loader, max_samples: int):
        self.loader = loader
        self.max_samples = int(max_samples)
        self._len: Optional[int] = None
        self._batch_size: Optional[int] = None

    def _infer_batch_size(self) -> int:
        """Infer batch size without consuming the iterator."""
        if self._batch_size is not None:
            return self._batch_size

        # Try to get batch_size from loader attribute (DataLoader has this)
        if hasattr(self.loader, "batch_size") and self.loader.batch_size is not None:
            self._batch_size = int(self.loader.batch_size)
            return self._batch_size

        # Fallback: assume batch_size = 32 (conservative)
        self._batch_size = 32
        return self._batch_size

    def __iter__(self):
        seen = 0
        for batch in self.loader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                raise ValueError("Dataloader batch must be (inputs, labels, ...).")
            x, y = batch[0], batch[1]
            bs = int(x.shape[0])

            if seen >= self.max_samples:
                break

            take = min(bs, self.max_samples - seen)
            if take < bs:
                x = x[:take]
                y = y[:take]
            seen += take
            yield (x, y)

            if seen >= self.max_samples:
                break

    def __len__(self) -> int:
        # FIX: Use inferred batch size instead of consuming iterator
        if self._len is None:
            bs = self._infer_batch_size()
            self._len = int(math.ceil(self.max_samples / max(1, bs)))
        return self._len


# -----------------------------------------------------------------------------
# Temperature scaling
# -----------------------------------------------------------------------------
@dataclass
class TemperatureScalingResult:
    temperature: float
    nll_before: float
    nll_after: float


def _nll_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, labels, reduction="mean")


def fit_temperature(
    calib_logits: np.ndarray,
    calib_labels: np.ndarray,
    max_iter: int = 100,
    lr: float = 0.05,
    device: str = "cpu",
) -> TemperatureScalingResult:
    """
    Multiclass temperature scaling (single scalar T > 0).
    Optimizes NLL on calibration split.
    """
    logits = torch.tensor(calib_logits, dtype=torch.float32, device=device)
    labels = torch.tensor(calib_labels, dtype=torch.long, device=device)

    # Parameterize by logT to ensure positivity
    logT = torch.nn.Parameter(torch.zeros((), dtype=torch.float32, device=device))

    optimizer = torch.optim.LBFGS(
        [logT], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    nll_before = float(_nll_from_logits(logits, labels).detach().cpu().item())

    def closure():
        optimizer.zero_grad()
        T = torch.exp(logT).clamp(min=1e-6, max=1e6)
        loss = _nll_from_logits(logits / T, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        T = float(torch.exp(logT).clamp(min=1e-6, max=1e6).cpu().item())
        nll_after = float(
            _nll_from_logits(logits / torch.tensor(T, device=device), labels).cpu().item()
        )

    return TemperatureScalingResult(temperature=T, nll_before=nll_before, nll_after=nll_after)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    T = float(max(1e-8, temperature))
    return logits / T


# -----------------------------------------------------------------------------
# Paired bootstrap
# -----------------------------------------------------------------------------
def _metric_from_probs(
    labels: np.ndarray,
    probs: np.ndarray,
    metric: str,
    num_classes: int,
    n_bins: int,
) -> float:
    """
    Compute a metric from probability predictions.
    
    FIX: Corrected fallback logic - probs are already probabilities, not logits.
    """
    m = metric.lower().strip()

    if m in {"auprc", "auprc_macro", "auprc_ovr_macro"}:
        from sklearn.metrics import average_precision_score

        y_onehot = np.eye(num_classes, dtype=np.float32)[labels.astype(int)]
        return float(average_precision_score(y_onehot, probs, average="macro"))

    if m in {"auc", "auc_roc", "roc_auc", "auc_roc_ovr_macro"}:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))

    if m in {"ece"}:
        return float(expected_calibration_error(labels, probs, n_bins=n_bins))

    if m in {"brier"}:
        return float(brier_score_multiclass(labels, probs, num_classes=num_classes))

    if m in {"acc", "accuracy"}:
        preds = probs.argmax(axis=1)
        return float((preds == labels).mean())

    if m in {"bal_acc", "balanced_accuracy"}:
        from sklearn.metrics import balanced_accuracy_score

        preds = probs.argmax(axis=1)
        return float(balanced_accuracy_score(labels, preds))

    raise ValueError(f"Unknown metric: {metric}")


def paired_bootstrap_delta_ci(
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    labels: np.ndarray,
    metric: str,
    num_classes: int,
    n_bins: int,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Paired bootstrap for delta = metric(A) - metric(B), resampling indices with replacement.
    Returns mean delta, 95% percentile CI, and a simple two-sided p-value estimate.
    """
    rng = np.random.default_rng(int(seed))
    n = int(labels.shape[0])

    probs_a = softmax_np(logits_a)
    probs_b = softmax_np(logits_b)

    deltas = np.zeros(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n, endpoint=False)
        la = labels[idx]
        pa = probs_a[idx]
        pb = probs_b[idx]
        deltas[i] = _metric_from_probs(
            la, pa, metric, num_classes, n_bins
        ) - _metric_from_probs(la, pb, metric, num_classes, n_bins)

    lo = float(np.percentile(deltas, 2.5))
    hi = float(np.percentile(deltas, 97.5))
    mean = float(deltas.mean())

    # Two-sided p-value: proportion of bootstrap deltas on the opposite side of 0
    p_pos = float((deltas >= 0.0).mean())
    p_neg = float((deltas <= 0.0).mean())
    p_two = float(2.0 * min(p_pos, p_neg))
    p_two = float(min(1.0, max(0.0, p_two)))

    return {
        "metric": metric,
        "delta_mean": mean,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "p_value_two_sided": p_two,
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


# -----------------------------------------------------------------------------
# Plotting helpers (optional but useful for paper figures)
# -----------------------------------------------------------------------------
def plot_reliability_diagram(
    labels: np.ndarray,
    logits: np.ndarray,
    n_bins: int,
    title: str,
    out_path: Path,
) -> None:
    """
    Saves a standard reliability diagram: accuracy vs confidence.
    """
    import matplotlib.pyplot as plt

    probs = softmax_np(logits)
    bins = calibration_bins(labels, probs, n_bins)
    bin_conf = np.array(bins["bin_conf"], dtype=np.float64)
    bin_acc = np.array(bins["bin_acc"], dtype=np.float64)
    bin_counts = np.array(bins["bin_counts"], dtype=np.int64)

    # Mask empty bins
    mask = bin_counts > 0
    bin_conf = bin_conf[mask]
    bin_acc = bin_acc[mask]

    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray", label="Perfect calibration")
    ax.plot(bin_conf, bin_acc, marker="o", linewidth=2, label="Model")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def _plot_calib_size_curve(
    rows: List[Dict[str, Any]], out_path: Path, title: str
) -> None:
    """
    FIX: Corrected dual-axis plotting with proper colors and combined legend.
    """
    import matplotlib.pyplot as plt

    rows = [r for r in rows if not math.isnan(float(r.get("val_auprc", float("nan"))))]
    if not rows:
        return

    x = np.array([r["calib_size"] for r in rows], dtype=np.float64)
    y1 = np.array([r.get("val_auprc", np.nan) for r in rows], dtype=np.float64)
    y2 = np.array([r.get("val_ece", np.nan) for r in rows], dtype=np.float64)

    ensure_dir(out_path.parent)

    fig, ax1 = plt.subplots(figsize=(7.5, 5.5))

    # Primary axis: AUPRC (blue)
    color1 = "tab:blue"
    ax1.set_xlabel("Calibration set size (images, log scale)")
    ax1.set_xscale("log")
    ax1.set_ylabel("AUPRC", color=color1)
    line1 = ax1.plot(
        x, y1, marker="o", linewidth=2, color=color1, label="Val AUPRC"
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0.0, 1.0)

    # Secondary axis: ECE (red)
    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("ECE", color=color2)
    line2 = ax2.plot(
        x, y2, marker="s", linewidth=2, color=color2, label="Val ECE"
    )
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0.0, max(0.5, np.nanmax(y2) * 1.1))

    # Combined legend
    lines = line1 + line2
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="center right")

    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def plot_reliability_comparison(
    labels: np.ndarray,
    logits_dict: Dict[str, np.ndarray],
    n_bins: int,
    title: str,
    out_path: Path,
) -> None:
    """
    Plot multiple reliability diagrams on the same axes for comparison.
    
    Args:
        labels: Ground truth labels
        logits_dict: Dict mapping regime name to logits array, e.g. {"FP32": logits_fp32, "PTQ": logits_ptq}
        n_bins: Number of calibration bins
        title: Plot title
        out_path: Output path for PNG
    """
    import matplotlib.pyplot as plt

    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(7, 6))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray", label="Perfect")

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    markers = ["o", "s", "^", "D", "v"]

    for i, (name, logits) in enumerate(logits_dict.items()):
        probs = softmax_np(logits)
        bins = calibration_bins(labels, probs, n_bins)
        bin_conf = np.array(bins["bin_conf"], dtype=np.float64)
        bin_acc = np.array(bins["bin_acc"], dtype=np.float64)
        bin_counts = np.array(bins["bin_counts"], dtype=np.int64)

        mask = bin_counts > 0
        ax.plot(
            bin_conf[mask],
            bin_acc[mask],
            marker=markers[i % len(markers)],
            linewidth=2,
            color=colors[i % len(colors)],
            label=name,
            markersize=8,
        )

    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------
def cmd_temperature_scaling(args: argparse.Namespace) -> None:
    calib_logits = load_npy(Path(args.calib_logits))
    calib_labels = load_npy(Path(args.calib_labels)).astype(np.int64)

    eval_logits = load_npy(Path(args.eval_logits))
    eval_labels = load_npy(Path(args.eval_labels)).astype(np.int64)

    num_classes = int(args.num_classes)
    n_bins = int(args.n_bins)

    ts = fit_temperature(
        calib_logits=calib_logits,
        calib_labels=calib_labels,
        max_iter=int(args.max_iter),
        lr=float(args.lr),
        device="cuda" if (args.ts_device == "cuda" and torch.cuda.is_available()) else "cpu",
    )

    eval_metrics_before = compute_all_metrics(
        eval_logits, eval_labels, num_classes=num_classes, n_bins=n_bins
    )
    eval_logits_scaled = apply_temperature(eval_logits, ts.temperature)
    eval_metrics_after = compute_all_metrics(
        eval_logits_scaled, eval_labels, num_classes=num_classes, n_bins=n_bins
    )

    out = {
        "temperature": float(ts.temperature),
        "calib_nll_before": float(ts.nll_before),
        "calib_nll_after": float(ts.nll_after),
        "eval_metrics_before": eval_metrics_before,
        "eval_metrics_after": eval_metrics_after,
    }

    if args.save_scaled_logits:
        save_npy(Path(args.save_scaled_logits), eval_logits_scaled)

    if args.reliability_png:
        plot_reliability_diagram(
            labels=eval_labels,
            logits=eval_logits_scaled,
            n_bins=n_bins,
            title=f"Reliability after TS (T={ts.temperature:.3g})",
            out_path=Path(args.reliability_png),
        )

    if args.out_json:
        save_json(Path(args.out_json), out)
        print(f"Saved: {args.out_json}")
    else:
        print(json.dumps(out, indent=2))


def cmd_paired_bootstrap(args: argparse.Namespace) -> None:
    logits_a = load_npy(Path(args.a_logits))
    logits_b = load_npy(Path(args.b_logits))
    labels = load_npy(Path(args.labels)).astype(np.int64)

    num_classes = int(args.num_classes)
    n_bins = int(args.n_bins)

    res = paired_bootstrap_delta_ci(
        logits_a=logits_a,
        logits_b=logits_b,
        labels=labels,
        metric=str(args.metric),
        num_classes=num_classes,
        n_bins=n_bins,
        n_boot=int(args.n_boot),
        seed=int(args.seed),
    )

    if args.out_json:
        save_json(Path(args.out_json), res)
        print(f"Saved: {args.out_json}")
    else:
        print(json.dumps(res, indent=2))


def _infer_fp32_ckpt_path(dataset: str, model: str, fold: int) -> Path:
    ckpt_dir = DEFAULT_ROOT / "checkpoints_quant" / dataset / model
    return ckpt_dir / f"fold{fold}_fp32.pt"


def cmd_calib_size_ablation(args: argparse.Namespace) -> None:
    set_local_caches(DEFAULT_ROOT)

    config = load_config(args.config)
    cfg_base_dir = (
        Path(args.config).resolve().parent if args.config else (REPO_ROOT / "configs")
    )
    normalize_paths_in_config(config, cfg_base_dir)

    results_root = resolve_results_dir(config)
    ensure_dir(results_root)

    dataset = str(args.dataset)
    model_key = str(args.model)
    backend = str(args.backend)
    device = get_device(str(args.device))

    # For brainmri, force fold=0 to align with your runners
    fold = int(args.fold)
    if dataset == "brainmri":
        fold = 0

    seed = (
        int(args.seed)
        if args.seed is not None
        else int(config.get("experiment", {}).get("random_seed", 42))
    )
    deterministic = bool(config.get("experiment", {}).get("deterministic", True))
    set_determinism(seed, deterministic=deterministic)

    num_classes = int(config["datasets"][dataset]["num_classes"])
    n_bins = int(config.get("evaluation", {}).get("calibration", {}).get("n_bins", 15))

    # Prepare output dir
    out_dir = (
        results_root
        / "extra_analyses"
        / "ptq_calib_size"
        / dataset
        / model_key
        / f"backend_{backend}"
        / f"fold{fold}"
    )
    ensure_dir(out_dir)

    # Data and splits (same logic as runners)
    image_paths, labels, _, _ = load_dataset(dataset, config)
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found for dataset={dataset}. Check config paths.")

    cv_cfg = config.get("cross_validation", {})
    test_ratio = float(cv_cfg.get("test_ratio", cv_cfg.get("holdout_test_ratio", 0.15)))
    test_indices = None

    if dataset in {"chestxray", "skincancer"}:
        splits, test_indices = _create_splits_with_holdout(
            image_paths, labels, config, test_ratio=test_ratio, dataset=dataset
        )
    else:
        splits = _create_splits(image_paths, labels, config, dataset=dataset)

    if fold < 0 or fold >= len(splits):
        raise ValueError(f"--fold must be in [0, {len(splits)-1}], got {fold}")
    train_idx, val_idx = splits[fold]
    train_loader, val_loader = _create_loaders(
        image_paths, labels, train_idx, val_idx, config, dataset
    )
    test_loader, test_info = build_test_loader(
        dataset, config, image_paths, labels, test_indices
    )

    # Load FP32 checkpoint
    ckpt_path = (
        Path(args.fp32_ckpt) if args.fp32_ckpt else _infer_fp32_ckpt_path(dataset, model_key, fold)
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"FP32 checkpoint not found at: {ckpt_path}. "
            f"Either run run_precision_experiment.py with --precision fp32 or pass --fp32_ckpt."
        )

    base_model = create_model(model_key, num_classes=num_classes, config=config, device=device)
    _ = count_parameters(base_model)
    load_checkpoint(ckpt_path, base_model, optimizer=None)

    # Baseline eval (optional, helps interpret ablation curves)
    _sync(device)
    fp32_val_logits, fp32_val_labels = evaluate_logits_labels(base_model, val_loader, device)
    _sync(device)
    fp32_val_metrics = compute_all_metrics(
        fp32_val_logits, fp32_val_labels, num_classes=num_classes, n_bins=n_bins
    )

    fp32_test_metrics = None
    fp32_test_logits = None
    fp32_test_labels = None
    if test_loader is not None:
        fp32_test_logits, fp32_test_labels = evaluate_logits_labels(
            base_model, test_loader, device
        )
        fp32_test_metrics = compute_all_metrics(
            fp32_test_logits, fp32_test_labels, num_classes=num_classes, n_bins=n_bins
        )

    # Run ablation points
    torch.backends.quantized.engine = backend

    calib_sizes = [
        int(x) for x in str(args.calib_sizes).split(",") if str(x).strip()
    ]
    calib_sizes = sorted(list(dict.fromkeys([c for c in calib_sizes if c > 0])))

    rows: List[Dict[str, Any]] = []
    detail: List[Dict[str, Any]] = []

    for calib_size in calib_sizes:
        print("=" * 70)
        print(
            f"PTQ calib-size ablation: dataset={dataset} model={model_key} "
            f"fold={fold} backend={backend} calib_size={calib_size}"
        )

        # CPU model copy for PTQ
        model_cpu = copy.deepcopy(base_model).cpu()
        model_cpu.eval()

        # Calibration subset
        limited_train = LimitedSamplesLoader(train_loader, max_samples=calib_size)
        calib_loader = CalibLoaderWrapper(limited_train)
        num_calib_batches = max(1, len(limited_train))

        # Convert
        t0 = _now()
        qmodel = quantize_int8_ptq(
            model=model_cpu,
            model_key=model_key,
            calib_loader=calib_loader,
            num_calib_batches=int(num_calib_batches),
        )
        convert_time_sec = float(_now() - t0)

        # Eval on VAL (CPU)
        t1 = _now()
        int8_val_logits, int8_val_labels = evaluate_logits_labels(
            qmodel, val_loader, torch.device("cpu")
        )
        val_eval_time_sec = float(_now() - t1)
        int8_val_metrics = compute_all_metrics(
            int8_val_logits, int8_val_labels, num_classes=num_classes, n_bins=n_bins
        )

        # Benchmark (CPU)
        bench = None
        if int(args.bench_batches) > 0:
            bench = benchmark_inference(
                qmodel,
                val_loader,
                torch.device("cpu"),
                num_warmup_batches=int(args.bench_warmup),
                num_meas_batches=int(args.bench_batches),
            )

        # Optional TEST eval
        int8_test_metrics = None
        ts_on_test = None
        if test_loader is not None:
            int8_test_logits, int8_test_labels = evaluate_logits_labels(
                qmodel, test_loader, torch.device("cpu")
            )
            int8_test_metrics = compute_all_metrics(
                int8_test_logits, int8_test_labels, num_classes=num_classes, n_bins=n_bins
            )

            # Temperature scaling: fit on VAL, report on TEST
            if args.do_temperature_scaling:
                ts = fit_temperature(
                    calib_logits=int8_val_logits,
                    calib_labels=int8_val_labels,
                    max_iter=int(args.ts_max_iter),
                    lr=float(args.ts_lr),
                    device="cpu",
                )
                int8_test_logits_scaled = apply_temperature(int8_test_logits, ts.temperature)
                int8_test_metrics_scaled = compute_all_metrics(
                    int8_test_logits_scaled,
                    int8_test_labels,
                    num_classes=num_classes,
                    n_bins=n_bins,
                )
                ts_on_test = {
                    "temperature": float(ts.temperature),
                    "calib_nll_before": float(ts.nll_before),
                    "calib_nll_after": float(ts.nll_after),
                    "test_metrics_after_ts": int8_test_metrics_scaled,
                }

        # Save per-point artifacts
        point = {
            "dataset": dataset,
            "model": model_key,
            "fold": int(fold),
            "backend": backend,
            "calib_size": int(calib_size),
            "num_calib_batches": int(num_calib_batches),
            "convert_time_sec": float(convert_time_sec),
            "val_eval_time_sec": float(val_eval_time_sec),
            "fp32_val_metrics": fp32_val_metrics,
            "int8_val_metrics": int8_val_metrics,
            "benchmark_val": bench,
            "test_info": test_info,
            "fp32_test_metrics": fp32_test_metrics,
            "int8_test_metrics": int8_test_metrics,
            "temp_scaling_test": ts_on_test,
            "env": get_env_metadata(),
        }
        detail.append(point)
        save_json(out_dir / f"calibsize_{calib_size}.json", point)

        # Minimal row for CSV
        row = {
            "dataset": dataset,
            "model": model_key,
            "fold": int(fold),
            "backend": backend,
            "calib_size": int(calib_size),
            "convert_time_sec": float(convert_time_sec),
            "val_auc": float(int8_val_metrics.get("auc_roc_ovr_macro_sklearn", np.nan)),
            "val_auprc": float(int8_val_metrics.get("auprc_ovr_macro_sklearn", np.nan)),
            "val_ece": float(int8_val_metrics.get("ece", np.nan)),
            "val_brier": float(int8_val_metrics.get("brier", np.nan)),
            "val_p50_ms": (
                float(bench.get("p50_batch_sec", np.nan) * 1000.0)
                if isinstance(bench, dict)
                else float("nan")
            ),
            "val_thr_sps": (
                float(bench.get("throughput_samples_per_sec", np.nan))
                if isinstance(bench, dict)
                else float("nan")
            ),
        }
        if fp32_test_metrics is not None and int8_test_metrics is not None:
            row.update(
                {
                    "test_auc": float(
                        int8_test_metrics.get("auc_roc_ovr_macro_sklearn", np.nan)
                    ),
                    "test_auprc": float(
                        int8_test_metrics.get("auprc_ovr_macro_sklearn", np.nan)
                    ),
                    "test_ece": float(int8_test_metrics.get("ece", np.nan)),
                    "test_brier": float(int8_test_metrics.get("brier", np.nan)),
                }
            )
        if ts_on_test is not None:
            tsm = ts_on_test["test_metrics_after_ts"]
            row.update(
                {
                    "test_ece_after_ts": float(tsm.get("ece", np.nan)),
                    "test_brier_after_ts": float(tsm.get("brier", np.nan)),
                    "test_auprc_after_ts": float(tsm.get("auprc_ovr_macro_sklearn", np.nan)),
                    "ts_temperature": float(ts_on_test.get("temperature", np.nan)),
                }
            )
        rows.append(row)

    # Aggregate outputs
    save_csv(out_dir / "summary.csv", rows)
    save_json(
        out_dir / "all_points.json",
        {
            "dataset": dataset,
            "model": model_key,
            "fold": int(fold),
            "backend": backend,
            "calib_sizes": calib_sizes,
            "fp32_val_metrics": fp32_val_metrics,
            "fp32_test_metrics": fp32_test_metrics,
            "points": detail,
            "env": get_env_metadata(),
        },
    )

    # Optional: make a simple curve plot (calib_size vs metric)
    if args.plot_png:
        _plot_calib_size_curve(
            rows, out_dir / "calib_size_curve.png", title=f"{dataset} {model_key} PTQ ({backend})"
        )

    print(f"Saved ablation results to: {out_dir}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extra analyses for CHIL quantization paper.")

    sub = p.add_subparsers(dest="cmd", required=True)

    # 1) Temperature scaling
    p_ts = sub.add_parser(
        "temperature_scaling",
        help="Fit temperature on calib logits and evaluate on eval logits.",
    )
    p_ts.add_argument("--calib_logits", required=True)
    p_ts.add_argument("--calib_labels", required=True)
    p_ts.add_argument("--eval_logits", required=True)
    p_ts.add_argument("--eval_labels", required=True)
    p_ts.add_argument("--num_classes", type=int, required=True)
    p_ts.add_argument("--n_bins", type=int, default=15)
    p_ts.add_argument("--max_iter", type=int, default=100)
    p_ts.add_argument("--lr", type=float, default=0.05)
    p_ts.add_argument(
        "--ts_device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for fitting temperature.",
    )
    p_ts.add_argument(
        "--save_scaled_logits",
        default=None,
        help="Optional path to save scaled eval logits (.npy).",
    )
    p_ts.add_argument(
        "--reliability_png",
        default=None,
        help="Optional path to save reliability diagram PNG.",
    )
    p_ts.add_argument("--out_json", default=None, help="Optional path to save JSON output.")
    p_ts.set_defaults(func=cmd_temperature_scaling)

    # 2) Paired bootstrap
    p_bs = sub.add_parser(
        "paired_bootstrap",
        help="Paired bootstrap CI for delta metric between two logits sets.",
    )
    p_bs.add_argument("--a_logits", required=True, help="Logits for method A (.npy)")
    p_bs.add_argument("--b_logits", required=True, help="Logits for method B (.npy)")
    p_bs.add_argument("--labels", required=True, help="Labels (.npy)")
    p_bs.add_argument(
        "--metric",
        required=True,
        choices=["auprc", "auc", "ece", "brier", "accuracy", "balanced_accuracy"],
    )
    p_bs.add_argument("--num_classes", type=int, required=True)
    p_bs.add_argument("--n_bins", type=int, default=15)
    p_bs.add_argument("--n_boot", type=int, default=2000)
    p_bs.add_argument("--seed", type=int, default=42)
    p_bs.add_argument("--out_json", default=None)
    p_bs.set_defaults(func=cmd_paired_bootstrap)

    # 3) PTQ calib-size ablation
    p_ab = sub.add_parser(
        "calib_size_ablation",
        help="Run INT8 PTQ with varying calibration-set sizes.",
    )
    p_ab.add_argument(
        "--dataset", required=True, choices=["brainmri", "chestxray", "skincancer"]
    )
    p_ab.add_argument("--model", required=True)
    p_ab.add_argument("--fold", type=int, default=0)
    p_ab.add_argument("--backend", default="fbgemm", choices=["fbgemm", "qnnpack"])
    p_ab.add_argument(
        "--calib_sizes",
        required=True,
        help="Comma-separated sizes, e.g. 16,32,64,128,256",
    )
    p_ab.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p_ab.add_argument("--seed", type=int, default=None)
    p_ab.add_argument("--config", type=str, default=None)
    p_ab.add_argument(
        "--fp32_ckpt",
        type=str,
        default=None,
        help="Optional explicit fp32 checkpoint path.",
    )
    p_ab.add_argument("--bench_warmup", type=int, default=5)
    p_ab.add_argument("--bench_batches", type=int, default=20)
    p_ab.add_argument(
        "--plot_png",
        action="store_true",
        help="Also write a simple calib-size curve PNG.",
    )
    p_ab.add_argument(
        "--do_temperature_scaling",
        action="store_true",
        help="Fit temperature on VAL and report TS metrics on TEST.",
    )
    p_ab.add_argument("--ts_max_iter", type=int, default=100)
    p_ab.add_argument("--ts_lr", type=float, default=0.05)
    p_ab.set_defaults(func=cmd_calib_size_ablation)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()