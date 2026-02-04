from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

# Prefer torch.ao.quantization when available
try:
    import torch.ao.quantization as quant
except Exception:
    import torch.quantization as quant

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

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
except ImportError:
    # Fallback to data_loader_helper if utils.data_loader not available
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
from utils.train_eval import (
    EarlyStopper,
    build_optimizer,
    load_checkpoint,
    save_checkpoint,
    set_determinism,
    train_one_epoch,
)

DEFAULT_ROOT = Path(os.environ.get("EXPERIMENT_ROOT", str(REPO_ROOT / "runs")))
# -------------------------
# General helpers
# -------------------------
def set_local_caches(root: Path) -> None:
    cache_root = root / ".cache"
    (cache_root / "torch").mkdir(parents=True, exist_ok=True)
    (cache_root / "hf").mkdir(parents=True, exist_ok=True)
    (cache_root / "xdg").mkdir(parents=True, exist_ok=True)
    (cache_root / "timm").mkdir(parents=True, exist_ok=True)

    os.environ["TORCH_HOME"] = str(cache_root / "torch")
    os.environ["HF_HOME"] = str(cache_root / "hf")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_root / "hf" / "hub")
    # TRANSFORMERS_CACHE is deprecated in newer Transformers; HF_HOME covers it.
    os.environ["XDG_CACHE_HOME"] = str(cache_root / "xdg")
    os.environ["TIMM_HOME"] = str(cache_root / "timm")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    cfg_path = Path(config_path) if config_path else (REPO_ROOT / "configs" / "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _expand_and_resolve_path(p: str, base_dir: Path) -> str:
    """Expand ~ and $VARS, and make relative paths absolute relative to base_dir."""
    v = os.path.expandvars(os.path.expanduser(p))
    pp = Path(v)
    if not pp.is_absolute():
        pp = (base_dir / pp).resolve()
    return str(pp)


def _normalize_paths_in_obj(obj: Any, base_dir: Path) -> Any:
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, str) and any(tok in k.lower() for tok in ["path", "dir", "root", "folder"]):
                obj[k] = _expand_and_resolve_path(v, base_dir)
            else:
                obj[k] = _normalize_paths_in_obj(v, base_dir)
        return obj
    if isinstance(obj, list):
        return [_normalize_paths_in_obj(v, base_dir) for v in obj]
    return obj


def normalize_paths_in_config(config: Dict[str, Any], base_dir: Path) -> None:
    """In-place normalization of path-like fields in config."""
    _normalize_paths_in_obj(config, base_dir)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_npy(path: Path, arr: np.ndarray) -> None:
    ensure_dir(path.parent)
    np.save(str(path), arr)


def get_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _now() -> float:
    return time.perf_counter()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def load_dataset(dataset: str, config: Dict[str, Any]) -> Tuple[list, list, list, list]:
    """Load dataset and return (train_paths, train_labels, test_paths, test_labels).
    
    Handles both old (3-value) and new (4-value) return signatures.
    """
    if dataset == "brainmri":
        result = load_brainmri_dataset(config)
    elif dataset == "chestxray":
        result = load_chestxray_dataset(config)
    elif dataset == "skincancer":
        result = load_skincancer_dataset(config)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Handle both old (3-value) and new (4-value) return signatures
    if len(result) == 3:
        # Old signature: (image_paths, labels, class_names) or similar
        return result[0], result[1], [], []
    elif len(result) == 4:
        # New signature: (train_paths, train_labels, test_paths, test_labels)
        return result
    else:
        raise ValueError(f"Unexpected return value from dataset loader: got {len(result)} values")


def resolve_results_dir(config: Dict[str, Any]) -> Path:
    # Force a stable, repo-consistent layout regardless of YAML settings.
    # Existing analysis assumes everything lives under DEFAULT_ROOT/results.
    return DEFAULT_ROOT / "results"


def _create_splits(image_paths: list, labels: list, config: Dict[str, Any], dataset: str = None):
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
    dataset: str = None,
):
    """Version-tolerant wrapper for CV+holdout splitting."""
    try:
        return create_cv_splits_with_holdout(image_paths, labels, config, test_ratio=test_ratio, dataset=dataset)
    except TypeError:
        try:
            return create_cv_splits_with_holdout(image_paths, labels, config, test_ratio=test_ratio)
        except TypeError:
            return create_cv_splits_with_holdout(image_paths, labels, config, test_ratio)


def build_test_loader(
    dataset: str,
    config: Dict[str, Any],
    image_paths: list,
    labels: list,
    test_indices: Optional[List[int]],
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Create a test loader when possible.

    - brainmri: uses separate Testing folder
    - chestxray/skincancer: uses holdout indices
    """
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
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT)).decode().strip()
        md["git_commit"] = commit
    except Exception:
        md["git_commit"] = None
    return md


# -------------------------
# Metrics helpers (same as precision script)
# -------------------------
def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
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
        from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

        out["auc_roc_ovr_macro_sklearn"] = float(
            roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        )

        y_onehot = np.eye(num_classes, dtype=np.float32)[labels.astype(int)]
        out["auprc_ovr_macro_sklearn"] = float(
            average_precision_score(y_onehot, probs, average="macro")
        )

        out["log_loss_sklearn"] = float(log_loss(labels, probs, labels=list(range(num_classes))))
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


# -------------------------
# Eval + Benchmark
# -------------------------
@torch.no_grad()
def evaluate_logits_labels(model: torch.nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
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
    mean_batch_sec = float(arr.mean())
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    throughput = float(n_samples / arr.sum())

    return {
        "device": str(device),
        "num_batches": int(n_batches),
        "num_samples": int(n_samples),
        "mean_batch_sec": mean_batch_sec,
        "p50_batch_sec": p50,
        "p90_batch_sec": p90,
        "throughput_samples_per_sec": throughput,
    }


# -------------------------
# QAT plumbing
# -------------------------
class QuantizableModelWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.quant = quant.QuantStub()
        self.model = model
        self.dequant = quant.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


def has_quant_stubs(model: nn.Module) -> bool:
    return hasattr(model, "quant") and hasattr(model, "dequant")


def _collect_fuse_patterns_in_sequential(seq: nn.Sequential, prefix: str) -> List[List[str]]:
    patterns: List[List[str]] = []
    keys = list(seq._modules.keys())
    i = 0
    while i < len(keys):
        k1 = keys[i]
        m1 = seq._modules[k1]
        n1 = f"{prefix}.{k1}" if prefix else k1

        # Conv2d + BN (+ ReLU)
        if isinstance(m1, nn.Conv2d) and i + 1 < len(keys):
            k2 = keys[i + 1]
            m2 = seq._modules[k2]
            n2 = f"{prefix}.{k2}" if prefix else k2
            if isinstance(m2, nn.BatchNorm2d):
                if i + 2 < len(keys):
                    k3 = keys[i + 2]
                    m3 = seq._modules[k3]
                    n3 = f"{prefix}.{k3}" if prefix else k3
                    if isinstance(m3, (nn.ReLU, nn.ReLU6)):
                        patterns.append([n1, n2, n3])
                        i += 3
                        continue
                patterns.append([n1, n2])
                i += 2
                continue

        i += 1
    return patterns


def collect_fuse_patterns(model: nn.Module, prefix: str = "") -> List[List[str]]:
    patterns: List[List[str]] = []
    for name, child in model.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Sequential):
            patterns.extend(_collect_fuse_patterns_in_sequential(child, child_prefix))
        patterns.extend(collect_fuse_patterns(child, child_prefix))
    return patterns


def maybe_fuse_model(model: nn.Module) -> None:
    inner = model.model if isinstance(model, QuantizableModelWrapper) else model
    if hasattr(inner, "fuse_model") and callable(getattr(inner, "fuse_model")):
        try:
            inner.fuse_model()
            return
        except Exception:
            pass

    patterns = collect_fuse_patterns(model)
    if not patterns:
        return
    try:
        quant.fuse_modules(model, patterns, inplace=True)
    except Exception:
        return


def freeze_bn_stats_if_available(model: nn.Module) -> bool:
    try:
        fn = torch.nn.intrinsic.qat.freeze_bn_stats  # type: ignore[attr-defined]
        model.apply(fn)
        return True
    except Exception:
        return False


def disable_observers(model: nn.Module) -> None:
    model.apply(quant.disable_observer)


def prepare_model_for_qat(model: nn.Module, backend: str, device: torch.device) -> nn.Module:
    torch.backends.quantized.engine = backend

    if not has_quant_stubs(model):
        model = QuantizableModelWrapper(model)

    model.to(device)
    model.train()

    model.qconfig = quant.get_default_qat_qconfig(backend)
    maybe_fuse_model(model)
    model_prepared = quant.prepare_qat(model, inplace=False)
    return model_prepared


def convert_to_int8(model_qat: nn.Module, backend: str) -> nn.Module:
    torch.backends.quantized.engine = backend
    model_qat.eval()
    model_qat = model_qat.cpu()
    with torch.no_grad():
        q = quant.convert(model_qat, inplace=False)
    return q


# -------------------------
# Main
# -------------------------
def main() -> None:
    set_local_caches(DEFAULT_ROOT)

    p = argparse.ArgumentParser(description="QAT runner with paper-ready metrics and timing.")
    p.add_argument("--dataset", required=True, choices=["brainmri", "chestxray", "skincancer"])
    p.add_argument("--model", required=True)
    p.add_argument("--fold", type=int, default=0, help="0-indexed fold id (e.g., 0..4 for 5-fold CV). Ignored for brainmri.")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--backend", default="fbgemm", choices=["fbgemm", "qnnpack"])
    p.add_argument("--fp32_epochs", type=int, default=10)
    p.add_argument("--qat_epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", action="store_true", help="Reuse existing checkpoints if available")
    p.add_argument("--skip_if_done", action="store_true", help="Skip if output JSON already exists")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--save_logits", action="store_true")
    p.add_argument("--bench_warmup", type=int, default=5)
    p.add_argument("--bench_batches", type=int, default=20)
    args = p.parse_args()
    
    # For brainmri, force fold=0 (no CV, just train/val split)
    if args.dataset == "brainmri":
        args.fold = 0

    config = load_config(args.config)
    cfg_base_dir = (Path(args.config).resolve().parent if args.config else (REPO_ROOT / "configs"))
    normalize_paths_in_config(config, cfg_base_dir)

    results_root = resolve_results_dir(config)
    checkpoints_root = DEFAULT_ROOT / "checkpoints_quant"
    ensure_dir(results_root)
    ensure_dir(checkpoints_root)

    # Keep the original on-disk layout (results/qat/...) and
    # disambiguate backends via the filename to prevent overwrites.
    out_dir = results_root / "qat" / args.dataset / args.model
    ckpt_dir = checkpoints_root / args.dataset / args.model
    ensure_dir(out_dir)
    ensure_dir(ckpt_dir)


    # Include backend in the filename as an extra guard against accidental overwrites.
    out_path = out_dir / f"fold{args.fold}_{args.backend}.json"

    # Avoid checkpoint collisions when running multiple backends in parallel.
    ckpt_fp32 = ckpt_dir / f"fold{args.fold}_fp32_{args.backend}.pt"
    ckpt_qat = ckpt_dir / f"fold{args.fold}_qat_{args.backend}.pt"

    if args.skip_if_done and out_path.exists():
        print(f"Skip: output already exists: {out_path}")
        return

    seed = int(args.seed) if args.seed is not None else int(config.get("experiment", {}).get("random_seed", 42))
    deterministic = bool(config.get("experiment", {}).get("deterministic", True))
    set_determinism(seed, deterministic=deterministic)

    device = get_device(args.device)

    # Data
    image_paths, labels, _, _ = load_dataset(args.dataset, config)
    if len(image_paths) == 0:
        raise RuntimeError(
            f"No images found for dataset={args.dataset}. Check config paths. PWD={os.getcwd()}"
        )

    num_classes = int(config["datasets"][args.dataset]["num_classes"])

    # -------------------------
    # Proper split: train/val for early stopping, plus held-out test metrics
    #   - brainmri: separate Testing folder (test is constant across folds)
    #   - chestxray/skincancer: hold out a test split, then run CV on the remaining trainval
    # -------------------------
    cv_cfg = config.get("cross_validation", {})
    test_ratio = float(cv_cfg.get("test_ratio", cv_cfg.get("holdout_test_ratio", 0.15)))
    test_indices = None

    if args.dataset in {"chestxray", "skincancer"}:
        splits, test_indices = _create_splits_with_holdout(image_paths, labels, config, test_ratio=test_ratio, dataset=args.dataset)
    else:
        splits = _create_splits(image_paths, labels, config, dataset=args.dataset)

    if args.fold < 0 or args.fold >= len(splits):
        raise ValueError(f"--fold must be in [0, {len(splits)-1}], got {args.fold}")
    train_idx, val_idx = splits[args.fold]
    train_loader, val_loader = _create_loaders(image_paths, labels, train_idx, val_idx, config, args.dataset)
    test_loader, test_split_info = build_test_loader(args.dataset, config, image_paths, labels, test_indices)

    # Model
    model = create_model(args.model, num_classes=num_classes, config=config, device=device)
    n_params = int(count_parameters(model))

    n_bins = int(config.get("evaluation", {}).get("calibration", {}).get("n_bins", 15))

    print("=" * 70)
    print("QAT RUN")
    print("=" * 70)
    print(f"dataset={args.dataset} model={args.model} fold={args.fold}")
    print(f"train_device={device} backend={args.backend} seed={seed}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"num_classes={num_classes} num_params={n_params:,}")
    print(f"fp32_epochs={args.fp32_epochs} qat_epochs={args.qat_epochs}")
    print("=" * 70)

    job_t0 = _now()

    # -------------------------
    # Phase 1: FP32 pretrain
    # -------------------------
    fp32_history: List[Dict[str, Any]] = []
    fp32_best_acc = -1e9
    fp32_train_time_sum = 0.0
    fp32_eval_time_sum = 0.0
    start_epoch_fp32 = 0

    fp32_optimizer = build_optimizer(model, config)

    if args.resume and ckpt_fp32.exists():
        ckpt = load_checkpoint(ckpt_fp32, model, fp32_optimizer)
        fp32_best_acc = float(ckpt.get("best_metric", -1e9))
        start_epoch_fp32 = int(ckpt.get("epoch", 0)) + 1
        print(f"Resumed FP32 from {ckpt_fp32} epoch={start_epoch_fp32} best_acc={fp32_best_acc:.6f}")

    fp32_epochs = int(args.fp32_epochs)
    
    # Only train if there are remaining epochs
    if start_epoch_fp32 < fp32_epochs:
        es_cfg = config.get("training", {}).get("early_stopping", {})
        stopper = EarlyStopper(
            patience=int(es_cfg.get("patience", 10)),
            min_delta=float(es_cfg.get("min_delta", 0.0)),
            mode="max",
        )
        # If resuming, we need to "catch up" the stopper
        if start_epoch_fp32 > 0:
            stopper.best = fp32_best_acc

        fp32_phase_t0 = _now()
        for epoch in range(start_epoch_fp32, fp32_epochs):
            epoch_t0 = _now()

            _sync(device)
            t_train0 = _now()
            train_loss = train_one_epoch(model, train_loader, fp32_optimizer, device, scaler=None)
            _sync(device)
            train_time_sec = _now() - t_train0

            _sync(device)
            t_eval0 = _now()
            val_logits, val_labels = evaluate_logits_labels(model, val_loader, device)
            _sync(device)
            eval_time_sec = _now() - t_eval0

            m = compute_all_metrics(val_logits, val_labels, num_classes=num_classes, n_bins=n_bins)
            m["epoch"] = int(epoch)
            m["train_loss"] = float(train_loss)
            m["train_time_sec"] = float(train_time_sec)
            m["eval_time_sec"] = float(eval_time_sec)
            m["epoch_time_sec"] = float(_now() - epoch_t0)
            fp32_history.append(m)

            fp32_train_time_sum += float(train_time_sec)
            fp32_eval_time_sum += float(eval_time_sec)

            val_acc = float(m.get("accuracy", 0.0))
            if val_acc > fp32_best_acc:
                fp32_best_acc = val_acc
                save_checkpoint(ckpt_fp32, model, fp32_optimizer, epoch, fp32_best_acc)

            print(
                f"FP32 Epoch {epoch+1}/{fp32_epochs} loss={train_loss:.4f} acc={val_acc*100:.2f}% "
                f"train={train_time_sec:.2f}s eval={eval_time_sec:.2f}s"
            )

            if bool(es_cfg.get("enabled", True)) and stopper.step(val_acc):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        fp32_phase_time_sec = float(_now() - fp32_phase_t0)
        print(f"FP32 phase time: {fp32_phase_time_sec:.2f}s")
    else:
        print(f"FP32 training already complete (epoch {start_epoch_fp32}/{fp32_epochs})")

    if ckpt_fp32.exists():
        load_checkpoint(ckpt_fp32, model, optimizer=None)

    _sync(device)
    t_fp32_eval0 = _now()
    fp32_logits, fp32_labels = evaluate_logits_labels(model, val_loader, device)
    _sync(device)
    fp32_final_eval_time_sec = float(_now() - t_fp32_eval0)

    fp32_metrics = compute_all_metrics(fp32_logits, fp32_labels, num_classes=num_classes, n_bins=n_bins)
    fp32_bench = benchmark_inference(
        model, val_loader, device, num_warmup_batches=args.bench_warmup, num_meas_batches=args.bench_batches
    )

    save_preds_cfg = bool(config.get("experiment", {}).get("save_predictions", False))
    if args.save_logits or save_preds_cfg:
        save_npy(out_dir / f"fold{args.fold}_fp32_val_logits.npy", fp32_logits)
        save_npy(out_dir / f"fold{args.fold}_fp32_val_labels.npy", fp32_labels)

    # FP32 eval on held-out test (if available)
    fp32_test_metrics: Optional[Dict[str, Any]] = None
    fp32_test_eval_time_sec: Optional[float] = None
    if test_loader is not None:
        _sync(device)
        t_fp32_test0 = _now()
        fp32_test_logits, fp32_test_labels = evaluate_logits_labels(model, test_loader, device)
        _sync(device)
        fp32_test_eval_time_sec = float(_now() - t_fp32_test0)
        fp32_test_metrics = compute_all_metrics(fp32_test_logits, fp32_test_labels, num_classes=num_classes, n_bins=n_bins)

        if args.save_logits or save_preds_cfg:
            save_npy(out_dir / f"fold{args.fold}_fp32_test_logits.npy", fp32_test_logits)
            save_npy(out_dir / f"fold{args.fold}_fp32_test_labels.npy", fp32_test_labels)

    # -------------------------
    # Phase 2: QAT fine-tune
    # -------------------------
    model_qat = prepare_model_for_qat(model, backend=args.backend, device=device)
    qat_optimizer = build_optimizer(model_qat, config)

    qat_history: List[Dict[str, Any]] = []
    qat_best_acc = -1e9
    qat_train_time_sum = 0.0
    qat_eval_time_sum = 0.0
    start_epoch_qat = 0

    if args.resume and ckpt_qat.exists():
        ckpt = load_checkpoint(ckpt_qat, model_qat, qat_optimizer)
        qat_best_acc = float(ckpt.get("best_metric", -1e9))
        start_epoch_qat = int(ckpt.get("epoch", 0)) + 1
        print(f"Resumed QAT from {ckpt_qat} epoch={start_epoch_qat} best_acc={qat_best_acc:.6f}")

    qat_epochs = int(args.qat_epochs)
    
    # Only train if there are remaining epochs
    if start_epoch_qat < qat_epochs:
        es_cfg_qat = config.get("training", {}).get("early_stopping", {})
        qat_stopper = EarlyStopper(
            patience=int(es_cfg_qat.get("patience", 10)),
            min_delta=float(es_cfg_qat.get("min_delta", 0.0)),
            mode="max",
        )
        use_qat_es = bool(es_cfg_qat.get("enabled", True))
        
        # If resuming, we need to "catch up" the stopper
        if start_epoch_qat > 0:
            qat_stopper.best = qat_best_acc

        # In QAT, it is common to freeze BN stats and observers after some epochs
        freeze_bn_epoch = max(2, qat_epochs // 4)
        freeze_obs_epoch = max(freeze_bn_epoch + 1, qat_epochs // 2)

        did_freeze_bn = start_epoch_qat >= freeze_bn_epoch
        did_freeze_obs = start_epoch_qat >= freeze_obs_epoch
        
        # Apply freezing if resuming past those epochs
        if did_freeze_bn:
            freeze_bn_stats_if_available(model_qat)
        if did_freeze_obs:
            disable_observers(model_qat)

        qat_phase_t0 = _now()
        for epoch in range(start_epoch_qat, qat_epochs):
            epoch_t0 = _now()

            if (not did_freeze_bn) and epoch >= freeze_bn_epoch:
                did_freeze_bn = freeze_bn_stats_if_available(model_qat)

            if (not did_freeze_obs) and epoch >= freeze_obs_epoch:
                disable_observers(model_qat)
                did_freeze_obs = True

            _sync(device)
            t_train0 = _now()
            train_loss = train_one_epoch(model_qat, train_loader, qat_optimizer, device, scaler=None)
            _sync(device)
            train_time_sec = _now() - t_train0

            _sync(device)
            t_eval0 = _now()
            val_logits, val_labels = evaluate_logits_labels(model_qat, val_loader, device)
            _sync(device)
            eval_time_sec = _now() - t_eval0

            m = compute_all_metrics(val_logits, val_labels, num_classes=num_classes, n_bins=n_bins)
            m["epoch"] = int(epoch)
            m["train_loss"] = float(train_loss)
            m["train_time_sec"] = float(train_time_sec)
            m["eval_time_sec"] = float(eval_time_sec)
            m["epoch_time_sec"] = float(_now() - epoch_t0)
            qat_history.append(m)

            qat_train_time_sum += float(train_time_sec)
            qat_eval_time_sum += float(eval_time_sec)

            val_acc = float(m.get("accuracy", 0.0))
            if val_acc > qat_best_acc:
                qat_best_acc = val_acc
                save_checkpoint(ckpt_qat, model_qat, qat_optimizer, epoch, qat_best_acc)

            print(
                f"QAT Epoch {epoch+1}/{qat_epochs} loss={train_loss:.4f} acc={val_acc*100:.2f}% "
                f"train={train_time_sec:.2f}s eval={eval_time_sec:.2f}s"
            )

            if use_qat_es and qat_stopper.step(val_acc):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        qat_phase_time_sec = float(_now() - qat_phase_t0)
        print(f"QAT phase time: {qat_phase_time_sec:.2f}s")
    else:
        print(f"QAT training already complete (epoch {start_epoch_qat}/{qat_epochs})")

    if ckpt_qat.exists():
        load_checkpoint(ckpt_qat, model_qat, optimizer=None)

    _sync(device)
    t_qat_eval0 = _now()
    qat_fake_logits, qat_fake_labels = evaluate_logits_labels(model_qat, val_loader, device)
    _sync(device)
    qat_fake_eval_time_sec = float(_now() - t_qat_eval0)

    qat_fake_metrics = compute_all_metrics(qat_fake_logits, qat_fake_labels, num_classes=num_classes, n_bins=n_bins)
    qat_fake_bench = benchmark_inference(
        model_qat, val_loader, device, num_warmup_batches=args.bench_warmup, num_meas_batches=args.bench_batches
    )

    if args.save_logits or save_preds_cfg:
        save_npy(out_dir / f"fold{args.fold}_qat_fake_val_logits.npy", qat_fake_logits)
        save_npy(out_dir / f"fold{args.fold}_qat_fake_val_labels.npy", qat_fake_labels)

    # QAT fake-quant eval on held-out test (if available)
    qat_fake_test_metrics: Optional[Dict[str, Any]] = None
    qat_fake_test_eval_time_sec: Optional[float] = None
    if test_loader is not None:
        _sync(device)
        t_qat_test0 = _now()
        qat_fake_test_logits, qat_fake_test_labels = evaluate_logits_labels(model_qat, test_loader, device)
        _sync(device)
        qat_fake_test_eval_time_sec = float(_now() - t_qat_test0)
        qat_fake_test_metrics = compute_all_metrics(
            qat_fake_test_logits, qat_fake_test_labels, num_classes=num_classes, n_bins=n_bins
        )

        if args.save_logits or save_preds_cfg:
            save_npy(out_dir / f"fold{args.fold}_qat_fake_test_logits.npy", qat_fake_test_logits)
            save_npy(out_dir / f"fold{args.fold}_qat_fake_test_labels.npy", qat_fake_test_labels)

    # -------------------------
    # Phase 3: Convert to real INT8 and eval on CPU
    # -------------------------
    int8_convert_time_sec: Optional[float] = None
    int8_eval_time_sec: Optional[float] = None
    int8_metrics: Optional[Dict[str, Any]] = None
    int8_bench: Optional[Dict[str, Any]] = None
    int8_test_metrics: Optional[Dict[str, Any]] = None
    int8_test_eval_time_sec: Optional[float] = None
    int8_conversion_error: Optional[str] = None
    
    try:
        t_conv0 = _now()
        model_int8 = convert_to_int8(model_qat, backend=args.backend)
        int8_convert_time_sec = float(_now() - t_conv0)

        t_int8_eval0 = _now()
        int8_logits, int8_labels = evaluate_logits_labels(model_int8, val_loader, torch.device("cpu"))
        int8_eval_time_sec = float(_now() - t_int8_eval0)

        int8_metrics = compute_all_metrics(int8_logits, int8_labels, num_classes=num_classes, n_bins=n_bins)
        int8_bench = benchmark_inference(
            model_int8, val_loader, torch.device("cpu"),
            num_warmup_batches=args.bench_warmup,
            num_meas_batches=args.bench_batches,
        )

        if args.save_logits or save_preds_cfg:
            save_npy(out_dir / f"fold{args.fold}_int8_val_logits.npy", int8_logits)
            save_npy(out_dir / f"fold{args.fold}_int8_val_labels.npy", int8_labels)

        # INT8 eval on held-out test (if available)
        if test_loader is not None:
            t_int8_test0 = _now()
            int8_test_logits, int8_test_labels = evaluate_logits_labels(model_int8, test_loader, torch.device("cpu"))
            int8_test_eval_time_sec = float(_now() - t_int8_test0)
            int8_test_metrics = compute_all_metrics(
                int8_test_logits, int8_test_labels, num_classes=num_classes, n_bins=n_bins
            )

            if args.save_logits or save_preds_cfg:
                save_npy(out_dir / f"fold{args.fold}_int8_test_logits.npy", int8_test_logits)
                save_npy(out_dir / f"fold{args.fold}_int8_test_labels.npy", int8_test_labels)
                
    except (NotImplementedError, RuntimeError) as e:
        # This can happen with timm models that use non-standard layers like BatchNormAct2d
        # which are not fully compatible with PyTorch's quantization framework
        int8_conversion_error = str(e)
        print(f"WARNING: INT8 conversion/evaluation failed: {e}")
        print("This is common with timm models that use BatchNormAct2d or other custom layers.")
        print("QAT fake-quant metrics are still valid. Skipping real INT8 evaluation.")

    runtime_sec = float(_now() - job_t0)

    # Train-time ratio, if available
    ratio = None
    if fp32_train_time_sum > 0:
        ratio = float(qat_train_time_sum / fp32_train_time_sum)

    out: Dict[str, Any] = {
        "method": "qat",
        "dataset": args.dataset,
        "model": args.model,
        "fold": int(args.fold),
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "num_classes": int(num_classes),
        "num_params": int(n_params),
        "train_device": str(device),
        "int8_backend": str(args.backend),
        "fp32_epochs": int(args.fp32_epochs),
        "qat_epochs": int(args.qat_epochs),
        "fp32_best_acc": float(fp32_best_acc),
        "qat_best_acc": float(qat_best_acc),
        "fp32_history": fp32_history,
        "qat_history": qat_history,
        "test_split_info": test_split_info,
        # Keep existing keys for backward-compatibility (these are VAL).
        "fp32_metrics": fp32_metrics,
        "fp32_val_metrics": fp32_metrics,
        "fp32_test_metrics": fp32_test_metrics,
        "fp32_benchmark": fp32_bench,
        "qat_fake_metrics": qat_fake_metrics,
        "qat_fake_val_metrics": qat_fake_metrics,
        "qat_fake_test_metrics": qat_fake_test_metrics,
        "qat_fake_benchmark": qat_fake_bench,
        "int8_metrics": int8_metrics,
        "int8_val_metrics": int8_metrics,
        "int8_test_metrics": int8_test_metrics,
        "int8_benchmark": int8_bench,
        "int8_conversion_error": int8_conversion_error,
        "checkpoint_fp32": str(ckpt_fp32),
        "checkpoint_qat": str(ckpt_qat),
        "timing": {
            "fp32_train_time_sum_sec": float(fp32_train_time_sum),
            "fp32_eval_time_sum_sec": float(fp32_eval_time_sum),
            "fp32_final_eval_time_sec": float(fp32_final_eval_time_sec),
            "fp32_test_eval_time_sec": fp32_test_eval_time_sec,
            "qat_train_time_sum_sec": float(qat_train_time_sum),
            "qat_eval_time_sum_sec": float(qat_eval_time_sum),
            "qat_fake_eval_time_sec": float(qat_fake_eval_time_sec),
            "qat_fake_test_eval_time_sec": qat_fake_test_eval_time_sec,
            "int8_convert_time_sec": int8_convert_time_sec,
            "int8_eval_time_sec": int8_eval_time_sec,
            "int8_test_eval_time_sec": int8_test_eval_time_sec,
            "train_time_ratio_qat_over_fp32": ratio,
        },
        "runtime_sec": float(runtime_sec),
        "args": vars(args),
        "env": get_env_metadata(),
        "config_path": str(args.config) if args.config else str(REPO_ROOT / "configs" / "config.yaml"),
        "config_snapshot": config,
    }

    save_json(out_path, out)
    print(f"Saved results: {out_path}")


if __name__ == "__main__":
    main()