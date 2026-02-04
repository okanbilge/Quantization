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
import yaml

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
from utils.quantization_utils import quantize_int8_ptq
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


def normalize_paths_in_config(config: Dict[str, Any], base_dir: Path) -> None:
    """Best-effort normalization of path-like fields in the config.

    This prevents silent failures when configs contain relative paths, ~, or $VARS.
    """

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
    # Always write under DEFAULT_ROOT/results, regardless of config.
    # This matches the repository's on-disk structure (results/{fp16,fp32,int8_ptq,qat}/...).
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
            # Older signature without dataset
            return create_cv_splits_with_holdout(image_paths, labels, config, test_ratio=test_ratio)
        except TypeError:
            # Even older signature without keyword
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
# Metrics helpers
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

        # ROC-AUC OVR macro
        out["auc_roc_ovr_macro_sklearn"] = float(
            roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        )

        # AUPRC OVR macro (one-vs-rest)
        y_onehot = np.eye(num_classes, dtype=np.float32)[labels.astype(int)]
        out["auprc_ovr_macro_sklearn"] = float(
            average_precision_score(y_onehot, probs, average="macro")
        )

        # Log loss
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
# Eval + Benchmark helpers
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


class CalibLoaderWrapper:
    """
    Ensures calibration data is on CPU for PTQ calibration.
    """
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for batch in self.loader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                raise ValueError("Dataloader batch must be (inputs, labels, ...).")
            x, y = batch[0], batch[1]
            yield (x.cpu(), y)

    def __len__(self):
        try:
            return len(self.loader)
        except Exception:
            return 0


# -------------------------
# Main
# -------------------------
def main() -> None:
    set_local_caches(DEFAULT_ROOT)

    p = argparse.ArgumentParser(description="FP32/FP16/PTQ runner with paper-ready metrics and timing.")
    p.add_argument("--dataset", required=True, choices=["brainmri", "chestxray", "skincancer"])
    p.add_argument("--model", required=True)
    p.add_argument("--fold", type=int, default=0, help="0-indexed fold id (e.g., 0..4 for 5-fold CV). Ignored for brainmri.")
    p.add_argument("--precision", required=True, choices=["fp32", "fp16", "int8_ptq"])
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--backend", default="fbgemm", choices=["fbgemm", "qnnpack"], help="PTQ INT8 backend target")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--calib_batches", type=int, default=16)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip_if_done", action="store_true", help="Skip if output JSON already exists")
    p.add_argument("--config", type=str, default=None, help="Optional config path (yaml)")
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

    # Output paths
    # Keep the original on-disk layout (results/{fp32,fp16,int8_ptq}/...) and
    # encode backend in the filename for INT8 PTQ to prevent overwrites.
    if args.precision == "int8_ptq":
        out_dir = results_root / "int8_ptq" / args.dataset / args.model
    else:
        out_dir = results_root / args.precision / args.dataset / args.model

    ckpt_dir = checkpoints_root / args.dataset / args.model
    ensure_dir(out_dir)
    ensure_dir(ckpt_dir)


    # Use backend in filename for INT8 PTQ to avoid accidental overwrites.
    if args.precision == "int8_ptq":
        out_path = out_dir / f"fold{args.fold}_{args.backend}.json"
        ckpt_path = ckpt_dir / f"fold{args.fold}_{args.precision}_{args.backend}.pt"
    else:
        out_path = out_dir / f"fold{args.fold}.json"
        ckpt_path = ckpt_dir / f"fold{args.fold}_{args.precision}.pt"

    if args.skip_if_done and out_path.exists():
        print(f"Skip: output already exists: {out_path}")
        return

    # Seed
    seed = int(args.seed) if args.seed is not None else int(config.get("experiment", {}).get("random_seed", 42))
    deterministic = bool(config.get("experiment", {}).get("deterministic", True))
    set_determinism(seed, deterministic=deterministic)

    device = get_device(args.device)

    if args.epochs is not None:
        config.setdefault("training", {})
        config["training"]["epochs"] = int(args.epochs)

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

    # Model + opt
    model = create_model(args.model, num_classes=num_classes, config=config, device=device)
    n_params = int(count_parameters(model))
    optimizer = build_optimizer(model, config)

    start_epoch = 0
    best_acc = -1e9
    if args.resume and ckpt_path.exists():
        ckpt = load_checkpoint(ckpt_path, model, optimizer)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_acc = float(ckpt.get("best_metric", -1e9))

    epochs = int(config["training"]["epochs"])
    use_amp = (args.precision == "fp16") and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    es_cfg = config.get("training", {}).get("early_stopping", {})
    stopper = EarlyStopper(
        patience=int(es_cfg.get("patience", 10)),
        min_delta=float(es_cfg.get("min_delta", 0.0)),
        mode="max",
    )
    # If resuming, initialize stopper with best accuracy so far
    if start_epoch > 0 and best_acc > -1e9:
        stopper.best = best_acc

    n_bins = int(config.get("evaluation", {}).get("calibration", {}).get("n_bins", 15))

    print("=" * 70)
    print("PRECISION RUN")
    print("=" * 70)
    print(f"dataset={args.dataset} model={args.model} fold={args.fold} precision={args.precision}")
    print(f"device={device} seed={seed} epochs={epochs} start_epoch={start_epoch} amp={use_amp}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"num_classes={num_classes} num_params={n_params:,}")
    print("=" * 70)

    job_t0 = _now()
    history: List[Dict[str, Any]] = []
    train_time_sum = 0.0
    eval_time_sum = 0.0

    # -------------------------
    # Train FP32/FP16 (baseline for PTQ too)
    # -------------------------
    train_phase_t0 = _now()
    for epoch in range(start_epoch, epochs):
        epoch_t0 = _now()

        _sync(device)
        t_train0 = _now()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
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

        history.append(m)
        train_time_sum += float(train_time_sec)
        eval_time_sum += float(eval_time_sec)

        val_acc = float(m.get("accuracy", 0.0))
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(ckpt_path, model, optimizer, epoch, best_acc)

        print(
            f"Epoch {epoch+1}/{epochs} loss={train_loss:.4f} acc={val_acc*100:.2f}% "
            f"train={train_time_sec:.2f}s eval={eval_time_sec:.2f}s"
        )

        if stopper.step(val_acc):
            break

    train_phase_time_sec = float(_now() - train_phase_t0)

    # Load best for final eval
    if ckpt_path.exists():
        load_checkpoint(ckpt_path, model, optimizer=None)

    # Final eval baseline (val)
    _sync(device)
    t_final_eval0 = _now()
    fp_logits, fp_labels = evaluate_logits_labels(model, val_loader, device)
    _sync(device)
    final_eval_time_sec = float(_now() - t_final_eval0)

    fp_metrics = compute_all_metrics(fp_logits, fp_labels, num_classes=num_classes, n_bins=n_bins)
    fp_bench = benchmark_inference(
        model, val_loader, device, num_warmup_batches=args.bench_warmup, num_meas_batches=args.bench_batches
    )

    save_preds_cfg = bool(config.get("experiment", {}).get("save_predictions", False))
    if args.save_logits or save_preds_cfg:
        save_npy(out_dir / f"fold{args.fold}_val_logits.npy", fp_logits)
        save_npy(out_dir / f"fold{args.fold}_val_labels.npy", fp_labels)

    # Final eval on held-out test (if available)
    test_eval_time_sec: Optional[float] = None
    fp_test_metrics: Optional[Dict[str, Any]] = None
    if test_loader is not None:
        _sync(device)
        t_test0 = _now()
        fp_test_logits, fp_test_labels = evaluate_logits_labels(model, test_loader, device)
        _sync(device)
        test_eval_time_sec = float(_now() - t_test0)

        fp_test_metrics = compute_all_metrics(fp_test_logits, fp_test_labels, num_classes=num_classes, n_bins=n_bins)

        if args.save_logits or save_preds_cfg:
            save_npy(out_dir / f"fold{args.fold}_test_logits.npy", fp_test_logits)
            save_npy(out_dir / f"fold{args.fold}_test_labels.npy", fp_test_labels)

    # -------------------------
    # PTQ INT8 (optional)
    # -------------------------
    torch.backends.quantized.engine = args.backend
    ptq_convert_time_sec: Optional[float] = None
    ptq_eval_time_sec: Optional[float] = None
    ptq_test_eval_time_sec: Optional[float] = None
    int8_metrics: Optional[Dict[str, Any]] = None
    int8_test_metrics: Optional[Dict[str, Any]] = None
    int8_bench: Optional[Dict[str, Any]] = None

    if args.precision == "int8_ptq":
        try:
            # Ensure CPU model for PTQ
            if device.type == "cuda":
                model_cpu = copy.deepcopy(model).cpu()
            else:
                model_cpu = model.cpu()
            model_cpu.eval()

            calib_loader = CalibLoaderWrapper(train_loader)

            t_ptq0 = _now()
            qmodel = quantize_int8_ptq(
                model=model_cpu,
                model_key=args.model,
                calib_loader=calib_loader,
                num_calib_batches=int(args.calib_batches),
            )
            ptq_convert_time_sec = float(_now() - t_ptq0)

            t_qeval0 = _now()
            q_logits, q_labels = evaluate_logits_labels(qmodel, val_loader, torch.device("cpu"))
            ptq_eval_time_sec = float(_now() - t_qeval0)

            int8_metrics = compute_all_metrics(q_logits, q_labels, num_classes=num_classes, n_bins=n_bins)
            int8_bench = benchmark_inference(
                qmodel, val_loader, torch.device("cpu"),
                num_warmup_batches=args.bench_warmup,
                num_meas_batches=args.bench_batches,
            )

            if args.save_logits or save_preds_cfg:
                save_npy(out_dir / f"fold{args.fold}_int8_val_logits.npy", q_logits)
                save_npy(out_dir / f"fold{args.fold}_int8_val_labels.npy", q_labels)

            if test_loader is not None:
                t_qtest0 = _now()
                q_test_logits, q_test_labels = evaluate_logits_labels(qmodel, test_loader, torch.device("cpu"))
                ptq_test_eval_time_sec = float(_now() - t_qtest0)
                int8_test_metrics = compute_all_metrics(q_test_logits, q_test_labels, num_classes=num_classes, n_bins=n_bins)

                if args.save_logits or save_preds_cfg:
                    save_npy(out_dir / f"fold{args.fold}_int8_test_logits.npy", q_test_logits)
                    save_npy(out_dir / f"fold{args.fold}_int8_test_labels.npy", q_test_labels)

        except (NotImplementedError, RuntimeError, Exception) as e:
            error_msg = str(e)
            int8_metrics = {"error": error_msg}
            print(f"WARNING: INT8 PTQ conversion/evaluation failed: {error_msg}")
            if "batch_norm" in error_msg.lower() or "BatchNorm" in error_msg:
                print("This is common with timm models that use BatchNormAct2d or other custom layers.")
            print("Skipping INT8 PTQ evaluation.")

    runtime_sec = float(_now() - job_t0)

    out: Dict[str, Any] = {
        "method": args.precision,
        "dataset": args.dataset,
        "model": args.model,
        "fold": int(args.fold),
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "num_classes": int(num_classes),
        "num_params": int(n_params),
        "train_device": str(device),
        "int8_backend": str(args.backend),
        "epochs_requested": int(epochs),
        "epochs_ran": int(len(history)),
        "best_acc": float(best_acc),
        "history": history,
        # Keep baseline_metrics for backward-compatibility (this is VAL).
        "baseline_metrics": fp_metrics,
        "val_metrics": fp_metrics,
        "test_metrics": fp_test_metrics,
        "test_split_info": test_split_info,
        "baseline_benchmark": fp_bench,
        # Keep int8_ptq_metrics for backward-compatibility (this is VAL).
        "int8_ptq_metrics": int8_metrics,
        "int8_ptq_val_metrics": int8_metrics,
        "int8_ptq_test_metrics": int8_test_metrics,
        "int8_ptq_benchmark": int8_bench,
        "checkpoint_path": str(ckpt_path),
        "timing": {
            "train_phase_time_sec": float(train_phase_time_sec),
            "train_time_sum_sec": float(train_time_sum),
            "eval_time_sum_sec": float(eval_time_sum),
            "final_eval_time_sec": float(final_eval_time_sec),
            "test_eval_time_sec": test_eval_time_sec,
            "ptq_convert_time_sec": ptq_convert_time_sec,
            "ptq_eval_time_sec": ptq_eval_time_sec,
            "ptq_test_eval_time_sec": ptq_test_eval_time_sec,
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