#!/usr/bin/env python3
"""
Main Training Script for Medical Image Quantization Experiments
==============================================================================

Runs training experiments for medical image classification with various precision levels:
- FP32: Full precision baseline
- FP16: Mixed precision training
- INT8-PTQ: Post-Training Quantization (fbgemm/qnnpack backends)
- INT8-QAT: Quantization-Aware Training (fbgemm/qnnpack backends)

Output Structure:
    results/<precision>/<dataset>/<model>/fold<N>.json
    results/<precision>/<dataset>/<model>/val_logits_fold<N>.npy
    results/<precision>/<dataset>/<model>/val_labels_fold<N>.npy
    results/<precision>/<dataset>/<model>/test_logits_fold<N>.npy  (if test set exists)
    results/<precision>/<dataset>/<model>/test_labels_fold<N>.npy

Usage:
    # FP32 training
    python run_job.py --dataset brainmri --model resnet18 --precision fp32 --fold 1
    
    # FP16 training
    python run_job.py --dataset chestxray --model resnet50 --precision fp16 --fold 1
    
    # INT8 PTQ (requires pre-trained FP32 checkpoint)
    python run_job.py --dataset brainmri --model resnet18 --precision int8_ptq \\
        --backend fbgemm --fp32_checkpoint results/fp32/brainmri/resnet18/fold1.pt
    
    # INT8 QAT
    python run_job.py --dataset brainmri --model resnet18 --precision int8_qat \\
        --backend fbgemm --fp32_checkpoint results/fp32/brainmri/resnet18/fold1.pt \\
        --qat_epochs 10

Author: Quantization Medical Imaging Study
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

# Prefer torch.ao.quantization when available (PyTorch 2.0+)
try:
    import torch.ao.quantization as quant
    from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert
except ImportError:
    import torch.quantization as quant
    from torch.quantization import get_default_qat_qconfig, prepare_qat, convert

# Add parent directory to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "utils"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from data_loader_helper import (
    load_brainmri_dataset,
    load_chestxray_dataset,
    load_skincancer_dataset,
    create_splits_brainmri,
    create_splits_chestxray,
    create_splits_skincancer,
    create_dataloaders,
    create_test_loader,
    get_dataset_and_splits,
)
from utils.metrics import (
    compute_classification_metrics,
    expected_calibration_error,
    brier_score_multiclass,
    softmax_np,
)
from utils.model_factory import create_model, count_parameters
from utils.train_eval import (
    EarlyStopper,
    build_optimizer,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    set_determinism,
    train_one_epoch,
)
from utils.qat_utils import (
    QuantizableWrapper,
    prepare_model_for_qat,
    convert_qat_to_quantized,
    freeze_batchnorm,
    load_fp32_weights_into_qat,
)

# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_ROOT = Path(os.environ.get("QUANT_ROOT", REPO_ROOT))
DEFAULT_CONFIG = DEFAULT_ROOT / "configs" / "config.yaml"
DEFAULT_RESULTS = DEFAULT_ROOT / "results"
DEFAULT_CHECKPOINTS = DEFAULT_ROOT / "checkpoints"

DATASETS = ["brainmri", "chestxray", "skincancer"]
MODELS = [
    "resnet18", "resnet50", "densenet121", "efficientnet_b0",
    "mobilenet_v2", "convnext_tiny", "swin_tiny", "vit_base"
]
PRECISIONS = ["fp32", "fp16", "int8_ptq", "int8_qat"]
BACKENDS = ["fbgemm", "qnnpack"]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = DEFAULT_CONFIG
    config_path = Path(config_path)
    
    if not config_path.exists():
        # Return minimal default config
        return {
            "experiment": {"random_seed": 42},
            "training": {
                "epochs": 50,
                "batch_size": 32,
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
                "optimizer": "adamw",
                "num_workers": 4,
                "early_stopping": {"patience": 10, "min_delta": 0.001}
            },
            "cv": {"n_folds": 5, "val_split": 0.2},
            "datasets": {
                "brainmri": {"num_classes": 4, "img_size": 224, "data_root": "data/brainmri"},
                "chestxray": {"num_classes": 4, "img_size": 224, "data_root": "data/chestxray"},
                "skincancer": {"num_classes": 7, "img_size": 224, "data_root": "data/skincancer", 
                              "metadata_path": "data/skincancer/HAM10000_metadata.csv"},
            },
            "models": {
                "deep_learning": {
                    "resnet18": {"architecture": "resnet18", "pretrained": True, "dropout": 0.0},
                    "resnet50": {"architecture": "resnet50", "pretrained": True, "dropout": 0.0},
                    "densenet121": {"architecture": "densenet121", "pretrained": True, "dropout": 0.0},
                    "efficientnet_b0": {"architecture": "efficientnet_b0", "pretrained": True, "dropout": 0.2},
                    "mobilenet_v2": {"architecture": "mobilenetv2_100", "pretrained": True, "dropout": 0.2},
                    "convnext_tiny": {"architecture": "convnext_tiny", "pretrained": True, "dropout": 0.0},
                    "swin_tiny": {"architecture": "swin_tiny_patch4_window7_224", "pretrained": True, "dropout": 0.0},
                    "vit_base": {"architecture": "vit_base_patch16_224", "pretrained": True, "dropout": 0.1},
                }
            },
            "qat": {"epochs": 10, "learning_rate": 1e-5},
            "evaluation": {"calibration": {"n_bins": 15}},
        }
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_local_caches(root: Path) -> None:
    """Set local cache directories for torch, huggingface, timm."""
    cache_root = root / ".cache"
    (cache_root / "torch").mkdir(parents=True, exist_ok=True)
    (cache_root / "hf").mkdir(parents=True, exist_ok=True)
    
    os.environ["TORCH_HOME"] = str(cache_root / "torch")
    os.environ["HF_HOME"] = str(cache_root / "hf")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_root / "hf" / "hub")


def get_system_info() -> Dict[str, Any]:
    """Get system information for logging."""
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    return info


def compute_all_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    n_bins: int = 15,
) -> Dict[str, Any]:
    """Compute all metrics from logits and labels."""
    probs = softmax_np(logits)
    preds = np.argmax(probs, axis=1)
    
    metrics = compute_classification_metrics(labels, preds, probs, num_classes)
    metrics["ece"] = expected_calibration_error(labels, probs, n_bins=n_bins)
    metrics["brier"] = brier_score_multiclass(labels, probs, num_classes)
    
    return metrics


# ============================================================================
# FP32 / FP16 TRAINING
# ============================================================================

def train_fp32_fp16(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    use_fp16: bool = False,
    checkpoint_path: Optional[Path] = None,
) -> Tuple[nn.Module, List[Dict[str, Any]], float]:
    """
    Train model in FP32 or FP16 (mixed precision).
    
    Returns:
        (trained_model, history, best_val_acc)
    """
    epochs = int(config["training"]["epochs"])
    es_cfg = config.get("training", {}).get("early_stopping", {})
    
    optimizer = build_optimizer(model, config)
    stopper = EarlyStopper(
        patience=int(es_cfg.get("patience", 10)),
        min_delta=float(es_cfg.get("min_delta", 0.001)),
        mode="max",
    )
    
    scaler = torch.cuda.amp.GradScaler() if (use_fp16 and device.type == "cuda") else None
    
    history = []
    best_val_acc = -1.0
    best_state = None
    
    for epoch in range(epochs):
        t0 = time.time()
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        
        # Evaluate
        val_logits, val_labels = evaluate(model, val_loader, device)
        val_probs = softmax_np(val_logits)
        val_preds = np.argmax(val_probs, axis=1)
        val_acc = float(np.mean(val_preds == val_labels))
        
        epoch_time = time.time() - t0
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_accuracy": val_acc,
            "epoch_time": epoch_time,
        })
        
        print(f"Epoch {epoch+1:3d}/{epochs}: loss={train_loss:.4f} val_acc={val_acc:.4f} time={epoch_time:.1f}s")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path:
                save_checkpoint(checkpoint_path, model, optimizer, epoch, best_val_acc)
        
        # Early stopping
        if stopper.step(val_acc):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, history, best_val_acc


# ============================================================================
# INT8 POST-TRAINING QUANTIZATION (PTQ)
# ============================================================================

def apply_ptq(
    model: nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    backend: str = "fbgemm",
    num_calibration_batches: int = 100,
) -> nn.Module:
    """
    Apply post-training static quantization.
    
    Args:
        model: Trained FP32 model
        calibration_loader: DataLoader for calibration
        backend: 'fbgemm' (x86) or 'qnnpack' (ARM)
        num_calibration_batches: Number of batches for calibration
    
    Returns:
        Quantized INT8 model
    """
    model_fp32 = copy.deepcopy(model)
    model_fp32.eval()
    model_fp32.cpu()
    
    # Wrap model with quant/dequant stubs
    wrapped = QuantizableWrapper(model_fp32)
    
    # Set backend and qconfig
    torch.backends.quantized.engine = backend
    wrapped.qconfig = quant.get_default_qconfig(backend)
    
    # Prepare for calibration
    quant.prepare(wrapped, inplace=True)
    
    # Calibration pass
    wrapped.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(calibration_loader):
            if i >= num_calibration_batches:
                break
            images = images.cpu()
            wrapped(images)
    
    # Convert to quantized model
    model_int8 = quant.convert(wrapped, inplace=False)
    
    return model_int8


# ============================================================================
# INT8 QUANTIZATION-AWARE TRAINING (QAT)
# ============================================================================

def train_qat(
    fp32_model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    backend: str = "fbgemm",
    qat_epochs: Optional[int] = None,
    qat_lr: Optional[float] = None,
) -> Tuple[nn.Module, nn.Module, List[Dict[str, Any]]]:
    """
    Quantization-Aware Training.
    
    Args:
        fp32_model: Pre-trained FP32 model
        train_loader: Training data
        val_loader: Validation data
        config: Configuration
        backend: 'fbgemm' or 'qnnpack'
        qat_epochs: Number of QAT epochs
        qat_lr: Learning rate for QAT
    
    Returns:
        (qat_model_fp32, qat_model_int8, history)
    """
    qat_epochs = qat_epochs or int(config.get("qat", {}).get("epochs", 10))
    qat_lr = qat_lr or float(config.get("qat", {}).get("learning_rate", 1e-5))
    
    # Prepare model for QAT
    model_qat = prepare_model_for_qat(fp32_model, backend)
    model_qat.train()
    
    # Use CPU for QAT (required for PyTorch quantization)
    device = torch.device("cpu")
    model_qat = model_qat.to(device)
    
    # Optimizer with lower LR
    optimizer = torch.optim.Adam(
        [p for p in model_qat.parameters() if p.requires_grad],
        lr=qat_lr,
        weight_decay=float(config["training"].get("weight_decay", 1e-5))
    )
    
    criterion = nn.CrossEntropyLoss()
    history = []
    best_val_acc = -1.0
    best_state = None
    
    for epoch in range(qat_epochs):
        t0 = time.time()
        
        # Train
        model_qat.train()
        running_loss = 0.0
        n_samples = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model_qat(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * labels.size(0)
            n_samples += labels.size(0)
        
        train_loss = running_loss / max(n_samples, 1)
        
        # Evaluate
        model_qat.eval()
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits = model_qat(images)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        val_logits = np.concatenate(all_logits)
        val_labels = np.concatenate(all_labels)
        val_probs = softmax_np(val_logits)
        val_preds = np.argmax(val_probs, axis=1)
        val_acc = float(np.mean(val_preds == val_labels))
        
        epoch_time = time.time() - t0
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_accuracy": val_acc,
            "epoch_time": epoch_time,
        })
        
        print(f"QAT Epoch {epoch+1:3d}/{qat_epochs}: loss={train_loss:.4f} val_acc={val_acc:.4f} time={epoch_time:.1f}s")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model_qat.state_dict())
    
    # Restore best
    if best_state is not None:
        model_qat.load_state_dict(best_state)
    
    # Convert to INT8
    model_qat.eval()
    model_int8 = convert_qat_to_quantized(model_qat)
    
    return model_qat, model_int8, history


# ============================================================================
# EVALUATION HELPERS
# ============================================================================

def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    n_bins: int = 15,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Evaluate model and compute metrics.
    
    Returns:
        (logits, labels, metrics_dict)
    """
    model.eval()
    model = model.to(device)
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu().float().numpy())
            all_labels.append(labels.cpu().numpy())
    
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    
    metrics = compute_all_metrics(logits, labels, num_classes, n_bins)
    
    return logits, labels, metrics


def evaluate_quantized_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    num_classes: int,
    n_bins: int = 15,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Evaluate quantized INT8 model (CPU only).
    """
    model.eval()
    model.cpu()
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.cpu()
            try:
                logits = model(images)
                all_logits.append(logits.float().numpy())
                all_labels.append(labels.numpy())
            except Exception as e:
                print(f"Warning: INT8 inference error: {e}")
                continue
    
    if not all_logits:
        return None, None, {"error": "INT8 inference failed"}
    
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    
    metrics = compute_all_metrics(logits, labels, num_classes, n_bins)
    
    return logits, labels, metrics


# ============================================================================
# MAIN JOB RUNNER
# ============================================================================

def run_job(args: argparse.Namespace) -> Dict[str, Any]:
    """Run a single training/evaluation job."""
    
    # Setup
    config = load_config(args.config)
    seed = int(config.get("experiment", {}).get("random_seed", 42))
    set_determinism(seed)
    set_local_caches(DEFAULT_ROOT)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine output paths
    precision_dir = args.precision
    if args.precision in ["int8_ptq", "int8_qat"]:
        precision_dir = f"{args.precision}_{args.backend}"
    
    results_dir = Path(args.results_dir) / precision_dir / args.dataset / args.model
    checkpoints_dir = Path(args.checkpoints_dir) / precision_dir / args.dataset / args.model
    ensure_dir(results_dir)
    ensure_dir(checkpoints_dir)
    
    # Output files
    fold_str = f"fold{args.fold}"
    result_json = results_dir / f"{fold_str}.json"
    checkpoint_path = checkpoints_dir / f"{fold_str}.pt"
    
    # Load dataset and splits
    print("=" * 70)
    print(f"JOB: {args.dataset} / {args.model} / {args.precision} / fold {args.fold}")
    print("=" * 70)
    
    num_classes = int(config["datasets"][args.dataset]["num_classes"])
    n_bins = int(config.get("evaluation", {}).get("calibration", {}).get("n_bins", 15))
    
    # Get data
    # get_dataset_and_splits returns: (image_paths, labels, test_paths, test_labels, (train_idx, val_idx))
    image_paths, labels, test_paths, test_labels, (train_idx, val_idx) = get_dataset_and_splits(
        args.dataset, config, fold=args.fold - 1
    )
    
    # Convert to lists if needed
    if not test_paths:
        test_paths = None
        test_labels = None
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        image_paths, labels, train_idx, val_idx, config, args.dataset
    )
    
    test_loader = None
    if test_paths:
        test_loader = create_test_loader(test_paths, test_labels, config, args.dataset)
    
    # Initialize result
    result = {
        "dataset": args.dataset,
        "model": args.model,
        "precision": args.precision,
        "fold": args.fold,
        "backend": args.backend if args.precision in ["int8_ptq", "int8_qat"] else None,
        "num_classes": num_classes,
        "system_info": get_system_info(),
        "config": {
            "epochs": config["training"]["epochs"],
            "batch_size": config["training"]["batch_size"],
            "learning_rate": config["training"]["learning_rate"],
        },
    }
    
    t0 = time.time()
    
    # =========================================================================
    # FP32 Training
    # =========================================================================
    if args.precision == "fp32":
        print("\n[Phase] FP32 Training")
        model = create_model(args.model, num_classes, config, device)
        result["num_params"] = count_parameters(model)
        
        model, history, best_acc = train_fp32_fp16(
            model, train_loader, val_loader, config, device,
            use_fp16=False, checkpoint_path=checkpoint_path
        )
        
        result["history"] = history
        result["epochs_ran"] = len(history)
        
        # Final evaluation
        print("\n[Phase] Final Evaluation")
        val_logits, val_labels_arr, val_metrics = evaluate_model(
            model, val_loader, device, num_classes, n_bins
        )
        result["fp32_metrics"] = val_metrics
        
        # Save logits
        np.save(results_dir / f"val_logits_{fold_str}.npy", val_logits)
        np.save(results_dir / f"val_labels_{fold_str}.npy", val_labels_arr)
        
        if test_loader:
            test_logits, test_labels_arr, test_metrics = evaluate_model(
                model, test_loader, device, num_classes, n_bins
            )
            result["test_metrics"] = test_metrics
            np.save(results_dir / f"test_logits_{fold_str}.npy", test_logits)
            np.save(results_dir / f"test_labels_{fold_str}.npy", test_labels_arr)
    
    # =========================================================================
    # FP16 Training
    # =========================================================================
    elif args.precision == "fp16":
        print("\n[Phase] FP16 (Mixed Precision) Training")
        model = create_model(args.model, num_classes, config, device)
        result["num_params"] = count_parameters(model)
        
        model, history, best_acc = train_fp32_fp16(
            model, train_loader, val_loader, config, device,
            use_fp16=True, checkpoint_path=checkpoint_path
        )
        
        result["history"] = history
        result["epochs_ran"] = len(history)
        
        # Final evaluation
        print("\n[Phase] Final Evaluation")
        val_logits, val_labels_arr, val_metrics = evaluate_model(
            model, val_loader, device, num_classes, n_bins
        )
        result["fp32_metrics"] = val_metrics  # Use same key for consistency
        
        np.save(results_dir / f"val_logits_{fold_str}.npy", val_logits)
        np.save(results_dir / f"val_labels_{fold_str}.npy", val_labels_arr)
        
        if test_loader:
            test_logits, test_labels_arr, test_metrics = evaluate_model(
                model, test_loader, device, num_classes, n_bins
            )
            result["test_metrics"] = test_metrics
            np.save(results_dir / f"test_logits_{fold_str}.npy", test_logits)
            np.save(results_dir / f"test_labels_{fold_str}.npy", test_labels_arr)
    
    # =========================================================================
    # INT8 Post-Training Quantization
    # =========================================================================
    elif args.precision == "int8_ptq":
        print(f"\n[Phase] INT8 PTQ ({args.backend})")
        
        # Load pre-trained FP32 model
        if not args.fp32_checkpoint:
            raise ValueError("--fp32_checkpoint required for int8_ptq")
        
        model = create_model(args.model, num_classes, config, device)
        result["num_params"] = count_parameters(model)
        
        ckpt = load_checkpoint(Path(args.fp32_checkpoint), model)
        print(f"Loaded FP32 checkpoint: {args.fp32_checkpoint}")
        
        # Evaluate FP32 baseline
        print("\n[Phase] FP32 Baseline Evaluation")
        val_logits_fp32, val_labels_arr, fp32_metrics = evaluate_model(
            model, val_loader, device, num_classes, n_bins
        )
        result["fp32_metrics"] = fp32_metrics
        np.save(results_dir / f"val_logits_fp32_{fold_str}.npy", val_logits_fp32)
        np.save(results_dir / f"val_labels_{fold_str}.npy", val_labels_arr)
        
        # Apply PTQ
        print(f"\n[Phase] Applying PTQ ({args.backend})")
        model_int8 = apply_ptq(
            model, train_loader, backend=args.backend,
            num_calibration_batches=100
        )
        
        # Evaluate INT8
        print("\n[Phase] INT8 Evaluation")
        val_logits_int8, _, int8_metrics = evaluate_quantized_model(
            model_int8, val_loader, num_classes, n_bins
        )
        result["int8_ptq_metrics"] = int8_metrics
        
        if val_logits_int8 is not None:
            np.save(results_dir / f"val_logits_int8_{fold_str}.npy", val_logits_int8)
        
        if test_loader:
            test_logits_fp32, test_labels_arr, test_fp32_metrics = evaluate_model(
                model, test_loader, device, num_classes, n_bins
            )
            result["test_fp32_metrics"] = test_fp32_metrics
            np.save(results_dir / f"test_logits_fp32_{fold_str}.npy", test_logits_fp32)
            np.save(results_dir / f"test_labels_{fold_str}.npy", test_labels_arr)
            
            test_logits_int8, _, test_int8_metrics = evaluate_quantized_model(
                model_int8, test_loader, num_classes, n_bins
            )
            result["test_int8_ptq_metrics"] = test_int8_metrics
            if test_logits_int8 is not None:
                np.save(results_dir / f"test_logits_int8_{fold_str}.npy", test_logits_int8)
    
    # =========================================================================
    # INT8 Quantization-Aware Training
    # =========================================================================
    elif args.precision == "int8_qat":
        print(f"\n[Phase] INT8 QAT ({args.backend})")
        
        # Load pre-trained FP32 model
        if not args.fp32_checkpoint:
            raise ValueError("--fp32_checkpoint required for int8_qat")
        
        model = create_model(args.model, num_classes, config, device)
        result["num_params"] = count_parameters(model)
        
        ckpt = load_checkpoint(Path(args.fp32_checkpoint), model)
        print(f"Loaded FP32 checkpoint: {args.fp32_checkpoint}")
        
        # Evaluate FP32 baseline
        print("\n[Phase] FP32 Baseline Evaluation")
        val_logits_fp32, val_labels_arr, fp32_metrics = evaluate_model(
            model, val_loader, device, num_classes, n_bins
        )
        result["fp32_metrics"] = fp32_metrics
        np.save(results_dir / f"val_logits_fp32_{fold_str}.npy", val_logits_fp32)
        np.save(results_dir / f"val_labels_{fold_str}.npy", val_labels_arr)
        
        # QAT Training
        print(f"\n[Phase] QAT Training ({args.backend}, {args.qat_epochs} epochs)")
        model_qat, model_int8, qat_history = train_qat(
            model, train_loader, val_loader, config,
            backend=args.backend,
            qat_epochs=args.qat_epochs,
            qat_lr=args.qat_lr
        )
        
        result["qat_history"] = qat_history
        result["qat_epochs_ran"] = len(qat_history)
        
        # Evaluate QAT model (before INT8 conversion)
        print("\n[Phase] QAT Model Evaluation (FP32 mode)")
        val_logits_qat, _, qat_metrics = evaluate_model(
            model_qat, val_loader, torch.device("cpu"), num_classes, n_bins
        )
        result["qat_fp32_metrics"] = qat_metrics
        np.save(results_dir / f"val_logits_qat_fp32_{fold_str}.npy", val_logits_qat)
        
        # Evaluate INT8
        print("\n[Phase] INT8 Evaluation")
        val_logits_int8, _, int8_metrics = evaluate_quantized_model(
            model_int8, val_loader, num_classes, n_bins
        )
        result["int8_qat_metrics"] = int8_metrics
        
        if val_logits_int8 is not None:
            np.save(results_dir / f"val_logits_int8_{fold_str}.npy", val_logits_int8)
        
        # Save QAT checkpoint
        torch.save({
            "qat_state_dict": model_qat.state_dict(),
            "int8_state_dict": model_int8.state_dict(),
        }, checkpoint_path)
        
        if test_loader:
            test_logits_fp32, test_labels_arr, test_fp32_metrics = evaluate_model(
                model, test_loader, device, num_classes, n_bins
            )
            result["test_fp32_metrics"] = test_fp32_metrics
            np.save(results_dir / f"test_logits_fp32_{fold_str}.npy", test_logits_fp32)
            np.save(results_dir / f"test_labels_{fold_str}.npy", test_labels_arr)
            
            test_logits_int8, _, test_int8_metrics = evaluate_quantized_model(
                model_int8, test_loader, num_classes, n_bins
            )
            result["test_int8_qat_metrics"] = test_int8_metrics
            if test_logits_int8 is not None:
                np.save(results_dir / f"test_logits_int8_{fold_str}.npy", test_logits_int8)
    
    else:
        raise ValueError(f"Unknown precision: {args.precision}")
    
    # Finalize
    result["runtime_sec"] = time.time() - t0
    
    # Save result JSON
    with open(result_json, "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print(f"Job completed in {result['runtime_sec']:.1f}s")
    print(f"Results saved to: {result_json}")
    print("=" * 70)
    
    return result


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run training job for medical image quantization experiments"
    )
    
    # Required arguments
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS,
                        help="Dataset to use")
    parser.add_argument("--model", type=str, required=True, choices=MODELS,
                        help="Model architecture")
    parser.add_argument("--precision", type=str, required=True, choices=PRECISIONS,
                        help="Precision level: fp32, fp16, int8_ptq, int8_qat")
    parser.add_argument("--fold", type=int, required=True,
                        help="Fold number (1-indexed)")
    
    # Optional arguments
    parser.add_argument("--backend", type=str, default="fbgemm", choices=BACKENDS,
                        help="Quantization backend (for int8_ptq/int8_qat)")
    parser.add_argument("--fp32_checkpoint", type=str, default=None,
                        help="Path to FP32 checkpoint (required for PTQ/QAT)")
    parser.add_argument("--qat_epochs", type=int, default=10,
                        help="Number of QAT epochs")
    parser.add_argument("--qat_lr", type=float, default=1e-5,
                        help="Learning rate for QAT")
    
    # Paths
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml")
    parser.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS),
                        help="Results output directory")
    parser.add_argument("--checkpoints_dir", type=str, default=str(DEFAULT_CHECKPOINTS),
                        help="Checkpoints directory")
    
    return parser.parse_args()


def main():
    args = parse_args()
    run_job(args)


if __name__ == "__main__":
    main()
