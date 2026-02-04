#!/usr/bin/env python3
"""
generate_all_figures_v2.py (FIXED)

CHIL 2026 Quantization Paper - Figure generator

Fixes (important):
- Uses the SAME metric extraction logic as experiments/paper_analysis.py:
  reads JSONs, infers dataset/model/variant/backend, extracts PRIMARY metrics
  (so Figure1 numbers match paper_analysis outputs)
- Robust percent conversion:
  if metric <= 1.0 -> treated as fraction and converted to percent
  if metric > 1.0  -> treated as already-percent and kept
- Figure1 prints ALL plotted numbers to stdout for verification
- Figure2 is wider, titles do not include CPU/GPU prefixes, and uses the same
  AUPRC values as Figure1 (so they must match)

Usage:
  python generate_all_figures_v2.py --results_dir /path/to/results --output_dir ./paper_figures

Note:
- This script focuses on fixing Figure1 and Figure2 consistency.
- Other figures can remain as in your original version if you want,
  but I kept them minimal here to avoid mixing incompatible loaders.
"""

import argparse
import json
import re
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =============================================================================
# Styling
# =============================================================================

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

DATASETS = ["brainmri", "chestxray", "skincancer"]
DATASET_LABELS = {
    "brainmri": "BrainMRI",
    "chestxray": "ChestXray",
    "skincancer": "SkinCancer",
}

MODELS = [
    "resnet18", "resnet50", "densenet121", "efficientnet_b0",
    "mobilenet_v2", "convnext_tiny", "swin_tiny", "vit_base"
]
MODEL_LABELS = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenet_v2": "MobileNetV2",
    "convnext_tiny": "ConvNeXt-T",
    "swin_tiny": "Swin-T",
    "vit_base": "ViT-B",
}

# Keep the same variant naming idea you used
VARIANTS = [
    "fp32", "fp16",
    "int8_ptq_fbgemm", "int8_ptq_qnnpack", "int8_ptq",
    "int8_qat_fbgemm", "int8_qat_qnnpack", "int8_qat"
]
VARIANT_LABELS = {
    "fp32": "FP32",
    "fp16": "FP16",
    "int8_ptq": "INT8 PTQ",
    "int8_qat": "INT8 QAT",
    "int8_ptq_fbgemm": "INT8 PTQ (fbgemm)",
    "int8_ptq_qnnpack": "INT8 PTQ (qnnpack)",
    "int8_qat_fbgemm": "INT8 QAT (fbgemm)",
    "int8_qat_qnnpack": "INT8 QAT (qnnpack)",
    "unknown": "Unknown",
}

VARIANT_COLORS = {
    "fp32": "#1f77b4",
    "fp16": "#ff7f0e",
    "int8_ptq_fbgemm": "#2ca02c",
    "int8_ptq_qnnpack": "#d62728",
    "int8_ptq": "#2ca02c",
    "int8_qat_fbgemm": "#9467bd",
    "int8_qat_qnnpack": "#8c564b",
    "int8_qat": "#9467bd",
    "unknown": "#7f7f7f",
}

# =============================================================================
# paper_analysis.py compatible metric extraction
# =============================================================================

CANONICAL = [
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "precision_macro",
    "recall_macro",
    "auc_roc_ovr_macro",
    "auprc_ovr_macro_sklearn",
    "log_loss",
    "ece",
    "brier",
]

SYNONYMS: Dict[str, List[str]] = {
    "accuracy": ["accuracy", "acc", "top1_acc", "top1"],
    "balanced_accuracy": ["balanced_accuracy", "bal_accuracy", "bal_acc", "balanced_acc"],
    "f1_macro": ["f1_macro", "f1", "f1_score_macro", "macro_f1"],
    "precision_macro": ["precision_macro", "macro_precision", "precision"],
    "recall_macro": ["recall_macro", "macro_recall", "recall", "sensitivity_macro"],
    "auc_roc_ovr_macro": [
        "auc_roc_ovr_macro_sklearn",
        "auc_roc_ovr_macro",
        "roc_auc_ovr_macro",
        "auc_roc_macro",
        "roc_auc_macro",
        "auc",
        "roc_auc",
    ],
    "auprc_ovr_macro_sklearn": [
        "auprc_ovr_macro_sklearn",
        "auprc_ovr_macro",
        "auprc_macro",
        "auprc",
        "average_precision_macro",
    ],
    "log_loss": ["log_loss_sklearn", "log_loss", "nll", "neg_log_likelihood"],
    "ece": ["ece", "expected_calibration_error"],
    "brier": ["brier", "brier_score", "brier_score_multiclass"],
}


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def load_json_robust(path: Path) -> Dict[str, Any]:
    """
    Handles:
    - normal JSON
    - JSON with extra trailing text
    - multiple JSON objects concatenated
    Returns the LAST successfully decoded JSON object.
    """
    txt = path.read_text(errors="ignore").strip()
    if not txt:
        raise ValueError("Empty file")

    try:
        return json.loads(txt)
    except Exception:
        pass

    dec = json.JSONDecoder()
    objs: List[Dict[str, Any]] = []

    i = 0
    n = len(txt)
    while i < n:
        while i < n and txt[i].isspace():
            i += 1
        if i >= n:
            break

        if txt[i] not in "{[":
            i += 1
            continue

        try:
            obj, j = dec.raw_decode(txt, i)
            if isinstance(obj, dict):
                objs.append(obj)
            i = j
        except Exception:
            i += 1

    if not objs:
        last = txt.rfind("}")
        if last != -1:
            try:
                return json.loads(txt[: last + 1])
            except Exception:
                pass
        raise ValueError("Could not decode any JSON object")

    return objs[-1]


def get_metric(metrics: Dict[str, Any], canonical_key: str) -> float:
    if not isinstance(metrics, dict):
        return float("nan")
    for k in SYNONYMS.get(canonical_key, [canonical_key]):
        if k in metrics:
            return safe_float(metrics.get(k))
    return float("nan")


def pick_metrics_dict(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    for k in keys:
        v = d.get(k, None)
        if isinstance(v, dict) and v:
            return v
    return {}


def find_backend_anywhere(d: Dict[str, Any], path: Path) -> str:
    for key in ["int8_backend", "backend", "quant_backend"]:
        b = d.get(key, None)
        if b is not None:
            return str(b).lower().strip()

    for nested_key in ["args", "config", "experiment"]:
        nested = d.get(nested_key, {})
        if isinstance(nested, dict):
            for key in ["int8_backend", "backend"]:
                b = nested.get(key, None)
                if b is not None:
                    return str(b).lower().strip()

    s = str(path).lower()
    if "qnnpack" in s:
        return "qnnpack"
    if "fbgemm" in s:
        return "fbgemm"
    return "unknown"


def infer_dataset_model(path: Path, d: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    dataset = d.get("dataset", None)
    model = d.get("model", None)

    parts = path.parts

    if dataset is None:
        for p in parts:
            p_lower = p.lower()
            if p_lower in set(DATASETS):
                dataset = p_lower
                break

    if model is None:
        parent_name = path.parent.name
        if parent_name and parent_name.lower() not in {
            "brainmri", "chestxray", "skincancer",
            "fp32", "fp16", "int8_ptq", "qat",
            "int8_ptq_fbgemm", "int8_ptq_qnnpack",
            "qat_fbgemm", "qat_qnnpack",
            "results", "results_quant", "results_additional_analyses",
        }:
            model = parent_name

        if model is None:
            stem = path.stem
            stem_clean = re.sub(r"_fold\d+$", "", stem, flags=re.IGNORECASE)
            if stem_clean:
                model = stem_clean

    if dataset is not None:
        dataset = str(dataset).strip().lower()
    if model is not None:
        model = str(model).strip().lower().replace("-", "_")

    return dataset, model


def infer_fold(path: Path, d: Dict[str, Any]) -> int:
    if "fold" in d and d["fold"] is not None:
        try:
            return int(d["fold"])
        except Exception:
            pass

    if "args" in d and isinstance(d["args"], dict):
        fold_val = d["args"].get("fold", None)
        if fold_val is not None:
            try:
                return int(fold_val)
            except Exception:
                pass

    m = re.search(r"_?fold(\d+)", path.stem.lower())
    if m:
        return int(m.group(1))
    return 0


def determine_variant(d: Dict[str, Any], backend: str, path: Path) -> str:
    backend = (backend or "unknown").lower().strip()

    # folder-based, same as paper_analysis
    for p in path.parts:
        pl = p.lower()
        if pl == "int8_ptq_fbgemm":
            return "int8_ptq_fbgemm"
        if pl == "int8_ptq_qnnpack":
            return "int8_ptq_qnnpack"
        if pl == "qat_fbgemm":
            return "int8_qat_fbgemm"
        if pl == "qat_qnnpack":
            return "int8_qat_qnnpack"
        if pl == "fp32":
            return "fp32"
        if pl == "fp16":
            return "fp16"

    method = d.get("method", None)
    if method is not None:
        m = str(method).lower().strip()
        if m == "fp32":
            return "fp32"
        if m == "fp16":
            return "fp16"
        if m in {"int8_ptq", "int8", "ptq"}:
            if backend in {"fbgemm", "qnnpack"}:
                return f"int8_ptq_{backend}"
            return "int8_ptq"
        if m in {"qat", "int8_qat"}:
            if backend in {"fbgemm", "qnnpack"}:
                return f"int8_qat_{backend}"
            return "int8_qat"

    precision = d.get("precision", None)
    if precision is not None:
        p = str(precision).lower().strip()
        if p == "fp32":
            return "fp32"
        if p == "fp16":
            return "fp16"
        if p in {"int8_ptq", "int8"}:
            if backend in {"fbgemm", "qnnpack"}:
                return f"int8_ptq_{backend}"
            return "int8_ptq"

    if "args" in d and isinstance(d["args"], dict):
        args = d["args"]
        if "precision" in args:
            p = str(args["precision"]).lower().strip()
            if p == "fp32":
                return "fp32"
            if p == "fp16":
                return "fp16"
            if p in {"int8_ptq", "int8"}:
                if backend in {"fbgemm", "qnnpack"}:
                    return f"int8_ptq_{backend}"
                return "int8_ptq"
        if "method" in args:
            m = str(args["method"]).lower().strip()
            if m in {"qat", "int8_qat"}:
                if backend in {"fbgemm", "qnnpack"}:
                    return f"int8_qat_{backend}"
                return "int8_qat"

    # key-based fallback
    if "qat_fake_metrics" in d or "qat_history" in d:
        if backend in {"fbgemm", "qnnpack"}:
            return f"int8_qat_{backend}"
        return "int8_qat"

    if "int8_ptq_metrics" in d or "int8_ptq_val_metrics" in d or "int8_ptq_test_metrics" in d:
        if backend in {"fbgemm", "qnnpack"}:
            return f"int8_ptq_{backend}"
        return "int8_ptq"

    s = str(path).lower()
    if "qat" in s:
        if backend in {"fbgemm", "qnnpack"}:
            return f"int8_qat_{backend}"
        return "int8_qat"
    if "ptq" in s or "int8" in s:
        if backend in {"fbgemm", "qnnpack"}:
            return f"int8_ptq_{backend}"
        return "int8_ptq"
    if "fp16" in s:
        return "fp16"
    if "fp32" in s:
        return "fp32"

    return "unknown"


@dataclass
class Record:
    dataset: str
    model: str
    variant: str
    fold: int
    roc_auc: float
    auprc: float


def extract_primary_metrics(d: Dict[str, Any], variant: str) -> Tuple[float, float]:
    """
    paper_analysis compatible primary metric selection:
    - fp32/fp16: from test_metrics/val_metrics/baseline_metrics...
    - int8_ptq: from int8_ptq_* first, else int8_metrics
    - int8_qat: from int8_* first, else qat_fake_*
    Returns: (roc_auc, auprc) as raw float (may be in [0,1] or already percent)
    """
    if variant in {"fp32", "fp16"}:
        src = pick_metrics_dict(d, ["test_metrics", "val_metrics", "baseline_metrics", "fp32_test_metrics", "fp32_metrics", "metrics"])
        roc = get_metric(src, "auc_roc_ovr_macro")
        prc = get_metric(src, "auprc_ovr_macro_sklearn")
        return roc, prc

    if variant.startswith("int8_ptq") or variant == "int8_ptq":
        int8_src = pick_metrics_dict(d, ["int8_ptq_test_metrics", "int8_ptq_val_metrics", "int8_ptq_metrics", "ptq_metrics", "int8_metrics"])
        roc = get_metric(int8_src, "auc_roc_ovr_macro")
        prc = get_metric(int8_src, "auprc_ovr_macro_sklearn")
        return roc, prc

    if variant.startswith("int8_qat") or variant == "int8_qat":
        int8_src = pick_metrics_dict(d, ["int8_test_metrics", "int8_val_metrics", "int8_metrics", "int8_qat_metrics"])
        if int8_src:
            roc = get_metric(int8_src, "auc_roc_ovr_macro")
            prc = get_metric(int8_src, "auprc_ovr_macro_sklearn")
            return roc, prc

        qat_fake_src = pick_metrics_dict(d, ["qat_fake_test_metrics", "qat_fake_val_metrics", "qat_fake_metrics", "qat_metrics_fake"])
        roc = get_metric(qat_fake_src, "auc_roc_ovr_macro")
        prc = get_metric(qat_fake_src, "auprc_ovr_macro_sklearn")
        return roc, prc

    # fallback
    src = pick_metrics_dict(d, ["test_metrics", "val_metrics", "baseline_metrics", "fp32_metrics", "metrics"])
    roc = get_metric(src, "auc_roc_ovr_macro")
    prc = get_metric(src, "auprc_ovr_macro_sklearn")
    return roc, prc


def to_percent_for_plot(x: float) -> float:
    """
    Robust conversion:
    - If x <= 1.0: interpret as fraction and convert to percent
    - Else: keep as already-percent
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    return x * 100.0 if x <= 1.0 else x


def discover_result_jsons(root: Path) -> List[Path]:
    """
    paper_analysis style scanning:
    - if root/results exists, scan it
    - if root/results_quant exists, scan it
    - also scan root itself
    Only collects *.json under precision folders/datasets/models.
    """
    candidates: List[Path] = []
    if (root / "results").exists():
        candidates.append(root / "results")
    if (root / "results_quant").exists():
        candidates.append(root / "results_quant")
    candidates.append(root)

    candidates = sorted(list({p.resolve() for p in candidates}))

    PRECISION_FOLDERS = [
        "fp32",
        "fp16",
        "int8_ptq",
        "int8_ptq_fbgemm",
        "int8_ptq_qnnpack",
        "qat",
        "qat_fbgemm",
        "qat_qnnpack",
    ]

    all_jsons: List[Path] = []

    for results_dir in candidates:
        for precision_folder in PRECISION_FOLDERS:
            precision_dir = results_dir / precision_folder
            if not precision_dir.exists():
                continue
            for dataset in DATASETS:
                dataset_dir = precision_dir / dataset
                if not dataset_dir.exists():
                    continue
                for model_dir in dataset_dir.iterdir():
                    if not model_dir.is_dir():
                        continue
                    for json_file in model_dir.glob("*.json"):
                        # skip mac artifacts
                        if "__MACOSX" in str(json_file):
                            continue
                        all_jsons.append(json_file)

    all_jsons = sorted(list(set(all_jsons)))
    return all_jsons


def load_primary_records(results_dir: Path) -> pd.DataFrame:
    """
    Load records with primary metrics (roc_auc, auprc) per dataset/model/variant/fold.
    """
    paths = discover_result_jsons(results_dir)
    print(f"  Found {len(paths)} result JSON files")

    rows: List[Dict[str, Any]] = []
    parse_errors = 0

    for p in paths:
        try:
            d = load_json_robust(p)
            dataset, model = infer_dataset_model(p, d)
            if dataset is None or model is None:
                continue

            backend = find_backend_anywhere(d, p)
            variant = determine_variant(d, backend, p)
            fold = infer_fold(p, d)

            roc, prc = extract_primary_metrics(d, variant)

            rows.append({
                "path": str(p),
                "dataset": dataset,
                "model": model,
                "variant": variant,
                "fold": int(fold),
                "roc_auc": roc,
                "auprc": prc,
            })
        except Exception:
            parse_errors += 1

    if parse_errors > 0:
        print(f"  Parse errors: {parse_errors} files (skipped)")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Keep only known datasets/models if possible
    df = df[df["dataset"].isin(DATASETS)].copy()
    # model normalization: your MODELS list uses underscores
    df = df[df["model"].isin(MODELS)].copy()

    # Keep variant order but allow "int8_ptq" and "int8_qat"
    df = df[df["variant"].isin(set(VARIANTS + ["unknown"]))].copy()

    print(f"  Parsed {len(df)} usable records")
    return df


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate by dataset, model, variant: mean/std/count for roc_auc and auprc
    Uses RAW values (may be [0,1] or already percent). Plot conversion handles it.
    """
    if df.empty:
        return df

    agg = df.groupby(["dataset", "model", "variant"]).agg(
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        roc_auc_count=("roc_auc", "count"),
        auprc_mean=("auprc", "mean"),
        auprc_std=("auprc", "std"),
        auprc_count=("auprc", "count"),
    ).reset_index()

    # fill std NaN for singletons
    for c in ["roc_auc_std", "auprc_std"]:
        if c in agg.columns:
            agg[c] = agg[c].fillna(0.0)

    return agg


# =============================================================================
# Figure 1: Bar charts, and PRINT plotted numbers
# =============================================================================

def print_figure1_numbers(agg_df: pd.DataFrame, dataset: str, metric: str) -> None:
    """
    Print table of plotted values for Figure1 verification.
    metric: "roc_auc" or "auprc"
    """
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    s = agg_df[agg_df["dataset"] == dataset].copy()
    if s.empty:
        print(f"\n[Figure1 PRINT] {dataset} {metric}: NO DATA")
        return

    # choose variants present in data but keep ordering
    variants_present = [v for v in VARIANTS if v in set(s["variant"].tolist())]
    models_present = [m for m in MODELS if m in set(s["model"].tolist())]

    print(f"\n[Figure1 PRINT] dataset={dataset} metric={metric}")
    header = ["model"] + variants_present
    print(" | ".join([h.ljust(20) for h in header]))
    print("-" * (23 * len(header)))

    for m in models_present:
        row_cells = [MODEL_LABELS.get(m, m).ljust(20)]
        for v in variants_present:
            r = s[(s["model"] == m) & (s["variant"] == v)]
            if r.empty:
                row_cells.append("NA".ljust(20))
                continue
            mu = float(r[mean_col].iloc[0])
            sd = float(r[std_col].iloc[0])
            mu_p = to_percent_for_plot(mu)
            sd_p = to_percent_for_plot(sd)
            row_cells.append(f"{mu_p:.2f} ± {sd_p:.2f}".ljust(20))
        print(" | ".join(row_cells))


def figure1_bars_single(
    agg_df: pd.DataFrame,
    output_dir: Path,
    dataset: str,
    metric: str,
    metric_label: str,
    filename: str
) -> bool:
    """
    Single dataset bar chart for metric (roc_auc or auprc).
    Uses robust percent conversion.
    """
    if agg_df.empty:
        print(f"    Skipping {filename}: No data")
        return False

    df_ds = agg_df[agg_df["dataset"] == dataset].copy()
    if df_ds.empty:
        print(f"    Skipping {filename}: No data for {dataset}")
        return False

    metric_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if metric_col not in df_ds.columns:
        print(f"    Skipping {filename}: No {metric_col} column")
        return False

    models_present = [m for m in MODELS if m in df_ds["model"].values]
    variants_present = [v for v in VARIANTS if v in df_ds["variant"].values]
    if not models_present or not variants_present:
        print(f"    Skipping {filename}: No models/variants")
        return False

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models_present))
    width = 0.8 / len(variants_present)

    for i, variant in enumerate(variants_present):
        df_v = df_ds[df_ds["variant"] == variant]
        means = []
        stds = []

        for model in models_present:
            row = df_v[df_v["model"] == model]
            if row.empty:
                means.append(np.nan)
                stds.append(0.0)
                continue

            val = float(row[metric_col].values[0])
            std_val = float(row[std_col].values[0]) if std_col in row.columns else 0.0

            means.append(to_percent_for_plot(val))
            stds.append(to_percent_for_plot(std_val))

        offset = (i - len(variants_present) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=VARIANT_LABELS.get(variant, variant),
            color=VARIANT_COLORS.get(variant, f"C{i}"),
            capsize=2,
            alpha=0.85,
        )

    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{metric_label} (%)", fontsize=12, fontweight="bold")
    ax.set_title(f"{DATASET_LABELS.get(dataset, dataset)}", fontsize=14, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models_present], rotation=45, ha="right", fontsize=10)

    # AUC often looks nicer with 40-100, AUPRC usually 0-100
    if metric == "roc_auc":
        ax.set_ylim(40, 100)
    else:
        ax.set_ylim(0, 100)

    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {filename}")
    return True


def figure1_all_bars(agg_df: pd.DataFrame, output_dir: Path) -> None:
    print("\n  Generating Figure 1 (AUC and AUPRC bar charts)...")

    count = 0
    for dataset in DATASETS:
        # Print values for verification
        print_figure1_numbers(agg_df, dataset, "roc_auc")
        print_figure1_numbers(agg_df, dataset, "auprc")

        if figure1_bars_single(agg_df, output_dir, dataset, "roc_auc", "AUC-ROC", f"figure1_auc_{dataset}.png"):
            count += 1
        if figure1_bars_single(agg_df, output_dir, dataset, "auprc", "AUPRC", f"figure1_auprc_{dataset}.png"):
            count += 1

    print(f"    Total: {count}/6 files generated")


# =============================================================================
# Figure 2: AUPRC vs Latency scatter (wider, titles without CPU/GPU text)
# =============================================================================

def load_benchmark_json(results_dir: Path, benchmark_path: Optional[Path] = None) -> Dict[str, Any]:
    if benchmark_path and benchmark_path.exists():
        print(f"  Loading benchmark from: {benchmark_path}")
        return json.loads(benchmark_path.read_text())

    candidates = [
        results_dir / "experiments" / "benchmark_results" / "benchmark_all.json",
        results_dir / "benchmark_results" / "benchmark_all.json",
        results_dir / "bench_fair_b1" / "benchmark_all.json",
        results_dir / "benchmark_all.json",
    ]
    for p in candidates:
        if p.exists():
            print(f"  Loading benchmark from: {p}")
            return json.loads(p.read_text())

    print("  WARNING: No benchmark JSON found")
    return {}



def figure2_latency_scatter(
    agg_df: pd.DataFrame,
    benchmark: Dict[str, Any],
    output_dir: Path,
    batch_size: int = 1,
):
    """
    Figure 2: Macro AUPRC vs Latency scatter plots.

    Layout:
      2×3 grid
        Top row: GPU (FP32, FP16)
        Bottom row: CPU (FP32, INT8-QAT)

    Notes:
      - AUPRC values come from agg_df["auprc_mean"] to match Figure 1.
      - CPU INT8-QAT latency: if your benchmark JSON does not include a QAT-specific CPU latency key,
        this uses cpu_int8_ptq as a proxy (common when measuring int8 runtime on CPU).
    """
    print("\n  Generating Figure 2 (Latency scatter)...")

    if not benchmark or "results" not in benchmark:
        print("    Skipping: No benchmark data")
        return

    if agg_df.empty:
        print("    Skipping: No aggregated results (need fold*.json files)")
        return

    if "auprc_mean" not in agg_df.columns:
        print("    Skipping: No AUPRC data in aggregated results")
        return

    # -----------------------------
    # Styling (bigger text, wider figure)
    # -----------------------------
    SUPTITLE_FS = 20
    TITLE_FS = 22
    AXIS_LABEL_FS = 20
    TICK_FS = 16
    LEGEND_FS = 20

    MARKER_SIZE = 120
    EDGE_LW = 0.8

    fig, axes = plt.subplots(2, 3, figsize=(24.0, 10.0), sharey=True)

    # -----------------------------
    # Build latency lookup
    # -----------------------------
    latency: Dict[Tuple[str, str, str], float] = {}
    for r in benchmark["results"]:
        ds = r.get("dataset")
        model = r.get("model")
        if ds is None or model is None:
            continue

        ds = str(ds).lower().strip()
        model = str(model).lower().strip().replace("-", "_")

        for regime, key in [
            ("gpu_fp32", "gpu_fp32"),
            ("gpu_fp16", "gpu_fp16"),
            ("cpu_fp32", "cpu_fp32"),
            ("cpu_int8", "cpu_int8_ptq"),  # proxy for int8 cpu latency (PTQ-like measurement)
        ]:
            data = r.get(key, {})
            if isinstance(data, dict) and str(batch_size) in data:
                p50 = data[str(batch_size)].get("p50_ms")
                if p50 is not None:
                    latency[(ds, model, regime)] = float(p50)

    print(f"    Loaded {len(latency)} latency entries")

    # Model colors
    cmap = plt.get_cmap("tab10")
    model_colors = {m: cmap(i % 10) for i, m in enumerate(MODELS)}

    def normalize_auprc(v: float) -> float:
        if v is None or np.isnan(v):
            return np.nan
        v = float(v)
        if v > 1.5:  # looks like percent
            return v / 100.0
        return v

    def set_log_ticks(ax: plt.Axes, xs: List[float]) -> None:
        xs = [float(x) for x in xs if x is not None and not np.isnan(x) and x > 0]
        if len(xs) == 0:
            return

        xmin = min(xs)
        xmax = max(xs)
        xmin_pad = xmin / 1.25
        xmax_pad = xmax * 1.25
        ax.set_xlim(xmin_pad, xmax_pad)

        candidates = [
            0.5, 1, 2, 5,
            10, 20, 50,
            100, 200, 500,
            1000, 2000, 5000,
            10000, 20000, 50000,
        ]
        ticks = [t for t in candidates if (t >= xmin_pad and t <= xmax_pad)]
        if len(ticks) < 2:
            mid = 10 ** ((np.log10(xmin_pad) + np.log10(xmax_pad)) / 2.0)
            ticks = sorted(list({xmin, mid, xmax}))

        ax.set_xticks(ticks)

        from matplotlib.ticker import FuncFormatter
        def fmt(x, pos):
            if x >= 10:
                return f"{int(round(x))}"
            return f"{x:.1f}".rstrip("0").rstrip(".")

        ax.xaxis.set_major_formatter(FuncFormatter(fmt))

    points_plotted = 0

    # -----------------------------
    # GPU row (top): FP32 vs FP16
    # -----------------------------
    for col, dataset in enumerate(DATASETS):
        ax = axes[0, col]
        xs_this_ax: List[float] = []

        for model in MODELS:
            # GPU FP32
            lat_fp32 = latency.get((dataset, model, "gpu_fp32"))
            row_fp32 = agg_df[
                (agg_df["dataset"] == dataset) &
                (agg_df["model"] == model) &
                (agg_df["variant"] == "fp32")
            ]
            if lat_fp32 is not None and not row_fp32.empty:
                auprc = normalize_auprc(float(row_fp32["auprc_mean"].values[0]))
                if not np.isnan(auprc):
                    ax.scatter(
                        lat_fp32, auprc,
                        marker="o", s=MARKER_SIZE,
                        color=model_colors[model], alpha=0.88,
                        edgecolors="black", linewidth=EDGE_LW
                    )
                    xs_this_ax.append(lat_fp32)
                    points_plotted += 1

            # GPU FP16
            lat_fp16 = latency.get((dataset, model, "gpu_fp16"))
            row_fp16 = agg_df[
                (agg_df["dataset"] == dataset) &
                (agg_df["model"] == model) &
                (agg_df["variant"] == "fp16")
            ]
            if lat_fp16 is not None and not row_fp16.empty:
                auprc = normalize_auprc(float(row_fp16["auprc_mean"].values[0]))
                if not np.isnan(auprc):
                    ax.scatter(
                        lat_fp16, auprc,
                        marker="^", s=MARKER_SIZE,
                        color=model_colors[model], alpha=0.88,
                        edgecolors="black", linewidth=EDGE_LW
                    )
                    xs_this_ax.append(lat_fp16)
                    points_plotted += 1

        ax.set_xscale("log")
        set_log_ticks(ax, xs_this_ax)

        ax.set_title(f"{DATASET_LABELS[dataset]}", fontweight="bold", fontsize=TITLE_FS)
        ax.set_xlabel("Batch latency p50 (ms)", fontsize=AXIS_LABEL_FS, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Macro AUPRC", fontsize=AXIS_LABEL_FS, fontweight="bold")
        else:
            ax.set_ylabel("")          # ya da: ax.set_ylabel(None)
            ax.tick_params(labelleft=True)  # kalsın istiyorsan
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.grid(True, alpha=0.3)

    # -----------------------------
    # CPU row (bottom): FP32 vs INT8-QAT
    # -----------------------------
    for col, dataset in enumerate(DATASETS):
        ax = axes[1, col]
        xs_this_ax: List[float] = []

        for model in MODELS:
            # CPU FP32
            lat_fp32 = latency.get((dataset, model, "cpu_fp32"))
            row_fp32 = agg_df[
                (agg_df["dataset"] == dataset) &
                (agg_df["model"] == model) &
                (agg_df["variant"] == "fp32")
            ]
            if lat_fp32 is not None and not row_fp32.empty:
                auprc = normalize_auprc(float(row_fp32["auprc_mean"].values[0]))
                if not np.isnan(auprc):
                    ax.scatter(
                        lat_fp32, auprc,
                        marker="o", s=MARKER_SIZE,
                        color=model_colors[model], alpha=0.88,
                        edgecolors="black", linewidth=EDGE_LW
                    )
                    xs_this_ax.append(lat_fp32)
                    points_plotted += 1

            # CPU INT8-QAT (use any available QAT variant in results)
            row_qat = agg_df[
                (agg_df["dataset"] == dataset) &
                (agg_df["model"] == model) &
                (agg_df["variant"].isin(["int8_qat_fbgemm", "int8_qat_qnnpack", "int8_qat"]))
            ]
            if not row_qat.empty:
                auprc = normalize_auprc(float(row_qat["auprc_mean"].values[0]))
                if np.isnan(auprc):
                    continue

                # Latency proxy: CPU int8 benchmark (often measured with PTQ-style int8 runtime)
                lat_int8 = latency.get((dataset, model, "cpu_int8"))
                if lat_int8 is None:
                    continue

                ax.scatter(
                    lat_int8, auprc,
                    marker="s", s=MARKER_SIZE,
                    color=model_colors[model], alpha=0.88,
                    edgecolors="black", linewidth=EDGE_LW
                )
                xs_this_ax.append(lat_int8)
                points_plotted += 1

        ax.set_xscale("log")
        set_log_ticks(ax, xs_this_ax)

        # ax.set_title(f"{DATASET_LABELS[dataset]}", fontweight="bold", fontsize=TITLE_FS)
        ax.set_xlabel("Batch latency p50 (ms)", fontsize=AXIS_LABEL_FS, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Macro AUPRC", fontsize=AXIS_LABEL_FS, fontweight="bold")
        else:
            ax.set_ylabel("")          # ya da: ax.set_ylabel(None)
            ax.tick_params(labelleft=True)  # kalsın istiyorsan

        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.grid(True, alpha=0.3)

    if points_plotted == 0:
        print("    No points plotted (missing overlap between benchmark and results)")
        plt.close(fig)
        return

    # Row labels (rotate 90 degrees)
    fig.text(
        0.03, 0.72, "GPU",
        rotation=90, fontsize=20, fontweight="bold",
        va="center", ha="center"
    )
    fig.text(
        0.03, 0.35, "CPU",
        rotation=90, fontsize=20, fontweight="bold",
        va="center", ha="center"
    )

    # Legends: regime markers + model colors
    variant_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color="gray", label="FP32", markersize=20),
        Line2D([0], [0], marker="^", linestyle="None", color="gray", label="FP16", markersize=20),
        Line2D([0], [0], marker="s", linestyle="None", color="gray", label="INT8-QAT", markersize=20),
    ]
    model_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color=model_colors[m],
               label=MODEL_LABELS.get(m, m), markersize=20)
        for m in MODELS
    ]

    fig.suptitle(
        "Macro AUPRC versus inference latency",
        fontsize=SUPTITLE_FS, fontweight="bold", y=1.02
    )

    fig.legend(
        handles=variant_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.01),
        frameon=True,
        fontsize=LEGEND_FS
    )
    fig.legend(
        handles=model_handles,
        loc="lower center",
        ncol=8,
        bbox_to_anchor=(0.5, -0.06),
        frameon=False,
        fontsize=LEGEND_FS
    )

    fig.tight_layout(rect=[0.04, 0.12, 1, 0.95])

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "figure2_latency_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: figure2_latency_scatter.png ({points_plotted} points)")

def figure2_appendix_cpu_fp32_vs_int8ptq(
    agg_df: pd.DataFrame,
    benchmark: Dict[str, Any],
    output_dir: Path,
    batch_size: int = 1,
    ptq_jitter_log10: float = 0.035,
):
    """
    Figure 2 (Appendix, CPU-only): FP32 vs INT8-PTQ (accuracy vs latency)

    Layout:
      1×3 grid (one panel per dataset)
      Each panel: FP32 (o), INT8-PTQ (D)

    Legends:
      - Marker legend: FP32 vs INT8-PTQ
      - Model legend: colors

    AUPRC values:
      - Uses agg_df["auprc_mean"] to match Figure 1.
      - Auto-normalizes percent (0..100) into 0..1 if needed.
    """
    print("\n  Generating Figure 2 Appendix (CPU-only, FP32 vs INT8-PTQ)...")

    if not benchmark or "results" not in benchmark:
        print("    Skipping: No benchmark data")
        return

    if agg_df.empty or "auprc_mean" not in agg_df.columns:
        print("    Skipping: No aggregated AUPRC data")
        return

    # Bigger text
    SUPTITLE_FS = 20
    TITLE_FS = 20
    AXIS_LABEL_FS = 20
    TICK_FS = 20
    LEGEND_FS =20  # make legend text bigger

    MARKER_SIZE = 130
    EDGE_LW = 0.9

    fig, axes = plt.subplots(1, 3, figsize=(24.0, 7.2), sharey=True)

    # latency lookup: (dataset, model, regime) -> p50_ms
    latency: Dict[Tuple[str, str, str], float] = {}
    for r in benchmark.get("results", []):
        ds = r.get("dataset")
        model = r.get("model")
        if ds is None or model is None:
            continue

        ds = str(ds).lower().strip()
        model = str(model).lower().strip().replace("-", "_")

        for regime, key in [("cpu_fp32", "cpu_fp32"), ("cpu_int8", "cpu_int8_ptq")]:
            data = r.get(key, {})
            if isinstance(data, dict) and str(batch_size) in data:
                p50 = data[str(batch_size)].get("p50_ms")
                if p50 is not None:
                    latency[(ds, model, regime)] = float(p50)

    print(f"    Loaded {len(latency)} latency entries")

    # Model colors
    cmap = plt.get_cmap("tab10")
    model_colors = {m: cmap(i % 10) for i, m in enumerate(MODELS)}

    def normalize_auprc(v: float) -> float:
        if v is None or np.isnan(v):
            return np.nan
        v = float(v)
        if v > 1.5:
            return v / 100.0
        return v

    def set_log_ticks(ax: plt.Axes, xs: List[float]) -> None:
        xs = [float(x) for x in xs if x is not None and not np.isnan(x) and x > 0]
        if len(xs) == 0:
            return

        xmin = min(xs)
        xmax = max(xs)
        xmin_pad = xmin / 1.25
        xmax_pad = xmax * 1.25
        ax.set_xlim(xmin_pad, xmax_pad)

        candidates = [
            0.5, 1, 2, 5,
            10, 20, 50,
            100, 200, 500,
            1000, 2000, 5000,
            10000, 20000, 50000,
        ]
        ticks = [t for t in candidates if (t >= xmin_pad and t <= xmax_pad)]
        if len(ticks) < 2:
            mid = 10 ** ((np.log10(xmin_pad) + np.log10(xmax_pad)) / 2.0)
            ticks = sorted(list({xmin, mid, xmax}))

        ax.set_xticks(ticks)

        from matplotlib.ticker import FuncFormatter
        def fmt(x, pos):
            if x >= 10:
                return f"{int(round(x))}"
            return f"{x:.1f}".rstrip("0").rstrip(".")

        ax.xaxis.set_major_formatter(FuncFormatter(fmt))
        ax.tick_params(axis="x", labelsize=TICK_FS)

    def apply_log_jitter(x: float, sign: float) -> float:
        return float(x) * (10 ** (sign * float(ptq_jitter_log10)))

    points_plotted = 0

    for col, dataset in enumerate(DATASETS):
        ax = axes[col]
        xs_this_ax: List[float] = []

        for model in MODELS:
            # FP32
            lat_fp32 = latency.get((dataset, model, "cpu_fp32"))
            row_fp32 = agg_df[
                (agg_df["dataset"] == dataset) &
                (agg_df["model"] == model) &
                (agg_df["variant"] == "fp32")
            ]
            if lat_fp32 is not None and not row_fp32.empty:
                auprc_fp32 = normalize_auprc(float(row_fp32["auprc_mean"].values[0]))
                if not np.isnan(auprc_fp32):
                    ax.scatter(
                        lat_fp32, auprc_fp32,
                        marker="o", s=MARKER_SIZE,
                        color=model_colors[model], alpha=0.88,
                        edgecolors="black", linewidth=EDGE_LW
                    )
                    xs_this_ax.append(lat_fp32)
                    points_plotted += 1

            # INT8-PTQ (pick any PTQ variant you have)
            row_ptq = agg_df[
                (agg_df["dataset"] == dataset) &
                (agg_df["model"] == model) &
                (agg_df["variant"].isin(["int8_ptq_fbgemm", "int8_ptq_qnnpack", "int8_ptq"]))
            ]
            if not row_ptq.empty:
                auprc_ptq = normalize_auprc(float(row_ptq["auprc_mean"].values[0]))
                if np.isnan(auprc_ptq):
                    continue

                lat_int8 = latency.get((dataset, model, "cpu_int8"))
                if lat_int8 is None:
                    continue

                lat_use = apply_log_jitter(lat_int8, sign=+1.0)
                ax.scatter(
                    lat_use, auprc_ptq,
                    marker="D", s=MARKER_SIZE,
                    color=model_colors[model], alpha=0.88,
                    edgecolors="black", linewidth=EDGE_LW
                )
                xs_this_ax.append(lat_use)
                points_plotted += 1

        ax.set_xscale("log")
        set_log_ticks(ax, xs_this_ax)

        ax.set_xlabel("Batch latency p50 (ms)", fontsize=AXIS_LABEL_FS, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Macro AUPRC", fontsize=AXIS_LABEL_FS, fontweight="bold")

        ax.set_title(f"{DATASET_LABELS[dataset]}", fontweight="bold", fontsize=TITLE_FS)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.grid(True, alpha=0.3)

    if points_plotted == 0:
        print("    No points plotted (missing overlap between benchmark and results)")
        plt.close(fig)
        return

    marker_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color="gray", label="FP32", markersize=15),
        Line2D([0], [0], marker="D", linestyle="None", color="gray", label="INT8-PTQ", markersize=15),
    ]
    model_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color=model_colors[m],
               label=MODEL_LABELS.get(m, m), markersize=15)
        for m in MODELS
    ]

    # fig.suptitle(
    #     "CPU: Macro AUPRC versus inference latency (FP32 vs INT8-PTQ)",
    #     fontsize=SUPTITLE_FS, fontweight="bold", y=1.03
    # )

    fig.legend(
        handles=marker_handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        fontsize=LEGEND_FS
    )
    fig.legend(
        handles=model_handles,
        loc="lower center",
        ncol=8,
        bbox_to_anchor=(0.5, -0.115),
        frameon=False,
        fontsize=LEGEND_FS
    )

    fig.tight_layout(rect=[0, 0.12, 1, 0.95])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "figure2_appendix_cpu_fp32_vs_int8ptq.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name} ({points_plotted} points)")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CHIL 2026 paper figures (fixed Figure1/Figure2)")
    parser.add_argument("--results_dir", type=str, required=True, help="Root directory containing experiment results")
    parser.add_argument("--output_dir", type=str, default="./paper_figures", help="Output directory for figures")
    parser.add_argument("--benchmark_path", type=str, default=None, help="Path to benchmark_all.json (optional)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for latency plots (default: 1)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    benchmark_path = Path(args.benchmark_path) if args.benchmark_path else None

    print("=" * 70)
    print("CHIL 2026 Paper Figure Generator (v2 FIXED)")
    print("=" * 70)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    if benchmark_path:
        print(f"Benchmark path: {benchmark_path}")
    print()

    print("[1/2] Loading data (paper_analysis compatible)...")
    raw_df = load_primary_records(results_dir)
    agg_df = aggregate_results(raw_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    if not agg_df.empty:
        agg_df.to_csv(output_dir / "summary_model_variant_mean_std.csv", index=False)
        print(f"  Saved: summary_model_variant_mean_std.csv ({len(agg_df)} rows)")
    else:
        print("  WARNING: No aggregated data produced!")

    benchmark = load_benchmark_json(results_dir, benchmark_path)

    print()
    print("[2/2] Generating Figure1 and Figure2...")
    figure1_all_bars(agg_df, output_dir)
    figure2_latency_scatter(agg_df, benchmark, output_dir, batch_size=args.batch_size)
    figure2_appendix_cpu_fp32_vs_int8ptq(
        agg_df,
        benchmark,
        output_dir,
        batch_size=args.batch_size,
    )

    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  ✓ {f.name}")
    for f in sorted(output_dir.glob("*.csv")):
        print(f"  ✓ {f.name}")

    print("\nDone.")
    print("=" * 70)


if __name__ == "__main__":
    main()
