"""
Data Loading Helper with Leakage-Free Split Strategies
========================================================

This module provides dataset loading and splitting for medical imaging datasets:
- BrainMRI: Train/Test pre-separated, split train into train/val
- ChestXray: No patient metadata, use StratifiedKFold
- SkinCancer: Lesion-level GroupKFold to prevent leakage

Author: Updated for leakage-free splits
Date: 2026-01-23
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split


# ============================================================================
# DATASET LOADERS
# ============================================================================

def load_brainmri_dataset(config: Dict[str, Any]) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Load BrainMRI dataset with pre-separated train/test sets.
    
    Directory structure:
        data/brainmri/
            Training/
                glioma/
                meningioma/
                notumor/
                pituitary/
            Testing/
                glioma/
                meningioma/
                notumor/
                pituitary/
    
    Returns:
        (train_paths, train_labels, test_paths, test_labels)
    """
    data_root = Path(config["datasets"]["brainmri"]["data_root"])
    
    train_dir = data_root / "Training"
    test_dir = data_root / "Testing"
    
    class_names = ["glioma", "meningioma", "notumor", "pituitary"]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # Load training data
    train_paths = []
    train_labels = []
    for class_name in class_names:
        class_dir = train_dir / class_name
        if not class_dir.exists():
            continue
        for img_path in class_dir.glob("*.jpg"):
            train_paths.append(str(img_path))
            train_labels.append(class_to_idx[class_name])
    
    # Load testing data
    test_paths = []
    test_labels = []
    for class_name in class_names:
        class_dir = test_dir / class_name
        if not class_dir.exists():
            continue
        for img_path in class_dir.glob("*.jpg"):
            test_paths.append(str(img_path))
            test_labels.append(class_to_idx[class_name])
    
    print(f"BrainMRI Dataset Loaded:")
    print(f"  Training: {len(train_paths)} images")
    print(f"  Testing: {len(test_paths)} images")
    print(f"  Classes: {class_names}")
    
    return train_paths, train_labels, test_paths, test_labels


def load_chestxray_dataset(config: Dict[str, Any]) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Load ChestXray COVID-19 Radiography Dataset.
    
    Directory structure:
        data/chestxray/COVID-19_Radiography_Dataset/
            COVID/images/
            Normal/images/
            Lung_Opacity/images/
            Viral Pneumonia/images/
    
    Returns:
        (all_paths, all_labels, [], [])
        Empty lists for test since we'll use CV
    """
    data_root = Path(config["datasets"]["chestxray"]["data_root"]) / "COVID-19_Radiography_Dataset"
    
    class_dirs = {
        "COVID": "COVID/images",
        "Normal": "Normal/images",
        "Lung_Opacity": "Lung_Opacity/images",
        "Viral Pneumonia": "Viral Pneumonia/images",
    }
    
    class_names = list(class_dirs.keys())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    all_paths = []
    all_labels = []
    
    for class_name, rel_path in class_dirs.items():
        class_dir = data_root / rel_path
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found")
            continue
        
        for img_path in class_dir.glob("*.png"):
            all_paths.append(str(img_path))
            all_labels.append(class_to_idx[class_name])
    
    print(f"ChestXray Dataset Loaded:")
    print(f"  Total: {len(all_paths)} images")
    print(f"  Classes: {class_names}")
    for class_name in class_names:
        count = sum(1 for l in all_labels if l == class_to_idx[class_name])
        print(f"    {class_name}: {count}")
    
    return all_paths, all_labels, [], []


def load_skincancer_dataset(config: Dict[str, Any]) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Load SkinCancer HAM10000 dataset.
    
    Directory structure:
        data/skincancer/
            HAM10000_images_part_1/
            HAM10000_images_part_2/
            HAM10000_metadata.csv
    
    Returns:
        (all_paths, all_labels, [], [])
        Empty lists for test since we'll use CV
    """
    data_root = Path(config["datasets"]["skincancer"]["data_root"])
    metadata_path = config["datasets"]["skincancer"]["metadata_path"]
    
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    
    # Class mapping
    class_names = sorted(metadata['dx'].unique().tolist())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # Find images in both directories
    img_dirs = [
        data_root / "HAM10000_images_part_1",
        data_root / "HAM10000_images_part_2",
    ]
    
    image_id_to_path = {}
    for img_dir in img_dirs:
        if not img_dir.exists():
            continue
        for img_path in img_dir.glob("*.jpg"):
            img_id = img_path.stem
            image_id_to_path[img_id] = str(img_path)
    
    # Match metadata with images
    all_paths = []
    all_labels = []
    missing = 0
    
    for _, row in metadata.iterrows():
        img_id = row['image_id']
        if img_id in image_id_to_path:
            all_paths.append(image_id_to_path[img_id])
            all_labels.append(class_to_idx[row['dx']])
        else:
            missing += 1
    
    print(f"SkinCancer Dataset Loaded:")
    print(f"  Total: {len(all_paths)} images")
    print(f"  Missing: {missing} images (in metadata but not found)")
    print(f"  Classes: {class_names}")
    for class_name in class_names:
        count = sum(1 for l in all_labels if l == class_to_idx[class_name])
        print(f"    {class_name}: {count}")
    
    return all_paths, all_labels, [], []


# ============================================================================
# SPLIT STRATEGIES
# ============================================================================

def create_splits_brainmri(
    train_paths: List[str],
    train_labels: List[int],
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    BrainMRI: Split training set into train/val for early stopping.
    Test set is already separated.
    
    Returns:
        (train_idx, val_idx) - indices into train_paths
    """
    seed = config.get("experiment", {}).get("random_seed", 42)
    val_split = config.get("cv", {}).get("val_split", 0.2)
    
    n_train = len(train_paths)
    
    # Stratified train/val split
    train_idx, val_idx = train_test_split(
        np.arange(n_train),
        test_size=val_split,
        random_state=seed,
        stratify=train_labels,
    )
    
    print(f"BrainMRI Split:")
    print(f"  Train: {len(train_idx)} samples ({len(train_idx)/n_train*100:.1f}%)")
    print(f"  Val: {len(val_idx)} samples ({len(val_idx)/n_train*100:.1f}%)")
    
    return train_idx, val_idx


def create_splits_chestxray(
    image_paths: List[str],
    labels: List[int],
    config: Dict[str, Any],
    fold: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ChestXray: Use StratifiedKFold or single train/val split.
    
    Args:
        fold: If None, use single train/val split. If 0-4, use that fold from 5-fold CV.
    
    Returns:
        (train_idx, val_idx)
    """
    seed = config.get("experiment", {}).get("random_seed", 42)
    
    if fold is None:
        # Single train/val split
        val_split = config.get("cv", {}).get("val_split", 0.2)
        train_idx, val_idx = train_test_split(
            np.arange(len(image_paths)),
            test_size=val_split,
            random_state=seed,
            stratify=labels,
        )
        print(f"ChestXray Single Split:")
        print(f"  Train: {len(train_idx)} samples")
        print(f"  Val: {len(val_idx)} samples")
    else:
        # K-fold CV
        n_folds = config.get("cv", {}).get("n_folds", 5)
        if not (0 <= fold < n_folds):
            raise ValueError(f"fold must be in [0, {n_folds-1}], got {fold}")
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(skf.split(image_paths, labels))
        train_idx, val_idx = splits[fold]
        
        print(f"ChestXray Fold {fold+1}/{n_folds}:")
        print(f"  Train: {len(train_idx)} samples")
        print(f"  Val: {len(val_idx)} samples")
    
    return train_idx, val_idx


def create_splits_skincancer(
    image_paths: List[str],
    labels: List[int],
    metadata_path: str,
    config: Dict[str, Any],
    fold: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SkinCancer: Use lesion-level GroupKFold to prevent leakage.
    
    Args:
        fold: If None, use single train/val split. If 0-4, use that fold from 5-fold CV.
    
    Returns:
        (train_idx, val_idx)
    """
    seed = config.get("experiment", {}).get("random_seed", 42)
    
    # Load metadata to get lesion IDs
    metadata = pd.read_csv(metadata_path)
    img_to_lesion = dict(zip(metadata['image_id'], metadata['lesion_id']))
    
    # Map image paths to lesion IDs
    lesion_ids = []
    for path in image_paths:
        img_id = Path(path).stem
        if img_id not in img_to_lesion:
            raise ValueError(f"Image {img_id} not found in metadata")
        lesion_ids.append(img_to_lesion[img_id])
    
    lesion_ids = np.array(lesion_ids)
    
    # Stats
    unique_lesions = len(np.unique(lesion_ids))
    print(f"SkinCancer Dataset Info:")
    print(f"  Images: {len(image_paths)}")
    print(f"  Unique lesions: {unique_lesions}")
    print(f"  Avg images/lesion: {len(image_paths)/unique_lesions:.2f}")
    
    if fold is None:
        # Single train/val split
        from sklearn.model_selection import GroupShuffleSplit
        
        val_split = config.get("cv", {}).get("val_split", 0.2)
        gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
        train_idx, val_idx = next(gss.split(image_paths, labels, groups=lesion_ids))
        
        train_lesions = len(np.unique(lesion_ids[train_idx]))
        val_lesions = len(np.unique(lesion_ids[val_idx]))
        
        print(f"SkinCancer Single Split (Lesion-Level):")
        print(f"  Train: {len(train_idx)} images ({train_lesions} lesions)")
        print(f"  Val: {len(val_idx)} images ({val_lesions} lesions)")
    else:
        # K-fold CV
        n_folds = config.get("cv", {}).get("n_folds", 5)
        if not (0 <= fold < n_folds):
            raise ValueError(f"fold must be in [0, {n_folds-1}], got {fold}")
        
        gkf = GroupKFold(n_splits=n_folds)
        splits = list(gkf.split(image_paths, labels, groups=lesion_ids))
        train_idx, val_idx = splits[fold]
        
        train_lesions = len(np.unique(lesion_ids[train_idx]))
        val_lesions = len(np.unique(lesion_ids[val_idx]))
        
        print(f"SkinCancer Fold {fold+1}/{n_folds} (Lesion-Level):")
        print(f"  Train: {len(train_idx)} images ({train_lesions} lesions)")
        print(f"  Val: {len(val_idx)} images ({val_lesions} lesions)")
        
        # Verify no leakage
        train_lesions_set = set(lesion_ids[train_idx])
        val_lesions_set = set(lesion_ids[val_idx])
        overlap = train_lesions_set & val_lesions_set
        if overlap:
            raise RuntimeError(f"LEAKAGE DETECTED: {len(overlap)} lesions in both train and val!")
    
    return train_idx, val_idx


# ============================================================================
# UNIFIED INTERFACE
# ============================================================================

def get_dataset_and_splits(
    dataset: str,
    config: Dict[str, Any],
    fold: Optional[int] = None,
) -> Tuple[List[str], List[int], List[str], List[int], Tuple[np.ndarray, np.ndarray]]:
    """
    Unified interface to load dataset and create splits.
    
    Args:
        dataset: 'brainmri', 'chestxray', or 'skincancer'
        config: Configuration dict
        fold: For CV datasets, which fold to use (0-indexed). None for single split.
    
    Returns:
        (train_paths, train_labels, test_paths, test_labels, (train_idx, val_idx))
        
        For brainmri:
            - train_paths/train_labels: Training directory images
            - test_paths/test_labels: Testing directory images (held-out)
            - train_idx/val_idx: Indices to split train_paths into train/val
        
        For chestxray/skincancer:
            - train_paths/train_labels: All images
            - test_paths/test_labels: Empty (no pre-separated test set)
            - train_idx/val_idx: Indices for CV or single split
    """
    
    if dataset == "brainmri":
        train_paths, train_labels, test_paths, test_labels = load_brainmri_dataset(config)
        train_idx, val_idx = create_splits_brainmri(train_paths, train_labels, config)
        return train_paths, train_labels, test_paths, test_labels, (train_idx, val_idx)
    
    elif dataset == "chestxray":
        all_paths, all_labels, _, _ = load_chestxray_dataset(config)
        train_idx, val_idx = create_splits_chestxray(all_paths, all_labels, config, fold=fold)
        return all_paths, all_labels, [], [], (train_idx, val_idx)
    
    elif dataset == "skincancer":
        all_paths, all_labels, _, _ = load_skincancer_dataset(config)
        metadata_path = config["datasets"]["skincancer"]["metadata_path"]
        train_idx, val_idx = create_splits_skincancer(all_paths, all_labels, metadata_path, config, fold=fold)
        return all_paths, all_labels, [], [], (train_idx, val_idx)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def verify_no_leakage(
    dataset: str,
    train_paths: List[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    metadata_path: Optional[str] = None,
) -> bool:
    """
    Verify no patient/lesion-level leakage between train and val.
    
    Returns:
        True if no leakage, raises error if leakage detected
    """
    if dataset == "skincancer":
        if metadata_path is None:
            print("Warning: Cannot verify leakage without metadata path")
            return True
        
        metadata = pd.read_csv(metadata_path)
        img_to_lesion = dict(zip(metadata['image_id'], metadata['lesion_id']))
        
        train_lesions = set()
        val_lesions = set()
        
        for idx in train_idx:
            img_id = Path(train_paths[idx]).stem
            train_lesions.add(img_to_lesion[img_id])
        
        for idx in val_idx:
            img_id = Path(train_paths[idx]).stem
            val_lesions.add(img_to_lesion[img_id])
        
        overlap = train_lesions & val_lesions
        if overlap:
            raise RuntimeError(
                f"LESION LEAKAGE DETECTED! {len(overlap)} lesions in both train and val. "
                f"This invalidates your results!"
            )
        
        print("✓ No lesion-level leakage detected")
        return True
    
    elif dataset == "brainmri":
        print("✓ BrainMRI: Train/test separated by dataset structure")
        return True
    
    elif dataset == "chestxray":
        print("✓ ChestXray: No patient metadata, assuming no leakage")
        return True
    
    return True


# ============================================================================
# COMPATIBILITY FUNCTIONS FOR run_precision_experiment.py / run_qat_experiment.py
# ============================================================================

def create_cv_splits(
    image_paths: List[str],
    labels: List[int],
    config: Dict[str, Any],
    dataset: Optional[str] = None,
    metadata_path: Optional[str] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create cross-validation splits for datasets.
    
    For brainmri: Single train/val split (test set is separate Testing folder).
    For skincancer: Uses GroupKFold with lesion_id to prevent data leakage.
    For chestxray: Uses StratifiedKFold.
    
    Args:
        image_paths: List of image file paths
        labels: List of labels
        config: Configuration dictionary
        dataset: Dataset name ('brainmri', 'chestxray', 'skincancer')
        metadata_path: Path to metadata CSV (required for skincancer)
    
    Returns:
        List of (train_idx, val_idx) tuples for each fold
    """
    seed = config.get("experiment", {}).get("random_seed", 42)
    n_folds = config.get("cv", {}).get("n_folds", 5)
    val_split = config.get("cv", {}).get("val_split", 0.2)
    
    # Special handling for brainmri - single train/val split (no CV needed)
    # Test set is already separate in the Testing folder
    if dataset == "brainmri":
        train_idx, val_idx = train_test_split(
            np.arange(len(image_paths)),
            test_size=val_split,
            random_state=seed,
            stratify=labels,
        )
        print(f"BrainMRI Split (single train/val, test is separate folder):")
        print(f"  Train: {len(train_idx)} samples ({100*(1-val_split):.0f}%)")
        print(f"  Val: {len(val_idx)} samples ({100*val_split:.0f}%)")
        
        # Return as single-element list so splits[0] works
        return [(np.array(train_idx), np.array(val_idx))]
    
    # Special handling for skincancer - use GroupKFold to prevent leakage
    if dataset == "skincancer":
        if metadata_path is None:
            metadata_path = config.get("datasets", {}).get("skincancer", {}).get("metadata_path")
        
        if metadata_path is None:
            raise ValueError("metadata_path required for skincancer dataset to prevent data leakage")
        
        metadata = pd.read_csv(metadata_path)
        img_to_lesion = dict(zip(metadata['image_id'], metadata['lesion_id']))
        
        # Map image paths to lesion IDs
        lesion_ids = []
        for path in image_paths:
            img_id = Path(path).stem
            if img_id not in img_to_lesion:
                raise ValueError(f"Image {img_id} not found in metadata")
            lesion_ids.append(img_to_lesion[img_id])
        
        lesion_ids = np.array(lesion_ids)
        
        # Use GroupKFold to ensure no lesion appears in both train and val
        gkf = GroupKFold(n_splits=n_folds)
        splits = list(gkf.split(image_paths, labels, groups=lesion_ids))
        
        print(f"SkinCancer CV Splits (Lesion-Level GroupKFold):")
        print(f"  {n_folds} folds, {len(np.unique(lesion_ids))} unique lesions")
        
        return [(np.array(train_idx), np.array(val_idx)) for train_idx, val_idx in splits]
    
    # For chestxray - use StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(skf.split(image_paths, labels))
    
    print(f"ChestXray CV Splits (StratifiedKFold):")
    print(f"  {n_folds} folds")
    
    return [(np.array(train_idx), np.array(val_idx)) for train_idx, val_idx in splits]


def create_cv_splits_with_holdout(
    image_paths: List[str],
    labels: List[int],
    config: Dict[str, Any],
    test_ratio: float = 0.15,
    dataset: Optional[str] = None,
    metadata_path: Optional[str] = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[int]]:
    """
    Create CV splits with a held-out test set.
    
    For skincancer: Uses GroupKFold with lesion_id to prevent data leakage.
    For others: Uses StratifiedKFold.
    
    Args:
        image_paths: List of image file paths
        labels: List of labels
        config: Configuration dictionary
        test_ratio: Fraction of data to hold out for testing
        dataset: Dataset name ('brainmri', 'chestxray', 'skincancer')
        metadata_path: Path to metadata CSV (required for skincancer)
    
    Returns:
        (splits, test_indices) where splits is list of (train_idx, val_idx)
    """
    seed = config.get("experiment", {}).get("random_seed", 42)
    n_folds = config.get("cv", {}).get("n_folds", 5)
    
    all_indices = np.arange(len(image_paths))
    labels_arr = np.array(labels)
    
    # Special handling for skincancer - split by lesion groups
    if dataset == "skincancer":
        if metadata_path is None:
            metadata_path = config.get("datasets", {}).get("skincancer", {}).get("metadata_path")
        
        if metadata_path is None:
            raise ValueError("metadata_path required for skincancer dataset to prevent data leakage")
        
        metadata = pd.read_csv(metadata_path)
        img_to_lesion = dict(zip(metadata['image_id'], metadata['lesion_id']))
        
        # Map image paths to lesion IDs
        lesion_ids = []
        for path in image_paths:
            img_id = Path(path).stem
            if img_id not in img_to_lesion:
                raise ValueError(f"Image {img_id} not found in metadata")
            lesion_ids.append(img_to_lesion[img_id])
        
        lesion_ids = np.array(lesion_ids)
        
        # First split off test set using GroupShuffleSplit (lesion-level)
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        trainval_idx, test_idx = next(gss.split(all_indices, labels_arr, groups=lesion_ids))
        
        # Then create CV splits on remaining data using GroupKFold
        trainval_lesion_ids = lesion_ids[trainval_idx]
        trainval_labels = labels_arr[trainval_idx]
        
        gkf = GroupKFold(n_splits=n_folds)
        splits = []
        for fold_train, fold_val in gkf.split(trainval_idx, trainval_labels, groups=trainval_lesion_ids):
            # Map back to original indices
            train_idx = trainval_idx[fold_train]
            val_idx = trainval_idx[fold_val]
            splits.append((np.array(train_idx), np.array(val_idx)))
        
        # Verify no leakage between train/val and test
        test_lesions = set(lesion_ids[test_idx])
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_lesions = set(lesion_ids[train_idx])
            val_lesions = set(lesion_ids[val_idx])
            
            if train_lesions & test_lesions:
                raise RuntimeError(f"Fold {fold_idx}: LEAKAGE between train and test!")
            if val_lesions & test_lesions:
                raise RuntimeError(f"Fold {fold_idx}: LEAKAGE between val and test!")
            if train_lesions & val_lesions:
                raise RuntimeError(f"Fold {fold_idx}: LEAKAGE between train and val!")
        
        print(f"SkinCancer CV+Holdout Splits (Lesion-Level):")
        print(f"  Test: {len(test_idx)} images ({len(test_lesions)} lesions)")
        print(f"  {n_folds} CV folds on remaining data")
        
        return splits, test_idx.tolist()
    
    # For chestxray and others - use stratified splits
    trainval_idx, test_idx = train_test_split(
        all_indices,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels_arr,
    )
    
    # Then create CV splits on remaining data
    trainval_labels = labels_arr[trainval_idx]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    splits = []
    for fold_train, fold_val in skf.split(trainval_idx, trainval_labels):
        # Map back to original indices
        train_idx = trainval_idx[fold_train]
        val_idx = trainval_idx[fold_val]
        splits.append((np.array(train_idx), np.array(val_idx)))
    
    return splits, test_idx.tolist()


def load_test_dataset(
    dataset: str,
    config: Dict[str, Any],
    image_paths: Optional[List[str]] = None,
    labels: Optional[List[int]] = None,
    test_indices: Optional[List[int]] = None,
) -> Tuple[List[str], List[int]]:
    """
    Load test dataset.
    
    For brainmri: loads from separate Testing folder
    For others: uses provided test_indices
    """
    if dataset == "brainmri":
        _, _, test_paths, test_labels = load_brainmri_dataset(config)
        return test_paths, test_labels
    
    if image_paths is None or labels is None or test_indices is None:
        raise ValueError("image_paths, labels, and test_indices required for non-brainmri datasets")
    
    test_paths = [image_paths[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    return test_paths, test_labels


# Dataset class defined at module level for pickling compatibility
# Note: We use a try/except to handle cases where torch is not installed
try:
    from torch.utils.data import Dataset as TorchDataset
    _TORCH_AVAILABLE = True
except ImportError:
    TorchDataset = object
    _TORCH_AVAILABLE = False


class _SimpleImageDataset(TorchDataset):
    """Simple image dataset for DataLoader."""
    
    def __init__(self, paths: List[str], labels: List[int], transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def create_dataloaders(
    image_paths: List[str],
    labels: List[int],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    config: Dict[str, Any],
    dataset: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Create train and validation dataloaders.
    
    Args:
        image_paths: List of all image paths
        labels: List of all labels
        train_idx: Indices for training set
        val_idx: Indices for validation set
        config: Configuration dictionary
        dataset: Dataset name (for getting correct img_size from config)
    
    Returns:
        (train_loader, val_loader)
    """
    try:
        from torch.utils.data import DataLoader
        import torchvision.transforms as T
    except ImportError as e:
        raise ImportError(f"Required packages not available: {e}")
    
    # Get config values
    batch_size = config.get("training", {}).get("batch_size", 32)
    num_workers = config.get("training", {}).get("num_workers", 4)
    
    # Get img_size from dataset-specific config or use default
    if dataset:
        img_size = config.get("datasets", {}).get(dataset, {}).get("img_size", 224)
    else:
        img_size = 224
    
    # Define transforms
    train_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create datasets
    train_paths = [image_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [image_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    
    train_dataset = _SimpleImageDataset(train_paths, train_labels, train_transform)
    val_dataset = _SimpleImageDataset(val_paths, val_labels, val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def create_test_loader(
    test_paths: List[str],
    test_labels: List[int],
    config: Dict[str, Any],
    dataset: Optional[str] = None,
) -> Any:
    """
    Create test dataloader.
    
    Args:
        test_paths: List of test image paths
        test_labels: List of test labels
        config: Configuration dictionary
        dataset: Dataset name (for getting correct img_size from config)
    
    Returns:
        test_loader
    """
    try:
        from torch.utils.data import DataLoader
        import torchvision.transforms as T
    except ImportError as e:
        raise ImportError(f"Required packages not available: {e}")
    
    batch_size = config.get("training", {}).get("batch_size", 32)
    num_workers = config.get("training", {}).get("num_workers", 4)
    
    # Get img_size from dataset-specific config or use default
    if dataset:
        img_size = config.get("datasets", {}).get(dataset, {}).get("img_size", 224)
    else:
        img_size = 224
    
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_dataset = _SimpleImageDataset(test_paths, test_labels, transform)
    
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )