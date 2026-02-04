"""
Single Modality Experiment Runner
Run experiments on one dataset at a time
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
from pathlib import Path

from utils.data_loader import (
    load_chestxray_dataset,
    load_brainmri_dataset,
    load_skincancer_dataset
)


def load_config():
    """Load configuration"""
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_single_modality(dataset_name, n_folds=None):
    """
    Run experiments on single modality
    
    Args:
        dataset_name: 'chestxray', 'brainmri', or 'skincancer'
        n_folds: Number of folds (default from config)
    """
    print("="*70)
    print(f"Single Modality Experiment: {dataset_name.upper()}")
    print("="*70)
    
    # Load config
    config = load_config()
    
    # Override n_folds if specified
    if n_folds:
        config['cross_validation']['n_folds'] = n_folds
    
    # Load dataset
    print(f"\nLoading {dataset_name} dataset...")
    
    if dataset_name == 'chestxray':
        image_paths, labels, class_names = load_chestxray_dataset(config)
    elif dataset_name == 'brainmri':
        image_paths, labels, class_names = load_brainmri_dataset(config)
    elif dataset_name == 'skincancer':
        image_paths, labels, class_names = load_skincancer_dataset(config)
    else:
        print(f"❌ Unknown dataset: {dataset_name}")
        print("   Valid options: chestxray, brainmri, skincancer")
        sys.exit(1)
    
    print(f"\n✓ Loaded {len(image_paths)} images from {len(class_names)} classes")
    
    # Print configuration
    print(f"\nExperiment Configuration:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Number of folds: {config['cross_validation']['n_folds']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Models: {list(config['models']['deep_learning'].keys())}")
    
    print("\n" + "="*70)
    print("NOTE: Full training code will be added")
    print("For now, this verifies data loading works correctly")
    print("="*70)
    
    print("\n✓ Data loading successful!")
    print(f"\nTo run full experiment, the pipeline will:")
    print(f"  1. Create {config['cross_validation']['n_folds']} CV folds")
    print(f"  2. Train deep learning models (FP32)")
    print(f"  3. Train classical ML models")
    print(f"  4. Apply quantization (FP16, INT8, INT4)")
    print(f"  5. Save all results and metrics")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Run single modality experiment')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['chestxray', 'brainmri', 'skincancer'],
                       help='Dataset to use')
    parser.add_argument('--folds', type=int, default=None,
                       help='Number of CV folds (default from config)')
    
    args = parser.parse_args()
    
    run_single_modality(args.dataset, args.folds)


if __name__ == '__main__':
    main()
