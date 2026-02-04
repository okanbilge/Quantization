"""
Quick Test Script
Runs a minimal experiment to verify everything works
Uses 1 fold, 1 model, 1 dataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from pathlib import Path

# Import utilities
from utils.data_loader import (
    load_brainmri_dataset, 
    create_cv_splits, 
    create_dataloaders,
    print_dataset_statistics
)


def load_config():
    """Load configuration"""
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config


def quick_test():
    """Run quick test"""
    print("="*70)
    print("Quick Test - Verifying Pipeline")
    print("="*70)
    print("\nThis will:")
    print("  1. Load Brain MRI dataset (smallest)")
    print("  2. Create 1-fold split")
    print("  3. Test data loading")
    print("  4. Verify transforms work")
    print("\nEstimated time: 2-3 minutes")
    print("="*70)
    
    # Load config
    config = load_config()
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load dataset
    print("\n" + "="*70)
    print("Loading Brain MRI Dataset")
    print("="*70)
    
    try:
        image_paths, labels, class_names = load_brainmri_dataset(config)
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        print("\nPlease verify datasets first:")
        print("  python experiments/verify_datasets.py")
        sys.exit(1)
    
    # Print statistics
    print_dataset_statistics(image_paths, labels, class_names)
    
    # Create 1 CV split
    print("\n" + "="*70)
    print("Creating 1-Fold Split")
    print("="*70)
    
    # Temporarily modify config for 1 fold
    config['cross_validation']['n_folds'] = 1
    splits = create_cv_splits(image_paths, labels, config)
    
    train_idx, val_idx = splits[0]
    print(f"\nTrain samples: {len(train_idx)}")
    print(f"Val samples: {len(val_idx)}")
    
    # Create dataloaders
    print("\n" + "="*70)
    print("Creating DataLoaders")
    print("="*70)
    
    try:
        train_loader, val_loader = create_dataloaders(
            image_paths, labels, train_idx, val_idx, config, 'brainmri'
        )
        
        print(f"✓ Train loader: {len(train_loader)} batches")
        print(f"✓ Val loader: {len(val_loader)} batches")
        
    except Exception as e:
        print(f"\n❌ Error creating dataloaders: {e}")
        sys.exit(1)
    
    # Test loading one batch
    print("\n" + "="*70)
    print("Testing Batch Loading")
    print("="*70)
    
    try:
        images, lbls = next(iter(train_loader))
        
        print(f"✓ Batch loaded successfully")
        print(f"  Image batch shape: {images.shape}")
        print(f"  Labels shape: {lbls.shape}")
        print(f"  Image dtype: {images.dtype}")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
        
        # Try moving to device
        images = images.to(device)
        lbls = lbls.to(device)
        print(f"✓ Data moved to {device} successfully")
        
    except Exception as e:
        print(f"\n❌ Error loading batch: {e}")
        sys.exit(1)
    
    # Summary
    print("\n" + "="*70)
    print("Quick Test Results")
    print("="*70)
    print("\n✓ All checks passed!")
    print("\nYour setup is ready for experiments.")
    print("\nNext steps:")
    print("  1. Run single modality test:")
    print("     python experiments/run_single_modality.py --dataset brainmri --folds 2")
    print("\n  2. Or run full multi-modal study:")
    print("     python experiments/run_all_experiments.py")
    print("\n" + "="*70)


if __name__ == '__main__':
    quick_test()
