"""
Dataset Verification Script
Checks if all datasets are properly downloaded and structured
"""

import os
import sys
from pathlib import Path
import yaml


def load_config():
    """Load configuration"""
    config_path = Path('configs/config.yaml')
    if not config_path.exists():
        print("❌ Config file not found: configs/config.yaml")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def verify_chestxray(config):
    """Verify Chest X-ray dataset"""
    print("\n" + "="*70)
    print("Verifying Chest X-ray Dataset")
    print("="*70)
    
    dataset_config = config['datasets']['chestxray']
    base_path = Path(dataset_config['path'])
    classes = dataset_config['classes']
    
    if not base_path.exists():
        print(f"❌ Dataset path not found: {base_path}")
        print(f"\nExpected structure:")
        print(f"  {base_path}/")
        for cls in classes:
            print(f"    ├── {cls}/images/")
        return False
    
    total_images = 0
    all_ok = True
    
    for class_name in classes:
        class_path = base_path / class_name / 'images'
        
        if not class_path.exists():
            print(f"❌ Class directory not found: {class_path}")
            all_ok = False
            continue
        
        # Count images
        images = list(class_path.glob('*.png')) + list(class_path.glob('*.jpg'))
        num_images = len(images)
        total_images += num_images
        
        if num_images == 0:
            print(f"❌ {class_name}: No images found")
            all_ok = False
        else:
            print(f"✓ {class_name}: {num_images} images")
    
    print(f"\nTotal Chest X-ray images: {total_images}")
    
    if total_images < 100:
        print("⚠️  Warning: Very few images found. Is the dataset fully extracted?")
        all_ok = False
    
    return all_ok


def verify_brainmri(config):
    """Verify Brain MRI dataset"""
    print("\n" + "="*70)
    print("Verifying Brain MRI Dataset")
    print("="*70)
    
    dataset_config = config['datasets']['brainmri']
    base_path = Path(dataset_config['path'])
    classes = dataset_config['classes']
    splits = ['Training', 'Testing']
    
    if not base_path.exists():
        print(f"❌ Dataset path not found: {base_path}")
        print(f"\nExpected structure:")
        print(f"  {base_path}/")
        for split in splits:
            print(f"    ├── {split}/")
            for cls in classes:
                print(f"    │   ├── {cls}/")
        return False
    
    total_images = 0
    all_ok = True
    
    for split in splits:
        split_path = base_path / split
        
        if not split_path.exists():
            print(f"❌ Split directory not found: {split_path}")
            all_ok = False
            continue
        
        print(f"\n{split} split:")
        split_total = 0
        
        for class_name in classes:
            class_path = split_path / class_name
            
            if not class_path.exists():
                print(f"  ❌ {class_name}: Directory not found")
                all_ok = False
                continue
            
            # Count images
            images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
            num_images = len(images)
            split_total += num_images
            
            if num_images == 0:
                print(f"  ❌ {class_name}: No images found")
                all_ok = False
            else:
                print(f"  ✓ {class_name}: {num_images} images")
        
        print(f"  Subtotal: {split_total} images")
        total_images += split_total
    
    print(f"\nTotal Brain MRI images: {total_images}")
    
    if total_images < 100:
        print("⚠️  Warning: Very few images found. Is the dataset fully extracted?")
        all_ok = False
    
    return all_ok


def verify_skincancer(config):
    """Verify Skin Cancer dataset"""
    print("\n" + "="*70)
    print("Verifying Skin Cancer Dataset")
    print("="*70)
    
    dataset_config = config['datasets']['skincancer']
    base_path = Path(dataset_config['path'])
    metadata_file = dataset_config['metadata_file']
    
    if not base_path.exists():
        print(f"❌ Dataset path not found: {base_path}")
        print(f"\nExpected structure:")
        print(f"  {base_path}/")
        print(f"    ├── HAM10000_images_part_1/")
        print(f"    ├── HAM10000_images_part_2/")
        print(f"    └── {metadata_file}")
        return False
    
    all_ok = True
    total_images = 0
    
    # Check image directories
    for part in ['part_1', 'part_2']:
        img_dir = base_path / f'HAM10000_images_{part}'
        
        if not img_dir.exists():
            print(f"❌ Image directory not found: {img_dir}")
            all_ok = False
            continue
        
        images = list(img_dir.glob('*.jpg'))
        num_images = len(images)
        total_images += num_images
        
        print(f"✓ HAM10000_images_{part}: {num_images} images")
    
    # Check metadata
    metadata_path = base_path / metadata_file
    if not metadata_path.exists():
        print(f"⚠️  Metadata file not found: {metadata_path}")
        print("   (Dataset can still work without it)")
    else:
        print(f"✓ Metadata file found: {metadata_file}")
    
    print(f"\nTotal Skin Cancer images: {total_images}")
    
    if total_images < 100:
        print("⚠️  Warning: Very few images found. Is the dataset fully extracted?")
        all_ok = False
    
    return all_ok


def main():
    """Main verification"""
    print("="*70)
    print("Medical Image Quantization - Dataset Verification")
    print("="*70)
    
    # Load config
    config = load_config()
    
    # Verify each dataset
    results = {
        'chestxray': verify_chestxray(config),
        'brainmri': verify_brainmri(config),
        'skincancer': verify_skincancer(config)
    }
    
    # Summary
    print("\n" + "="*70)
    print("Verification Summary")
    print("="*70)
    
    for dataset, status in results.items():
        status_str = "✓ PASS" if status else "❌ FAIL"
        print(f"{dataset.upper()}: {status_str}")
    
    all_pass = all(results.values())
    
    print("\n" + "="*70)
    
    if all_pass:
        print("✓ All datasets verified successfully!")
        print("\nYou can now run experiments:")
        print("  python experiments/quick_test.py")
        print("  python experiments/run_single_modality.py --dataset brainmri")
        print("  python experiments/run_all_experiments.py")
    else:
        print("❌ Some datasets have issues. Please check:")
        print("\n1. Are the ZIP files extracted?")
        print("2. Are they in the correct directories?")
        print("3. See DATASETS.md for expected structure")
        print("\nTo extract:")
        print("  cd data")
        print("  unzip covid19-radiography-database.zip -d chestxray/")
        print("  unzip brain-tumor-mri-dataset.zip -d brainmri/")
        print("  unzip skin-cancer-mnist-ham10000.zip -d skincancer/")
        sys.exit(1)
    
    print("="*70)


if __name__ == '__main__':
    main()
