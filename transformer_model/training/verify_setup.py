#!/usr/bin/env python3
"""
Quick setup verification script.
Checks that all required files and dependencies are in place.
"""

import sys
from pathlib import Path
import torch


def check_file_exists(path, description):
    """Check if a file exists."""
    if Path(path).exists():
        print(f"✓ {description}")
        return True
    else:
        print(f"✗ {description} - NOT FOUND: {path}")
        return False


def check_directory_exists(path, description):
    """Check if a directory exists."""
    if Path(path).is_dir():
        print(f"✓ {description}")
        return True
    else:
        print(f"✗ {description} - NOT FOUND: {path}")
        return False


def main():
    """Run verification checks."""
    print("="*80)
    print("TRAJECTORY PREDICTION SETUP VERIFICATION")
    print("="*80)
    
    # Get project root
    script_dir = Path(__file__).parent
    # Go up one level from training/ or utils/ to root
    project_dir = script_dir.parent if script_dir.name in ('scripts', 'training', 'utils') else script_dir
    
    print(f"\nProject root: {project_dir}\n")
    
    all_good = True
    
    # Check Python version
    print("Python & Dependencies:")
    py_version = sys.version_info
    if py_version.major >= 3 and py_version.minor >= 9:
        print(f"✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        print(f"✗ Python {py_version.major}.{py_version.minor} (requires 3.9+)")
        all_good = False
    
    # Check PyTorch
    try:
        torch_version = torch.__version__
        print(f"✓ PyTorch {torch_version}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available ({torch.cuda.get_device_name(0)})")
        else:
            print("⚠ CUDA not available (will use CPU, training will be slow)")
    except ImportError:
        print("✗ PyTorch not installed")
        all_good = False
    
    # Check required files
    print("\nRequired Files:")
    files_to_check = [
        ('training/model.py', 'Model architecture'),
        ('training/dataset.py', 'Dataset utilities'),
        ('training/train.py', 'Training script'),
        ('training/eval.py', 'Evaluation script'),
        ('training/example_inference.py', 'Inference script'),
        ('utils/preprocess.py', 'Preprocessing script'),
        ('utils/visualize.py', 'Visualization tools'),
        ('utils/inspect_data.py', 'Data inspection'),
    ]
    
    for file_path, desc in files_to_check:
        full_path = project_dir / file_path
        if not check_file_exists(full_path, desc):
            all_good = False
    
    # Check data files
    print("\nData Files:")
    data_files = [
        ('data/train_x.pt', 'Input trajectories (train_x.pt)'),
        ('data/train_y.pt', 'Target trajectories (train_y.pt)'),
        ('data/scene_ids.pt', 'Scene IDs (scene_ids.pt)'),
    ]
    
    for file_path, desc in data_files:
        full_path = project_dir / file_path
        if check_file_exists(full_path, desc):
            # Show file size
            size_mb = full_path.stat().st_size / (1024**2)
            print(f"  └─ Size: {size_mb:.1f} MB")
        else:
            all_good = False
    
    # Check directories
    print("\nDirectories:")
    dirs_to_check = [
        ('training', 'Training scripts directory'),
        ('utils', 'Utility scripts directory'),
        ('data', 'Data directory'),
    ]
    
    for dir_path, desc in dirs_to_check:
        full_path = project_dir / dir_path
        if not check_directory_exists(full_path, desc):
            all_good = False
    
    # Optional: Check dataset in archive
    print("\nOptional (archived materials):")
    if (project_dir / 'extra' / 'nuscenes_data').is_dir():
        print(f"✓ nuScenes dataset (in extra/)")
    else:
        print(f"⚠ nuScenes dataset not found (archived, not needed for training)")
    
    # Check/create checkpoints directory
    checkpoints_dir = project_dir / 'checkpoints'
    if not checkpoints_dir.exists():
        print(f"⚠ Checkpoints directory not found, creating...")
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created checkpoints directory")
    else:
        print(f"✓ Checkpoints directory")
    
    # Try loading data
    print("\nData Verification:")
    try:
        data_dir = project_dir / 'data'
        train_x = torch.load(data_dir / 'train_x.pt', weights_only=True)
        train_y = torch.load(data_dir / 'train_y.pt', weights_only=True)
        scene_ids = torch.load(data_dir / 'scene_ids.pt', weights_only=True)
        
        print(f"✓ Loaded train_x: {train_x.shape}")
        print(f"✓ Loaded train_y: {train_y.shape}")
        print(f"✓ Loaded scene_ids: {scene_ids.shape}")
        
        # Verify shapes
        if train_x.shape == (233948, 4, 2):
            print(f"✓ train_x shape correct")
        else:
            print(f"✗ train_x has unexpected shape: {train_x.shape}")
            all_good = False
        
        if train_y.shape == (233948, 12, 2):
            print(f"✓ train_y shape correct")
        else:
            print(f"✗ train_y has unexpected shape: {train_y.shape}")
            all_good = False
            
        if scene_ids.shape == (233948,):
            print(f"✓ scene_ids shape correct")
        else:
            print(f"✗ scene_ids has unexpected shape: {scene_ids.shape}")
            all_good = False
    
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        all_good = False
    
    # Summary
    print("\n" + "="*80)
    if all_good:
        print("✓ ALL CHECKS PASSED - Ready to train!")
        print("\nNext steps:")
        print("  1. python training/train.py     # Train model")
        print("  2. python training/eval.py      # Evaluate model")
        print("  3. python utils/inspect_data.py # Inspect data")
        return 0
    else:
        print("✗ SOME CHECKS FAILED - Please fix issues above")
        return 1


if __name__ == '__main__':
    sys.exit(main())
