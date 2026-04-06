# Trajectory Prediction Project - Changelog

## Overview
This document tracks all changes, improvements, and optimizations made to the trajectory prediction preprocessing and visualization pipeline.

---

## Phase 8: Transformer Training Pipeline (Mar 12)

### ✅ Implemented end-to-end training/eval workflow

**Added:**
- `transformer_model/training/model.py` (Transformer architecture)
- `transformer_model/training/train.py` (training + checkpointing)
- `transformer_model/training/eval.py` (metrics/evaluation)
- `transformer_model/training/dataset.py` (loaders/splits)
- `transformer_model/training/example_inference.py` (inference demo)

**Impact:**
- Full train/eval/inference loop available in one project path.

---

## Phase 9: Architecture Correction (Temporal Tokens) (Mar 12)

### ✅ Fixed transformer tokenization bug

**The problem:**
- Earlier model flattened all history frames into one token before encoder input.
- This prevented meaningful temporal self-attention across history frames.

**The fix:**
- Updated model to embed each history frame as its own token.
- Encoder now attends across temporal sequence as intended.

**Compatibility update:**
- Added legacy checkpoint loading support in evaluation/inference for older flattened checkpoints.

**Impact:**
- Model behavior now aligns with true sequence-transformer design.

---

## Phase 10: Flexible Features + Reporting Outputs (Apr 6)

### ✅ Added plug-and-play feature handling and report-ready outputs

**Training updates (`transformer_model/training/train.py`):**
- CSV epoch logger: `transformer_model/checkpoints/training_metrics.csv`
- Extra best-weight export: `transformer_model/checkpoints/transformer_best.pth`
- Safer long-run defaults (lower LR, longer schedule handling)

**Evaluation updates (`transformer_model/training/eval.py`):**
- Added ADE/FDE min/max statistics
- Added spreadsheet one-line print row
- Added raw prediction tensor export: `transformer_raw_predictions.pt`
- Added trajectory plot exports: `checkpoints/trajectory_plots/*.png`

**Impact:**
- Metrics and artifacts are now ready for final report tables/figures.

---

## Phase 11: Modular Experiment Switchboard (Apr 6)

### ✅ One script supports all phase combinations

**CLI flags added:**
- `--agent {car,pedestrian}`
- `--features {baseline,velocity,map,probabilistic}`

**Where:**
- `transformer_model/training/train.py`
- `transformer_model/training/eval.py`

**Behavior:**
- Auto-resolves experiment data directory first (`data/{agent}/{features}`)
- Falls back safely to baseline `data/` if experiment-specific files are missing
- Uses experiment-specific checkpoint directories by default

**Impact:**
- No separate `train_map.py` / `train_pedestrian.py` scripts needed.

---

## Phase 12: Repository Consolidation (Apr 6)

### ✅ Made root `ece1508` the canonical repo source of truth

**What changed:**
- Resolved nested-repo confusion by consolidating to a single active root repo.
- Preserved migration safety copies in `_migration_backups/`.
- Added ignore rules to avoid accidental backup commits.

**Impact:**
- Single working/push location, reduced branch confusion, safer collaboration.

---

## Phase 1: Data Preprocessing Foundation

### ✅ Created `preprocess.py` (Feb 17)
**Purpose:** Extract vehicle trajectories from nuScenes dataset

**What it does:**
- Loads all 850 scenes from nuScenes v1.0-trainval
- Extracts 2-second history (4 frames) + 6-second future (12 frames) for each vehicle
- Applies ego-centric transformation (rotation + translation)
- Saves as PyTorch tensors

**Output files:**
- `train_x.pt` (7.1 MB) - history trajectories
- `train_y.pt` (22 MB) - future trajectories

**Result:** 231,909 complete trajectories extracted

---

## Phase 2: Performance Optimization

### ✅ Fixed "Moving Anchor" Bug (Feb 20)

**The Problem:**
- Original code got a new `ego_pose` for EVERY frame in the trajectory
- This warped trajectories as the ego vehicle moved
- Transformer would learn "weird physics" instead of real motion

**The Solution:**
- Anchor entire 8-second sequence to ONE fixed ego position (the "present" moment)
- All frames in trajectory transformed relative to same anchor point
- Now physics is mathematically correct

**Impact:** ⚡ Transformer gets honest, realistic training data

### ✅ Killed CPU Bottleneck (Feb 20)

**The Problem:**
- Original code looped through `sample['anns']` multiple times searching for each instance
- Very slow: O(n²) searching through annotations

**The Solution:**
- Use nuScenes' built-in `'next'` token to hop directly between annotations
- Chain annotations together instantly (O(n) traversal)

**Code Change:**
```python
# BEFORE (slow)
for ann_token in sample['anns']:  # Search every time
    if ann['instance_token'] == target:
        ...
        
# AFTER (fast)
current_ann_token = start_ann_token
while current_ann_token:
    ann = nusc.get('sample_annotation', current_ann_token)
    ...
    current_ann_token = ann['next']  # Hop to next instantly
```

**Impact:** ⚡ Processing time reduced by ~50%

---

## Phase 3: Zero-Overlap Train/Test Split

### ✅ Added Scene ID Tracking (Mar 9)

**The Problem:**
```python
# OLD: Random split could cause data leakage
train_set, test_set = random_split(data, 0.8/0.2)

# BAD: Same scene (same car) in both sets!
# Training:  [Frame 5 Scene 0, Frame 12 Scene 0, Frame 35 Scene 0]
# Test:      [Frame 18 Scene 0]  ← Model already saw this car!
```

**The Solution:**
- Track which scene (0-849) each trajectory came from
- Split by SCENES, not random frames
- NO overlap possible - different scenes = different cars, roads, drivers

```python
# NEW: Clean scene-based split
train_mask = scene_ids < 700  # Scenes 0-699
test_mask = scene_ids >= 700   # Scenes 700-849

# GOOD: Complete scene separation
# Training: ALL frames from Scenes 0-699
# Test:     ALL frames from Scenes 700-849
```

**New File:** `scene_ids.pt` (long tensor)
- Records which scene each trajectory came from
- Enables bulletproof train/test split

**Impact:** 🎯 Results are now honest and reproducible

---

## Phase 4: Visualization Tools

### ✅ Created `visualize.py` (Mar 3)

**Purpose:** Multiple visualization options for trajectory inspection

**Available Tools:**
1. **Matplotlib** - Static 2D plots (fast)
2. **Plotly** - Interactive HTML plots (beautiful)
3. **Open3D** - 3D visualization (cool)
4. **TensorBoard** - Statistics & logging (detailed)

**Usage:**
```bash
python scripts/visualize.py --tool matplotlib --trajectories 10
python scripts/visualize.py --tool plotly --trajectories 10
python scripts/visualize.py --tool open3d --trajectories 10
python scripts/visualize.py --tool tensorboard
```

**Color Scheme:**
- 🔵 Blue = history (2 seconds)
- 🔴 Red = future (6 seconds)
- 🟢 Green = ego vehicle at origin

---

## Phase 5: File Organization

### ✅ Organized Project Structure (Mar 8)

**Before:**
```
/media/robotswithai/data1/ece1508/
  ├── preprocess.py (root)
  ├── train_x.pt    (root)
  ├── train_y.pt    (root)
  └── scattered files
```

**After:**
```
/media/robotswithai/data1/ece1508/
  ├── scripts/
  │   ├── preprocess.py
  │   ├── visualize.py
  │   ├── inspect_data.py
  │   └── preprocess_scene_ids.log
  ├── data/
  │   ├── train_x.pt
  │   ├── train_y.pt
  │   └── scene_ids.pt
  ├── nuscenes_data/
  └── nuscenes-devkit/
```

**Impact:** ✅ Clean, organized, professional structure

---

## Phase 6: Portable Configuration

### ✅ Removed Hardcoded Paths (Mar 11)

**The Problem:**
```python
# Hardcoded paths tied to YOUR machine
DATAROOT = '/media/robotswithai/data1/ece1508/nuscenes_data'
OUTPUT_DIR = '/media/robotswithai/data1/ece1508/data'

# If project moves anywhere else = BROKEN
```

**The Solution:**
```python
# Relative paths based on script location
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent
DATAROOT = PROJECT_DIR / 'nuscenes_data'
OUTPUT_DIR = PROJECT_DIR / 'data'

# Works ANYWHERE the folder structure is preserved
```

**Benefits:**
- ✅ Portable between machines
- ✅ Shareable on GitHub
- ✅ Works for teammates with different folder locations
- ✅ Professional code

---

## Phase 7: Data Inspection

### ✅ Updated `inspect_data.py` (Mar 11)

**Purpose:** Verify preprocessed data integrity

**Now Shows:**
- File shapes and data types
- Statistics (min, max, mean, std)
- **Scene ID distribution** (trajectories per scene)
- **Train/Test split demonstration**
- **Zero-overlap verification**

**Usage:**
```bash
python inspect_data.py
```

**Output Includes:**
```
Scene ID range: 0 to 849
Total scenes: 850
Total trajectories: 231,909

Training set:  Scenes 0-699   (X trajectories)
Test set:      Scenes 700-849 (Y trajectories)
✓ NO OVERLAP: Training and test sets use completely different scenes
```

---

## GitHub Integration

### ✅ Pushed to GitHub (Mar 4-11)

**Repository:** https://github.com/ploewen/Trajectory_Prediction
**Branch:** `transformer-model`

**Commits:**
1. Initial `preprocess.py` with optimizations
2. Scene ID tracking implementation
3. Relative paths for portability

**Files Shared:**
- `preprocess.py` - complete preprocessing pipeline
- `visualize.py` - visualization tools
- `inspect_data.py` - data inspection

---

## Summary of Key Improvements

| Feature | Before | After | Impact |
|---------|--------|-------|--------|
| **Trajectory Physics** | ❌ Warped (moving anchor) | ✅ Correct (fixed anchor) | Math is honest |
| **Processing Speed** | ❌ Slow (searching) | ✅ Fast (token hopping) | 50% faster |
| **Train/Test Split** | ❌ Random (data leakage) | ✅ Scene-based (no overlap) | Results trustworthy |
| **File Organization** | ❌ Scattered | ✅ Organized | Professional |
| **Portability** | ❌ Hardcoded paths | ✅ Relative paths | Shareable/portable |
| **Visualization** | ❌ Basic plots | ✅ 4 tools available | Flexible inspection |
| **Documentation** | ❌ None | ✅ Full inspection | Easy verification |

---

## Data Pipeline Summary

```
Raw nuScenes Data (850 scenes, 39,000 samples)
           ↓
    [preprocess.py]
           ↓
    ┌─────┴─────┬──────────┐
    ↓           ↓          ↓
 train_x.pt  train_y.pt  scene_ids.pt
 (history)   (future)    (anti-cheat)
    ↓           ↓          ↓
    └─────┬─────┴──────────┘
          ↓
    [inspect_data.py]
          ↓
   ┌──────┴──────┐
   ↓             ↓
Training      Testing
(Scenes 0-699) (Scenes 700-849)
```

---

## Next Steps

- [ ] Add experiment-specific prepared datasets for each `--agent/--features` combination
- [ ] Run full benchmark table for all 6 planned phases
- [ ] Add per-agent/per-feature comparison plots for report appendix
- [ ] Optional: tune hyperparameters separately for car vs pedestrian splits

---

**Last Updated:** April 6, 2026
**Project Status:** Modular transformer pipeline ✅ | Report artifact generation ✅
