"""
Inspect preprocessed trajectory data including scene IDs
"""
import torch
from pathlib import Path

# Use relative paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / 'data'

# Load the data
train_x = torch.load(DATA_DIR / 'train_x.pt')
train_y = torch.load(DATA_DIR / 'train_y.pt')
scene_ids = torch.load(DATA_DIR / 'scene_ids.pt')

print("=" * 80)
print("TRAJECTORY DATA INSPECTION")
print("=" * 80)

print("\n[INFO] Data shapes:")
print(f"  train_x (history):  {train_x.shape}")      # (num_traj, 4, 2)
print(f"  train_y (future):   {train_y.shape}")      # (num_traj, 12, 2)
print(f"  scene_ids:          {scene_ids.shape}")    # (num_traj,)

print("\n" + "=" * 80)
print("FIRST 5 TRAJECTORIES")
print("=" * 80)
print("\nFirst 5 histories from train_x:")
print(train_x[:5])
print("\nFirst 5 futures from train_y:")
print(train_y[:5])
print("\nFirst 5 scene IDs:")
print(scene_ids[:5])

print("\n" + "=" * 80)
print("DATA STATISTICS")
print("=" * 80)
print(f"\ntrain_x (history):")
print(f"  min:  {train_x.min():7.3f}")
print(f"  max:  {train_x.max():7.3f}")
print(f"  mean: {train_x.mean():7.3f}")
print(f"  std:  {train_x.std():7.3f}")

print(f"\ntrain_y (future):")
print(f"  min:  {train_y.min():7.3f}")
print(f"  max:  {train_y.max():7.3f}")
print(f"  mean: {train_y.mean():7.3f}")
print(f"  std:  {train_y.std():7.3f}")

print("\n" + "=" * 80)
print("SCENE ID STATISTICS (Anti-Cheating File)")
print("=" * 80)
print(f"\nScene ID range: {scene_ids.min().item()} to {scene_ids.max().item()}")
print(f"Total scenes represented: {scene_ids.max().item() + 1}")
print(f"Total trajectories: {len(scene_ids)}")

# Count trajectories per scene
from collections import Counter
scene_counts = Counter(scene_ids.numpy())
print(f"\nTrajectories per scene (first 10):")
for scene_id in sorted(scene_counts.keys())[:10]:
    print(f"  Scene {scene_id:3d}: {scene_counts[scene_id]:5d} trajectories")

print(f"\nMin trajectories in a scene: {min(scene_counts.values())}")
print(f"Max trajectories in a scene: {max(scene_counts.values())}")
print(f"Avg trajectories per scene: {len(scene_ids) / (scene_ids.max().item() + 1):.1f}")

print("\n" + "=" * 80)
print("TRAIN/TEST SPLIT (Using Scene IDs)")
print("=" * 80)

# Demonstrate clean split
split_point = 700
train_mask = scene_ids < split_point
test_mask = scene_ids >= split_point

print(f"\nSplit point: Scene {split_point}")
print(f"Training set:  Scenes 0-{split_point-1}")
print(f"Test set:      Scenes {split_point}-{scene_ids.max().item()}")

print(f"\nTraining trajectories: {train_mask.sum().item()} (from scenes 0-{split_point-1})")
print(f"Test trajectories:     {test_mask.sum().item()} (from scenes {split_point}-{scene_ids.max().item()})")

x_train = train_x[train_mask]
y_train = train_y[train_mask]
x_test = train_x[test_mask]
y_test = train_y[test_mask]

print(f"\nTraining data shapes:")
print(f"  x_train: {x_train.shape}")
print(f"  y_train: {y_train.shape}")
print(f"\nTest data shapes:")
print(f"  x_test: {x_test.shape}")
print(f"  y_test: {y_test.shape}")

print("\n" + "=" * 80)
print("✓ NO OVERLAP: Training and test sets use completely different scenes")
print("=" * 80)
