"""
Dataset and DataLoader utilities for trajectory prediction.
Handles loading and splitting data using scene IDs to prevent data leakage.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for trajectory prediction."""

    def __init__(self, x, y, device="cpu"):
        """
        Args:
            x: Input trajectories of shape (N, num_input_frames, num_features)
            y: Target trajectories of shape (N, num_output_frames, num_features)
            device: Device to load tensors onto
        """
        self.x = x.to(device)
        self.y = y.to(device)
        self.device = device

        assert x.shape[0] == y.shape[0], "Mismatched number of samples"
        assert len(x.shape) == 3, f"x must be 3D (N, frames, features), got {x.shape}"
        assert len(y.shape) == 3, f"y must be 3D (N, frames, features), got {y.shape}"
        assert x.shape[2] == y.shape[2], (
            f"Feature mismatch: x has {x.shape[2]}, y has {y.shape[2]}"
        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def load_data(data_dir=None, device="cpu"):
    """
    Load trajectory data with flexible feature stacking.

    TRULY PLUG-AND-PLAY:
    - Always loads: train_x.pt (position: x, y)
    - Auto-detects & loads: extra_features_*.pt files in sorted order
      (e.g., extra_features_continuous.pt, extra_features_categorical.pt, etc.)
    - Concatenates everything: [x, y] + [extra_1] + [extra_2] + ...

    Example scenarios:
    1. Position only:
       Files: train_x.pt
       Result: (N, 4, 2)

    2. Position + Velocity:
       Files: train_x.pt, extra_features_continuous.pt
       Result: (N, 4, 2+2) = (N, 4, 4) → [x, y, vx, vy]

    3. Position + Type (one-hot):
       Files: train_x.pt, extra_features_categorical.pt
       Result: (N, 4, 2+2) = (N, 4, 4) → [x, y, is_car, is_ped]

    4. Position + Velocity + Type:
       Files: train_x.pt, extra_features_categorical.pt, extra_features_continuous.pt
       Result: (N, 4, 2+2+2) = (N, 4, 6) → [x, y, is_car, is_ped, vx, vy]

    Args:
        data_dir: Directory containing data files. If None, uses relative path
        device: Device to load tensors onto

    Returns:
        Tuple of (train_x, train_y, scene_ids) as tensors on specified device
        train_x shape depends on what extra_features files are present
    """
    if data_dir is None:
        script_dir = Path(__file__).parent
        project_dir = script_dir.parent
        data_dir = project_dir / "data"
    else:
        data_dir = Path(data_dir)

    # Load position data (always required)
    train_x = torch.load(data_dir / "train_x.pt", weights_only=True).to(device)
    train_y = torch.load(data_dir / "train_y.pt", weights_only=True).to(device)
    scene_ids = torch.load(data_dir / "scene_ids.pt", weights_only=True).to(device)

    print("  Base features: 2 (position: x, y)")

    # Auto-detect and load extra features in sorted order
    extra_files = sorted(data_dir.glob("extra_features_*.pt"))

    if extra_files:
        feature_list = ["position (x, y)"]
        for extra_file in extra_files:
            extra_features = torch.load(extra_file, weights_only=True).to(device)
            feature_name = extra_file.stem.replace("extra_features_", "")

            # Skip stale tensors that do not match the base dataset size.
            if extra_features.shape[0] != train_x.shape[0]:
                print(
                    f"  [SKIP] {feature_name}: sample count {extra_features.shape[0]} does not match train_x {train_x.shape[0]}"
                )
                continue

            # Handle shape: (N, frames, features) or (N, features) with no frame dim
            if extra_features.dim() == 2:
                # No frame dimension - expand it
                num_frames = train_x.shape[1]
                extra_features = extra_features.unsqueeze(1).expand(-1, num_frames, -1)
            elif (
                extra_features.dim() == 3
                and extra_features.shape[1] != train_x.shape[1]
            ):
                print(
                    f"  [SKIP] {feature_name}: frame count {extra_features.shape[1]} does not match train_x {train_x.shape[1]}"
                )
                continue

            num_extra_features = extra_features.shape[2]
            train_x = torch.cat([train_x, extra_features], dim=2)
            feature_list.append(f"{feature_name} ({num_extra_features} dims)")
            print(f"  [FEATURE LOADED] {feature_name}: +{num_extra_features} features")

        print(f"  Total features: {train_x.shape[2]} ({', '.join(feature_list)})")
    else:
        print(
            f"  No extra features found. To add features, place extra_features_*.pt files in {data_dir}/"
        )

    return train_x, train_y, scene_ids


def create_dataloaders(
    train_x,
    train_y,
    scene_ids,
    train_scene_cutoff=700,
    batch_size=32,
    num_workers=0,
    device="cpu",
    shuffle_train=True,
):
    """
    Create train and test DataLoaders using scene-based split.

    This prevents data leakage by ensuring trajectories from the same scene
    don't appear in both training and test sets.

    Args:
        train_x: Full history tensor (N, 4, 2)
        train_y: Full target tensor (N, 12, 2)
        scene_ids: Scene IDs for each trajectory (N,)
        train_scene_cutoff: Scenes < cutoff are training, >= cutoff are test
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        device: Device for data loading
        shuffle_train: Whether to shuffle training data

    Returns:
        Tuple of (train_loader, test_loader, train_indices, test_indices)
    """
    # Create train/test split masks
    train_mask = scene_ids < train_scene_cutoff
    test_mask = scene_ids >= train_scene_cutoff

    # Get indices
    train_indices = torch.where(train_mask)[0]
    test_indices = torch.where(test_mask)[0]

    # Create subsets
    x_train = train_x[train_indices]
    y_train = train_y[train_indices]
    x_test = train_x[test_indices]
    y_test = train_y[test_indices]

    print(
        f"Train set: {len(x_train):,} trajectories (scenes 0-{train_scene_cutoff - 1})"
    )
    print(f"Test set:  {len(x_test):,} trajectories (scenes {train_scene_cutoff}-849)")

    # Create datasets
    train_dataset = TrajectoryDataset(x_train, y_train, device=device)
    test_dataset = TrajectoryDataset(x_test, y_test, device=device)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader, train_indices, test_indices


if __name__ == "__main__":
    # Test data loading
    print("Loading trajectory data...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_x, train_y, scene_ids = load_data(device=device)
    print(f"Loaded data on device: {device}")
    print(f"  train_x: {train_x.shape}")
    print(f"  train_y: {train_y.shape}")
    print(f"  scene_ids: {scene_ids.shape}")

    print("\nCreating dataloaders...")
    train_loader, test_loader, train_idx, test_idx = create_dataloaders(
        train_x, train_y, scene_ids, batch_size=32, device=device
    )

    print(f"\nTrain loader: {len(train_loader)} batches of size 32")
    print(f"Test loader:  {len(test_loader)} batches of size 32")

    # Inspect first batch
    x_batch, y_batch = next(iter(train_loader))
    print("\nFirst batch:")
    print(f"  x_batch: {x_batch.shape}")
    print(f"  y_batch: {y_batch.shape}")
