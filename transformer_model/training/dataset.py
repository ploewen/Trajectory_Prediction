"""
Dataset and DataLoader utilities for trajectory prediction.
Uses a flat per-agent data layout and explicit feature files.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for trajectory prediction."""
    
    def __init__(self, x, y, device='cpu', features='baseline'):
        """
        Args:
            x: Input trajectories of shape (N, num_input_frames, num_features)
            y: Target trajectories of shape (N, num_output_frames, num_features)
            device: Device to load tensors onto
        """
        self.x = x.to(device)
        self.y = y.to(device)
        self.device = device
        self.features = features
        
        assert x.shape[0] == y.shape[0], "Mismatched number of samples"
        assert len(x.shape) == 3, f"x must be 3D (N, frames, features), got {x.shape}"
        assert len(y.shape) == 3, f"y must be 3D (N, frames, features), got {y.shape}"
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def load_data(data_dir=None, agent='car', features='baseline', device='cpu'):
    """
    Args:
        data_dir: Directory containing data files. If None, uses data/{agent}/
        agent: Agent type folder name
        features: Explicit feature mode
        device: Device to load tensors onto
    
    Returns:
        Tuple of (train_x, train_y, scene_ids) as tensors on specified device
    """
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    if data_dir is None:
        data_dir = project_dir / 'data' / agent
    else:
        data_dir = Path(data_dir)

    required = ['train_x.pt', 'train_y.pt', 'scene_ids.pt']
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required data files in {data_dir}: {', '.join(missing)}"
        )
    
    # Load position data (always required)
    train_x = torch.load(data_dir / 'train_x.pt', weights_only=True).to(device)
    train_y = torch.load(data_dir / 'train_y.pt', weights_only=True).to(device)
    scene_ids = torch.load(data_dir / 'scene_ids.pt', weights_only=True).to(device)
    
    print(f"  Base features: 2 (position: x, y)")

    use_velocity = features in ['velocity', 'probabilistic_velocity']
    use_map = features == 'map'

    if use_velocity:
        extra_path = data_dir / 'velocity.pt'
        if not extra_path.exists():
            raise FileNotFoundError(
                f"Requested velocity features, but {extra_path} does not exist. "
                f"Run preprocess.py --agent {agent} --extract velocity first."
            )
        extra_features = torch.load(extra_path, weights_only=True).to(device)
        if extra_features.dim() == 2:
            extra_features = extra_features.unsqueeze(1).expand(-1, train_x.shape[1], -1)
        train_x = torch.cat([train_x, extra_features], dim=2)
        print(f"  [FEATURE LOADED] velocity.pt: +{extra_features.shape[2]} features")
        print(f"  Total features: {train_x.shape[2]} (position + velocity)")
    elif use_map:
        extra_path = data_dir / 'map.pt'
        if not extra_path.exists():
            raise FileNotFoundError(
                f"Requested map features, but {extra_path} does not exist."
            )
        extra_features = torch.load(extra_path, weights_only=True).to(device)
        if extra_features.dim() == 2:
            extra_features = extra_features.unsqueeze(1).expand(-1, train_x.shape[1], -1)
        train_x = torch.cat([train_x, extra_features], dim=2)
        print(f"  [FEATURE LOADED] map.pt: +{extra_features.shape[2]} features")
        print(f"  Total features: {train_x.shape[2]} (position + map)")
    else:
        print(f"  Using base data only for features={features}")
    
    return train_x, train_y, scene_ids


def create_dataloaders(
    train_x,
    train_y,
    scene_ids,
    train_scene_cutoff=700,
    batch_size=32,
    num_workers=0,
    device='cpu',
    shuffle_train=True
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
    
    print(f"Train set: {len(x_train):,} trajectories (scenes 0-{train_scene_cutoff-1})")
    print(f"Test set:  {len(x_test):,} trajectories (scenes {train_scene_cutoff}-849)")
    
    # Create datasets
    train_dataset = TrajectoryDataset(x_train, y_train, device=device, features='baseline')
    test_dataset = TrajectoryDataset(x_test, y_test, device=device, features='baseline')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader, train_indices, test_indices


if __name__ == '__main__':
    # Test data loading
    print("Loading trajectory data...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_x, train_y, scene_ids = load_data(agent='car', features='baseline', device=device)
    print(f"Loaded data on device: {device}")
    print(f"  train_x: {train_x.shape}")
    print(f"  train_y: {train_y.shape}")
    print(f"  scene_ids: {scene_ids.shape}")
    
    print("\nCreating dataloaders...")
    train_loader, test_loader, train_idx, test_idx = create_dataloaders(
        train_x, train_y, scene_ids,
        batch_size=32,
        device=device
    )
    
    print(f"\nTrain loader: {len(train_loader)} batches of size 32")
    print(f"Test loader:  {len(test_loader)} batches of size 32")
    
    # Inspect first batch
    x_batch, y_batch = next(iter(train_loader))
    print(f"\nFirst batch:")
    print(f"  x_batch: {x_batch.shape}")
    print(f"  y_batch: {y_batch.shape}")
