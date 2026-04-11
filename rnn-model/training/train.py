"""
Training script for RNN-based trajectory prediction model with staged experiments.
Includes deterministic and probabilistic (Gaussian) variants.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import time
from tqdm import tqdm
from dataclasses import dataclass
import argparse
import numpy as np
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt

from model import create_model, LSTMGRUPredictor, ProbabilisticLSTMGRUPredictor
from dataset import load_data, create_dataloaders


@dataclass
class ExperimentStage:
    """Configuration for an experiment stage."""

    name: str  # e.g., "cars_baseline"
    description: str  # e.g., "Cars with position only"
    agent_types: List[str]  # ["vehicle"] or ["pedestrian"]
    include_velocity: bool  # Whether to add velocity features
    selected_map_features: Optional[List[str]]  # None or list of map feature names
    probabilistic: bool = False  # Whether to train with Gaussian uncertainty output
    append_map_to_existing: bool = False  # Preserve existing non-position features


# Define experiment stages
DEFAULT_EXPERIMENT_STAGES = [
    ExperimentStage(
        name="cars_baseline",
        description="Cars: position only",
        agent_types=["vehicle"],
        include_velocity=False,
        selected_map_features=None,
    ),
    ExperimentStage(
        name="cars_velocity",
        description="Cars: position + velocity",
        agent_types=["vehicle"],
        include_velocity=True,
        selected_map_features=None,
    ),
    ExperimentStage(
        name="cars_map",
        description="Cars: position + lane_type_road + drivable_area",
        agent_types=["vehicle"],
        include_velocity=False,
        selected_map_features=["lane_type_road", "drivable_area"],
    ),
    ExperimentStage(
        name="pedestrian_baseline",
        description="Pedestrian: position only",
        agent_types=["pedestrian"],
        include_velocity=False,
        selected_map_features=None,
    ),
    ExperimentStage(
        name="pedestrian_map",
        description="Pedestrian: position + has_crosswalk + drivable_area",
        agent_types=["pedestrian"],
        include_velocity=False,
        selected_map_features=["has_crosswalk", "drivable_area"],
    ),
    ExperimentStage(
        name="cars_probabilistic",
        description="Cars: probabilistic Gaussian output (mean + uncertainty)",
        agent_types=["vehicle"],
        include_velocity=False,
        selected_map_features=None,
        probabilistic=True,
    ),
]


def extract_agent_types_from_nuscenes():
    """
    Extract agent types (vehicle vs pedestrian) from nuScenes data.
    Creates a tensor aligned with train_x.

    Agent type encoding: 0=pedestrian, 1=vehicle (from metadata)

    Returns:
        agent_types: Tensor of shape (N,) with values 0=pedestrian, 1=vehicle
    """
    from nuscenes.nuscenes import NuScenes

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / "data"

    print("[INFO] Extracting agent types from nuScenes dataset...")

    # Load NuScenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(data_dir), verbose=False)
    train_x = torch.load(data_dir / "train_x.pt", weights_only=True)
    scene_ids = torch.load(data_dir / "scene_ids.pt", weights_only=True)

    # Reverse mapping: for each scene_id, track which annotations we've seen
    all_agent_types = []
    instance_cache = {}
    category_cache = {}

    # Map scene indices to nuScenes scene tokens
    scene_tokens = [nusc.scene[i]["token"] for i in range(len(nusc.scene))]

    sample_idx = 0
    for scene_idx, scene_token in enumerate(scene_tokens):
        scene = nusc.get("scene", scene_token)
        first_sample_token = scene["first_sample_token"]
        sample_token = first_sample_token

        while sample_token:
            sample = nusc.get("sample", sample_token)

            # Get all annotations in this sample
            for ann_token in sample["anns"]:
                if ann_token not in instance_cache:
                    instance = nusc.get("instance", ann_token)
                    instance_cache[ann_token] = instance
                else:
                    instance = instance_cache[ann_token]

                # Get category
                category_token = instance["category_token"]
                if category_token not in category_cache:
                    category = nusc.get("category", category_token)
                    category_cache[category_token] = category
                else:
                    category = category_cache[category_token]

                category_name = category["name"]

                # Skip non-vehicle/non-pedestrian
                if not ("vehicle" in category_name or "human" in category_name):
                    continue

                # Track agent type: 0=pedestrian (human.*), 1=vehicle
                agent_type = 0 if "human" in category_name else 1
                all_agent_types.append(agent_type)

            sample_token = sample["next"]

    agent_types = torch.tensor(all_agent_types[: len(train_x)], dtype=torch.long)

    # Pad if needed (shouldn't happen if extraction is correct)
    if len(agent_types) < len(train_x):
        print(
            f"[WARNING] Extracted {len(agent_types)} agent types but train_x has {len(train_x)} samples"
        )
        padding = torch.ones(len(train_x) - len(agent_types), dtype=torch.long)
        agent_types = torch.cat([agent_types, padding])

    print(f"[INFO] Extracted agent types. Shape: {agent_types.shape}")
    print(f"  - Pedestrians: {(agent_types == 0).sum().item()}")
    print(f"  - Vehicles: {(agent_types == 1).sum().item()}")

    # Save
    torch.save(agent_types, data_dir / "agent_types.pt")

    return agent_types


def load_agent_types():
    """Load agent types tensor, creating it if needed."""
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / "data"
    agent_types_path = data_dir / "agent_types.pt"

    if agent_types_path.exists():
        agent_types = torch.load(agent_types_path, weights_only=True)

        # Check if file contains valid data (has & pedestrians)
        # If it's all 1's (vehicles), it's likely stale/default data that needs regeneration
        num_pedestrians = (agent_types == 0).sum().item()
        if num_pedestrians == 0 and len(agent_types) > 0:
            print("[WARNING] Agent types file contains no pedestrians (all vehicles).")
            print("[INFO] This may be stale data. Regenerating agent types...")
            agent_types = extract_agent_types_from_nuscenes()
            if agent_types is not None:
                return agent_types

        return agent_types
    else:
        print("Agent types not found. Extracting from nuScenes...")
        agent_types = extract_agent_types_from_nuscenes()
        if agent_types is None:
            # Fallback: assume all vehicles (type 1)
            train_x = torch.load(data_dir / "train_x.pt", weights_only=True)
            agent_types = torch.ones(len(train_x), dtype=torch.long)
            torch.save(agent_types, agent_types_path)
        return agent_types


def load_agent_type_metadata():
    """Load agent type metadata to identify vehicle vs pedestrian."""
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / "data"
    metadata_path = data_dir / "agent_type_metadata.pt"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Agent type metadata not found at {metadata_path}")

    metadata = torch.load(metadata_path, weights_only=False)
    return metadata


def filter_by_agent_type(train_x, train_y, agent_types_tensor, agent_types_to_keep):
    """
    Filter data by agent type.

    Args:
        train_x: Input trajectories
        train_y: Target trajectories
        agent_types_tensor: Tensor with agent type for each sample (0=vehicle, 1=pedestrian)
        agent_types_to_keep: List of agent type names (["vehicle"], ["pedestrian"], etc.)

    Returns:
        Filtered (train_x, train_y, mask_indices)
    """
    # Create mask for desired agent types
    mask = torch.zeros(len(agent_types_tensor), dtype=torch.bool)

    metadata = load_agent_type_metadata()
    type_to_idx = metadata["agent_type_to_idx"]  # {'pedestrian': 0, 'vehicle': 1}

    for agent_type in agent_types_to_keep:
        if agent_type in type_to_idx:
            idx = type_to_idx[agent_type]
            mask |= agent_types_tensor == idx

    # Apply mask
    filtered_x = train_x[mask]
    filtered_y = train_y[mask]
    mask_indices = torch.where(mask)[0]

    return filtered_x, filtered_y, mask_indices


def maybe_append_velocity_features(train_x, train_y):
    """
    Compute and append velocity features (vx, vy) to trajectory data.

    Velocity is computed as the difference between consecutive frames.

    Args:
        train_x: Input trajectories of shape (N, T_in, 2)
        train_y: Target trajectories of shape (N, T_out, 2)

    Returns:
        (train_x_with_vel, train_y_with_vel) with shape (N, T, 4)
    """
    # Compute velocity as frame-to-frame delta
    # For input: velocity at frame i = position[i] - position[i-1]
    # Use zero as velocity for first frame

    # Compute velocity for train_x
    x_pos = train_x[:, :, :2].clone()
    x_vel = torch.zeros_like(x_pos)
    x_vel[:, 1:, :] = x_pos[:, 1:, :] - x_pos[:, :-1, :]  # Delta from previous frame
    x_with_vel = torch.cat([x_pos, x_vel], dim=2)  # (N, T_in, 4)

    # Compute velocity for train_y
    y_pos = train_y[:, :, :2].clone()
    y_vel = torch.zeros_like(y_pos)
    y_vel[:, 1:, :] = y_pos[:, 1:, :] - y_pos[:, :-1, :]
    y_with_vel = torch.cat([y_pos, y_vel], dim=2)  # (N, T_out, 4)

    return x_with_vel, y_with_vel


def apply_map_feature_selection(
    train_x, train_y, selected_features, sample_indices=None, append_to_existing=False
):
    """
    Select specific map features from the full feature set.

    If map features are unavailable or have mismatched sample count,
    returns the input data unchanged with a warning.

    Assumes: train_x/train_y have shape (N, T, num_features)
    where first 2 features are (x, y) and remaining are additional features.

    Args:
        train_x: Input trajectories
        train_y: Target trajectories
        selected_features: List of feature names to select (e.g., ["lane_type_road", "drivable_area"])
        sample_indices: Optional indices into the full dataset for the current subset
        append_to_existing: If True, append selected map features after existing features

    Returns:
        (train_x_subset, train_y_subset) with position + selected map features,
        or (train_x, train_y) unchanged if map features unavailable
    """
    # Load the full map features tensor
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / "data"
    map_features_path = data_dir / "extra_features_map.pt"

    if not map_features_path.exists():
        print(f"    !! Map features file not found: {map_features_path}")
        print(f"    Continuing with position only")
        return train_x, train_y

    try:
        full_map_features = torch.load(map_features_path, weights_only=True)
    except Exception as e:
        print(f"    !! Could not load map features: {e}")
        print(f"    Continuing with position only")
        return train_x, train_y

    target_device = train_x.device

    # Align map features to filtered subset when provided.
    if full_map_features.shape[0] != train_x.shape[0]:
        if sample_indices is not None and len(sample_indices) == train_x.shape[0]:
            try:
                sample_indices = sample_indices.to(full_map_features.device)
                full_map_features = full_map_features[sample_indices]
            except Exception as e:
                print(f"    !! Could not align map features with filtered indices: {e}")
                print(f"    Continuing with position only")
                return train_x, train_y
        else:
            print(
                f"    !! Map features sample count ({full_map_features.shape[0]}) != train_x ({train_x.shape[0]})"
            )
            print(
                f"    Map features appear stale or misaligned. Continuing with position only"
            )
            return train_x, train_y

    # Get map feature names from metadata
    metadata = load_agent_type_metadata()
    feature_index_map = {feat: idx for idx, feat in enumerate(metadata["map_features"])}

    # Extract selected feature indices
    selected_indices = []
    for feat in selected_features:
        if feat in feature_index_map:
            selected_indices.append(feature_index_map[feat])
        else:
            print(
                f"    Warning: Map feature '{feat}' not found. Available: {list(feature_index_map.keys())}"
            )

    if not selected_indices:
        print(f"    No valid map features found. Continuing with position only")
        return train_x, train_y

    # Keep map features on the same device as train_x/train_y.
    full_map_features = full_map_features.to(target_device)

    # Extract position from train_x/y
    x_pos = train_x[:, :, :2]
    y_pos = train_y[:, :, :2]

    # Extract selected map features for all frames
    # Map features shape: (N, num_map_features)
    # Expand to all frames by repeating across time dimension
    x_selected_map = (
        full_map_features[:, selected_indices]
        .unsqueeze(1)
        .expand(-1, train_x.shape[1], -1)
    )
    y_selected_map = (
        full_map_features[:, selected_indices]
        .unsqueeze(1)
        .expand(-1, train_y.shape[1], -1)
    )

    # Concatenate desired features.
    if append_to_existing and train_x.shape[2] > 2:
        x_existing = train_x[:, :, 2:]
        y_existing = train_y[:, :, 2:] if train_y.shape[2] > 2 else None
        x_with_map = torch.cat([x_pos, x_existing, x_selected_map], dim=2)
        if y_existing is not None:
            y_with_map = torch.cat([y_pos, y_existing, y_selected_map], dim=2)
        else:
            y_with_map = torch.cat([y_pos, y_selected_map], dim=2)
    else:
        x_with_map = torch.cat([x_pos, x_selected_map], dim=2)
        y_with_map = torch.cat([y_pos, y_selected_map], dim=2)

    print(
        f"    Selected {len(selected_indices)} map features: {[metadata['map_features'][i] for i in selected_indices]}"
    )

    return x_with_map, y_with_map


def save_training_curves(history: Dict, output_dir: Path):
    """
    Save training and validation curves as PNG.

    Args:
        history: Dictionary with epoch, train_loss, val_loss, train_ade, val_ade, etc.
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = history["epoch"]

    # Loss curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    ax.plot(epochs, history["val_loss"], label="Val Loss", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png", dpi=150)
    plt.close()

    # ADE/FDE curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ADE
    ax1.plot(epochs, history["train_ade"], label="Train ADE", marker="o", markersize=3)
    ax1.plot(epochs, history["val_ade"], label="Val ADE", marker="s", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("ADE (meters)")
    ax1.set_title("Average Displacement Error")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # FDE
    ax2.plot(epochs, history["train_fde"], label="Train FDE", marker="o", markersize=3)
    ax2.plot(epochs, history["val_fde"], label="Val FDE", marker="s", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("FDE (meters)")
    ax2.set_title("Final Displacement Error")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "ade_fde_curves.png", dpi=150)
    plt.close()


def save_sample_predictions_plot(
    model, test_loader, output_dir: Path, device, num_samples=4, probabilistic=False
):
    """
    Save sample predictions with ground truth for visual inspection.

    Args:
        model: Trained model
        test_loader: Test DataLoader
        output_dir: Directory to save plots
        device: Device model is on
        num_samples: Number of sample predictions to visualize
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    all_x = []
    all_y_pred = []
    all_y_true = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            model_output = model(x_batch)
            y_pred, _ = unpack_model_output(model_output)

            all_x.append(x_batch.cpu().numpy())
            all_y_pred.append(y_pred.cpu().numpy())
            all_y_true.append(y_batch.cpu().numpy())

            if sum(len(x) for x in all_x) >= num_samples:
                break

    x_all = np.concatenate(all_x, axis=0)[:num_samples]
    y_pred_all = np.concatenate(all_y_pred, axis=0)[:num_samples]
    y_true_all = np.concatenate(all_y_true, axis=0)[:num_samples]

    # Validate shapes
    print(f"    Visualization data shapes:")
    print(f"      x_all (history): {x_all.shape} - expected (num_samples, 4, 2)")
    print(f"      y_pred_all: {y_pred_all.shape} - expected (num_samples, 12, 2)")
    print(f"      y_true_all: {y_true_all.shape} - expected (num_samples, 12, 2)")

    # Create subplot for each sample
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i in range(min(num_samples, 4)):
        ax = axes[i]

        # Extract position (first 2 features)
        hist_x = x_all[i, :, 0]
        hist_y = x_all[i, :, 1]
        pred_x = y_pred_all[i, :, 0]
        pred_y = y_pred_all[i, :, 1]
        true_x = y_true_all[i, :, 0]
        true_y = y_true_all[i, :, 1]

        # Calculate FDE for this sample
        fde = np.sqrt((pred_x[-1] - true_x[-1]) ** 2 + (pred_y[-1] - true_y[-1]) ** 2)

        # Plot trajectories with enhanced visibility
        ax.plot(
            hist_x,
            hist_y,
            "o-",
            label="History (4 frames)",
            color="blue",
            markersize=5,
            linewidth=2,
        )
        ax.plot(
            true_x,
            true_y,
            "s-",
            label="Ground Truth (12 frames)",
            color="green",
            markersize=4,
            linewidth=1.5,
            alpha=0.8,
        )
        ax.plot(
            pred_x,
            pred_y,
            "^--",
            label="Prediction (12 frames)",
            color="red",
            markersize=4,
            linewidth=1.5,
            alpha=0.8,
        )

        # Mark start and end points more clearly
        ax.plot(hist_x[0], hist_y[0], "bo", markersize=10, label="History Start")
        ax.plot(hist_x[-1], hist_y[-1], "b*", markersize=15, label="History End")
        ax.plot(true_x[-1], true_y[-1], "gs", markersize=10, label="GT End")
        ax.plot(pred_x[-1], pred_y[-1], "r^", markersize=10, label="Pred End")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"Sample {i + 1} (FDE={fde:.3f}m)")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

    plt.tight_layout()
    plt.savefig(output_dir / "sample_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()


def compute_metrics_full(predictions, targets):
    """
    Compute trajectory prediction metrics with optional per-category breakdown.

    FLEXIBLE EVALUATION:
    - Base metrics: ADE, FDE computed on position features (x, y) only
    - If categorical features exist (one-hot): Computes separate ADE per category

    Args:
        predictions: Model predictions of shape (batch_size, num_frames, num_features)
        targets: Ground truth of shape (batch_size, num_frames, num_features)

    Returns:
        Dictionary with metrics:
        - mse: Mean Squared Error (on all features)
        - rmse: Root Mean Squared Error (on all features)
        - ade: Average Displacement Error (only position: x, y)
        - fde: Final Displacement Error (only position: x, y)
        - ade_per_category: Dict with ADE for each category (if one-hot present)
          e.g., {'car': 1.8, 'pedestrian': 3.2}
    """
    num_features = predictions.shape[2]
    batch_size = predictions.shape[0]

    # MSE: mean squared error across all coordinates
    mse = torch.mean((predictions - targets) ** 2).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()

    # ADE/FDE: Only use position features (first 2) to avoid mixing units
    pred_pos = predictions[:, :, :2]  # Get position only (x, y)
    target_pos = targets[:, :, :2]  # Get position only (x, y)

    # ADE: Average Displacement Error (mean L2 distance to ground truth)
    distances = torch.norm(pred_pos - target_pos, dim=2)  # (batch, num_frames)
    ade = torch.mean(distances).item()

    # FDE: Final Displacement Error (L2 distance at final frame)
    fde = torch.mean(distances[:, -1]).item()

    metrics = {"mse": mse, "rmse": rmse, "ade": ade, "fde": fde}

    # If categorical features exist (one-hot encoded), compute ADE per category
    if num_features > 2:
        num_categories = num_features - 2  # Number of categorical features

        # Extract one-hot categories from targets (position-independent, same across frames)
        categories_onehot = targets[
            0, 0, 2 : 2 + num_categories
        ]  # Shape: (num_categories,)
        category_idx = torch.argmax(categories_onehot).item()

        # Get all samples and their categories
        all_categories_onehot = targets[
            :, 0, 2 : 2 + num_categories
        ]  # (batch, num_categories)
        category_indices = torch.argmax(all_categories_onehot, dim=1)  # (batch,)

        # Compute ADE per category
        ade_per_category = {}
        for cat_id in range(num_categories):
            mask = category_indices == cat_id
            if mask.sum() > 0:
                cat_distances = distances[mask]
                cat_ade = torch.mean(cat_distances).item()
                ade_per_category[f"category_{cat_id}"] = cat_ade

        metrics["ade_per_category"] = ade_per_category

    return metrics


def compute_metrics(predictions, targets):
    """
    Compute trajectory prediction metrics.

    Args:
        predictions: Model predictions of shape (batch_size, num_frames, 2)
        targets: Ground truth of shape (batch_size, num_frames, 2)

    Returns:
        Dictionary with metrics: mse, rmse, ade, fde
    """
    # MSE:mean squared error
    mse = torch.mean((predictions - targets) ** 2).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()

    # ADE/FDE: Only use position features (first 2)
    pred_pos = predictions[:, :, :2]
    target_pos = targets[:, :, :2]

    # ADE: Average Displacement Error
    distances = torch.norm(pred_pos - target_pos, dim=2)
    ade = torch.mean(distances).item()

    # FDE: Final Displacement Error
    fde = torch.mean(distances[:, -1]).item()

    return {"mse": mse, "rmse": rmse, "ade": ade, "fde": fde}


def unpack_model_output(model_output):
    """Normalize model outputs to (mean_prediction, logvar_or_none)."""
    if isinstance(model_output, tuple):
        return model_output
    return model_output, None


def gaussian_nll_loss(pred_mean, pred_logvar, targets):
    """Negative log-likelihood for diagonal Gaussian trajectory outputs."""
    var = torch.exp(pred_logvar)
    nll = 0.5 * (
        pred_logvar + ((targets - pred_mean) ** 2) / (var + 1e-8) + np.log(2.0 * np.pi)
    )
    return nll.mean()


def train_epoch(model, train_loader, optimizer, criterion, device, probabilistic=False):
    """
    Train for one epoch.

    Args:
        model: Trajectory prediction model
        train_loader: Training DataLoader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on

    Returns:
        Dictionary with epoch statistics
    """
    model.train()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for x_batch, y_batch in pbar:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        optimizer.zero_grad()
        model_output = model(x_batch)
        y_pred, y_logvar = unpack_model_output(model_output)

        # Compute loss
        if probabilistic:
            loss = gaussian_nll_loss(y_pred, y_logvar, y_batch)
        else:
            loss = criterion(y_pred, y_batch)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Compute metrics
        metrics = compute_metrics(y_pred.detach(), y_batch)

        total_loss += loss.item()
        total_ade += metrics["ade"]
        total_fde += metrics["fde"]
        num_batches += 1

        pbar.update(1)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return {
        "loss": total_loss / num_batches,
        "ade": total_ade / num_batches,
        "fde": total_fde / num_batches,
    }


def validate(model, test_loader, criterion, device, probabilistic=False):
    """
    Validate model on test set.

    Args:
        model: Trajectory prediction model
        test_loader: Test DataLoader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Dictionary with validation statistics
    """
    model.eval()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Validation", leave=False)
        for x_batch, y_batch in pbar:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            model_output = model(x_batch)
            y_pred, y_logvar = unpack_model_output(model_output)

            # Compute loss
            if probabilistic:
                loss = gaussian_nll_loss(y_pred, y_logvar, y_batch)
            else:
                loss = criterion(y_pred, y_batch)

            # Compute metrics
            metrics = compute_metrics(y_pred, y_batch)

            total_loss += loss.item()
            total_ade += metrics["ade"]
            total_fde += metrics["fde"]
            num_batches += 1

            pbar.update(1)

    return {
        "loss": total_loss / num_batches,
        "ade": total_ade / num_batches,
        "fde": total_fde / num_batches,
    }


def train(
    num_epochs=25,
    batch_size=32,
    learning_rate=1e-3,
    device=None,
    checkpoint_dir=None,
    d_model=256,
    nhead=8,
    num_layers=4,
    num_workers=4,
):
    """
    Full training pipeline.

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        device: Device to train on (defaults to GPU if available)
        checkpoint_dir: Directory to save checkpoints
        d_model: Hidden dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        num_workers: Number of DataLoader worker processes
    """
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Setup checkpoint directory
    if checkpoint_dir is None:
        script_dir = Path(__file__).parent
        project_dir = script_dir.parent
        checkpoint_dir = project_dir / "checkpoints"
    else:
        checkpoint_dir = Path(checkpoint_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Load data
    print("\nLoading data...")
    train_x, train_y, scene_ids = load_data(device=device)

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, test_loader, train_idx, test_idx = create_dataloaders(
        train_x,
        train_y,
        scene_ids,
        batch_size=batch_size,
        num_workers=0,  # Use 0 to avoid multiprocessing CUDA issues
        device=device,
        shuffle_train=True,
    )
    print(f"Data shape: x={train_x.shape}, y={train_y.shape}")
    print(f"Train set: {len(train_idx):,} trajectories")
    print(f"Test set:  {len(test_idx):,} trajectories")

    # Extract dimensions from data (flexible to any feature set)
    num_input_frames, num_input_features = train_x.shape[1:]
    num_output_frames, num_output_features = train_y.shape[1:]

    # Create model
    print("\nCreating model...")
    print(f"  Input: {num_input_frames} frames × {num_input_features} features")
    print(f"  Output: {num_output_frames} frames × {num_output_features} features")

    model = LSTMGRUPredictor(
        history_frames=num_input_frames,
        future_frames=num_output_frames,
        hidden_dim=d_model,
        input_features=num_input_features,
        output_features=num_output_features,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    # Training loop
    print(f"\nTraining for {num_epochs} epochs...\n")
    history = {
        "epoch": [],
        "train_loss": [],
        "train_ade": [],
        "train_fde": [],
        "val_loss": [],
        "val_ade": [],
        "val_fde": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0

    start_time = time.time()

    for epoch in range(num_epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_metrics = validate(model, test_loader, criterion, device)

        # Update history
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_metrics["loss"])
        history["train_ade"].append(train_metrics["ade"])
        history["train_fde"].append(train_metrics["fde"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_ade"].append(val_metrics["ade"])
        history["val_fde"].append(val_metrics["fde"])

        # Learning rate scheduling
        scheduler.step(val_metrics["loss"])

        # Print progress
        print(
            f"Epoch {epoch + 1:3d}/{num_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} (ADE: {train_metrics['ade']:.4f}) | "
            f"Val Loss: {val_metrics['loss']:.4f} (ADE: {val_metrics['ade']:.4f})"
        )

        # Print per-category metrics if available
        if "ade_per_category" in val_metrics:
            for cat_name, cat_ade in val_metrics["ade_per_category"].items():
                print(f"    {cat_name}: {cat_ade:.4f}")

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "hyperparameters": {
                    "d_model": d_model,
                    "nhead": nhead,
                    "num_layers": num_layers,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                },
            }

            checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"  → Saved best model (Val Loss: {val_metrics['loss']:.4f})")

    # Save final model
    final_checkpoint = {
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "hyperparameters": {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        },
    }
    torch.save(final_checkpoint, checkpoint_dir / "final_model.pt")

    # Save history
    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Summary
    elapsed_time = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"Training completed in {elapsed_time:.1f} seconds")
    print(f"Best model at epoch {best_epoch} with Val Loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"{'=' * 80}")

    return model, history


def run_experiment_stages(
    stages: List[ExperimentStage],
    num_epochs=25,
    batch_size=32,
    learning_rate=1e-3,
    device=None,
    output_base_dir=None,
    num_workers=4,
):
    """
    Run a sequence of experiment stages sequentially.

    Args:
        stages: List of ExperimentStage definitions
        num_epochs: Number of epochs per stage
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        output_base_dir: Base directory for stage outputs
        num_workers: Number of DataLoader workers
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    if output_base_dir is None:
        script_dir = Path(__file__).parent
        project_dir = script_dir.parent
        output_base_dir = project_dir / "experiments"
    else:
        output_base_dir = Path(output_base_dir)

    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Load base data once
    print("Loading base data...")
    train_x, train_y, scene_ids = load_data(device=device)
    print(f"Base data shape: x={train_x.shape}, y={train_y.shape}")

    # Load agent types
    print("Loading agent types...")
    agent_types = load_agent_types()
    # Keep agent_types on CPU (filtering doesn't benefit from GPU, avoids device mismatch)
    print(f"Agent types shape: {agent_types.shape}\n")

    # Run each stage
    for stage_idx, stage in enumerate(stages, 1):
        print(f"\n{'=' * 80}")
        print(f"Stage {stage_idx}/{len(stages)}: {stage.name}")
        print(f"Description: {stage.description}")
        print(f"{'=' * 80}\n")

        # Create stage output directory
        stage_output_dir = output_base_dir / stage.name
        stage_output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Filter by agent type
        print(f"Filtering by agent type: {stage.agent_types}...")
        stage_x, stage_y, mask_indices = filter_by_agent_type(
            train_x, train_y, agent_types, stage.agent_types
        )
        print(f"  Filtered data shape: x={stage_x.shape}, y={stage_y.shape}")
        print(f"  Samples: {len(mask_indices):,}")

        # Skip stage if no samples after filtering
        if len(mask_indices) == 0:
            print(
                f"\n⚠️  SKIPPING Stage {stage_idx}: No samples found for agent types {stage.agent_types}"
            )
            print(f"  Dataset may not contain {stage.agent_types} trajectories.\n")
            continue

        # Step 2: Optionally add velocity features
        if stage.include_velocity:
            print("Adding velocity features...")
            stage_x, stage_y = maybe_append_velocity_features(stage_x, stage_y)
            print(f"  With velocity shape: x={stage_x.shape}, y={stage_y.shape}")

        # Step 3: Optionally select map features
        if stage.selected_map_features:
            print(f"Selecting map features: {stage.selected_map_features}...")
            try:
                stage_x, stage_y = apply_map_feature_selection(
                    stage_x,
                    stage_y,
                    stage.selected_map_features,
                    sample_indices=mask_indices,
                    append_to_existing=stage.append_map_to_existing,
                )
                print(
                    f"  With map features shape: x={stage_x.shape}, y={stage_y.shape}"
                )
            except Exception as e:
                print(f"  Warning: Could not load map features: {e}")
                print(f"  Continuing with position only")

        # Step 4: Create dataloaders for this stage
        print("\nCreating dataloaders...")
        # Use scene-based split but only for the filtered data
        filtered_scene_ids = scene_ids[mask_indices]

        train_loader, test_loader, train_idx, test_idx = create_dataloaders(
            stage_x,
            stage_y,
            filtered_scene_ids,
            batch_size=batch_size,
            num_workers=0,  # Use 0 to avoid multiprocessing CUDA issues on Kaggle
            device=device,
            shuffle_train=True,
        )
        print(f"  Train: {len(train_idx):,}, Test: {len(test_idx):,}")

        # Step 5: Create and train model
        num_input_frames, num_input_features = stage_x.shape[1:]
        num_output_frames, num_output_features = stage_y.shape[1:]

        print(f"\nCreating model...")
        print(f"  Input: {num_input_frames} frames × {num_input_features} features")
        print(f"  Output: {num_output_frames} frames × {num_output_features} features")

        if stage.probabilistic:
            model = ProbabilisticLSTMGRUPredictor(
                history_frames=num_input_frames,
                future_frames=num_output_frames,
                hidden_dim=256,
                input_features=num_input_features,
                output_features=num_output_features,
            ).to(device)
            print("  Model type: Probabilistic Gaussian output")
        else:
            model = LSTMGRUPredictor(
                history_frames=num_input_frames,
                future_frames=num_output_frames,
                hidden_dim=256,
                input_features=num_input_features,
                output_features=num_output_features,
            ).to(device)
            print("  Model type: Deterministic")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {total_params:,}")

        # Training setup
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        criterion = nn.MSELoss()

        # Training loop
        print(f"\nTraining for {num_epochs} epochs...\n")
        history = {
            "epoch": [],
            "train_loss": [],
            "train_ade": [],
            "train_fde": [],
            "val_loss": [],
            "val_ade": [],
            "val_fde": [],
        }

        best_val_loss = float("inf")
        best_epoch = 0
        start_time = time.time()

        for epoch in range(num_epochs):
            # Train
            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                probabilistic=stage.probabilistic,
            )

            # Validate
            val_metrics = validate(
                model,
                test_loader,
                criterion,
                device,
                probabilistic=stage.probabilistic,
            )

            # Update history
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(train_metrics["loss"])
            history["train_ade"].append(train_metrics["ade"])
            history["train_fde"].append(train_metrics["fde"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_ade"].append(val_metrics["ade"])
            history["val_fde"].append(val_metrics["fde"])

            # Learning rate scheduling
            scheduler.step(val_metrics["loss"])

            # Print progress
            print(
                f"Epoch {epoch + 1:3d}/{num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} (ADE: {train_metrics['ade']:.4f}) | "
                f"Val Loss: {val_metrics['loss']:.4f} (ADE: {val_metrics['ade']:.4f})"
            )

            # Save best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch + 1

                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "stage": stage.name,
                }

                checkpoint_path = stage_output_dir / "best_model.pt"
                torch.save(checkpoint, checkpoint_path)
                print(f"  → Saved best model (Val Loss: {val_metrics['loss']:.4f})")

        # Save artifacts
        elapsed_time = time.time() - start_time
        print(f"\nStage completed in {elapsed_time:.1f} seconds")
        print(f"Best model at epoch {best_epoch} with Val Loss: {best_val_loss:.4f}")

        # Save training curves
        print("\nSaving artifacts...")
        save_training_curves(history, stage_output_dir)
        save_sample_predictions_plot(
            model,
            test_loader,
            stage_output_dir,
            device,
            num_samples=4,
            probabilistic=stage.probabilistic,
        )

        # Save history JSON
        history_path = stage_output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Save stage metadata
        metadata = {
            "stage_name": stage.name,
            "description": stage.description,
            "agent_types": stage.agent_types,
            "include_velocity": stage.include_velocity,
            "selected_map_features": stage.selected_map_features,
            "probabilistic": stage.probabilistic,
            "append_map_to_existing": stage.append_map_to_existing,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_training_samples": len(train_idx),
            "num_test_samples": len(test_idx),
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "training_time_seconds": elapsed_time,
        }

        with open(stage_output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Artifacts saved to: {stage_output_dir}")

    print(f"\n{'=' * 80}")
    print(f"All stages completed!")
    print(f"Results saved to: {output_base_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train trajectory prediction model with experiment stages"
    )
    parser.add_argument(
        "--run-stages",
        action="store_true",
        help="Run all configured experiment stages sequentially",
    )
    parser.add_argument(
        "--stage",
        type=str,
        help="Run a specific stage by name (e.g., 'cars_baseline')",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of epochs per stage",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for experiments",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes",
    )

    args = parser.parse_args()

    if args.run_stages:
        # Run all stages
        run_experiment_stages(
            stages=DEFAULT_EXPERIMENT_STAGES,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_base_dir=args.output_dir,
            num_workers=args.workers,
        )
    elif args.stage:
        # Run a specific stage
        stage_map = {stage.name: stage for stage in DEFAULT_EXPERIMENT_STAGES}
        if args.stage not in stage_map:
            print(f"Error: Unknown stage '{args.stage}'")
            print(f"Available stages: {list(stage_map.keys())}")
            exit(1)

        run_experiment_stages(
            stages=[stage_map[args.stage]],
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_base_dir=args.output_dir,
            num_workers=args.workers,
        )
    else:
        # Default: Run all stages
        print("No stage specified. Running all configured experiment stages...")
        print("Use --help for options.\n")
        run_experiment_stages(
            stages=DEFAULT_EXPERIMENT_STAGES,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_base_dir=args.output_dir,
            num_workers=args.workers,
        )
