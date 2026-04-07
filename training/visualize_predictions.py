"""
Visualize predictions from a trained model checkpoint without retraining.
Usage: python visualize_predictions.py <stage_name> [--checkpoint <path>]
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from dataclasses import dataclass
from typing import List, Optional

from model import LSTMGRUPredictor, ProbabilisticLSTMGRUPredictor
from dataset import load_data, create_dataloaders
from train import (
    load_agent_types,
    filter_by_agent_type,
    maybe_append_velocity_features,
    apply_map_feature_selection,
)


# Stage configurations (mirroring train.py)
@dataclass
class ExperimentStage:
    """Configuration for an experiment stage."""

    name: str
    description: str
    agent_types: List[str]
    include_velocity: bool
    selected_map_features: Optional[List[str]]
    probabilistic: bool = False
    append_map_to_existing: bool = False


STAGE_CONFIGS = {
    "cars_baseline": ExperimentStage(
        name="cars_baseline",
        description="Cars: position only",
        agent_types=["vehicle"],
        include_velocity=False,
        selected_map_features=None,
    ),
    "cars_velocity": ExperimentStage(
        name="cars_velocity",
        description="Cars: position + velocity",
        agent_types=["vehicle"],
        include_velocity=True,
        selected_map_features=None,
    ),
    "cars_map": ExperimentStage(
        name="cars_map",
        description="Cars: position + map features",
        agent_types=["vehicle"],
        include_velocity=False,
        selected_map_features=["lane_type_road", "drivable_area"],
    ),
    "pedestrian_baseline": ExperimentStage(
        name="pedestrian_baseline",
        description="Pedestrian: position only",
        agent_types=["pedestrian"],
        include_velocity=False,
        selected_map_features=None,
    ),
    "pedestrian_map": ExperimentStage(
        name="pedestrian_map",
        description="Pedestrian: position + map features",
        agent_types=["pedestrian"],
        include_velocity=False,
        selected_map_features=["has_crosswalk", "drivable_area"],
    ),
    "cars_probabilistic": ExperimentStage(
        name="cars_probabilistic",
        description="Cars: probabilistic Gaussian output",
        agent_types=["vehicle"],
        include_velocity=False,
        selected_map_features=None,
        probabilistic=True,
    ),
}


def load_checkpoint(checkpoint_path, device="cpu"):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    # Infer architecture from state_dict so loading works for any hidden size/features.
    input_features = state_dict["encoder.weight_ih_l0"].shape[1]
    hidden_dim = state_dict["encoder.weight_hh_l0"].shape[1]
    is_probabilistic = "mean_head.weight" in state_dict

    if is_probabilistic:
        output_features = state_dict["mean_head.weight"].shape[0]
        model = ProbabilisticLSTMGRUPredictor(
            history_frames=4,
            future_frames=12,
            hidden_dim=hidden_dim,
            input_features=input_features,
            output_features=output_features,
        ).to(device)
    else:
        output_features = state_dict["output_layer.weight"].shape[0]
        model = LSTMGRUPredictor(
            history_frames=4,
            future_frames=12,
            hidden_dim=hidden_dim,
            input_features=input_features,
            output_features=output_features,
        ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    return model


def unpack_model_output(model_output):
    """Normalize model outputs to (mean_prediction, logvar_or_none)."""
    if isinstance(model_output, tuple):
        return model_output
    return model_output, None


def generate_and_save_visualization(
    stage_name, checkpoint_path=None, device=None, batch_size=128
):
    """
    Generate and save visualization for a specific stage.

    Args:
        stage_name: Name of the stage (e.g., 'cars_baseline')
        checkpoint_path: Path to model checkpoint (auto-detected if None)
        device: Device to use
        batch_size: Batch size for inference
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    if checkpoint_path is None:
        checkpoint_path = project_dir / "experiments" / stage_name / "best_model.pt"

    checkpoint_path = Path(checkpoint_path)
    output_dir = checkpoint_path.parent

    print(f"Stage: {stage_name}")
    print(f"Loading checkpoint: {checkpoint_path}")

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    # Get stage configuration
    if stage_name not in STAGE_CONFIGS:
        print(f"Error: Unknown stage '{stage_name}'")
        print(f"Available stages: {list(STAGE_CONFIGS.keys())}")
        return

    stage = STAGE_CONFIGS[stage_name]

    # Load model
    model = load_checkpoint(checkpoint_path, device=device)
    print("Model loaded successfully")

    # Load and build the same stage-specific tensors as training.
    print("Loading data...")
    train_x, train_y, scene_ids = load_data(device=device)
    agent_types = load_agent_types()

    stage_x, stage_y, mask_indices = filter_by_agent_type(
        train_x, train_y, agent_types, stage.agent_types
    )

    if len(mask_indices) == 0:
        print(f"Error: no samples found for stage '{stage_name}'")
        return

    if stage.include_velocity:
        stage_x, stage_y = maybe_append_velocity_features(stage_x, stage_y)

    if stage.selected_map_features:
        stage_x, stage_y = apply_map_feature_selection(
            stage_x,
            stage_y,
            stage.selected_map_features,
            sample_indices=mask_indices,
            append_to_existing=stage.append_map_to_existing,
        )

    filtered_scene_ids = scene_ids[mask_indices]

    # Create test loader
    train_loader, test_loader, train_idx, test_idx = create_dataloaders(
        stage_x,
        stage_y,
        filtered_scene_ids,
        batch_size=batch_size,
        num_workers=0,  # No workers for evaluation
        device=device,
        shuffle_train=False,
    )

    print(f"Test set: {len(test_idx):,} samples")

    # Collect predictions
    print("Generating predictions...")
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

    x_all = np.concatenate(all_x, axis=0)
    y_pred_all = np.concatenate(all_y_pred, axis=0)
    y_true_all = np.concatenate(all_y_true, axis=0)

    # Randomly choose 10 samples
    num_samples = min(20, len(x_all))
    random_indices = np.random.choice(len(x_all), size=num_samples, replace=False)

    x_all = x_all[random_indices]
    y_pred_all = y_pred_all[random_indices]
    y_true_all = y_true_all[random_indices]

    print("Data shapes:")
    print(f"  x_all (history): {x_all.shape}")
    print(f"  y_pred_all: {y_pred_all.shape}")
    print(f"  y_true_all: {y_true_all.shape}")

    # Create visualization
    print("Creating visualization...")
    fig, axes = plt.subplots(num_samples // 2, 2, figsize=(14, 1.8 * num_samples))
    axes = axes.flatten()

    for i in range(num_samples):
        ax = axes[i]

        # Extract position
        hist_x = x_all[i, :, 0]
        hist_y = x_all[i, :, 1]
        pred_x = y_pred_all[i, :, 0]
        pred_y = y_pred_all[i, :, 1]
        true_x = y_true_all[i, :, 0]
        true_y = y_true_all[i, :, 1]

        # Calculate FDE
        fde = np.sqrt((pred_x[-1] - true_x[-1]) ** 2 + (pred_y[-1] - true_y[-1]) ** 2)

        # Plot trajectories
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

        # Mark key points
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
    output_path = output_dir / "sample_predictions_updated.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Visualization saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize predictions from trained model"
    )
    parser.add_argument(
        "stage",
        nargs="?",
        type=str,
        default=None,
        help="Stage name (e.g., 'cars_baseline')",
    )
    parser.add_argument(
        "--stage",
        dest="stage_flag",
        type=str,
        default=None,
        help="Stage name (alternative to positional argument)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint (auto-detected if not provided)",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")

    args = parser.parse_args()

    stage_name = args.stage_flag or args.stage
    if stage_name is None:
        parser.error("Please provide a stage name as positional arg or --stage")

    generate_and_save_visualization(
        stage_name=stage_name,
        checkpoint_path=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
    )
