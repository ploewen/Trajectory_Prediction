#!/usr/bin/env python3
"""Sample inference demo for the trajectory prediction models.

The script loads one sample from the dataset, runs it through a saved
checkpoint if one is available, prints the observed and predicted
trajectories, and saves a comparison plot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import create_dataloaders
from model import LSTMGRUPredictor, ProbabilisticLSTMGRUPredictor
from train import (
    apply_map_feature_selection,
    filter_by_agent_type,
    load_agent_types,
    maybe_append_velocity_features,
)


def find_checkpoint(
    project_root: Path, explicit_checkpoint: Optional[str]
) -> Optional[Path]:
    if explicit_checkpoint:
        checkpoint_path = Path(explicit_checkpoint)
        return checkpoint_path if checkpoint_path.exists() else None

    preferred = project_root / "experiments" / "cars_velocity" / "best_model.pt"
    if preferred.exists():
        return preferred

    candidates = sorted((project_root / "experiments").glob("*/best_model.pt"))
    return candidates[0] if candidates else None


def load_model_from_checkpoint(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

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
        )
    else:
        output_features = state_dict["output_layer.weight"].shape[0]
        model = LSTMGRUPredictor(
            history_frames=4,
            future_frames=12,
            hidden_dim=hidden_dim,
            input_features=input_features,
            output_features=output_features,
        )

    model.load_state_dict(state_dict)
    return model.to(device).eval()


def load_stage_metadata(checkpoint_path: Path) -> dict:
    metadata_path = checkpoint_path.parent / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as file_handle:
            return json.load(file_handle)

    return {
        "stage_name": checkpoint_path.parent.name,
        "agent_types": ["vehicle"],
        "include_velocity": False,
        "selected_map_features": None,
        "append_map_to_existing": False,
        "probabilistic": False,
    }


def build_stage_data(project_root: Path, stage_metadata: dict, device: str):
    data_dir = project_root / "data"
    train_x = torch.load(data_dir / "train_x.pt", weights_only=True).to(device)
    train_y = torch.load(data_dir / "train_y.pt", weights_only=True).to(device)
    scene_ids = torch.load(data_dir / "scene_ids.pt", weights_only=True).to(device)

    agent_types = load_agent_types().to(train_x.device)

    stage_x, stage_y, mask_indices = filter_by_agent_type(
        train_x, train_y, agent_types, stage_metadata.get("agent_types", ["vehicle"])
    )

    if stage_metadata.get("include_velocity", False):
        stage_x, stage_y = maybe_append_velocity_features(stage_x, stage_y)

    selected_map_features = stage_metadata.get("selected_map_features")
    if selected_map_features:
        stage_x, stage_y = apply_map_feature_selection(
            stage_x,
            stage_y,
            selected_map_features,
            sample_indices=mask_indices,
            append_to_existing=stage_metadata.get("append_map_to_existing", False),
        )

    filtered_scene_ids = scene_ids[mask_indices]

    return stage_x, stage_y, filtered_scene_ids


def create_fallback_model(train_x: torch.Tensor, train_y: torch.Tensor, device: str):
    return (
        LSTMGRUPredictor(
            history_frames=train_x.shape[1],
            future_frames=train_y.shape[1],
            hidden_dim=128,
            input_features=train_x.shape[2],
            output_features=train_y.shape[2],
        )
        .to(device)
        .eval()
    )


def format_trajectory(trajectory: torch.Tensor) -> str:
    return np.array2string(
        trajectory.detach().cpu().numpy(),
        precision=3,
        suppress_small=True,
    )


def save_plot(
    history: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    output_path: Path,
):
    history_xy = history[:, :2].detach().cpu().numpy()
    prediction_xy = prediction[:, :2].detach().cpu().numpy()
    target_xy = target[:, :2].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(history_xy[:, 0], history_xy[:, 1], "b-", label="history", linewidth=2)
    ax.plot(target_xy[:, 0], target_xy[:, 1], "gs--", label="ground truth", linewidth=2)
    ax.plot(
        prediction_xy[:, 0],
        prediction_xy[:, 1],
        "r^--",
        label="prediction",
        linewidth=2,
    )
    ax.scatter(history_xy[0, 0], history_xy[0, 1], s=80, label="start")
    ax.set_title("Sample trajectory input-output")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def select_most_dynamic_sample(
    test_x: torch.Tensor, test_y: torch.Tensor, test_indices: torch.Tensor
):
    history_xy = test_x[:, :, :2]
    future_xy = test_y[:, :, :2]

    history_motion = torch.norm(history_xy[:, 1:, :] - history_xy[:, :-1, :], dim=2)
    future_motion = torch.norm(future_xy[:, 1:, :] - future_xy[:, :-1, :], dim=2)
    motion_score = history_motion.sum(dim=1) + future_motion.sum(dim=1)

    sample_pos = int(torch.argmax(motion_score).item())
    dataset_index = int(test_indices[sample_pos].item())

    return sample_pos, dataset_index, float(motion_score[sample_pos].item())


def main():
    parser = argparse.ArgumentParser(
        description="Run a sample trajectory prediction demo"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a saved model checkpoint (defaults to the first best_model.pt found)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference (cpu or cuda)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for the saved comparison plot",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = find_checkpoint(project_root, args.checkpoint)

    print("Trajectory prediction demo")
    print(f"Device: {device}")

    stage_metadata = (
        load_stage_metadata(checkpoint_path)
        if checkpoint_path
        else {
            "stage_name": "cars_velocity",
            "agent_types": ["vehicle"],
            "include_velocity": True,
            "selected_map_features": None,
            "append_map_to_existing": False,
            "probabilistic": False,
        }
    )

    print(f"Stage: {stage_metadata.get('stage_name', 'unknown')}")

    train_x, train_y, scene_ids = build_stage_data(
        project_root, stage_metadata, device="cpu"
    )
    _, _, _, test_indices = create_dataloaders(
        train_x,
        train_y,
        scene_ids,
        batch_size=1,
        num_workers=0,
        device="cpu",
        shuffle_train=False,
    )

    test_x = train_x[test_indices]
    test_y = train_y[test_indices]

    sample_pos, dataset_index, motion_score = select_most_dynamic_sample(
        test_x, test_y, test_indices
    )
    history = test_x[sample_pos]
    target = test_y[sample_pos]

    print(
        f"Selected sample: test[{sample_pos}] / dataset[{dataset_index}] "
        f"(motion score={motion_score:.3f})"
    )
    print(f"Sample history shape: {tuple(history.shape)}")
    print(f"Sample target shape: {tuple(target.shape)}")

    if checkpoint_path is not None:
        print(f"Checkpoint: {checkpoint_path}")
        model = load_model_from_checkpoint(checkpoint_path, device)
    else:
        print("No checkpoint found. Using an untrained fallback model.")
        model = create_fallback_model(train_x, train_y, device)

    with torch.no_grad():
        model_output = model(history.unsqueeze(0).to(device))
        prediction = (
            model_output[0] if isinstance(model_output, tuple) else model_output
        )
        prediction = prediction.squeeze(0).cpu()

    ade = torch.norm(prediction[:, :2] - target[:, :2], dim=1).mean().item()
    fde = torch.norm(prediction[-1, :2] - target[-1, :2]).item()

    print("\nObserved history:")
    print(format_trajectory(history[:, :2]))
    print("\nGround truth future:")
    print(format_trajectory(target[:, :2]))
    print("\nPredicted future:")
    print(format_trajectory(prediction[:, :2]))
    print(f"\nSample ADE: {ade:.4f}")
    print(f"Sample FDE: {fde:.4f}")

    output_path = (
        Path(args.output)
        if args.output
        else (project_root / "experiments" / "sample_prediction_demo.png")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_plot(history, prediction, target, output_path)
    print(f"Saved comparison plot to: {output_path}")


if __name__ == "__main__":
    main()
