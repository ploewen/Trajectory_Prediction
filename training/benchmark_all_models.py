"""Benchmark all trained experiment models on their corresponding test splits."""

from pathlib import Path
import json
import csv

import torch
import torch.nn as nn

from model import LSTMGRUPredictor, ProbabilisticLSTMGRUPredictor
from dataset import create_dataloaders, load_data
from train import (
    load_agent_types,
    filter_by_agent_type,
    maybe_append_velocity_features,
    apply_map_feature_selection,
    unpack_model_output,
    compute_metrics,
    gaussian_nll_loss,
)


def build_stage_data(stage_meta, train_x, train_y, scene_ids, agent_types):
    """Apply the same stage preprocessing used during training."""
    stage_x, stage_y, mask_indices = filter_by_agent_type(
        train_x, train_y, agent_types, stage_meta["agent_types"]
    )

    if stage_meta.get("include_velocity", False):
        stage_x, stage_y = maybe_append_velocity_features(stage_x, stage_y)

    selected = stage_meta.get("selected_map_features")
    if selected:
        stage_x, stage_y = apply_map_feature_selection(
            stage_x,
            stage_y,
            selected,
            sample_indices=mask_indices,
            append_to_existing=stage_meta.get("append_map_to_existing", False),
        )

    filtered_scene_ids = scene_ids[mask_indices]
    return stage_x, stage_y, filtered_scene_ids


def infer_hidden_dim(state_dict):
    """Infer hidden dimension from LSTM encoder weight shape."""
    return state_dict["encoder.weight_hh_l0"].shape[1]


def evaluate_experiment(
    exp_dir,
    base_x_raw,
    base_y_raw,
    scene_ids_raw,
    base_x_enriched,
    base_y_enriched,
    scene_ids_enriched,
    agent_types,
    device,
):
    metadata_path = exp_dir / "metadata.json"
    checkpoint_path = exp_dir / "best_model.pt"

    if not metadata_path.exists() or not checkpoint_path.exists():
        return None

    with open(metadata_path, "r") as f:
        meta = json.load(f)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    probabilistic = bool(meta.get("probabilistic", False))
    required_input_features = state_dict["encoder.weight_ih_l0"].shape[1]

    stage_x, stage_y, stage_scene_ids = build_stage_data(
        meta, base_x_raw, base_y_raw, scene_ids_raw, agent_types
    )

    if stage_x.shape[2] != required_input_features:
        stage_x, stage_y, stage_scene_ids = build_stage_data(
            meta,
            base_x_enriched,
            base_y_enriched,
            scene_ids_enriched,
            agent_types,
        )

    if stage_x.shape[2] != required_input_features:
        raise RuntimeError(
            f"Could not match input features for {meta['stage_name']}: "
            f"checkpoint expects {required_input_features}, prepared {stage_x.shape[2]}"
        )

    _, test_loader, _, test_idx = create_dataloaders(
        stage_x,
        stage_y,
        stage_scene_ids,
        batch_size=int(meta.get("batch_size", 512)),
        num_workers=0,
        device=device,
        shuffle_train=False,
    )

    hidden_dim = infer_hidden_dim(state_dict)
    input_features = required_input_features
    output_features = stage_y.shape[2]

    if probabilistic:
        model = ProbabilisticLSTMGRUPredictor(
            history_frames=stage_x.shape[1],
            future_frames=stage_y.shape[1],
            hidden_dim=hidden_dim,
            input_features=input_features,
            output_features=output_features,
        ).to(device)
    else:
        model = LSTMGRUPredictor(
            history_frames=stage_x.shape[1],
            future_frames=stage_y.shape[1],
            hidden_dim=hidden_dim,
            input_features=input_features,
            output_features=output_features,
        ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    all_pred = []
    all_true = []
    total_loss = 0.0
    num_batches = 0
    mse_criterion = nn.MSELoss()

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            model_output = model(x_batch)
            y_pred, y_logvar = unpack_model_output(model_output)

            all_pred.append(y_pred)
            all_true.append(y_batch)

            if probabilistic:
                loss = gaussian_nll_loss(y_pred, y_logvar, y_batch)
            else:
                loss = mse_criterion(y_pred, y_batch)

            total_loss += float(loss.item())
            num_batches += 1

    pred = torch.cat(all_pred, dim=0)
    true = torch.cat(all_true, dim=0)
    metrics = compute_metrics(pred, true)

    return {
        "stage": meta["stage_name"],
        "description": meta.get("description", ""),
        "probabilistic": probabilistic,
        "test_samples": int(len(test_idx)),
        "eval_loss": total_loss / max(1, num_batches),
        "mse": float(metrics["mse"]),
        "rmse": float(metrics["rmse"]),
        "ade": float(metrics["ade"]),
        "fde": float(metrics["fde"]),
        "best_epoch": int(meta.get("best_epoch", -1)),
        "best_val_loss": float(meta.get("best_val_loss", float("nan"))),
        "checkpoint": str(checkpoint_path),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmark device: {device}")

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"

    # Raw position-only base tensors.
    train_x_raw = torch.load(data_dir / "train_x.pt", weights_only=True).to(device)
    train_y_raw = torch.load(data_dir / "train_y.pt", weights_only=True).to(device)
    scene_ids_raw = torch.load(data_dir / "scene_ids.pt", weights_only=True).to(device)

    # Enriched base tensors using auto-loaded extra_features_*.pt.
    train_x_enriched, train_y_enriched, scene_ids_enriched = load_data(device=device)

    agent_types = load_agent_types().to(train_x_raw.device)

    exp_root = project_root / "experiments"
    exp_dirs = sorted([p for p in exp_root.iterdir() if p.is_dir()])

    results = []
    for exp_dir in exp_dirs:
        print(f"\nEvaluating: {exp_dir.name}")
        try:
            result = evaluate_experiment(
                exp_dir,
                train_x_raw,
                train_y_raw,
                scene_ids_raw,
                train_x_enriched,
                train_y_enriched,
                scene_ids_enriched,
                agent_types,
                device,
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue

        if result is None:
            print("  skipped (missing metadata/checkpoint)")
            continue

        results.append(result)
        print(
            f"  ADE={result['ade']:.4f} | FDE={result['fde']:.4f} | "
            f"RMSE={result['rmse']:.4f} | n={result['test_samples']}"
        )

    if not results:
        print("\nNo benchmark results produced.")
        return

    results.sort(key=lambda row: row["ade"])

    output_json = exp_root / "benchmark_results.json"
    output_csv = exp_root / "benchmark_results.csv"

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 96)
    print("BENCHMARK SUMMARY (sorted by ADE ascending)")
    print("=" * 96)
    print(
        f"{'Stage':38s} {'Prob':5s} {'ADE':>8s} {'FDE':>8s} "
        f"{'RMSE':>8s} {'Loss':>10s} {'Ntest':>8s}"
    )
    print("-" * 96)

    for row in results:
        print(
            f"{row['stage'][:38]:38s} {str(row['probabilistic'])[:5]:5s} "
            f"{row['ade']:8.4f} {row['fde']:8.4f} {row['rmse']:8.4f} "
            f"{row['eval_loss']:10.4f} {row['test_samples']:8d}"
        )

    print("=" * 96)
    print(f"Wrote: {output_json}")
    print(f"Wrote: {output_csv}")


if __name__ == "__main__":
    main()
