"""
Evaluation script for trained trajectory prediction models.
Computes metrics like ADE, FDE, and MSE on test set.
"""

import torch
import torch.nn as nn
from pathlib import Path
import argparse
import json
import numpy as np
from tqdm import tqdm

from model import TransformerTrajectoryPredictor, LegacyFlattenedTransformerTrajectoryPredictor
from dataset import load_data, create_dataloaders


def resolve_data_dir(project_dir, agent, features):
    """Resolve data directory for an experiment with fallback to baseline data/."""
    candidate = project_dir / 'data' / agent / features
    required = ['train_x.pt', 'train_y.pt', 'scene_ids.pt']

    if candidate.exists() and all((candidate / name).exists() for name in required):
        print(f"Using experiment-specific data dir: {candidate}")
        return candidate

    fallback = project_dir / 'data'
    print(f"Using fallback data dir: {fallback}")
    print(f"(Expected {candidate} for agent={agent}, features={features})")
    return fallback


def save_trajectory_plot(history, ground_truth, prediction, index, output_dir):
    """Save a single trajectory plot (history vs ground truth vs prediction)."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 8))
    # Blue line for history
    plt.plot(history[:, 0], history[:, 1], 'b-', label='History', linewidth=2)
    # Green line for actual future
    plt.plot(ground_truth[:, 0], ground_truth[:, 1], 'g-', label='Ground Truth', linewidth=2)
    # Red dashed line for model prediction
    plt.plot(prediction[:, 0], prediction[:, 1], 'r--', label='Prediction', linewidth=2)

    plt.legend()
    plt.title(f"Trajectory Prediction {index}")
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.axis('equal')
    plt.grid(alpha=0.3)
    plt.savefig(output_dir / f"plot_sample_{index}.png", dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_model(checkpoint_path, device=None, batch_size=64, data_dir=None, agent='car', features='baseline'):
    """
    Evaluate a trained model on the test set.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to evaluate on
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary with evaluation metrics
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    if data_dir is None:
        data_dir = resolve_data_dir(project_dir, agent, features)
    else:
        data_dir = Path(data_dir)
        print(f"Using explicit data dir: {data_dir}")
    
    # Load data first to detect actual dimensions
    print("\nLoading data...")
    train_x, train_y, scene_ids = load_data(data_dir=data_dir, device=device)
    
    # Create dataloaders
    train_loader, test_loader, train_idx, test_idx = create_dataloaders(
        train_x, train_y, scene_ids,
        batch_size=batch_size,
        device=device,
        shuffle_train=False
    )
    
    # Load checkpoint
    print(f"\nLoading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Reconstruct model using actual data dimensions.
    # New checkpoints use one token per history frame.
    # Old checkpoints used a flattened single-token variant, so we keep compatibility here.
    hyperparams = checkpoint.get('hyperparameters', {})
    d_model = hyperparams.get('d_model', 64)
    nhead = hyperparams.get('nhead', 8)
    num_layers = hyperparams.get('num_layers', 4)
    
    num_input_frames, num_input_features = train_x.shape[1:]
    num_output_frames, num_output_features = train_y.shape[1:]

    architecture_version = hyperparams.get('architecture_version')
    input_embedding_weight = checkpoint['model_state_dict']['input_embedding.weight']
    uses_legacy_flattened_model = (
        architecture_version is None and input_embedding_weight.shape[1] == num_input_frames * num_input_features
    )

    model_cls = LegacyFlattenedTransformerTrajectoryPredictor if uses_legacy_flattened_model else TransformerTrajectoryPredictor

    model = model_cls(
        num_input_frames=num_input_frames,
        num_output_frames=num_output_frames,
        num_input_features=num_input_features,
        num_output_features=num_output_features,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded (Epoch {checkpoint.get('epoch', 'N/A')})")
    print(f"  Architecture: {'legacy_flattened_v1' if uses_legacy_flattened_model else 'temporal_tokens_v2'}")
    print(f"  Input: {num_input_frames} frames × {num_input_features} features")
    print(f"  Output: {num_output_frames} frames × {num_output_features} features")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    all_predictions = []
    all_targets = []
    all_losses = []
    
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader, desc='Evaluating'):
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            
            all_predictions.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
            all_losses.append(loss.item())
    
    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Compute metrics
    print("\nComputing metrics...")
    
    # MSE
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    
    # ADE: Average Displacement Error
    distances = np.linalg.norm(predictions[:, :, :2] - targets[:, :, :2], axis=2)
    ade = np.mean(distances)
    ade_std = np.std(distances)
    
    # FDE: Final Displacement Error
    final_distances = distances[:, -1]
    fde = np.mean(final_distances)
    fde_std = np.std(final_distances)

    # Min/max stats for reporting spreadsheets
    ade_min = np.min(distances)
    ade_max = np.max(distances)
    fde_min = np.min(final_distances)
    fde_max = np.max(final_distances)
    
    # MR: Miss Rate (proportion of predictions with error > threshold)
    thresholds = [0.5, 1.0, 2.0]
    miss_rates = {}
    for thresh in thresholds:
        mr = np.mean(distances[:, -1] > thresh)
        miss_rates[f'{thresh}m'] = float(mr)
    
    # Per-frame metrics
    per_frame_ade = np.mean(distances, axis=0)
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'ade': float(ade),
        'ade_std': float(ade_std),
        'ade_min': float(ade_min),
        'ade_max': float(ade_max),
        'fde': float(fde),
        'fde_std': float(fde_std),
        'fde_min': float(fde_min),
        'fde_max': float(fde_max),
        'miss_rates': miss_rates,
        'per_frame_ade': per_frame_ade.tolist(),
        'num_test_samples': len(predictions)
    }
    
    return metrics, predictions, targets


def main(checkpoint_path=None, agent='car', features='baseline', batch_size=64, data_dir=None):
    """Main evaluation function."""

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    experiment_name = f"{agent}_{features}"

    # Find best model if no checkpoint specified
    if checkpoint_path is None:
        experiment_ckpt = project_dir / 'checkpoints' / experiment_name / 'best_model.pt'
        fallback_ckpt = project_dir / 'checkpoints' / 'best_model.pt'
        checkpoint_path = experiment_ckpt if experiment_ckpt.exists() else fallback_ckpt
    
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("\nUsage: python eval.py [checkpoint_path]")
        print("Example: python eval.py checkpoints/best_model.pt")
        return
    
    # Evaluate
    metrics, predictions, targets = evaluate_model(
        checkpoint_path,
        batch_size=batch_size,
        data_dir=data_dir,
        agent=agent,
        features=features
    )
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nTest set size: {metrics['num_test_samples']:,} samples")
    print(f"\nLoss Metrics:")
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"\nTrajectory Prediction Metrics:")
    print(f"  ADE (Average Displacement Error): {metrics['ade']:.4f}m (±{metrics['ade_std']:.4f})")
    print(f"  ADE min/max: {metrics['ade_min']:.4f}m / {metrics['ade_max']:.4f}m")
    print(f"  FDE (Final Displacement Error):   {metrics['fde']:.4f}m (±{metrics['fde_std']:.4f})")
    print(f"  FDE min/max: {metrics['fde_min']:.4f}m / {metrics['fde_max']:.4f}m")
    print(f"\nMiss Rates (proportion exceeding threshold):")
    for thresh, mr in metrics['miss_rates'].items():
        print(f"  > {thresh}: {mr:.2%}")
    print(f"\nPer-frame ADE:")
    for i, ade in enumerate(metrics['per_frame_ade'], 1):
        print(f"  Frame {i:2d} (t={i*0.5:.1f}s): {ade:.4f}m")
    print("="*80)
    
    # Save metrics
    metrics_path = checkpoint_path.parent / 'evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    # Save raw prediction tensor for downstream analysis.
    raw_preds_path = checkpoint_path.parent / 'transformer_raw_predictions.pt'
    torch.save(torch.tensor(predictions), raw_preds_path)
    print(f"Raw predictions saved to: {raw_preds_path}")

    # Optional: save a few qualitative plots (history/GT/prediction).
    try:
        script_dir = Path(__file__).parent
        project_dir = script_dir.parent
        train_x, _, scene_ids = load_data(device='cpu')
        train_mask = scene_ids < 700
        test_indices = torch.where(~train_mask)[0]
        plots_dir = checkpoint_path.parent / 'trajectory_plots'

        num_plots = min(5, len(predictions), len(test_indices))
        for i in range(num_plots):
            history = train_x[test_indices[i]].cpu().numpy()[:, :2]
            gt = targets[i][:, :2]
            pred = predictions[i][:, :2]
            save_trajectory_plot(history, gt, pred, i, plots_dir)
        print(f"Saved {num_plots} trajectory plots to: {plots_dir}")
    except Exception as e:
        print(f"Plot generation skipped: {e}")

    # Excel/CSV one-line summary for spreadsheet paste.
    per_frame_str = ",".join([str(x) for x in metrics['per_frame_ade']])
    print("\nSpreadsheet row:")
    print(
        f"{agent},transformer_{features},{metrics['num_test_samples']},{metrics['ade']},{metrics['ade_std']},"
        f"{metrics['ade_min']},{metrics['ade_max']},{metrics['fde']},{metrics['fde_std']},"
        f"{metrics['fde_min']},{metrics['fde_max']},{per_frame_str}"
    )
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate transformer trajectory predictor')
    parser.add_argument('checkpoint_path', nargs='?', default=None,
                        help='Optional checkpoint path; defaults to checkpoints/{agent}_{features}/best_model.pt')
    parser.add_argument('--agent', choices=['car', 'pedestrian'], default='car',
                        help='Agent type experiment key')
    parser.add_argument('--features', choices=['baseline', 'velocity', 'map', 'probabilistic'],
                        default='baseline', help='Feature set experiment key')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Optional explicit data directory')
    args = parser.parse_args()

    main(
        checkpoint_path=args.checkpoint_path,
        agent=args.agent,
        features=args.features,
        batch_size=args.batch_size,
        data_dir=args.data_dir
    )
