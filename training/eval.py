"""
Evaluation script for trained trajectory prediction models.
Computes metrics like ADE, FDE, and MSE on test set.
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

from model import TransformerTrajectoryPredictor
from dataset import load_data, create_dataloaders


def evaluate_model(checkpoint_path, device=None, batch_size=64):
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
    
    # Load checkpoint
    print(f"\nLoading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct model
    hyperparams = checkpoint.get('hyperparameters', {})
    d_model = hyperparams.get('d_model', 64)
    nhead = hyperparams.get('nhead', 8)
    num_layers = hyperparams.get('num_layers', 4)
    
    model = TransformerTrajectoryPredictor(
        input_dim=8,
        output_dim=24,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded (Epoch {checkpoint.get('epoch', 'N/A')})")
    
    # Load data
    print("\nLoading data...")
    train_x, train_y, scene_ids = load_data(device=device)
    
    # Create dataloaders
    train_loader, test_loader, train_idx, test_idx = create_dataloaders(
        train_x, train_y, scene_ids,
        batch_size=batch_size,
        device=device,
        shuffle_train=False
    )
    
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
    predictions = np.concatenate(all_predictions, axis=0)  # (N, 12, 2)
    targets = np.concatenate(all_targets, axis=0)  # (N, 12, 2)
    
    # Compute metrics
    print("\nComputing metrics...")
    
    # MSE
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    
    # ADE: Average Displacement Error
    distances = np.linalg.norm(predictions - targets, axis=2)  # (N, 12)
    ade = np.mean(distances)
    ade_std = np.std(distances)
    
    # FDE: Final Displacement Error  
    fde = np.mean(distances[:, -1])
    fde_std = np.std(distances[:, -1])
    
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
        'fde': float(fde),
        'fde_std': float(fde_std),
        'miss_rates': miss_rates,
        'per_frame_ade': per_frame_ade.tolist(),
        'num_test_samples': len(predictions)
    }
    
    return metrics, predictions, targets


def main(checkpoint_path=None):
    """Main evaluation function."""
    
    # Find best model if no checkpoint specified
    if checkpoint_path is None:
        script_dir = Path(__file__).parent
        project_dir = script_dir.parent
        checkpoint_dir = project_dir / 'checkpoints'
        checkpoint_path = checkpoint_dir / 'best_model.pt'
    
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("\nUsage: python eval.py [checkpoint_path]")
        print("Example: python eval.py checkpoints/best_model.pt")
        return
    
    # Evaluate
    metrics, predictions, targets = evaluate_model(checkpoint_path)
    
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
    print(f"  FDE (Final Displacement Error):   {metrics['fde']:.4f}m (±{metrics['fde_std']:.4f})")
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
    
    return metrics


if __name__ == '__main__':
    import sys
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(checkpoint_path)
