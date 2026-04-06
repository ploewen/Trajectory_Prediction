"""
Training script for Transformer-based trajectory prediction model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
import json
import time
from tqdm import tqdm

from model import TransformerTrajectoryPredictor, create_model
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


def compute_metrics(predictions, targets):
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
    target_pos = targets[:, :, :2]     # Get position only (x, y)
    
    # ADE: Average Displacement Error (mean L2 distance to ground truth)
    distances = torch.norm(pred_pos - target_pos, dim=2)  # (batch, num_frames)
    ade = torch.mean(distances).item()
    
    # FDE: Final Displacement Error (L2 distance at final frame)
    fde = torch.mean(distances[:, -1]).item()
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'ade': ade,
        'fde': fde
    }
    
    # If categorical features exist (one-hot encoded), compute ADE per category
    if num_features > 2:
        num_categories = num_features - 2  # Number of categorical features
        
        # Extract one-hot categories from targets (position-independent, same across frames)
        categories_onehot = targets[0, 0, 2:2+num_categories]  # Shape: (num_categories,)
        category_idx = torch.argmax(categories_onehot).item()
        
        # Get all samples and their categories
        all_categories_onehot = targets[:, 0, 2:2+num_categories]  # (batch, num_categories)
        category_indices = torch.argmax(all_categories_onehot, dim=1)  # (batch,)
        
        # Compute ADE per category
        ade_per_category = {}
        for cat_id in range(num_categories):
            mask = category_indices == cat_id
            if mask.sum() > 0:
                cat_distances = distances[mask]
                cat_ade = torch.mean(cat_distances).item()
                ade_per_category[f'category_{cat_id}'] = cat_ade
        
        metrics['ade_per_category'] = ade_per_category
    
    return metrics


def train_epoch(model, train_loader, optimizer, criterion, device):
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
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for x_batch, y_batch in pbar:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        y_pred = model(x_batch)
        
        # Compute loss
        loss = criterion(y_pred, y_batch)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Compute metrics
        metrics = compute_metrics(y_pred.detach(), y_batch)
        
        total_loss += loss.item()
        total_ade += metrics['ade']
        total_fde += metrics['fde']
        num_batches += 1
        
        pbar.update(1)
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return {
        'loss': total_loss / num_batches,
        'ade': total_ade / num_batches,
        'fde': total_fde / num_batches
    }


def validate(model, test_loader, criterion, device):
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
        pbar = tqdm(test_loader, desc='Validation', leave=False)
        for x_batch, y_batch in pbar:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            y_pred = model(x_batch)
            
            # Compute loss
            loss = criterion(y_pred, y_batch)
            
            # Compute metrics
            metrics = compute_metrics(y_pred, y_batch)
            
            total_loss += loss.item()
            total_ade += metrics['ade']
            total_fde += metrics['fde']
            num_batches += 1
            
            pbar.update(1)
    
    return {
        'loss': total_loss / num_batches,
        'ade': total_ade / num_batches,
        'fde': total_fde / num_batches
    }


def train(
    num_epochs=100,
    batch_size=32,
    learning_rate=5e-4,
    device=None,
    checkpoint_dir=None,
    data_dir=None,
    agent='car',
    features='baseline',
    d_model=64,
    nhead=8,
    num_layers=4
):
    """
    Full training pipeline.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        device: Device to train on (defaults to GPU if available)
        checkpoint_dir: Directory to save checkpoints (if None, uses checkpoints/{agent}_{features})
        data_dir: Directory containing train_x.pt/train_y.pt/scene_ids.pt
        agent: Agent type experiment key (e.g., car, pedestrian)
        features: Feature set experiment key (e.g., baseline, velocity, map, probabilistic)
        d_model: Hidden dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
    """
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Setup checkpoint directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    experiment_name = f"{agent}_{features}"

    if checkpoint_dir is None:
        checkpoint_dir = project_dir / 'checkpoints' / experiment_name
    else:
        checkpoint_dir = Path(checkpoint_dir)
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Experiment: agent={agent}, features={features}")

    if data_dir is None:
        data_dir = resolve_data_dir(project_dir, agent, features)
    else:
        data_dir = Path(data_dir)
        print(f"Using explicit data dir: {data_dir}")
    
    # Load data
    print("\nLoading data...")
    train_x, train_y, scene_ids = load_data(data_dir=data_dir, device=device)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, test_loader, train_idx, test_idx = create_dataloaders(
        train_x, train_y, scene_ids,
        batch_size=batch_size,
        device=device,
        shuffle_train=True
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
    
    model = TransformerTrajectoryPredictor(
        num_input_frames=num_input_frames,
        num_output_frames=num_output_frames,
        num_input_features=num_input_features,
        num_output_features=num_output_features,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=256,
        dropout=0.1
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,
        patience=10,
        cooldown=2,
        min_lr=1e-6
    )
    criterion = nn.MSELoss()
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...\n")
    csv_metrics_path = checkpoint_dir / 'training_metrics.csv'
    with open(csv_metrics_path, 'w') as f:
        f.write("epoch,train_loss,val_loss\n")

    history = {
        'epoch': [],
        'train_loss': [],
        'train_ade': [],
        'train_fde': [],
        'val_loss': [],
        'val_ade': [],
        'val_fde': []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_metrics = validate(model, test_loader, criterion, device)
        
        # Update history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_metrics['loss'])
        history['train_ade'].append(train_metrics['ade'])
        history['train_fde'].append(train_metrics['fde'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_ade'].append(val_metrics['ade'])
        history['val_fde'].append(val_metrics['fde'])
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} (ADE: {train_metrics['ade']:.4f}) | "
              f"Val Loss: {val_metrics['loss']:.4f} (ADE: {val_metrics['ade']:.4f})")
        
        # Print per-category metrics if available
        if 'ade_per_category' in val_metrics:
            for cat_name, cat_ade in val_metrics['ade_per_category'].items():
                print(f"    {cat_name}: {cat_ade:.4f}")

        # Append CSV row for easy spreadsheet/report usage
        with open(csv_metrics_path, 'a') as f:
            f.write(f"{epoch + 1},{train_metrics['loss']},{val_metrics['loss']}\n")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch + 1
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'hyperparameters': {
                    'd_model': d_model,
                    'nhead': nhead,
                    'num_layers': num_layers,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'agent': agent,
                    'features': features,
                    'num_input_frames': num_input_frames,
                    'num_input_features': num_input_features,
                    'num_output_frames': num_output_frames,
                    'num_output_features': num_output_features,
                    'architecture_version': 'temporal_tokens_v2'
                }
            }
            
            checkpoint_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, checkpoint_path)

            # Extra lightweight best-model state dict for quick loading.
            torch.save(model.state_dict(), checkpoint_dir / 'transformer_best.pth')
            print(f"  → Saved best model (Val Loss: {val_metrics['loss']:.4f})")
    
    # Save final model
    final_checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hyperparameters': {
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'agent': agent,
            'features': features,
            'num_input_frames': num_input_frames,
            'num_input_features': num_input_features,
            'num_output_frames': num_output_frames,
            'num_output_features': num_output_features,
            'architecture_version': 'temporal_tokens_v2'
        }
    }
    torch.save(final_checkpoint, checkpoint_dir / 'final_model.pt')
    
    # Save history
    history_path = checkpoint_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Summary
    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Training completed in {elapsed_time:.1f} seconds")
    print(f"Best model at epoch {best_epoch} with Val Loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"CSV metrics saved to: {csv_metrics_path}")
    print(f"{'='*80}")
    
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train transformer trajectory predictor')
    parser.add_argument('--agent', choices=['car', 'pedestrian'], default='car',
                        help='Agent type experiment key')
    parser.add_argument('--features', choices=['baseline', 'velocity', 'map', 'probabilistic'],
                        default='baseline', help='Feature set experiment key')
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Optional explicit data directory')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Optional explicit checkpoint directory')
    args = parser.parse_args()

    train(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        agent=args.agent,
        features=args.features,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers
    )
