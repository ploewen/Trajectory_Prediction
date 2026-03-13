#!/usr/bin/env python3
"""
Example usage script showing how to use the trained model for inference.
"""

import torch
from pathlib import Path
from model import TransformerTrajectoryPredictor, LegacyFlattenedTransformerTrajectoryPredictor


def load_checkpoint(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get hyperparameters
    hyperparams = checkpoint.get('hyperparameters', {})
    
    num_input_frames = hyperparams.get('num_input_frames', 4)
    num_input_features = hyperparams.get('num_input_features', 2)
    num_output_frames = hyperparams.get('num_output_frames', 12)
    num_output_features = hyperparams.get('num_output_features', 2)

    input_embedding_weight = checkpoint['model_state_dict']['input_embedding.weight']
    uses_legacy_flattened_model = input_embedding_weight.shape[1] == num_input_frames * num_input_features
    model_cls = LegacyFlattenedTransformerTrajectoryPredictor if uses_legacy_flattened_model else TransformerTrajectoryPredictor

    model = model_cls(
        num_input_frames=num_input_frames,
        num_output_frames=num_output_frames,
        num_input_features=num_input_features,
        num_output_features=num_output_features,
        d_model=hyperparams.get('d_model', 64),
        nhead=hyperparams.get('nhead', 8),
        num_layers=hyperparams.get('num_layers', 4),
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def predict_trajectory(model, history, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Predict future trajectory from history.
    
    Args:
        model: Trained model
        history: Input tensor of shape (1, 4, 2) or (4, 2)
                - 4 frames of agent motion history
                - 2 coordinates (x, y) in ego-centric frame
        device: Device to run inference on
    
    Returns:
        Predicted trajectory of shape (1, 12, 2) or (12, 2)
                - 12 frames of predicted future motion
                - 2 coordinates (x, y) in ego-centric frame
    """
    # Handle single frame (4, 2) -> (1, 4, 2)
    unsqueezed = False
    if history.dim() == 2:
        history = history.unsqueeze(0)
        unsqueezed = True
    
    history = history.to(device)
    
    with torch.no_grad():
        prediction = model(history)
    
    # Remove batch dimension if input was single frame
    if unsqueezed:
        prediction = prediction.squeeze(0)
    
    return prediction


def main():
    """Example usage."""
    print("Trajectory Prediction Example\n")
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    checkpoint_path = project_root / 'checkpoints' / 'best_model.pt'
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found. Train the model first:")
        print(f"  python training/train.py")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load model
    print("Loading trained model...")
    model = load_checkpoint(checkpoint_path, device=device)
    print(f"✓ Model loaded from {checkpoint_path}\n")
    
    # Example 1: Single trajectory prediction
    print("=" * 60)
    print("Example 1: Single Trajectory Prediction")
    print("=" * 60)
    
    # Create dummy history matching the default baseline setup
    history = torch.randn(4, 2)
    print(f"History shape: {history.shape}")
    print(f"History (first 2 frames):\n{history[:2]}\n")
    
    # Predict
    prediction = predict_trajectory(model, history, device=device)
    print(f"Prediction shape: {prediction.shape}")
    print(f"Predicted trajectory (first 2 frames):\n{prediction[:2]}\n")
    
    # Example 2: Batch prediction
    print("=" * 60)
    print("Example 2: Batch Prediction (32 trajectories)")
    print("=" * 60)
    
    # Create batch of histories
    batch_history = torch.randn(32, 4, 2)
    print(f"Batch shape: {batch_history.shape}")
    
    # Predict
    batch_history_gpu = batch_history.to(device)
    with torch.no_grad():
        batch_prediction = model(batch_history_gpu)
    
    print(f"Predictions shape: {batch_prediction.shape}")
    print(f"Average prediction at final frame:\n  x: {batch_prediction[:, -1, 0].mean():.4f}")
    print(f"  y: {batch_prediction[:, -1, 1].mean():.4f}\n")
    
    # Example 3: Trajectory sequence
    print("=" * 60)
    print("Example 3: Understanding the Data Format")
    print("=" * 60)
    
    print("""
Input Trajectory (4 frames):
  Frame 0: Position 2.0 seconds in the past (oldest)
  Frame 1: Position 1.5 seconds in the past
  Frame 2: Position 1.0 seconds in the past
  Frame 3: Position 0.5 seconds in the past (most recent)
  
Output Trajectory (12 frames):
  Frame 0: Predicted position 0.5 seconds in the future
  Frame 1: Predicted position 1.0 seconds in the future
  ...
  Frame 11: Predicted position 6.0 seconds in the future
  
Coordinate System:
  (x, y) - Ego-centric coordinates relative to agent
  x - Forward direction
  y - Left direction
  """)
    
    print("=" * 60)
    print("✓ Example completed successfully!")
    print("\nTo train your own model:")
    print("  python training/train.py")
    print("\nTo evaluate model performance:")
    print("  python training/eval.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
