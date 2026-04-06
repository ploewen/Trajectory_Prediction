"""
Transformer-based trajectory prediction model.
Predicts future 12 frames (6 seconds) of agent motion from 4 history frames (2 seconds).
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Injects positional information into embeddings."""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args:
            d_model: Dimension of embeddings
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Embeddings of shape (batch_size, seq_len, d_model)
        
        Returns:
            Embeddings with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerTrajectoryPredictor(nn.Module):
    """Transformer-based trajectory prediction model."""
    
    def __init__(
        self,
        num_input_frames=4,  # Number of history frames
        num_output_frames=12, # Number of prediction frames
        num_input_features=2,  # Number of input features per frame (x, y, vx, vy, etc.)
        num_output_features=2, # Number of output features per frame (usually x, y)
        d_model=64,          # Hidden dimension
        nhead=8,             # Number of attention heads
        num_layers=4,        # Number of transformer encoder layers
        dim_feedforward=256,  # Feedforward network dimension
        dropout=0.1,
        activation='relu'
    ):
        """
        Args:
            num_input_frames: Number of history frames
            num_output_frames: Number of prediction frames
            num_input_features: Number of input features per frame
            num_output_features: Number of output features per frame
            d_model: Hidden dimension of transformer
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()
        
        # Calculate input/output dimensions from frame/feature counts
        self.num_input_frames = num_input_frames
        self.num_output_frames = num_output_frames
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features
        self.input_dim = num_input_frames * num_input_features
        self.output_dim = num_output_frames * num_output_features
        self.d_model = d_model
        
        # Input embedding layer: projects each frame independently to embedding dimension.
        # This creates one transformer token per history frame so attention operates across time.
        self.input_embedding = nn.Linear(self.num_input_features, d_model)
        
        # Positional encoding for sequence order
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=5000,
            dropout=dropout
        )
        
        # Transformer encoder: learns contextual patterns
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output head: projects back to trajectory space
        self.output_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, self.output_dim)
        )
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input trajectories of shape (batch_size, num_input_frames, num_input_features)
        
        Returns:
            Predicted future trajectories of shape (batch_size, num_output_frames, num_output_features)
        """
        batch_size = x.shape[0]

        # Embed each history frame independently: (batch, H, F_in) -> (batch, H, d_model)
        x_embed = self.input_embedding(x)

        # Add positional encoding so the model knows frame order.
        x_encoded = self.positional_encoding(x_embed)

        # Transformer encoder attends across the history frames: (batch, H, d_model)
        x_transformer = self.transformer_encoder(x_encoded)

        # Use the final history token as the summary representation.
        sequence_summary = x_transformer[:, -1, :]

        # Project to future trajectory space: (batch, d_model) -> (batch, T * F_out)
        output = self.output_head(sequence_summary)

        # Reshape to trajectory format: (batch, T * F_out) -> (batch, T, F_out)
        output = output.reshape(batch_size, self.num_output_frames, self.num_output_features)
        
        return output


class LegacyFlattenedTransformerTrajectoryPredictor(nn.Module):
    """Previous flattened single-token variant kept for old checkpoint loading."""

    def __init__(
        self,
        num_input_frames=4,
        num_output_frames=12,
        num_input_features=2,
        num_output_features=2,
        d_model=64,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        activation='relu'
    ):
        super().__init__()

        self.num_input_frames = num_input_frames
        self.num_output_frames = num_output_frames
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features
        self.input_dim = num_input_frames * num_input_features
        self.output_dim = num_output_frames * num_output_features
        self.d_model = d_model

        self.input_embedding = nn.Linear(self.input_dim, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=5000,
            dropout=dropout
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.output_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, self.output_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        x_embed = self.input_embedding(x_flat).unsqueeze(1)
        x_encoded = self.positional_encoding(x_embed)
        x_transformer = self.transformer_encoder(x_encoded)
        output = self.output_head(x_transformer.squeeze(1))
        return output.reshape(batch_size, self.num_output_frames, self.num_output_features)


def create_model(num_input_frames=4, num_output_frames=12,
                 num_input_features=2, num_output_features=2,
                 device='cuda' if torch.cuda.is_available() else 'cpu', **kwargs):
    """
    Factory function to create and initialize the model.
    
    Args:
        num_input_frames: Number of history frames
        num_output_frames: Number of prediction frames
        num_input_features: Number of input features per frame
        num_output_features: Number of output features per frame
        device: Device to place model on ('cuda' or 'cpu')
        **kwargs: Additional arguments to pass to TransformerTrajectoryPredictor
    
    Returns:
        Model on specified device
    """
    model = TransformerTrajectoryPredictor(
        num_input_frames=num_input_frames,
        num_output_frames=num_output_frames,
        num_input_features=num_input_features,
        num_output_features=num_output_features,
        **kwargs
    )
    model = model.to(device)
    return model


if __name__ == '__main__':
    # Test the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Example 1: Default (4 input frames, 12 output frames, 2 features)
    print("Example 1: Standard (x, y)")
    model = create_model(
        num_input_frames=4,
        num_output_frames=12,
        num_input_features=2,
        num_output_features=2,
        device=device
    )
    x = torch.randn(4, 4, 2).to(device)
    
    # Forward pass
    y = model(x)
    
    print(f"Device: {device}")
    print(f"\nInput shape: {x.shape}, Output shape: {y.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable_params:,}\n")
    
    # Example 2: With velocity (x, y, vx, vy)
    print("Example 2: With velocity input (x, y, vx, vy) and position-only output")
    model_with_vel = create_model(
        num_input_frames=4,
        num_output_frames=12,
        num_input_features=4,
        num_output_features=2,
        device=device
    )
    x_vel = torch.randn(4, 4, 4).to(device)  # 4 features
    y_vel = model_with_vel(x_vel)
    print(f"Input shape: {x_vel.shape}, Output shape: {y_vel.shape}")
    total_params_vel = sum(p.numel() for p in model_with_vel.parameters())
    print(f"Parameters: {total_params_vel:,}\n")
    
    # Example 3: More prediction frames
    print("Example 3: More prediction frames (20 frames ahead instead of 12)")
    model_20frames = create_model(
        num_input_frames=4,
        num_output_frames=20,
        num_input_features=2,
        num_output_features=2,
        device=device
    )
    y_20 = model_20frames(x)
    print(f"Input shape: {x.shape}, Output shape: {y_20.shape}")
    total_params_20 = sum(p.numel() for p in model_20frames.parameters())
    print(f"Parameters: {total_params_20:,}")
