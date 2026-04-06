"""
RNN-based trajectory prediction model.
Predicts future 12 frames (6 seconds) of agent motion from 4 history frames (2 seconds).
Uses an LSTM encoder and GRU decoder.
"""

import torch
import torch.nn as nn


class LSTMGRUPredictor(nn.Module):
    """
    LSTM-GRU based trajectory predictor.
    Encodes history with LSTM and decodes predictions with GRU.

    Input shape:  (B, 4, 2)  - B batch size, 4 history frames, 2 coordinates (x, y)
    Output shape: (B, 12, 2) - B batch size, 12 future frames, 2 coordinates (x, y)
    """

    def __init__(
        self,
        history_frames: int = 4,
        future_frames: int = 12,
        hidden_dim: int = 32,
    ) -> None:
        """
        Args:
            history_frames: Number of history frames (default 4)
            future_frames: Number of future frames to predict (default 12)
            hidden_dim: Hidden dimension size (default 32)
        """
        super().__init__()
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.hidden_dim = hidden_dim

        # LSTM encoder: processes trajectory history
        self.encoder = nn.LSTM(
            input_size=2,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Feature fusion: combines encoder hidden state and context
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        # GRU decoder: generates future trajectory
        self.decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Input projection for decoder
        self.decoder_input = nn.Linear(2, hidden_dim)

        # Output projection to (x, y) coordinates
        self.output_layer = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input trajectories of shape (batch_size, history_frames, 2)

        Returns:
            Predicted trajectories of shape (batch_size, future_frames, 2)
        """
        # Encode history
        encoder_outputs, (hidden_state, _) = self.encoder(x)
        encoder_summary = hidden_state[-1]  # (B, hidden_dim)

        # Extract context from encoder outputs
        context, _ = torch.max(encoder_outputs, dim=1)  # (B, hidden_dim)

        # Fuse encoder summary and context
        decoder_seed = self.feature_fusion(
            torch.cat([encoder_summary, context], dim=-1)
        )  # (B, hidden_dim)

        # Initialize decoder hidden state
        decoder_hidden = decoder_seed.unsqueeze(0)  # (1, B, hidden_dim)

        # Start with last position from history
        prev_point = x[:, -1, :]  # (B, 2)
        predictions = []

        # Decode future trajectory
        for _ in range(self.future_frames):
            # Project previous point to decoder input dimension
            decoder_input = self.decoder_input(prev_point).unsqueeze(
                1
            )  # (B, 1, hidden_dim)

            # GRU step
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            # Project to (x, y) coordinates
            next_point = self.output_layer(decoder_output.squeeze(1))  # (B, 2)

            predictions.append(next_point.unsqueeze(1))
            prev_point = next_point

        return torch.cat(predictions, dim=1)  # (B, future_frames, 2)


def create_model(
    num_input_frames=4,
    num_output_frames=12,
    num_features=2,
    device="cuda" if torch.cuda.is_available() else "cpu",
    hidden_dim=32,
    **kwargs,
):
    """
    Factory function to create an RNN trajectory prediction model.

    Args:
        num_input_frames: Number of history frames (default 4)
        num_output_frames: Number of future frames (default 12)
        num_features: Number of features per frame (must be 2 for this model: x, y)
        device: Device to place model on ('cuda' or 'cpu')
        hidden_dim: Hidden dimension size for LSTM/GRU (default 32)
        **kwargs: Additional arguments (ignored)

    Returns:
        LSTMGRUPredictor model on specified device
    """
    if num_features != 2:
        raise ValueError(
            f"LSTMGRUPredictor only supports 2 features (x, y), got {num_features}"
        )

    model = LSTMGRUPredictor(
        history_frames=num_input_frames,
        future_frames=num_output_frames,
        hidden_dim=hidden_dim,
    )
    model = model.to(device)
    return model


if __name__ == "__main__":
    # Test the model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Testing LSTMGRUPredictor model")
    print("=" * 60)

    # Create model
    model = LSTMGRUPredictor(history_frames=4, future_frames=12, hidden_dim=32).to(
        device
    )

    # Test input
    x = torch.randn(8, 4, 2).to(device)  # 8 samples, 4 frames, 2 coordinates

    # Forward pass
    y = model(x)

    print(f"Device: {device}")
    print(f"Input shape:  {x.shape}  (batch_size, history_frames, features)")
    print(f"Output shape: {y.shape} (batch_size, future_frames, features)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"\nModel parameters: {trainable_params:,} (trainable) / {total_params:,} (total)"
    )
