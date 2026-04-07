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

    Input shape:  (B, 4, F)  - B batch size, 4 history frames, F features
    Output shape: (B, 12, O) - B batch size, 12 future frames, O output features
    """

    def __init__(
        self,
        history_frames: int = 4,
        future_frames: int = 12,
        hidden_dim: int = 256,
        input_features: int = 2,
        output_features: int = 2,
    ) -> None:
        """
        Args:
            history_frames: Number of history frames (default 4)
            future_frames: Number of future frames to predict (default 12)
            hidden_dim: Hidden dimension size (default 32)
            input_features: Number of input features per frame
            output_features: Number of output features per frame (default 2 for x, y)
        """
        super().__init__()
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.hidden_dim = hidden_dim
        self.input_features = input_features
        self.output_features = output_features

        # LSTM encoder: processes trajectory history
        self.encoder = nn.LSTM(
            input_size=input_features,
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
        self.decoder_input = nn.Linear(output_features, hidden_dim)

        # Output projection to features
        self.output_layer = nn.Linear(hidden_dim, output_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input trajectories of shape (batch_size, history_frames, input_features)

        Returns:
            Predicted trajectories of shape (batch_size, future_frames, output_features)
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

        # Start decoder from the last available target-sized slice (typically x,y).
        prev_point = x[:, -1, : self.output_features]
        predictions = []

        # Decode future trajectory
        for frame_idx in range(self.future_frames):
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


class ProbabilisticLSTMGRUPredictor(nn.Module):
    """
    LSTM-GRU based probabilistic trajectory predictor.
    Predicts Gaussian parameters for each future coordinate.

    Input shape:  (B, 4, F)
    Output shape: mean/logvar each (B, 12, O)
    """

    def __init__(
        self,
        history_frames: int = 4,
        future_frames: int = 12,
        hidden_dim: int = 256,
        input_features: int = 2,
        output_features: int = 2,
    ) -> None:
        super().__init__()
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.hidden_dim = hidden_dim
        self.input_features = input_features
        self.output_features = output_features

        self.encoder = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        self.decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.decoder_input = nn.Linear(output_features, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, output_features)
        self.logvar_head = nn.Linear(hidden_dim, output_features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_outputs, (hidden_state, _) = self.encoder(x)
        encoder_summary = hidden_state[-1]
        context, _ = torch.max(encoder_outputs, dim=1)

        decoder_seed = self.feature_fusion(
            torch.cat([encoder_summary, context], dim=-1)
        )
        decoder_hidden = decoder_seed.unsqueeze(0)

        prev_point = x[:, -1, : self.output_features]
        pred_mean = []
        pred_logvar = []

        for _ in range(self.future_frames):
            decoder_input = self.decoder_input(prev_point).unsqueeze(1)
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            step_hidden = decoder_output.squeeze(1)

            step_mean = self.mean_head(step_hidden)
            # Clamp log-variance for numerical stability.
            step_logvar = torch.clamp(self.logvar_head(step_hidden), min=-8.0, max=4.0)

            pred_mean.append(step_mean.unsqueeze(1))
            pred_logvar.append(step_logvar.unsqueeze(1))
            prev_point = step_mean

        return torch.cat(pred_mean, dim=1), torch.cat(pred_logvar, dim=1)


def create_model(
    num_input_frames=4,
    num_output_frames=12,
    num_features=2,
    num_output_features=2,
    device="cuda" if torch.cuda.is_available() else "cpu",
    hidden_dim=32,
    **kwargs,
):
    """
    Factory function to create an RNN trajectory prediction model.

    Args:
        num_input_frames: Number of history frames (default 4)
        num_output_frames: Number of future frames (default 12)
        num_features: Number of input features per frame
        num_output_features: Number of output features per frame
        device: Device to place model on ('cuda' or 'cpu')
        hidden_dim: Hidden dimension size for LSTM/GRU (default 32)
        **kwargs: Additional arguments (ignored)

    Returns:
        LSTMGRUPredictor model on specified device
    """
    model = LSTMGRUPredictor(
        history_frames=num_input_frames,
        future_frames=num_output_frames,
        hidden_dim=hidden_dim,
        input_features=num_features,
        output_features=num_output_features,
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
