"""
Multi-tool trajectory visualization script
Select visualization tool via command-line argument

Usage:
    python visualize.py --tool matplotlib [--trajectories 10]
    python visualize.py --tool plotly [--trajectories 10]
    python visualize.py --tool open3d [--trajectories 10]
    python visualize.py --tool tensorboard [--logdir logs]
"""

import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration (relative to script location)
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent  # Go up from /scripts to root
DATA_DIR = PROJECT_DIR / 'data'
HISTORY_FRAMES = 4
FUTURE_FRAMES = 12

print(f"[INFO] Project directory: {PROJECT_DIR}")
print(f"[INFO] Data directory: {DATA_DIR}")


def load_data():
    """Load preprocessed trajectory data"""
    print("[INFO] Loading trajectory data...")
    train_x = torch.load(DATA_DIR / 'train_x.pt')
    train_y = torch.load(DATA_DIR / 'train_y.pt')
    print(f"[INFO] Loaded {len(train_x)} trajectories")
    print(f"[INFO] train_x shape: {train_x.shape}")
    print(f"[INFO] train_y shape: {train_y.shape}")
    return train_x, train_y


def visualize_matplotlib(train_x, train_y, num_trajectories=10):
    """Visualize using Matplotlib (static plots)"""
    print(f"[INFO] Visualizing {num_trajectories} trajectories with Matplotlib...")
    
    num_cols = 5
    num_rows = (num_trajectories + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 4*num_rows))
    axes = axes.flatten()
    
    for idx in range(min(num_trajectories, len(train_x))):
        ax = axes[idx]
        
        history = train_x[idx].numpy()  # (4, 2)
        future = train_y[idx].numpy()   # (12, 2)
        
        # Plot history (blue)
        ax.plot(history[:, 0], history[:, 1], 'b-o', linewidth=2, markersize=6, label='History (2s)')
        
        # Plot future (red)
        ax.plot(future[:, 0], future[:, 1], 'r-o', linewidth=2, markersize=6, label='Future (6s)')
        
        # Plot ego position (green circle at origin)
        ax.plot(0, 0, 'g*', markersize=20, label='Ego Vehicle')
        
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title(f'Trajectory #{idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    # Hide unused subplots
    for idx in range(num_trajectories, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def visualize_plotly(train_x, train_y, num_trajectories=10):
    """Visualize using Plotly (interactive plots)"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("[ERROR] Plotly not installed. Install with: pip install plotly")
        return
    
    print(f"[INFO] Visualizing {num_trajectories} trajectories with Plotly...")
    
    num_cols = 3
    num_rows = (min(num_trajectories, len(train_x)) + num_cols - 1) // num_cols
    
    fig = make_subplots(
        rows=num_rows, cols=num_cols,
        subplot_titles=[f'Trajectory #{i}' for i in range(min(num_trajectories, len(train_x)))]
    )
    
    for idx in range(min(num_trajectories, len(train_x))):
        row = idx // num_cols + 1
        col = idx % num_cols + 1
        
        history = train_x[idx].numpy()
        future = train_y[idx].numpy()
        
        # History
        fig.add_trace(
            go.Scatter(x=history[:, 0], y=history[:, 1], mode='lines+markers',
                      name='History', line=dict(color='blue')),
            row=row, col=col
        )
        
        # Future
        fig.add_trace(
            go.Scatter(x=future[:, 0], y=future[:, 1], mode='lines+markers',
                      name='Future', line=dict(color='red')),
            row=row, col=col
        )
        
        # Ego position
        fig.add_trace(
            go.Scatter(x=[0], y=[0], mode='markers', marker=dict(size=20, color='green'),
                      name='Ego', showlegend=(idx==0)),
            row=row, col=col
        )
    
    fig.update_xaxes(title_text="X (meters)")
    fig.update_yaxes(title_text="Y (meters)")
    fig.update_layout(height=300*num_rows, width=1200, title_text="Trajectory Visualization")
    fig.show()


def visualize_open3d(train_x, train_y, num_trajectories=10):
    """Visualize using Open3D (3D visualization)"""
    try:
        import open3d as o3d
    except ImportError:
        print("[ERROR] Open3D not installed. Install with: pip install open3d")
        return
    
    print(f"[INFO] Visualizing {num_trajectories} trajectories with Open3D...")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Trajectory 3D Visualization")
    
    for idx in range(min(num_trajectories, len(train_x))):
        history = train_x[idx].numpy()
        future = train_y[idx].numpy()
        
        # Combine history and future
        trajectory = np.vstack([history, future])
        
        # Create line set
        points = o3d.utility.Vector3dVector(np.column_stack([trajectory, np.zeros(len(trajectory))]))
        lines = o3d.utility.Vector2iVector(
            [[i, i+1] for i in range(len(trajectory)-1)]
        )
        
        line_set = o3d.geometry.LineSet(points, lines)
        
        # Color: blue for history, red for future
        colors = np.zeros((len(trajectory), 3))
        colors[:HISTORY_FRAMES] = [0, 0, 1]  # Blue
        colors[HISTORY_FRAMES:] = [1, 0, 0]  # Red
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        vis.add_geometry(line_set)
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
    vis.add_geometry(coord_frame)
    
    vis.run()
    vis.destroy_window()


def visualize_tensorboard(train_x, train_y, logdir=None):
    """Log trajectory statistics to TensorBoard"""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("[ERROR] TensorBoard not installed. Install with: pip install tensorboard")
        return
    
    if logdir is None:
        logdir = PROJECT_DIR / 'logs'
    
    print(f"[INFO] Logging to TensorBoard at {logdir}...")
    
    writer = SummaryWriter(log_dir=str(logdir))
    
    # Log statistics
    history_x_mean = train_x[:, :, 0].mean().item()
    history_y_mean = train_x[:, :, 1].mean().item()
    future_x_mean = train_y[:, :, 0].mean().item()
    future_y_mean = train_y[:, :, 1].mean().item()
    
    writer.add_scalar('Statistics/history_x_mean', history_x_mean)
    writer.add_scalar('Statistics/history_y_mean', history_y_mean)
    writer.add_scalar('Statistics/future_x_mean', future_x_mean)
    writer.add_scalar('Statistics/future_y_mean', future_y_mean)
    
    # Log histograms
    writer.add_histogram('History/X_values', train_x[:, :, 0])
    writer.add_histogram('History/Y_values', train_x[:, :, 1])
    writer.add_histogram('Future/X_values', train_y[:, :, 0])
    writer.add_histogram('Future/Y_values', train_y[:, :, 1])
    
    # Log sample trajectories as images
    for idx in range(min(5, len(train_x))):
        fig, ax = plt.subplots()
        history = train_x[idx].numpy()
        future = train_y[idx].numpy()
        ax.plot(history[:, 0], history[:, 1], 'b-o', label='History')
        ax.plot(future[:, 0], future[:, 1], 'r-o', label='Future')
        ax.plot(0, 0, 'g*', markersize=20, label='Ego')
        ax.legend()
        ax.set_title(f'Trajectory #{idx}')
        writer.add_figure(f'Sample_Trajectories/traj_{idx}', fig, global_step=idx)
        plt.close(fig)
    
    writer.flush()
    writer.close()
    
    print(f"[SUCCESS] TensorBoard logs saved to {logdir}")
    print(f"[INFO] View with: tensorboard --logdir={logdir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize trajectory data')
    parser.add_argument('--tool', type=str, required=True, 
                       choices=['matplotlib', 'plotly', 'open3d', 'tensorboard'],
                       help='Visualization tool to use')
    parser.add_argument('--trajectories', type=int, default=10,
                       help='Number of trajectories to visualize (for matplotlib/plotly/open3d)')
    parser.add_argument('--logdir', type=str, default=None,
                       help='TensorBoard log directory (default: PROJECT_DIR/logs)')
    
    args = parser.parse_args()
    
    # Load data
    train_x, train_y = load_data()
    
    # Visualize based on tool selection
    if args.tool == 'matplotlib':
        visualize_matplotlib(train_x, train_y, args.trajectories)
    elif args.tool == 'plotly':
        visualize_plotly(train_x, train_y, args.trajectories)
    elif args.tool == 'open3d':
        visualize_open3d(train_x, train_y, args.trajectories)
    elif args.tool == 'tensorboard':
        logdir = Path(args.logdir) if args.logdir else None
        visualize_tensorboard(train_x, train_y, logdir)
    
    print("[DONE] Visualization complete!")


if __name__ == '__main__':
    main()
