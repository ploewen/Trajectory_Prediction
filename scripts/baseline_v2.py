from pathlib import Path
import sys
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from metrics.ade_fde import ade, fde

PROCESSED_DIR = PROJECT_ROOT / "processed"

STEP_DT = 0.5
FUTURE_STEPS = 12


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed"

STEP_DT = 0.5      # 2Hz
FUTURE_STEPS = 12


def constant_velocity_predict_from_history(history_xy, future_steps=12, step_dt=0.5):
    """
    history_xy: (H, 2), here H=4
    Use the last two observed points to estimate constant velocity.
    """
    history_xy = np.asarray(history_xy, dtype=float)

    if history_xy.shape[0] < 2:
        raise ValueError("Need at least 2 history points for CV baseline.")

    p_last = history_xy[-1]                  # latest observed position
    p_prev = history_xy[-2]
    v = (p_last - p_prev) / step_dt          # m/s in ego frame

    ts = np.arange(1, future_steps + 1, dtype=float) * step_dt
    pred = p_last[None, :] + ts[:, None] * v[None, :]
    return pred


def main():
    test_x = torch.load(PROCESSED_DIR / "test_x.pt")
    test_y = torch.load(PROCESSED_DIR / "test_y.pt")

    test_x = test_x.cpu().numpy()   # (N, 4, 2)
    test_y = test_y.cpu().numpy()   # (N, 12, 2)

    ades = []
    fdes = []

    for i in range(len(test_x)):
        history = test_x[i]
        gt_future = test_y[i]

        pred_future = constant_velocity_predict_from_history(
            history_xy=history,
            future_steps=FUTURE_STEPS,
            step_dt=STEP_DT,
        )

        ades.append(ade(pred_future, gt_future))
        fdes.append(fde(pred_future, gt_future))

    print(f"CV baseline on preprocessed tensors: N={len(test_x)}")
    print(f"Mean ADE: {np.mean(ades):.4f} m")
    print(f"Mean FDE: {np.mean(fdes):.4f} m")


if __name__ == "__main__":
    main()