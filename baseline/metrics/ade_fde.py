import numpy as np

def ade(pred_xy: np.ndarray, gt_xy: np.ndarray) -> float:
    """
    Average Displacement Error (ADE).
    pred_xy, gt_xy: shape (T, 2)
    """
    pred_xy = np.asarray(pred_xy, dtype=float)
    gt_xy = np.asarray(gt_xy, dtype=float)
    assert pred_xy.shape == gt_xy.shape and pred_xy.shape[1] == 2
    return float(np.mean(np.linalg.norm(pred_xy - gt_xy, axis=1)))

def fde(pred_xy: np.ndarray, gt_xy: np.ndarray) -> float:
    """
    Final Displacement Error (FDE).
    """
    pred_xy = np.asarray(pred_xy, dtype=float)
    gt_xy = np.asarray(gt_xy, dtype=float)
    assert pred_xy.shape == gt_xy.shape and pred_xy.shape[1] == 2
    return float(np.linalg.norm(pred_xy[-1] - gt_xy[-1]))