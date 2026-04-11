from pathlib import Path
import sys
import json
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from metrics.ade_fde import ade, fde

PROCESSED_DIR = PROJECT_ROOT / "processed"

STEP_DT = 0.5
FUTURE_STEPS = 12


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# =========================
# Predictors
# =========================
def stationary_predict(history_xy, future_steps=12, step_dt=0.5):
    p_last = np.asarray(history_xy, dtype=float)[-1]
    return np.repeat(p_last[None, :], future_steps, axis=0)


def constant_velocity_predict(history_xy, future_steps=12, step_dt=0.5):
    history_xy = np.asarray(history_xy, dtype=float)
    p_last = history_xy[-1]
    p_prev = history_xy[-2]
    v = (p_last - p_prev) / step_dt
    ts = np.arange(1, future_steps + 1, dtype=float) * step_dt
    return p_last[None, :] + ts[:, None] * v[None, :]


def average_velocity_predict(history_xy, future_steps=12, step_dt=0.5):
    history_xy = np.asarray(history_xy, dtype=float)
    velocities = np.diff(history_xy, axis=0) / step_dt   # (H-1, 2)
    v = velocities.mean(axis=0)
    p_last = history_xy[-1]
    ts = np.arange(1, future_steps + 1, dtype=float) * step_dt
    return p_last[None, :] + ts[:, None] * v[None, :]


def linear_fit_predict(history_xy, future_steps=12, step_dt=0.5):
    """
    Fit x(t), y(t) separately with least squares over history frames.
    """
    history_xy = np.asarray(history_xy, dtype=float)
    H = history_xy.shape[0]

    t_hist = np.arange(-(H - 1), 1, dtype=float) * step_dt   # e.g. [-1.5, -1.0, -0.5, 0.0]
    t_future = np.arange(1, future_steps + 1, dtype=float) * step_dt

    coeff_x = np.polyfit(t_hist, history_xy[:, 0], deg=1)
    coeff_y = np.polyfit(t_hist, history_xy[:, 1], deg=1)

    pred_x = np.polyval(coeff_x, t_future)
    pred_y = np.polyval(coeff_y, t_future)

    return np.stack([pred_x, pred_y], axis=1)


def constant_acceleration_predict(history_xy, future_steps=12, step_dt=0.5):
    history_xy = np.asarray(history_xy, dtype=float)
    velocities = np.diff(history_xy, axis=0) / step_dt  # (H-1, 2)

    # use last velocity and average acceleration over observed velocities
    v_last = velocities[-1]
    if velocities.shape[0] >= 2:
        accels = np.diff(velocities, axis=0) / step_dt
        a = accels.mean(axis=0)
    else:
        a = np.zeros(2, dtype=float)

    p_last = history_xy[-1]
    ts = np.arange(1, future_steps + 1, dtype=float) * step_dt

    pred = []
    for t in ts:
        p = p_last + v_last * t + 0.5 * a * (t ** 2)
        pred.append(p)
    return np.asarray(pred)


# pedestrian-specific
def pedestrian_cv_predict(history_xy, future_steps=12, step_dt=0.5):
    return constant_velocity_predict(history_xy, future_steps, step_dt)


def pedestrian_damped_cv_predict(history_xy, future_steps=12, step_dt=0.5, alpha=0.6):
    history_xy = np.asarray(history_xy, dtype=float)
    p_last = history_xy[-1]
    p_prev = history_xy[-2]
    v = (p_last - p_prev) / step_dt
    v = alpha * v
    ts = np.arange(1, future_steps + 1, dtype=float) * step_dt
    return p_last[None, :] + ts[:, None] * v[None, :]


def pedestrian_heading_reduced_speed_predict(history_xy, future_steps=12, step_dt=0.5, speed_scale=0.6):
    history_xy = np.asarray(history_xy, dtype=float)

    displacements = np.diff(history_xy, axis=0)
    speeds = np.linalg.norm(displacements / step_dt, axis=1)

    # heading from overall motion across the full observed history
    overall_disp = history_xy[-1] - history_xy[0]
    norm = np.linalg.norm(overall_disp)
    if norm < 1e-8:
        direction = np.zeros(2, dtype=float)
    else:
        direction = overall_disp / norm

    avg_speed = speeds.mean() if len(speeds) > 0 else 0.0
    speed = speed_scale * avg_speed

    p_last = history_xy[-1]
    ts = np.arange(1, future_steps + 1, dtype=float) * step_dt

    pred = []
    for t in ts:
        pred.append(p_last + direction * speed * t)
    return np.asarray(pred)


# =========================
# Evaluation
# =========================
def per_timestep_errors(pred_future, gt_future):
    return np.linalg.norm(pred_future - gt_future, axis=1)


def summarize_metrics(ades, fdes, all_step_errors):
    all_step_errors = np.asarray(all_step_errors)  # (N, T)

    out = {
        "mean_ADE": float(np.mean(ades)),
        "std_ADE": float(np.std(ades)),
        "min_ADE": float(np.min(ades)),
        "max_ADE": float(np.max(ades)),
        "mean_FDE": float(np.mean(fdes)),
        "std_FDE": float(np.std(fdes)),
        "min_FDE": float(np.min(fdes)),
        "max_FDE": float(np.max(fdes)),
        "per_horizon_ADE": {
            f"t={(i+1)*STEP_DT:.1f}s": float(all_step_errors[:, i].mean())
            for i in range(all_step_errors.shape[1])
        },
        "num_samples": int(len(ades)),
    }
    return out


def evaluate_subset(test_x, test_y, categories, predictor_fn, subset_category=None):
    ades = []
    fdes = []
    all_step_errors = []

    for i in range(len(test_x)):
        if subset_category is not None and categories[i] != subset_category:
            continue

        history = test_x[i]
        gt_future = test_y[i]

        pred_future = predictor_fn(
            history_xy=history,
            future_steps=FUTURE_STEPS,
            step_dt=STEP_DT,
        )

        ades.append(ade(pred_future, gt_future))
        fdes.append(fde(pred_future, gt_future))
        all_step_errors.append(per_timestep_errors(pred_future, gt_future))

    if len(ades) == 0:
        return None

    return summarize_metrics(ades, fdes, all_step_errors)


def print_metrics(title, metrics):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"N:        {metrics['num_samples']}")
    print(f"Mean ADE: {metrics['mean_ADE']:.4f} m")
    print(f"Std ADE:  {metrics['std_ADE']:.4f} m")
    print(f"Min ADE:  {metrics['min_ADE']:.4f} m")
    print(f"Max ADE:  {metrics['max_ADE']:.4f} m")
    print(f"Mean FDE: {metrics['mean_FDE']:.4f} m")
    print(f"Std FDE:  {metrics['std_FDE']:.4f} m")
    print(f"Min FDE:  {metrics['min_FDE']:.4f} m")
    print(f"Max FDE:  {metrics['max_FDE']:.4f} m")

    print("Per-horizon ADE:")
    for k, v in metrics["per_horizon_ADE"].items():
        print(f"  {k}: {v:.4f} m")


def main():
    test_x = torch.load(PROCESSED_DIR / "test_x.pt").cpu().numpy()
    test_y = torch.load(PROCESSED_DIR / "test_y.pt").cpu().numpy()
    test_categories = load_json(PROCESSED_DIR / "test_categories.json")

    predictors = {
        # general baselines
        "stationary": stationary_predict,
        "cv": constant_velocity_predict,
        "cv_avg": average_velocity_predict,                    # 2A
        "linear_fit": linear_fit_predict,                      # 2B
        "constant_acceleration": constant_acceleration_predict,# 2C

        # pedestrian-specific
        "ped_cv": pedestrian_cv_predict,                       # 5A
        "ped_damped_cv": pedestrian_damped_cv_predict,         # 5B
        "ped_heading_reduced_speed": pedestrian_heading_reduced_speed_predict,  # 5C
    }

    results = {}

    # car experiments
    car_methods = ["stationary", "cv", "cv_avg", "linear_fit", "constant_acceleration"]
    for name in car_methods:
        metrics = evaluate_subset(
            test_x, test_y, test_categories,
            predictor_fn=predictors[name],
            subset_category="car"
        )
        results[f"car::{name}"] = metrics
        print_metrics(f"CAR :: {name}", metrics)

    # pedestrian experiments
    ped_methods = ["stationary", "ped_cv", "ped_damped_cv", "ped_heading_reduced_speed"]
    for name in ped_methods:
        metrics = evaluate_subset(
            test_x, test_y, test_categories,
            predictor_fn=predictors[name],
            subset_category="pedestrian"
        )
        results[f"pedestrian::{name}"] = metrics
        print_metrics(f"PEDESTRIAN :: {name}", metrics)

    with open(PROCESSED_DIR / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[SAVED] Results written to: {PROCESSED_DIR / 'baseline_results.json'}")


if __name__ == "__main__":
    main()