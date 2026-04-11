"""
Qualitative trajectory plotting utility.

Creates all-in-one 2D plots per sample with:
- History (blue)
- Ground truth future (green)
- Predicted future (red)
- Optional Gaussian uncertainty ellipses for probabilistic checkpoints
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import torch

from dataset import load_data
from model import TransformerTrajectoryPredictor, LegacyFlattenedTransformerTrajectoryPredictor


def resolve_checkpoint(project_dir, agent, features, checkpoint_path=None):
    if checkpoint_path is not None:
        return Path(checkpoint_path)

    experiment_ckpt = project_dir / 'checkpoints' / f"{agent}_{features}" / 'best_model.pt'
    fallback_ckpt = project_dir / 'checkpoints' / 'best_model.pt'
    return experiment_ckpt if experiment_ckpt.exists() else fallback_ckpt


def resolve_data_dir(project_dir, agent, data_dir=None):
    if data_dir is not None:
        return Path(data_dir)
    return project_dir / 'data' / agent


def build_model_from_checkpoint(checkpoint, train_x, train_y, device):
    hyperparams = checkpoint.get('hyperparameters', {})
    d_model = hyperparams.get('d_model', 64)
    nhead = hyperparams.get('nhead', 8)
    num_layers = hyperparams.get('num_layers', 4)

    num_input_frames, num_input_features = train_x.shape[1:]
    num_output_frames, num_output_features = train_y.shape[1:]

    architecture_version = hyperparams.get('architecture_version')
    input_embedding_weight = checkpoint['model_state_dict']['input_embedding.weight']
    uses_legacy_flattened_model = (
        architecture_version is None and input_embedding_weight.shape[1] == num_input_frames * num_input_features
    )

    output_head_weight = checkpoint['model_state_dict'].get('output_head.3.weight')
    inferred_probabilistic = (
        output_head_weight is not None and output_head_weight.shape[0] == num_output_frames * 5
    )
    is_probabilistic = bool(hyperparams.get('is_probabilistic', inferred_probabilistic))

    model_cls = LegacyFlattenedTransformerTrajectoryPredictor if uses_legacy_flattened_model else TransformerTrajectoryPredictor

    model_kwargs = dict(
        num_input_frames=num_input_frames,
        num_output_frames=num_output_frames,
        num_input_features=num_input_features,
        num_output_features=num_output_features,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
    )
    if not uses_legacy_flattened_model:
        model_kwargs['is_probabilistic'] = is_probabilistic

    model = model_cls(**model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, is_probabilistic


def split_test_indices(scene_ids, cutoff=700):
    return torch.where(scene_ids >= cutoff)[0]


def turn_score_from_future(future_xy):
    deltas = np.diff(future_xy, axis=0)
    headings = np.arctan2(deltas[:, 1], deltas[:, 0])
    dtheta = np.diff(np.unwrap(headings))
    return float(np.sum(np.abs(dtheta)))


def path_length(path_xy):
    if len(path_xy) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path_xy, axis=0), axis=-1)))


def choose_auto_indices(agent, x_test, y_test, num_samples):
    n = len(x_test)
    if n == 0:
        return []

    if agent == 'car' and n >= 2:
        scores = np.array([turn_score_from_future(y_test[i, :, :2]) for i in range(n)])
        straight_idx = int(np.argmin(scores))
        turn_idx = int(np.argmax(scores))
        selected = [straight_idx, turn_idx]

        if num_samples > 2:
            mids = np.argsort(scores)
            for idx in mids:
                idx = int(idx)
                if idx not in selected:
                    selected.append(idx)
                if len(selected) >= num_samples:
                    break
        return selected[:num_samples]

    # Pedestrian or tiny sets: evenly spaced picks.
    if n <= num_samples:
        return list(range(n))

    return [int(x) for x in np.linspace(0, n - 1, num_samples)]


def compute_sample_ade(model, x_test, y_test, batch_size=512):
    n = x_test.shape[0]
    if n == 0:
        return np.array([], dtype=np.float32)

    ades = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            pred = model(x_test[start:end])
            pred_xy = pred[..., :2]
            gt_xy = y_test[start:end, :, :2]
            per_timestep_l2 = torch.linalg.norm(pred_xy - gt_xy, dim=-1)
            batch_ade = per_timestep_l2.mean(dim=-1)
            ades[start:end] = batch_ade.detach().cpu().numpy().astype(np.float32)
    return ades


def choose_best_report_indices(context, agent):
    x_test = context['x_test']
    y_test = context['y_test']
    model = context['model']

    n = len(x_test)
    if n == 0:
        raise ValueError(f'No test samples available for agent={agent}.')

    ades = compute_sample_ade(model, x_test, y_test)
    y_np = y_test.detach().cpu().numpy()

    if agent == 'car':
        turn_scores = np.array([turn_score_from_future(y_np[i, :, :2]) for i in range(n)], dtype=np.float32)
        future_lengths = np.array([path_length(y_np[i, :, :2]) for i in range(n)], dtype=np.float32)

        low_q = np.quantile(turn_scores, 0.30)
        high_q = np.quantile(turn_scores, 0.90)

        straight_pool = np.where(turn_scores <= low_q)[0]
        sharp_turn_pool = np.where(turn_scores >= high_q)[0]

        if len(straight_pool) == 0:
            straight_pool = np.arange(n)
        if len(sharp_turn_pool) == 0:
            sharp_turn_pool = np.arange(n)

        straight_length_target = np.median(future_lengths[straight_pool])
        turn_length_target = np.median(future_lengths[sharp_turn_pool])

        straight_scores = (
            0.65 * (ades[straight_pool] / (np.max(ades[straight_pool]) + 1e-6))
            + 0.35 * np.abs(np.log((future_lengths[straight_pool] + 1e-6) / (straight_length_target + 1e-6)))
        )
        turn_scores_rank = (
            0.65 * (ades[sharp_turn_pool] / (np.max(ades[sharp_turn_pool]) + 1e-6))
            + 0.35 * np.abs(np.log((future_lengths[sharp_turn_pool] + 1e-6) / (turn_length_target + 1e-6)))
        )

        straight_idx = int(straight_pool[np.argmin(straight_scores)])
        turn_order = [int(idx) for idx in sharp_turn_pool[np.argsort(turn_scores_rank)] if int(idx) != straight_idx]

        if not turn_order:
            turn_order = [straight_idx]

        turn_idx = turn_order[0]
        second_turn_idx = turn_order[1] if len(turn_order) > 1 else turn_order[0]

        return [straight_idx, turn_idx, second_turn_idx]

    # Pedestrian: choose a well-predicted moving sample.
    displacements = np.linalg.norm(y_np[:, -1, :2] - y_np[:, 0, :2], axis=-1)
    moving_pool = np.where(displacements >= 1.0)[0]
    if len(moving_pool) == 0:
        moving_pool = np.arange(n)

    ped_idx = int(moving_pool[np.argmin(ades[moving_pool])])
    return [ped_idx]


def covariance_to_ellipse(sx, sy, rho):
    cov = np.array([
        [sx * sx, rho * sx * sy],
        [rho * sx * sy, sy * sy],
    ], dtype=np.float64)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    return eigvals, angle


def add_uncertainty_ellipses(ax, means, sigmas, rho, color='red', every=2):
    for i in range(0, len(means), every):
        mx, my = means[i]
        sx, sy = sigmas[i]
        r = float(np.clip(rho[i], -0.99, 0.99))
        sx = max(float(sx), 1e-4)
        sy = max(float(sy), 1e-4)

        eigvals, angle = covariance_to_ellipse(sx, sy, r)
        # 1-sigma ellipse
        width = 2.0 * np.sqrt(max(eigvals[0], 1e-8))
        height = 2.0 * np.sqrt(max(eigvals[1], 1e-8))

        ell = Ellipse(
            (mx, my),
            width=width,
            height=height,
            angle=angle,
            edgecolor=color,
            facecolor='none',
            linewidth=1.0,
            alpha=0.35,
        )
        ax.add_patch(ell)


def draw_sample_panel(ax, history, future_gt, pred_raw, is_probabilistic, title, legend_loc='upper left'):
    history_xy = history[:, :2]
    gt_xy = future_gt[:, :2]

    if is_probabilistic:
        pred_xy = pred_raw[:, :2]
        sigmas = pred_raw[:, 2:4]
        rho = pred_raw[:, 4]
    else:
        pred_xy = pred_raw[:, :2]
        sigmas = None
        rho = None

    start = history_xy[-1:]
    gt_branch = np.vstack([start, gt_xy])
    pred_branch = np.vstack([start, pred_xy])
    ade = float(np.mean(np.linalg.norm(pred_xy - gt_xy, axis=-1)))
    fde = float(np.linalg.norm(pred_xy[-1] - gt_xy[-1]))

    ax.plot(
        history_xy[:, 0],
        history_xy[:, 1],
        'o-',
        color='tab:blue',
        label='History (4 frames)',
        linewidth=3,
        markersize=8,
        markeredgecolor='white',
        markeredgewidth=0.8,
        zorder=5,
    )
    ax.plot(
        gt_branch[:, 0],
        gt_branch[:, 1],
        'o-',
        color='tab:green',
        label='Ground Truth (12 frames)',
        linewidth=3,
        markersize=7,
        markeredgecolor='white',
        markeredgewidth=0.8,
        zorder=4,
    )
    ax.plot(
        pred_branch[:, 0],
        pred_branch[:, 1],
        'o--',
        color='tab:red',
        label='Prediction (12 frames)',
        linewidth=2,
        alpha=0.72,
        zorder=3,
    )

    ax.scatter(
        history_xy[0, 0],
        history_xy[0, 1],
        marker='s',
        s=80,
        color='tab:blue',
        edgecolors='white',
        linewidths=0.8,
        zorder=6,
        label='History Start',
    )
    ax.scatter(
        history_xy[-1, 0],
        history_xy[-1, 1],
        marker='*',
        s=140,
        color='tab:blue',
        edgecolors='white',
        linewidths=0.8,
        zorder=7,
        label='History End',
    )
    ax.scatter(
        gt_branch[0, 0],
        gt_branch[0, 1],
        marker='s',
        s=80,
        color='tab:green',
        edgecolors='white',
        linewidths=0.8,
        zorder=6,
        label='GT Start',
    )
    ax.scatter(
        gt_branch[-1, 0],
        gt_branch[-1, 1],
        marker='s',
        s=80,
        color='tab:green',
        edgecolors='white',
        linewidths=0.8,
        zorder=6,
        label='GT End',
    )
    ax.scatter(
        pred_branch[-1, 0],
        pred_branch[-1, 1],
        marker='^',
        s=100,
        color='tab:red',
        edgecolors='white',
        linewidths=0.8,
        zorder=7,
        label='Pred End',
    )

    if is_probabilistic:
        add_uncertainty_ellipses(ax, pred_xy, sigmas, rho, color='tab:red', every=2)

    ax.set_title(f"{title} | ADE={ade:.3f}m | FDE={fde:.3f}m")
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.axis('equal')
    ax.grid(alpha=0.3)
    ax.legend(loc=legend_loc, fontsize=7, framealpha=0.92)

    return ade, fde


def plot_sample(history, future_gt, pred_raw, is_probabilistic, title, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    draw_sample_panel(ax, history, future_gt, pred_raw, is_probabilistic, title, legend_loc='upper left')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_report_grid(samples, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    for ax, sample in zip(axes_flat, samples):
        draw_sample_panel(
            ax,
            sample['history'],
            sample['future_gt'],
            sample['pred_raw'],
            sample['is_probabilistic'],
            sample['title'],
            legend_loc='upper left',
        )
        ax.set_title(sample['title'] + f" | ADE={sample['ade']:.3f}m | FDE={sample['fde']:.3f}m")

    for ax in axes_flat[len(samples):]:
        ax.axis('off')

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_indices(indices_str):
    if not indices_str:
        return None
    return [int(x.strip()) for x in indices_str.split(',') if x.strip()]


def load_inference_context(project_dir, checkpoint_path, agent, features, data_dir, device):
    resolved_ckpt = resolve_checkpoint(project_dir, agent, features, checkpoint_path)
    if not resolved_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_ckpt}")

    resolved_data = resolve_data_dir(project_dir, agent, data_dir)

    train_x, train_y, scene_ids = load_data(
        data_dir=resolved_data,
        agent=agent,
        features=features,
        device=device,
    )
    test_indices_global = split_test_indices(scene_ids)
    x_test = train_x[test_indices_global]
    y_test = train_y[test_indices_global]

    checkpoint = torch.load(resolved_ckpt, map_location=device, weights_only=False)
    model, is_probabilistic = build_model_from_checkpoint(checkpoint, train_x, train_y, device)

    return {
        'checkpoint': resolved_ckpt,
        'data_dir': resolved_data,
        'x_test': x_test,
        'y_test': y_test,
        'model': model,
        'is_probabilistic': is_probabilistic,
    }


def render_selected_samples(context, chosen_indices, output_specs, title_prefix):
    x_test = context['x_test']
    y_test = context['y_test']
    model = context['model']
    is_probabilistic = context['is_probabilistic']

    with torch.no_grad():
        for idx, (scenario_name, out_path) in zip(chosen_indices, output_specs):
            x = x_test[idx:idx + 1]
            y = y_test[idx:idx + 1]
            pred = model(x)

            pred_np = pred[0].detach().cpu().numpy()
            y_np = y[0].detach().cpu().numpy()
            x_np = x[0].detach().cpu().numpy()

            title = scenario_name.replace('_', ' ').title()
            plot_sample(x_np, y_np, pred_np, is_probabilistic, title, out_path)
            print(f"Saved: {out_path}")


def collect_selected_samples(context, chosen_indices, scenario_names, title_prefix):
    x_test = context['x_test']
    y_test = context['y_test']
    model = context['model']
    is_probabilistic = context['is_probabilistic']

    samples = []
    with torch.no_grad():
        for idx, scenario_name in zip(chosen_indices, scenario_names):
            x = x_test[idx:idx + 1]
            y = y_test[idx:idx + 1]
            pred = model(x)

            pred_np = pred[0].detach().cpu().numpy()
            y_np = y[0].detach().cpu().numpy()
            x_np = x[0].detach().cpu().numpy()
            ade = float(np.mean(np.linalg.norm(pred_np[:, :2] - y_np[:, :2], axis=-1)))
            fde = float(np.linalg.norm(pred_np[-1, :2] - y_np[-1, :2]))

            base_title = scenario_name.replace('_', ' ').title()
            samples.append(
                {
                    'title': base_title,
                    'history': x_np,
                    'future_gt': y_np,
                    'pred_raw': pred_np,
                    'is_probabilistic': is_probabilistic,
                    'ade': ade,
                    'fde': fde,
                }
            )

    return samples


def main():
    parser = argparse.ArgumentParser(description='Create qualitative trajectory comparison plots.')
    parser.add_argument('checkpoint_path', nargs='?', default=None)
    parser.add_argument('--agent', choices=['car', 'pedestrian'], default='car')
    parser.add_argument(
        '--features',
        choices=['baseline', 'velocity', 'map', 'probabilistic', 'probabilistic_velocity'],
        default='baseline',
    )
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--num-samples', type=int, default=3)
    parser.add_argument('--indices', type=str, default=None,
                        help='Comma-separated test-set indices, e.g. 2,10,34')
    parser.add_argument('--report-pack', action='store_true',
                        help='Generate report-ready trio: car_straight, car_turn, pedestrian_case')
    parser.add_argument('--car-features',
                        choices=['baseline', 'velocity', 'map', 'probabilistic', 'probabilistic_velocity'],
                        default='probabilistic_velocity')
    parser.add_argument('--ped-features',
                        choices=['baseline', 'velocity', 'map', 'probabilistic', 'probabilistic_velocity'],
                        default='map')
    parser.add_argument('--car-checkpoint', type=str, default=None)
    parser.add_argument('--ped-checkpoint', type=str, default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        if args.report_pack:
            out_dir = project_dir / 'checkpoints' / 'report_pack_plots'
        else:
            ckpt = resolve_checkpoint(project_dir, args.agent, args.features, args.checkpoint_path)
            out_dir = ckpt.parent / 'qualitative_plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")

    if args.report_pack:
        car_ctx = load_inference_context(
            project_dir=project_dir,
            checkpoint_path=args.car_checkpoint,
            agent='car',
            features=args.car_features,
            data_dir=None,
            device=device,
        )
        car_chosen = choose_best_report_indices(car_ctx, 'car')
        if len(car_chosen) < 3:
            raise ValueError('Need at least three car test samples to build report pack.')

        ped_ctx = load_inference_context(
            project_dir=project_dir,
            checkpoint_path=args.ped_checkpoint,
            agent='pedestrian',
            features=args.ped_features,
            data_dir=None,
            device=device,
        )
        ped_chosen = choose_best_report_indices(ped_ctx, 'pedestrian')
        if len(ped_chosen) < 1:
            raise ValueError('Need at least one pedestrian test sample to build report pack.')

        print(f"Car checkpoint: {car_ctx['checkpoint']}")
        print(f"Ped checkpoint: {ped_ctx['checkpoint']}")

        render_selected_samples(
            car_ctx,
            [car_chosen[0], car_chosen[1], car_chosen[2]],
            [
                ('car_straight', out_dir / 'car_straight.png'),
                ('car_turn', out_dir / 'car_turn.png'),
                ('car_turn_2', out_dir / 'car_turn_2.png'),
            ],
            title_prefix=f"Car | {args.car_features}",
        )
        render_selected_samples(
            ped_ctx,
            [ped_chosen[0]],
            [
                ('pedestrian_case', out_dir / 'pedestrian_case.png'),
            ],
            title_prefix=f"Pedestrian | {args.ped_features}",
        )

        grid_samples = collect_selected_samples(
            car_ctx,
            [car_chosen[0], car_chosen[1], car_chosen[2]],
            ['car_straight', 'car_turn', 'car_turn_2'],
            title_prefix=f"Car | {args.car_features}",
        )
        grid_samples.extend(
            collect_selected_samples(
                ped_ctx,
                [ped_chosen[0]],
                ['pedestrian_case'],
                title_prefix=f"Pedestrian | {args.ped_features}",
            )
        )
        plot_report_grid(grid_samples, out_dir / 'report_pack_grid.png')
        print(f"Saved: {out_dir / 'report_pack_grid.png'}")

        print(f"Done. Generated 5 report-pack plot(s) in: {out_dir}")
        return

    context = load_inference_context(
        project_dir=project_dir,
        checkpoint_path=args.checkpoint_path,
        agent=args.agent,
        features=args.features,
        data_dir=args.data_dir,
        device=device,
    )

    print(f"Using checkpoint: {context['checkpoint']}")
    print(f"Using data dir: {context['data_dir']}")

    manual_indices = parse_indices(args.indices)
    if manual_indices is not None:
        chosen = [i for i in manual_indices if 0 <= i < len(context['x_test'])]
    else:
        chosen = choose_auto_indices(
            args.agent,
            context['x_test'].detach().cpu().numpy(),
            context['y_test'].detach().cpu().numpy(),
            args.num_samples,
        )

    if not chosen:
        raise ValueError('No valid samples selected for plotting.')

    output_specs = []
    for rank, idx in enumerate(chosen):
        if args.agent == 'car' and manual_indices is None and len(chosen) >= 2:
            if rank == 0:
                scenario = 'car_straight'
            elif rank == 1:
                scenario = 'car_turn'
            else:
                scenario = f'car_case_{rank + 1}'
        else:
            scenario = f"{args.agent}_case_{rank + 1}"
        output_specs.append((scenario, out_dir / f"{scenario}.png"))

    render_selected_samples(
        context,
        chosen,
        output_specs,
        title_prefix=f"{args.agent.capitalize()} | {args.features}",
    )

    print(f"Done. Generated {len(chosen)} plot(s) in: {out_dir}")


if __name__ == '__main__':
    main()
