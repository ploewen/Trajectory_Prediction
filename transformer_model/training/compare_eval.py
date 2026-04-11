"""
Compare multiple trajectory model variants by evaluation values.

This script can:
1) Read saved metrics from checkpoints/*/evaluation_metrics.json, or
2) Run evaluation live when metrics are missing or --force-run is used.

Outputs:
- Console table sorted by ADE (lower is better)
- CSV summary under checkpoints/comparisons/
"""

import argparse
import csv
import json
from pathlib import Path

from eval import evaluate_model


DEFAULT_FEATURES = [
    'baseline',
    'velocity',
    'map',
    'probabilistic',
    'probabilistic_velocity',
]


def resolve_checkpoint(project_dir, agent, feature, checkpoint_override=None):
    if checkpoint_override:
        return Path(checkpoint_override)
    return project_dir / 'checkpoints' / f"{agent}_{feature}" / 'best_model.pt'


def read_saved_metrics(metrics_path):
    if not metrics_path.exists():
        return None
    with open(metrics_path, 'r') as f:
        return json.load(f)


def evaluate_feature(project_dir, agent, feature, batch_size, force_run=False):
    ckpt_path = resolve_checkpoint(project_dir, agent, feature)
    if not ckpt_path.exists():
        return {
            'feature': feature,
            'status': f'missing checkpoint: {ckpt_path}',
        }

    metrics_path = ckpt_path.parent / 'evaluation_metrics.json'
    metrics = None

    if not force_run:
        metrics = read_saved_metrics(metrics_path)

    if metrics is None:
        try:
            metrics, _, _ = evaluate_model(
                checkpoint_path=ckpt_path,
                batch_size=batch_size,
                data_dir=None,
                agent=agent,
                features=feature,
            )
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            return {
                'feature': feature,
                'status': f'eval failed: {e}',
            }

    miss_rates = metrics.get('miss_rates', {})
    return {
        'feature': feature,
        'status': 'ok',
        'checkpoint': str(ckpt_path),
        'num_test_samples': metrics.get('num_test_samples', -1),
        'rmse': float(metrics.get('rmse', float('nan'))),
        'ade': float(metrics.get('ade', float('nan'))),
        'fde': float(metrics.get('fde', float('nan'))),
        'mr_0_5m': float(miss_rates.get('0.5m', float('nan'))),
        'mr_1_0m': float(miss_rates.get('1.0m', float('nan'))),
        'mr_2_0m': float(miss_rates.get('2.0m', float('nan'))),
    }


def print_table(rows):
    ok_rows = [r for r in rows if r.get('status') == 'ok']
    bad_rows = [r for r in rows if r.get('status') != 'ok']

    if ok_rows:
        ok_rows = sorted(ok_rows, key=lambda r: r['ade'])
        baseline_row = next((r for r in ok_rows if r['feature'] == 'baseline'), None)

        print("\nModel comparison (sorted by ADE):")
        print("feature,ade,fde,rmse,mr_2.0m,delta_ade_vs_baseline")
        for r in ok_rows:
            if baseline_row is None:
                delta = float('nan')
            else:
                delta = r['ade'] - baseline_row['ade']
            print(
                f"{r['feature']},{r['ade']:.4f},{r['fde']:.4f},{r['rmse']:.4f},"
                f"{r['mr_2_0m']:.4f},{delta:.4f}"
            )

    if bad_rows:
        print("\nSkipped / failed:")
        for r in bad_rows:
            print(f"- {r['feature']}: {r['status']}")


def save_csv(project_dir, agent, rows):
    out_dir = project_dir / 'checkpoints' / 'comparisons'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{agent}_feature_comparison.csv"

    fieldnames = [
        'feature',
        'status',
        'checkpoint',
        'num_test_samples',
        'rmse',
        'ade',
        'fde',
        'mr_0_5m',
        'mr_1_0m',
        'mr_2_0m',
    ]

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in fieldnames})

    return out_csv


def parse_features(s):
    if not s:
        return DEFAULT_FEATURES
    return [x.strip() for x in s.split(',') if x.strip()]


def main():
    parser = argparse.ArgumentParser(description='Compare feature modes by evaluation values.')
    parser.add_argument('--agent', choices=['car', 'pedestrian'], default='car')
    parser.add_argument(
        '--features',
        type=str,
        default=','.join(DEFAULT_FEATURES),
        help='Comma-separated feature list, e.g. baseline,velocity,map',
    )
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--force-run', action='store_true', help='Re-run evaluation even if metrics JSON exists')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    features = parse_features(args.features)

    rows = []
    for feature in features:
        rows.append(
            evaluate_feature(
                project_dir=project_dir,
                agent=args.agent,
                feature=feature,
                batch_size=args.batch_size,
                force_run=args.force_run,
            )
        )

    print_table(rows)
    out_csv = save_csv(project_dir, args.agent, rows)
    print(f"\nSaved CSV: {out_csv}")


if __name__ == '__main__':
    main()
