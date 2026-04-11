# Extended Documentation

This folder contains supporting documentation for the trajectory prediction project.

## What the code does

The repository trains and evaluates trajectory forecasting models on preprocessed nuScenes data. The default pipeline predicts 12 future frames from 4 observed history frames, with optional stage-specific features for velocity, map context, and agent type filtering.

## Experiment stages

The training script defines the following stages:

- `cars_baseline`: vehicle trajectories with position only
- `cars_velocity`: vehicle trajectories with position plus velocity
- `cars_map`: vehicle trajectories with selected map features
- `pedestrian_baseline`: pedestrian trajectories with position only
- `pedestrian_map`: pedestrian trajectories with selected map features
- `cars_probabilistic`: vehicle trajectories with probabilistic Gaussian outputs

## Data flow

1. `training/dataset.py` loads `train_x.pt`, `train_y.pt`, and `scene_ids.pt`.
2. Optional `extra_features_*.pt` tensors are concatenated when present.
3. `training/train.py` filters by agent type, appends velocity if required, and selects map features for the chosen stage.
4. `training/train.py` saves checkpoints and metadata into `experiments/<stage>/`.
5. `training/eval.py`, `training/benchmark_all_models.py`, and `training/visualize_predictions.py` reuse the saved checkpoints for evaluation and plotting.

## Demo script

`training/example_inference.py` loads a saved checkpoint, prints a sample history/target pair, runs inference, and saves a comparison plot so you can inspect input-output behavior without retraining.

## Troubleshooting

- If `nuscenes-devkit` fails to import, reinstall the dependencies in a fresh virtual environment.
- If a stage cannot find map features, verify that `data/extra_features_map.pt` exists and matches the sample count in `data/train_x.pt`.
- If `training/verify_setup.py` reports missing files, confirm that the preprocessed tensors are present in `data/` and that the checkpoints exist under `experiments/`.