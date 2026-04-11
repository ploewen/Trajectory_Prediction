# Trajectory Prediction

Trajectory prediction experiments for nuScenes-based agent motion forecasting. The repository contains the preprocessed dataset, model code, training and evaluation scripts, benchmark outputs, and visualization utilities for deterministic and probabilistic LSTM/GRU models.

## Setup

The project targets Python 3.9 or newer.

1. Create and activate a virtual environment.

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the required packages.

   ```bash
   pip install -r requirements.txt
   ```

3. Place the downloaded nuScenes data under `data/`.

   The repository expects the nuScenes root directory to be the local `data/` folder. After downloading the nuScenes trainval tables and map expansion assets, the layout should look like this:

   ```text
   data/
     v1.0-trainval/
     maps/
   ```

   In other words, `v1.0-trainval` and `maps` should sit directly inside `data/`, so the code can resolve `NuScenes(version="v1.0-trainval", dataroot="data", ...)` without any extra path changes.

4. Verify the workspace.

   ```bash
   python training/verify_setup.py
   ```

## Repository Layout

- `data/` contains the preprocessed tensors and nuScenes metadata used by the training scripts.
- `training/` contains model definitions, dataloaders, training, evaluation, benchmarking, and demo scripts.
- `utils/` contains data inspection, preprocessing, and visualization helpers.
- `experiments/` stores trained checkpoints, training histories, benchmark outputs, and generated plots.
- `doc/` contains extended documentation.

## Data Format

The core dataset uses 4 history frames and 12 future frames:

- `data/train_x.pt`: input trajectories with shape `(N, 4, F)`.
- `data/train_y.pt`: target trajectories with shape `(N, 12, O)`.
- `data/scene_ids.pt`: scene index for each trajectory, used for the train/test split.
- Optional feature tensors named `data/extra_features_*.pt` are auto-detected and concatenated by `training/dataset.py`.

The preprocessing and training code also uses:

- `data/agent_types.pt`
- `data/agent_type_metadata.pt`
- `data/extra_features_map.pt`
- `data/extra_features_categorical.pt`

## Common Commands

Train all configured stages:

```bash
python training/train.py --run-stages
```

Train one stage:

```bash
python training/train.py --stage cars_velocity --epochs 25 --batch-size 32
```

Evaluate a checkpoint:

```bash
python training/eval.py experiments/cars_velocity/best_model.pt
```

Benchmark all saved experiments:

```bash
python training/benchmark_all_models.py
```

Visualize predictions for a saved stage:

```bash
python training/visualize_predictions.py cars_velocity --checkpoint experiments/cars_velocity/best_model.pt
```

Run the sample input-output demo:

```bash
python training/example_inference.py --checkpoint experiments/cars_velocity/best_model.pt
```

## Outputs

Training and benchmarking scripts write their artifacts into `experiments/<stage>/`, including:

- `best_model.pt`
- `final_model.pt`
- `training_history.json`
- `metadata.json`
- `benchmark_results.csv` and `benchmark_results.json`

The visualization scripts also save prediction plots alongside the checkpoint or experiment folder.

## Extended Documentation

See [doc/README.md](doc/README.md) for more detail on the data pipeline, experiment stages, and troubleshooting.