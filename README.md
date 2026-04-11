# Trajectory Prediction for Autonomous Driving

This repository contains three trajectory-prediction baselines for nuScenes-based motion forecasting:

- A constant-velocity baseline in [baseline/](baseline/)
- An RNN/LSTM/GRU track in [rnn-model/](rnn-model/)
- A Transformer track in [transformer_model/](transformer_model/)

The common task is to predict 12 future ego-centric trajectory points from 4 observed history frames. The preprocessing pipeline standardizes the data so the models can be compared on the same inputs and splits.

## Repository Layout
- [baseline/](baseline/) contains the constant-velocity baseline and summary scripts.
- [rnn-model/](rnn-model/) contains the RNN experiments, training scripts, saved checkpoints, and documentation.
- [transformer_model/](transformer_model/) contains the Transformer experiments and training scripts.

## Setup

The project targets Python 3.13 or newer.

1. Create and activate a virtual environment.

	```bash
	python3 -m venv .venv
	source .venv/bin/activate
	```

2. Install dependencies.

	```bash
	pip install -r requirements.txt
	```

	If you prefer `uv`, you can install from the project metadata instead:

	```bash
	uv sync
	```

3. Make sure the nuScenes data is available under `data/`.

	The repository expects the dataset layout below:

	```text
	data/
	  v1.0-trainval/
	  maps/
	```

	The preprocessed tensors used by the training code should also be present in `data/`.

4. Verify the setup.

	```bash
	python rnn-model/training/verify_setup.py
	```

## Common Commands

Run the baseline:

```bash
python baseline/scripts/baseline.py
```

Summarize baseline outputs:

```bash
python baseline/scripts/summary_baseline_results.py
```

Train an RNN experiment:

```bash
python rnn-model/training/train.py
```

Evaluate or visualize an RNN checkpoint:

```bash
python rnn-model/training/eval.py
python rnn-model/training/visualize_predictions.py
```

Train a Transformer experiment:

```bash
python transformer_model/training/train.py
```

Run the Transformer demo inference script:

```bash
python transformer_model/training/example_inference.py
```