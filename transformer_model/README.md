# Transformer Model Quick Commands

Use these commands from the repo root:

```bash
cd "$(git rev-parse --show-toplevel)"
```

## Python executable

```bash
PY=python
```

If your shell `python` is not your intended environment, replace it with your full interpreter path.

## 1) Preprocess data

### Car baseline data
```bash
$PY transformer_model/preprocess.py --agent car --features baseline
```

### Car velocity data
```bash
$PY transformer_model/preprocess.py --agent car --features velocity
```

### Car map data
```bash
$PY transformer_model/preprocess.py --agent car --features map
```

### Pedestrian baseline data
```bash
$PY transformer_model/preprocess.py --agent pedestrian --features baseline
```

### Pedestrian map data
```bash
$PY transformer_model/preprocess.py --agent pedestrian --features map
```

## 2) Train

### Car
```bash
$PY transformer_model/training/train.py --agent car --features baseline --seed 42
$PY transformer_model/training/train.py --agent car --features velocity --seed 42
$PY transformer_model/training/train.py --agent car --features map --seed 42
$PY transformer_model/training/train.py --agent car --features probabilistic --seed 42
$PY transformer_model/training/train.py --agent car --features probabilistic_velocity --seed 42
```

### Pedestrian
```bash
$PY transformer_model/training/train.py --agent pedestrian --features baseline --seed 42
$PY transformer_model/training/train.py --agent pedestrian --features map --seed 42
```

### Run 2 seeds in one command
```bash
$PY transformer_model/training/train.py --agent car --features velocity --seeds 42,123
```

## 3) Evaluate

### Car
```bash
$PY transformer_model/training/eval.py --agent car --features baseline
$PY transformer_model/training/eval.py --agent car --features velocity
$PY transformer_model/training/eval.py --agent car --features map
$PY transformer_model/training/eval.py --agent car --features probabilistic
$PY transformer_model/training/eval.py --agent car --features probabilistic_velocity
```

### Pedestrian
```bash
$PY transformer_model/training/eval.py --agent pedestrian --features baseline
$PY transformer_model/training/eval.py --agent pedestrian --features map
```

## 4) Compare feature variants by value

### Car
```bash
$PY transformer_model/training/compare_eval.py --agent car
```

### Pedestrian
```bash
$PY transformer_model/training/compare_eval.py --agent pedestrian
```

## 5) Qualitative plots

### Report pack (individual + 2x2 grid)
```bash
$PY transformer_model/training/plot_qualitative.py --report-pack --car-features velocity --ped-features baseline
```

Outputs are saved in:

- `transformer_model/checkpoints/report_pack_plots/`
