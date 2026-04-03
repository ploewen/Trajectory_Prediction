"""
preprocess_v2.py

Updated preprocessing pipeline for the NuScenes trajectory prediction project.

Changes compared to preprocess_v1:
--------------------------------------------------
1. Unified preprocessing pipeline
   - This script follows the shared preprocessing format used by the team,
     ensuring all members generate identical input tensors.

2. Dataset compatibility
   - Designed for NuScenes v1.0 trainval dataset.
   - Also compatible with v1.0-mini for local testing.

3. Category filtering
   - Static objects and animals are excluded from the dataset.
   - Only the following categories are kept for trajectory prediction:
        - vehicle
        - # human 
        - # movable_object

4. Output format
   - Generates tensors for trajectory prediction:
        train_x : (N, 4, 2)  -> observed trajectory (past positions)
        train_y : (N, 12, 2) -> future trajectory (prediction target)

5. Saved outputs
   - Processed tensors are stored in the `processed/` directory.

Purpose
--------------------------------------------------
This preprocessing pipeline ensures consistent training and evaluation
inputs for baseline models and future deep learning models (e.g., Transformer).

Author: Yongxin Guan
Branch: yongxin-baseline
"""

from pathlib import Path
from collections import defaultdict
import warnings
import json

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes

warnings.filterwarnings("ignore")


# =========================
# Config
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATAROOT = PROJECT_ROOT / "data"

# dataset: mini / trainval
# VERSION = "v1.0-mini"
VERSION = "v1.0-trainval"

OUTPUT_DIR = PROJECT_ROOT / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 2           # 2Hz => every 0.5 sec
STEP_DT = 1.0 / SAMPLE_RATE
HISTORY_FRAMES = 4        # past 2 sec
FUTURE_FRAMES = 12        # future 6 sec
TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def quaternion_to_rotation_matrix(q):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),   1 - 2 * (x**2 + y**2)],
    ], dtype=float)


def get_ego_pose_at_sample(nusc, sample_token):
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)
    ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])

    translation = np.array(ego_pose["translation"][:2], dtype=float)
    rotation = quaternion_to_rotation_matrix(ego_pose["rotation"])[:2, :2]
    return translation, rotation


def transform_to_ego_frame(global_xy, ego_translation, ego_rotation):
    relative = global_xy - ego_translation
    return ego_rotation.T @ relative


def valid_category(category_name: str) -> bool:
    """
    Keep:
      - vehicle.*
      - human.*
      - movable_object.*
    Exclude:
      - static_object.*
      - animal.*
    """
    if category_name.startswith("static_object"):
        return False
    if category_name.startswith("animal"):
        return False
    if category_name.startswith("human"):
        return False
    if category_name.startswith("movable_object"):
        return False
    return (
        category_name.startswith("vehicle") 
        # or
        # category_name.startswith("human") or
        # category_name.startswith("movable_object")
    )


def build_scene_instances(nusc, scene):
    """
    For each instance in a scene, collect a chronological list of
    (sample_token, ann_token, category_name).
    """
    scene_instances = defaultdict(list)

    current_sample_token = scene["first_sample_token"]
    while current_sample_token:
        sample = nusc.get("sample", current_sample_token)

        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            category_name = ann["category_name"]

            if not valid_category(category_name):
                continue

            instance_token = ann["instance_token"]
            scene_instances[instance_token].append(
                (current_sample_token, ann_token, category_name)
            )

        current_sample_token = sample["next"]

    return scene_instances


def ann_xy(nusc, ann_token):
    ann = nusc.get("sample_annotation", ann_token)
    return np.array(ann["translation"][:2], dtype=float)


def build_one_sample(nusc, records, anchor_idx):
    """
    records: chronological list of (sample_token, ann_token, category_name)
    anchor_idx: current/present time index

    history: [anchor_idx - HISTORY_FRAMES, ..., anchor_idx - 1]  -> 4 frames
    future : [anchor_idx + 1, ..., anchor_idx + FUTURE_FRAMES]   -> 12 frames
    both transformed into the anchor ego frame
    """
    anchor_sample_token, anchor_ann_token, category_name = records[anchor_idx]

    if anchor_idx < HISTORY_FRAMES:
        return None
    if anchor_idx + FUTURE_FRAMES >= len(records):
        return None

    hist_records = records[anchor_idx - HISTORY_FRAMES: anchor_idx]
    fut_records = records[anchor_idx + 1: anchor_idx + 1 + FUTURE_FRAMES]

    if len(hist_records) != HISTORY_FRAMES or len(fut_records) != FUTURE_FRAMES:
        return None

    ego_translation, ego_rotation = get_ego_pose_at_sample(nusc, anchor_sample_token)

    history_xy = []
    future_xy = []

    for _, ann_token, _ in hist_records:
        global_xy = ann_xy(nusc, ann_token)
        history_xy.append(transform_to_ego_frame(global_xy, ego_translation, ego_rotation))

    for _, ann_token, _ in fut_records:
        global_xy = ann_xy(nusc, ann_token)
        future_xy.append(transform_to_ego_frame(global_xy, ego_translation, ego_rotation))

    history_xy = np.asarray(history_xy, dtype=np.float32)   # (4, 2)
    future_xy = np.asarray(future_xy, dtype=np.float32)     # (12, 2)

    return history_xy, future_xy, category_name


def process_scenes():
    print(f"[INFO] Loading nuScenes {VERSION} from: {DATAROOT}")
    nusc = NuScenes(version=VERSION, dataroot=str(DATAROOT), verbose=False)

    all_histories = []
    all_futures = []
    all_categories = []

    total_scenes = len(nusc.scene)
    print(f"[INFO] Processing {total_scenes} scenes...\n")

    for scene_idx, scene in enumerate(nusc.scene):
        if scene_idx == 0 or (scene_idx + 1) % 10 == 0:
            print(f"[PROGRESS] Scene {scene_idx + 1}/{total_scenes}")

        scene_instances = build_scene_instances(nusc, scene)

        for _, records in scene_instances.items():
            for anchor_idx in range(len(records)):
                sample_out = build_one_sample(nusc, records, anchor_idx)
                if sample_out is None:
                    continue

                history_xy, future_xy, category_name = sample_out
                all_histories.append(history_xy)
                all_futures.append(future_xy)
                all_categories.append(category_name)

    n = len(all_histories)
    print(f"\n[INFO] Extracted {n} valid trajectory samples.")

    if n == 0:
        raise RuntimeError("No valid samples extracted. Check dataroot/version/category filter.")

    x = torch.tensor(np.asarray(all_histories), dtype=torch.float32)  # (N, 4, 2)
    y = torch.tensor(np.asarray(all_futures), dtype=torch.float32)    # (N, 12, 2)

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    indices = torch.randperm(n, generator=generator)

    split = int(TRAIN_RATIO * n)
    train_idx = indices[:split]
    test_idx = indices[split:]

    train_x = x[train_idx]
    train_y = y[train_idx]
    test_x = x[test_idx]
    test_y = y[test_idx]

    train_categories = [all_categories[i] for i in train_idx.tolist()]
    test_categories = [all_categories[i] for i in test_idx.tolist()]

    print(f"[INFO] train_x shape: {tuple(train_x.shape)}")
    print(f"[INFO] train_y shape: {tuple(train_y.shape)}")
    print(f"[INFO] test_x shape:  {tuple(test_x.shape)}")
    print(f"[INFO] test_y shape:  {tuple(test_y.shape)}")

    torch.save(train_x, OUTPUT_DIR / "train_x.pt")
    torch.save(train_y, OUTPUT_DIR / "train_y.pt")
    torch.save(test_x, OUTPUT_DIR / "test_x.pt")
    torch.save(test_y, OUTPUT_DIR / "test_y.pt")

    with open(OUTPUT_DIR / "train_categories.json", "w") as f:
        json.dump(train_categories, f, indent=2)

    with open(OUTPUT_DIR / "test_categories.json", "w") as f:
        json.dump(test_categories, f, indent=2)

    meta = {
        "version": VERSION,
        "dataroot": str(DATAROOT),
        "sample_rate_hz": SAMPLE_RATE,
        "step_dt": STEP_DT,
        "history_frames": HISTORY_FRAMES,
        "future_frames": FUTURE_FRAMES,
        "train_ratio": TRAIN_RATIO,
        "random_seed": RANDOM_SEED,
        "kept_prefixes": ["vehicle", "human", "movable_object"],
        "excluded_prefixes": ["static_object", "animal"],
        "train_size": int(train_x.shape[0]),
        "test_size": int(test_x.shape[0]),
    }

    with open(OUTPUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[SUCCESS] Saved processed tensors to: {OUTPUT_DIR}")


if __name__ == "__main__":
    process_scenes()