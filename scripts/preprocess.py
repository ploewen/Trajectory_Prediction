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

VERSION = "v1.0-trainval"
OUTPUT_DIR = PROJECT_ROOT / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 2
STEP_DT = 1.0 / SAMPLE_RATE
HISTORY_FRAMES = 4
FUTURE_FRAMES = 12

# scene-based split: 700 train / 150 val
TRAIN_SCENE_CUTOFF = 700


def quaternion_to_rotation_matrix(q):
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


def coarse_category(category_name: str):
    """
    Keep only:
      - vehicle.*   -> car
      - human.*     -> pedestrian
    Exclude:
      - movable_object.*
      - static_object.*
      - animal.*
    """
    if category_name.startswith("vehicle"):
        return "car"
    if category_name.startswith("human"):
        return "pedestrian"
    return None


def build_scene_instances(nusc, scene):
    scene_instances = defaultdict(list)

    current_sample_token = scene["first_sample_token"]
    while current_sample_token:
        sample = nusc.get("sample", current_sample_token)

        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            category_name = ann["category_name"]

            coarse = coarse_category(category_name)
            if coarse is None:
                continue

            instance_token = ann["instance_token"]
            scene_instances[instance_token].append(
                (current_sample_token, ann_token, coarse)
            )

        current_sample_token = sample["next"]

    return scene_instances


def ann_xy(nusc, ann_token):
    ann = nusc.get("sample_annotation", ann_token)
    return np.array(ann["translation"][:2], dtype=float)


def build_one_sample(nusc, records, anchor_idx):
    """
    records: chronological list of (sample_token, ann_token, coarse_category)
    """
    anchor_sample_token, anchor_ann_token, coarse = records[anchor_idx]

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

    history_xy = np.asarray(history_xy, dtype=np.float32)
    future_xy = np.asarray(future_xy, dtype=np.float32)

    return history_xy, future_xy, coarse


def process_scenes():
    print(f"[INFO] Loading nuScenes {VERSION} from: {DATAROOT}")
    nusc = NuScenes(version=VERSION, dataroot=str(DATAROOT), verbose=False)

    train_histories, train_futures = [], []
    train_categories, train_scene_ids = [], []

    test_histories, test_futures = [], []
    test_categories, test_scene_ids = [], []

    total_scenes = len(nusc.scene)
    print(f"[INFO] Processing {total_scenes} scenes...\n")

    for scene_idx, scene in enumerate(nusc.scene):
        if scene_idx == 0 or (scene_idx + 1) % 10 == 0:
            print(f"[PROGRESS] Scene {scene_idx + 1}/{total_scenes}")

        scene_instances = build_scene_instances(nusc, scene)
        is_train_scene = scene_idx < TRAIN_SCENE_CUTOFF

        for _, records in scene_instances.items():
            for anchor_idx in range(len(records)):
                sample_out = build_one_sample(nusc, records, anchor_idx)
                if sample_out is None:
                    continue

                history_xy, future_xy, coarse = sample_out

                if is_train_scene:
                    train_histories.append(history_xy)
                    train_futures.append(future_xy)
                    train_categories.append(coarse)
                    train_scene_ids.append(scene_idx)
                else:
                    test_histories.append(history_xy)
                    test_futures.append(future_xy)
                    test_categories.append(coarse)
                    test_scene_ids.append(scene_idx)

    if len(train_histories) == 0 or len(test_histories) == 0:
        raise RuntimeError("No valid samples extracted. Check dataroot/version/category filter.")

    train_x = torch.tensor(np.asarray(train_histories), dtype=torch.float32)
    train_y = torch.tensor(np.asarray(train_futures), dtype=torch.float32)
    train_scene_ids = torch.tensor(np.asarray(train_scene_ids), dtype=torch.long)

    test_x = torch.tensor(np.asarray(test_histories), dtype=torch.float32)
    test_y = torch.tensor(np.asarray(test_futures), dtype=torch.float32)
    test_scene_ids = torch.tensor(np.asarray(test_scene_ids), dtype=torch.long)

    print(f"[INFO] train_x shape: {tuple(train_x.shape)}")
    print(f"[INFO] train_y shape: {tuple(train_y.shape)}")
    print(f"[INFO] test_x shape:  {tuple(test_x.shape)}")
    print(f"[INFO] test_y shape:  {tuple(test_y.shape)}")

    torch.save(train_x, OUTPUT_DIR / "train_x.pt")
    torch.save(train_y, OUTPUT_DIR / "train_y.pt")
    torch.save(train_scene_ids, OUTPUT_DIR / "train_scene_ids.pt")

    torch.save(test_x, OUTPUT_DIR / "test_x.pt")
    torch.save(test_y, OUTPUT_DIR / "test_y.pt")
    torch.save(test_scene_ids, OUTPUT_DIR / "test_scene_ids.pt")

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
        "train_scene_cutoff": TRAIN_SCENE_CUTOFF,
        "kept_categories": ["vehicle -> car", "human -> pedestrian"],
        "excluded_categories": ["movable_object", "static_object", "animal"],
        "train_size": int(train_x.shape[0]),
        "test_size": int(test_x.shape[0]),
    }

    with open(OUTPUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    def count_categories(name_list):
        out = {}
        for c in name_list:
            out[c] = out.get(c, 0) + 1
        return out

    print("[INFO] Train category counts:", count_categories(train_categories))
    print("[INFO] Test category counts: ", count_categories(test_categories))
    print(f"\n[SUCCESS] Saved processed tensors to: {OUTPUT_DIR}")


if __name__ == "__main__":
    process_scenes()