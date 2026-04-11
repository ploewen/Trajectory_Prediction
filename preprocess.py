"""
NuScenes Data Preprocessing for Trajectory Prediction
Extracts 2-second history and 6-second future for all vehicles
Applies ego-centric transformation (rotation + translation)
Saves as PyTorch tensors: train_x.pt (history) and train_y.pt (future)
"""

import torch
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from collections import defaultdict
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Configuration (relative to script location)
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent  # Go up from /scripts to root
DATAROOT = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "data"
VERSION = "v1.0-trainval"
SAMPLE_RATE = 2  # Hz (samples every 0.5 sec)
HISTORY_FRAMES = 4  # 2 seconds at 2Hz
FUTURE_FRAMES = 12  # 6 seconds at 2Hz

print(f"[INFO] Project directory: {PROJECT_DIR}")
print(f"[INFO] Data root: {DATAROOT}")
print(f"[INFO] Output dir: {OUTPUT_DIR}")

print("[INFO] Loading nuScenes dataset...")
nusc = NuScenes(version=VERSION, dataroot=str(DATAROOT), verbose=False)


def quaternion_to_rotation_matrix(q):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q

    rotation_matrix = np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ]
    )

    return rotation_matrix


def get_ego_pose_at_sample(sample_token):
    """Get ego vehicle pose (translation and rotation) at a given sample."""
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)
    ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])

    translation = np.array(ego_pose["translation"])[:2]  # x, y only
    rotation = quaternion_to_rotation_matrix(ego_pose["rotation"])[
        :2, :2
    ]  # 2x2 rotation

    return translation, rotation


def transform_to_ego_frame(position, ego_translation, ego_rotation):
    """
    Transform global position to ego-centric frame:
    1. Subtract ego position (center on ego vehicle)
    2. Rotate so ego vehicle faces "north" (up)
    """
    # Relative to ego
    relative_pos = position - ego_translation

    # Rotate to ego frame (invert rotation matrix)
    ego_frame_pos = ego_rotation.T @ relative_pos

    return ego_frame_pos


def extract_trajectory_for_instance(
    start_ann_token,
    anchor_sample_token,
    nusc_obj,
    num_frames=HISTORY_FRAMES + FUTURE_FRAMES,
):
    """
    Extract trajectory using the fast 'next' token link.
    Anchors the entire trajectory to ONE fixed ego-pose (the present moment).
    This fixes the 'Moving Anchor' bug and maxes out speed.
    """
    trajectory = []
    current_ann_token = start_ann_token

    # 1. Get the anchor pose ONCE (This is the "present" moment)
    anchor_translation, anchor_rotation = get_ego_pose_at_sample(anchor_sample_token)

    for _ in range(num_frames):
        if not current_ann_token:
            return None  # Trajectory ended early

        # 2. Grab the annotation directly (No searching!)
        ann = nusc_obj.get("sample_annotation", current_ann_token)

        # 3. Extract global position
        global_pos = np.array(ann["translation"])[:2]  # x, y only

        # 4. Transform to ego frame using the FIXED anchor
        ego_pos = transform_to_ego_frame(
            global_pos, anchor_translation, anchor_rotation
        )
        trajectory.append(ego_pos)

        # 5. Hop to the next annotation instantly (no searching!)
        current_ann_token = ann["next"]

    return np.array(trajectory)  # Shape: (num_frames, 2)


def valid_category(category_name: str) -> bool:
    """
    Keep:
      - vehicle.*
      - human.*
    Exclude:
      - static_object.*
      - movable_object.*
      - animal.*
    """
    if (
        category_name.startswith("static_object")
        or category_name.startswith("animal")
        or category_name.startswith("movable_object")
    ):
        return False

    return category_name.startswith("vehicle") or category_name.startswith("human")


def get_agent_type_category(category_name: str) -> str:
    """
    Map nuScenes category to broad agent type.

    Examples:
      vehicle.car -> vehicle
      vehicle.truck -> vehicle
      human.pedestrian.adult -> pedestrian
      human.pedestrian.child -> pedestrian
      human.cyclist -> pedestrian

    Returns:
        One of: 'vehicle', 'pedestrian'
    """
    if category_name.startswith("vehicle"):
        return "vehicle"
    elif category_name.startswith("human"):
        return "pedestrian"
    else:
        return "unknown"


def extract_map_features(
    ego_pos, ego_rotation, nusc_obj, sample_token=None, map_radius=30.0
):
    """
    Extract map-based features around the ego vehicle.

    Returns:
        dict with keys:
        - 'drivable_area': 1.0 if position is on drivable area, 0.0 otherwise
        - 'lane_type_road': 1.0 if on road lane
        - 'lane_type_parking': 1.0 if on parking lane
        - 'lane_type_shoulder': 1.0 if on shoulder
        - 'speed_limit': normalized speed limit [0, 1] (0-130 km/h range)
        - 'is_intersection': 1.0 if in intersection, 0.0 otherwise
        - 'has_stop_line': 1.0 if stop line detected, 0.0 otherwise
        - 'has_crosswalk': 1.0 if crosswalk nearby, 0.0 otherwise
    """
    try:
        # Get map for this location
        if sample_token:
            sample = nusc_obj.get("sample", sample_token)
            lidar_token = sample["data"]["LIDAR_TOP"]
            lidar_data = nusc_obj.get("sample_data", lidar_token)
            log_token = lidar_data["log_token"]
            log = nusc_obj.get("log", log_token)
            map_name = log["location"]
        else:
            map_name = nusc_obj.get(
                "log", nusc_obj.get("scene", nusc_obj.scene[0])["log_token"]
            )["location"]
    except:
        # Fallback: return neutral features
        return _get_default_map_features()

    try:
        from nuscenes.map_expansion.map_api import NuScenesMap

        nusc_map = NuScenesMap(dataroot=str(DATAROOT), map_name=map_name)

        # Check if ego position is on drivable area
        x, y = ego_pos[0], ego_pos[1]
        drivable_area = 1.0 if nusc_map.is_in_drivable_area([x, y]) else 0.0

        # Get lane type at ego position
        lanes = nusc_map.get_lanes_at([x, y])
        lane_type_road = 0.0
        lane_type_parking = 0.0
        lane_type_shoulder = 0.0
        speed_limit_norm = 0.33

        if lanes:
            primary_lane = nusc_map.get_lane(lanes[0])
            lane_type = primary_lane.get("lane_type", "unknown").lower()
            if "parking" in lane_type:
                lane_type_parking = 1.0
            elif "shoulder" in lane_type or "sidewalk" in lane_type:
                lane_type_shoulder = 1.0
            else:
                lane_type_road = 1.0

            # Extract speed limit (normalize to [0, 1], assuming range 0-130 km/h)
            speed_limit_kmh = float(primary_lane.get("speed_limit_kmh", 50.0))
            speed_limit_norm = np.clip(speed_limit_kmh / 130.0, 0.0, 1.0)
        else:
            # Default to equal probability if not on lane
            lane_type_road = 0.33
            lane_type_parking = 0.33
            lane_type_shoulder = 0.34

        # Check if in intersection
        is_intersection = 1.0 if nusc_map.is_in_intersection_polygon([x, y]) else 0.0

        # Check for stop line (typically at intersections)
        has_stop_line = 0.0
        if is_intersection == 1.0:
            # At intersections, assume stop line presence with higher probability
            has_stop_line = 0.7
        else:
            has_stop_line = 0.0

        # Check for crosswalk/pedestrian crossing
        has_crosswalk = 0.0
        try:
            crosswalks = nusc_map.get_crosswalk_tokens_at([x, y])
            if crosswalks:
                has_crosswalk = 1.0
        except:
            pass

        return {
            "drivable_area": drivable_area,
            "lane_type_road": lane_type_road,
            "lane_type_parking": lane_type_parking,
            "lane_type_shoulder": lane_type_shoulder,
            "speed_limit": speed_limit_norm,
            "is_intersection": is_intersection,
            "has_stop_line": has_stop_line,
            "has_crosswalk": has_crosswalk,
        }
    except Exception as e:
        # Fallback for any errors
        return _get_default_map_features()


def _get_default_map_features():
    """Return default map features when extraction fails."""
    return {
        "drivable_area": 0.5,
        "lane_type_road": 0.33,
        "lane_type_parking": 0.33,
        "lane_type_shoulder": 0.34,
        "speed_limit": 0.4,
        "is_intersection": 0.0,
        "has_stop_line": 0.1,
        "has_crosswalk": 0.1,
    }


def extract_social_features(
    ego_pos,
    ego_rotation,
    sample_token,
    nusc_obj,
    num_neighbors=5,
    radius=30.0,
):
    """
    Extract social features: nearby agents and their properties.

    Returns:
        Array of shape (num_neighbors, 4) containing for each neighbor:
        - distance: L2 distance to neighbor
        - angle: angle from ego to neighbor (in radians)
        - type_vehicle: 1.0 if neighbor is vehicle, 0.0 otherwise
        - type_pedestrian: 1.0 if neighbor is pedestrian, 0.0 otherwise

        If fewer neighbors exist, pad with zeros.
    """
    sample = nusc_obj.get("sample", sample_token)

    neighbors_data = []

    # Find all annotations in this sample
    for ann_token in sample["anns"]:
        ann = nusc_obj.get("sample_annotation", ann_token)

        # Get instance and category
        instance = nusc_obj.get("instance", ann["instance_token"])
        category = nusc_obj.get("category", instance["category_token"])
        category_name = category["name"]

        # Skip ego vehicle and invalid categories
        if not valid_category(category_name):
            continue

        # Get neighbor position in global frame
        neighbor_global_pos = np.array(ann["translation"])[:2]

        # Transform to ego frame
        relative_pos = neighbor_global_pos - ego_pos
        neighbor_ego_pos = ego_rotation.T @ relative_pos

        # Calculate distance and angle
        distance = np.linalg.norm(neighbor_ego_pos)

        # Skip neighbors outside radius
        if (
            distance > radius or distance < 0.5
        ):  # Skip if too far or too close (ego itself)
            continue

        # Calculate angle (atan2 gives angle in ego frame where y is forward)
        angle = np.arctan2(
            neighbor_ego_pos[0], neighbor_ego_pos[1]
        )  # x=lateral, y=forward

        # Normalize distance to [0, 1]
        norm_distance = distance / radius

        # Determine agent type
        agent_type = get_agent_type_category(category_name)
        is_vehicle = 1.0 if agent_type == "vehicle" else 0.0
        is_pedestrian = 1.0 if agent_type == "pedestrian" else 0.0

        neighbors_data.append([norm_distance, angle, is_vehicle, is_pedestrian])

    # Sort by distance (nearest first)
    if neighbors_data:
        neighbors_data = sorted(neighbors_data, key=lambda x: x[0])

    # Pad or truncate to num_neighbors
    result = np.zeros((num_neighbors, 4), dtype=np.float32)
    for i, neighbor in enumerate(neighbors_data[:num_neighbors]):
        result[i] = neighbor

    return result


def process_scenes():
    """
    Main processing loop optimized for speed and fixed-anchor transformations.
    - Stores both sample_token AND ann_token to avoid searching
    - Uses fixed anchor pose for entire 8-second sequence (fixes physics bug)
    - Uses 'next' token to hop through data instantly (kills CPU bottleneck)
    - Tracks scene ID for each trajectory (no train/test overlap)
    - Extracts agent type (category) for each trajectory
    - Extracts map features (lane type, drivable area) at anchor frame
    - Extracts social features (nearby agents) at anchor frame
    """
    all_histories = []
    all_futures = []
    all_scene_ids = []  # Track which scene each trajectory came from
    all_agent_types = []  # Track agent type (vehicle or pedestrian)
    all_category_names = []  # Track full nuScenes category name
    all_map_features = []  # Track map features (drivable area, lane type)
    all_social_features = []  # Track social features (nearby agents)

    total_scenes = len(nusc.scene)
    print(f"[INFO] Processing {total_scenes} scenes...\n")

    for scene_idx, scene in enumerate(nusc.scene):
        if (scene_idx + 1) % 10 == 0 or scene_idx == 0:
            print(f"[PROGRESS] Scene {scene_idx + 1}/{total_scenes}")

        # Store tuples of (sample_token, ann_token) so we don't search later
        scene_instances = defaultdict(list)
        instance_categories = {}  # Map instance_token -> category_name

        current_sample_token = scene["first_sample_token"]
        while current_sample_token:
            sample = nusc.get("sample", current_sample_token)

            for ann_token in sample["anns"]:
                ann = nusc.get("sample_annotation", ann_token)
                instance_token = ann["instance_token"]

                # Get category info
                instance = nusc.get("instance", instance_token)
                category = nusc.get("category", instance["category_token"])
                category_name = category["name"]

                # Check if valid agent type
                if not valid_category(category_name):
                    continue

                # Store category for later (only store once per instance)
                if instance_token not in instance_categories:
                    instance_categories[instance_token] = category_name

                # Save BOTH tokens so we don't have to search later
                scene_instances[instance_token].append(
                    (current_sample_token, ann_token)
                )

            current_sample_token = sample["next"]

        # Extract trajectories
        for instance_token, sample_records in scene_instances.items():
            category_name = instance_categories.get(instance_token, "unknown")
            agent_type = get_agent_type_category(category_name)

            for i, (sample_token, ann_token) in enumerate(sample_records):
                if i < HISTORY_FRAMES or i >= len(sample_records) - FUTURE_FRAMES:
                    continue  # Not enough history or future

                # The start annotation is HISTORY_FRAMES steps back
                start_sample_token, start_ann_token = sample_records[i - HISTORY_FRAMES]

                # The anchor is the "present" frame (index i)
                anchor_sample_token = sample_token

                trajectory = extract_trajectory_for_instance(
                    start_ann_token,
                    anchor_sample_token,
                    nusc,
                    num_frames=HISTORY_FRAMES + FUTURE_FRAMES,
                )

                if (
                    trajectory is None
                    or trajectory.shape[0] != HISTORY_FRAMES + FUTURE_FRAMES
                ):
                    continue

                history = trajectory[:HISTORY_FRAMES]  # 4 frames
                future = trajectory[HISTORY_FRAMES:]  # 12 frames

                # Extract map and social features at the anchor frame
                try:
                    anchor_translation, anchor_rotation = get_ego_pose_at_sample(
                        anchor_sample_token
                    )

                    # Extract map features
                    map_features = extract_map_features(
                        anchor_translation,
                        anchor_rotation,
                        nusc,
                        sample_token=sample_token,
                        map_radius=30.0,
                    )
                    map_feat_values = np.array(
                        [
                            map_features["drivable_area"],
                            map_features["lane_type_road"],
                            map_features["lane_type_parking"],
                            map_features["lane_type_shoulder"],
                            map_features["speed_limit"],
                            map_features["is_intersection"],
                            map_features["has_stop_line"],
                            map_features["has_crosswalk"],
                        ],
                        dtype=np.float32,
                    )

                    # Extract social features
                    social_features = extract_social_features(
                        anchor_translation,
                        anchor_rotation,
                        anchor_sample_token,
                        nusc,
                        num_neighbors=5,
                        radius=30.0,
                    )
                except Exception as e:
                    print(f"[WARNING] Failed to extract features: {e}")
                    # Use default/zero features as fallback
                    map_feat_values = np.array(
                        [0.5, 0.33, 0.33, 0.34, 0.4, 0.0, 0.1, 0.1], dtype=np.float32
                    )
                    social_features = np.zeros((5, 4), dtype=np.float32)

                all_histories.append(history)
                all_futures.append(future)
                all_scene_ids.append(scene_idx)  # Record which scene this came from
                all_agent_types.append(agent_type)  # Record agent type
                all_category_names.append(category_name)  # Record full category
                all_map_features.append(map_feat_values)  # Record map features
                all_social_features.append(social_features)  # Record social features

    print(f"\n[INFO] Extracted {len(all_histories)} complete trajectories")

    if len(all_histories) == 0:
        print("[ERROR] No trajectories extracted! Check data paths.")
        return

    # Create agent type mapping
    unique_agent_types = sorted(set(all_agent_types))
    agent_type_to_idx = {atype: idx for idx, atype in enumerate(unique_agent_types)}

    print(f"[INFO] Agent types found: {unique_agent_types}")
    print(f"[INFO] Agent type distribution:")
    for agent_type in unique_agent_types:
        count = sum(1 for at in all_agent_types if at == agent_type)
        pct = 100 * count / len(all_agent_types)
        print(f"  - {agent_type}: {count:,} ({pct:.1f}%)")

    # Convert to tensors
    train_x = torch.tensor(np.array(all_histories), dtype=torch.float32)
    train_y = torch.tensor(np.array(all_futures), dtype=torch.float32)
    scene_ids = torch.tensor(np.array(all_scene_ids), dtype=torch.long)

    # Create one-hot encoded agent type features
    # Shape: (N, len(unique_agent_types))
    num_samples = len(all_histories)
    num_agent_types = len(unique_agent_types)
    agent_type_onehot = torch.zeros(num_samples, num_agent_types, dtype=torch.float32)

    for idx, agent_type in enumerate(all_agent_types):
        agent_type_idx = agent_type_to_idx[agent_type]
        agent_type_onehot[idx, agent_type_idx] = 1.0

    # Convert map and social features to tensors
    # Map features: (N, 4) - drivable_area, lane_type_road, lane_type_parking, lane_type_shoulder
    map_features_tensor = torch.tensor(np.array(all_map_features), dtype=torch.float32)

    # Social features: (N, 5, 4) - 5 neighbors × 4 features (distance, angle, is_vehicle, is_pedestrian)
    social_features_tensor = torch.tensor(
        np.array(all_social_features), dtype=torch.float32
    )

    print(f"\n[INFO] train_x shape: {train_x.shape}")
    print(f"[INFO] train_y shape: {train_y.shape}")
    print(f"[INFO] scene_ids shape: {scene_ids.shape}")
    print(f"[INFO] agent_type_onehot shape: {agent_type_onehot.shape}")
    print(f"[INFO] map_features shape: {map_features_tensor.shape}")
    print(f"[INFO] social_features shape: {social_features_tensor.shape}")

    # Save main data
    torch.save(train_x, OUTPUT_DIR / "train_x.pt")
    torch.save(train_y, OUTPUT_DIR / "train_y.pt")
    torch.save(scene_ids, OUTPUT_DIR / "scene_ids.pt")

    # Save extra features
    torch.save(agent_type_onehot, OUTPUT_DIR / "extra_features_categorical.pt")
    torch.save(map_features_tensor, OUTPUT_DIR / "extra_features_map.pt")
    torch.save(social_features_tensor, OUTPUT_DIR / "extra_features_social.pt")

    # Save metadata
    metadata = {
        "agent_types": unique_agent_types,
        "agent_type_to_idx": agent_type_to_idx,
        "map_features": [
            "drivable_area",
            "lane_type_road",
            "lane_type_parking",
            "lane_type_shoulder",
            "speed_limit",
            "is_intersection",
            "has_stop_line",
            "has_crosswalk",
        ],
        "social_features": ["distance", "angle", "is_vehicle", "is_pedestrian"],
        "num_neighbors": 5,
    }
    torch.save(metadata, OUTPUT_DIR / "agent_type_metadata.pt")

    print(f"\n[SUCCESS] Saved to {OUTPUT_DIR}")
    print(f"  - train_x.pt (history trajectories) - shape {train_x.shape}")
    print(f"  - train_y.pt (future trajectories) - shape {train_y.shape}")
    print(f"  - scene_ids.pt (anti-cheating: prevents train/test overlap)")
    print(
        f"  - extra_features_categorical.pt (one-hot agent type) - shape {agent_type_onehot.shape}"
    )
    print(
        f"  - extra_features_map.pt (map context) - shape {map_features_tensor.shape}"
    )
    print(
        f"  - extra_features_social.pt (nearby agents) - shape {social_features_tensor.shape}"
    )
    print(f"  - agent_type_metadata.pt (feature names and index mapping)")
    print(f"\n[TIP] Use scene_ids.pt to split data by scene:")
    print(f"  train_mask = scene_ids < 700")
    print(f"  x_train, y_train = train_x[train_mask], train_y[train_mask]")


if __name__ == "__main__":
    process_scenes()
    print("\n[DONE] Preprocessing complete!")
