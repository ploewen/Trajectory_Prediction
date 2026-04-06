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
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATAROOT = '/media/robotswithai/data1/ece1508/nuscenes_data'
VERSION = 'v1.0-trainval'
SAMPLE_RATE = 2  # Hz (samples every 0.5 sec)
HISTORY_FRAMES = 4  # 2 seconds at 2Hz
FUTURE_FRAMES = 12  # 6 seconds at 2Hz
OUTPUT_DIR = '/media/robotswithai/data1/ece1508/data'

print("[INFO] Loading nuScenes dataset...")
nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)


def quaternion_to_rotation_matrix(q):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    
    rotation_matrix = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x**2 + z**2),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    
    return rotation_matrix


def get_ego_pose_at_sample(sample_token):
    """Get ego vehicle pose (translation and rotation) at a given sample."""
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    
    translation = np.array(ego_pose['translation'])[:2]  # x, y only
    rotation = quaternion_to_rotation_matrix(ego_pose['rotation'])[:2, :2]  # 2x2 rotation
    
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


def extract_trajectory_for_instance(start_ann_token, anchor_sample_token, nusc_obj, num_frames=HISTORY_FRAMES + FUTURE_FRAMES):
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
        ann = nusc_obj.get('sample_annotation', current_ann_token)
        
        # 3. Extract global position
        global_pos = np.array(ann['translation'])[:2]  # x, y only
        
        # 4. Transform to ego frame using the FIXED anchor
        ego_pos = transform_to_ego_frame(global_pos, anchor_translation, anchor_rotation)
        trajectory.append(ego_pos)
        
        # 5. Hop to the next annotation instantly (no searching!)
        current_ann_token = ann['next']
        
    return np.array(trajectory)  # Shape: (num_frames, 2)


def process_scenes():
    """
    Main processing loop optimized for speed and fixed-anchor transformations.
    - Stores both sample_token AND ann_token to avoid searching
    - Uses fixed anchor pose for entire 8-second sequence (fixes physics bug)
    - Uses 'next' token to hop through data instantly (kills CPU bottleneck)
    - Tracks scene ID for each trajectory (no train/test overlap)
    """
    all_histories = []
    all_futures = []
    all_scene_ids = []  # Track which scene each trajectory came from
    
    total_scenes = len(nusc.scene)
    print(f"[INFO] Processing {total_scenes} scenes...\n")
    
    for scene_idx, scene in enumerate(nusc.scene):
        if (scene_idx + 1) % 10 == 0 or scene_idx == 0:
            print(f"[PROGRESS] Scene {scene_idx + 1}/{total_scenes}")
        
        # Store tuples of (sample_token, ann_token) so we don't search later
        scene_instances = defaultdict(list)
        
        current_sample_token = scene['first_sample_token']
        while current_sample_token:
            sample = nusc.get('sample', current_sample_token)
            
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                instance_token = ann['instance_token']
                
                # Fast category check
                instance = nusc.get('instance', instance_token)
                category = nusc.get('category', instance['category_token'])
                if 'vehicle' not in category['name']:
                    continue
                
                # Save BOTH tokens so we don't have to search later
                scene_instances[instance_token].append((current_sample_token, ann_token))
            
            current_sample_token = sample['next']
        
        # Extract trajectories
        for instance_token, sample_records in scene_instances.items():
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
                    num_frames=HISTORY_FRAMES + FUTURE_FRAMES
                )
                
                if trajectory is None or trajectory.shape[0] != HISTORY_FRAMES + FUTURE_FRAMES:
                    continue
                
                history = trajectory[:HISTORY_FRAMES]  # 4 frames
                future = trajectory[HISTORY_FRAMES:]   # 12 frames
                
                all_histories.append(history)
                all_futures.append(future)
                all_scene_ids.append(scene_idx)  # Record which scene this came from
    
    print(f"\n[INFO] Extracted {len(all_histories)} complete trajectories")
    
    if len(all_histories) == 0:
        print("[ERROR] No trajectories extracted! Check data paths.")
        return
    
    train_x = torch.tensor(np.array(all_histories), dtype=torch.float32)
    train_y = torch.tensor(np.array(all_futures), dtype=torch.float32)
    scene_ids = torch.tensor(np.array(all_scene_ids), dtype=torch.long)  # Scene tracker
    
    print(f"[INFO] train_x shape: {train_x.shape}")
    print(f"[INFO] train_y shape: {train_y.shape}")
    print(f"[INFO] scene_ids shape: {scene_ids.shape}")
    
    torch.save(train_x, f'{OUTPUT_DIR}/train_x.pt')
    torch.save(train_y, f'{OUTPUT_DIR}/train_y.pt')
    torch.save(scene_ids, f'{OUTPUT_DIR}/scene_ids.pt')
    
    print(f"\n[SUCCESS] Saved to {OUTPUT_DIR}")
    print(f"  - train_x.pt (history trajectories)")
    print(f"  - train_y.pt (future trajectories)")
    print(f"  - scene_ids.pt (anti-cheating: prevents train/test overlap)")
    print(f"\n[TIP] Use scene_ids.pt to split data by scene:")
    print(f"  train_mask = scene_ids < 700")
    print(f"  x_train, y_train = train_x[train_mask], train_y[train_mask]")
    print(f"  x_test, y_test = train_x[~train_mask], train_y[~train_mask]")


if __name__ == '__main__':
    process_scenes()
    print("\n[DONE] Preprocessing complete!")
