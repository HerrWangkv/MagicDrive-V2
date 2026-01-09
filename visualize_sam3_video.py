import cv2
import numpy as np
import argparse
import pickle
import os
import sys
from pathlib import Path
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

# Camera Layout
ROW1 = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']
ROW2 = ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
LAYOUT = [ROW1, ROW2]

def get_mask_path(image_path, data_root, mask_root):
    """
    Reconstruct mask path from image path.
    Image: [data_root]/samples/CAM_FRONT/file.jpg
    Mask:  [mask_root]/human/samples/CAM_FRONT/file.png
    """
    abs_img = os.path.abspath(image_path)
    abs_data = os.path.abspath(data_root)
    
    if abs_img.startswith(abs_data):
        rel_path = abs_img[len(abs_data):].strip(os.sep)
    else:
        # Fallback for manual paths not starting with data_root
        parts = image_path.split(os.sep)
        # Find 'samples' or 'sweeps'
        idx = -1
        if 'samples' in parts: idx = parts.index('samples')
        elif 'sweeps' in parts: idx = parts.index('sweeps')
        
        if idx != -1:
            rel_path = os.path.join(*parts[idx:])
        else:
            rel_path = os.path.join(*parts[-3:])

    mask_rel = os.path.splitext(rel_path)[0] + ".png"
    return os.path.join(mask_root, "human", mask_rel)

def overlay_mask(image, mask, color=(0, 0, 255), alpha=0.5):
    """Overlay binary mask on image with color (BGR)."""
    # Resize mask to image size if needed
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    
    mask_indices = mask > 0
    if not np.any(mask_indices):
        return image
        
    overlay = image.copy()
    overlay[mask_indices] = cv2.addWeighted(
        image[mask_indices], 1 - alpha,
        colored_mask[mask_indices], alpha,
        0
    )
    return overlay

def load_scene_frames(nusc, pkl_path, scene_idx):
    """
    Load 12Hz frames for a specific scene index using the pickle info.
    """
    if scene_idx >= len(nusc.scene):
        print(f"Error: Scene index {scene_idx} out of bounds.")
        sys.exit(1)

    scene = nusc.scene[scene_idx]
    scene_token = scene['token']
    scene_name = scene['name']
    
    # Get time range
    first = nusc.get('sample', scene['first_sample_token'])
    last = nusc.get('sample', scene['last_sample_token'])
    start_ts = first['timestamp']
    end_ts = last['timestamp']
    
    print(f"Loading frames for scene {scene_idx}: {scene_name}")
    
    # Try different split files if specific one fails
    splits_to_try = [pkl_path]
    # Infer other splits if pkl_path has a specific naming convention
    base_dir = os.path.dirname(pkl_path)
    splits_to_try.append(os.path.join(base_dir, "nuscenes_interp_12Hz_infos_val_with_bid.pkl"))
    splits_to_try.append(os.path.join(base_dir, "nuscenes_interp_12Hz_infos_train_with_bid.pkl"))
    
    # Remove duplicates and filter existing
    valid_pkls = []
    seen = set()
    for p in splits_to_try:
        if p not in seen and os.path.exists(p):
            valid_pkls.append(p)
            seen.add(p)
    
    if not valid_pkls:
        print(f"Error: No valid pickle info files found. Searched based on {pkl_path}")
        sys.exit(1)

    scene_frames = []
    
    for current_pkl in valid_pkls:
        print(f"  Checking {current_pkl}...")
        with open(current_pkl, 'rb') as f:
            data_dict = pickle.load(f)
        
        raw_infos = data_dict['data_list'] if 'data_list' in data_dict else data_dict.get('infos', data_dict)
        
        # Filter by timestamp (with buffer)
        # 1e6 us = 1 second buffer
        count = 0
        for info in raw_infos:
            ts = info['timestamp']
            if start_ts - 1e6 <= ts <= end_ts + 1e6:
                scene_frames.append(info)
                count += 1
        
        if count > 0:
            print(f"    Found {count} frames in this split.")

    scene_frames.sort(key=lambda x: x['timestamp'])
    # Remove duplicates if same frame in multiple splits (unlikely but safe)
    # Using timestamp and first camera token as unique key
    unique_frames = []
    seen_tokens = set()
    for f in scene_frames:
        # Just use timestamp as quick key
        key = f['timestamp'] 
        if key not in seen_tokens:
             unique_frames.append(f)
             seen_tokens.add(key)
             
    print(f"  Total Unique Frames: {len(unique_frames)}")
    return unique_frames, scene_name

def main():
    parser = argparse.ArgumentParser(description="Generate Video of SAM 3 Masks (12Hz)")
    parser.add_argument('--data_root', default='data/nuscenes')
    parser.add_argument('--mask_root', default='data/nuscenes_masks')
    parser.add_argument('--pkl_root', default='data/nuscenes_mmdet3d-12Hz')
    parser.add_argument('--split', default='val', help="val or train")
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--scene_idx', type=int, default=0)
    parser.add_argument('--output', default='output_video.mp4')
    parser.add_argument('--fps', type=int, default=12)
    parser.add_argument('--scale', type=float, default=0.5)
    args = parser.parse_args()

    # 1. Setup NuScenes
    print("Initializing NuScenes...")
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
    
    # 2. Load Frames
    pkl_name = f"nuscenes_interp_12Hz_infos_{args.split}_with_bid.pkl"
    pkl_path = os.path.join(args.pkl_root, pkl_name)
    
    frames, scene_name = load_scene_frames(nusc, pkl_path, args.scene_idx)
    
    if not frames:
        print("No frames found for this scene.")
        return

    # 3. Setup Video Writer
    # Determine size from first frame
    # NuScenes original: 1600x900
    # Grid: 3 cols, 2 rows.
    W = 1600
    H = 900
    grid_w = int(W * 3 * args.scale)
    grid_h = int(H * 2 * args.scale)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(args.output, fourcc, args.fps, (grid_w, grid_h))
    print(f"Writing video to {args.output} ({grid_w}x{grid_h} @ {args.fps}fps)...")

    # 4. Process Frames
    for info in tqdm(frames):
        rows_imgs = []
        for row_cams in LAYOUT:
            row_frames = []
            for cam_name in row_cams:
                if cam_name in info['cams']:
                    # Path handling similar to extraction script
                    rel_path = info['cams'][cam_name]['data_path']
                    if rel_path.startswith("./"): rel_path = rel_path[2:]
                    
                    full_path = os.path.abspath(os.path.join(os.getcwd(), rel_path))
                    if not os.path.exists(full_path):
                         full_path = os.path.join(args.data_root, rel_path)

                    mask_path = get_mask_path(full_path, args.data_root, args.mask_root)
                    
                    # Read Image
                    if os.path.exists(full_path):
                        img = cv2.imread(full_path)
                    else:
                        img = np.zeros((H, W, 3), dtype=np.uint8)
                        
                    # Read Mask & Overlay
                    if mask_path and os.path.exists(mask_path):
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            img = overlay_mask(img, mask)
                            
                    # Resize
                    img_small = cv2.resize(img, (0,0), fx=args.scale, fy=args.scale)
                else:
                    # Missing camera in this frame?
                    single_w = int(W * args.scale)
                    single_h = int(H * args.scale)
                    img_small = np.zeros((single_h, single_w, 3), dtype=np.uint8)
                
                # Add Label
                cv2.putText(img_small, cam_name, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                row_frames.append(img_small)
            
            rows_imgs.append(np.hstack(row_frames))
        
        final_grid = np.vstack(rows_imgs)
        video.write(final_grid)

    video.release()
    print("Done.")

if __name__ == "__main__":
    main()
