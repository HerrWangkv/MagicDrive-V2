"""
@file   check_progress.py
@brief  Scans the output directory to determine the resume index for a specific range.
"""
import argparse
import mmengine
import sys
import os
from pathlib import Path
from nuscenes.nuscenes import NuScenes

# Redirect print to stderr so we can capture stdout cleanly in bash
def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_root', type=str, default="data/nuscenes")
    parser.add_argument('--pkl_root', type=str, default="data/nuscenes_mmdet3d-12Hz")
    parser.add_argument('--save_root', type=str, required=True)
    parser.add_argument('--splits', type=str, default='train,val')
    
    # New arguments for parallel execution
    parser.add_argument('--range_start', type=int, default=0, help="Start index of the chunk")
    parser.add_argument('--range_end', type=int, default=10000, help="End index of the chunk")
    parser.add_argument('--quiet', action='store_true', help="Output only the resume index to stdout")
    args = parser.parse_args()

    save_root = Path(args.save_root)
    
    # If folder doesn't exist, just start from the beginning of the range
    if not save_root.exists():
        if args.quiet: print(args.range_start)
        else: log(f"Folder not found. Start: {args.range_start}")
        return

    # 1. Init NuScenes (Quietly)
    if not args.quiet: log("Initializing NuScenes...")
    # Redirect stdout to devnull during nusc init to silence SDK spam
    with open(os.devnull, 'w') as fnull:
        sys.stdout = fnull
        nusc = NuScenes(version='v1.0-trainval', dataroot=args.image_root, verbose=False)
        sys.stdout = sys.__stdout__ # Restore stdout

    # 2. Filter scenes to current range
    target_indices = list(range(args.range_start, args.range_end + 1))
    valid_scenes = []
    
    # Build scene intervals only for target range to save time
    # But we need correct global indices, so we iterate all but filter storage
    scene_intervals = []
    for i, scene in enumerate(nusc.scene):
        if i < args.range_start or i > args.range_end:
            continue
            
        first = nusc.get('sample', scene['first_sample_token'])
        last = nusc.get('sample', scene['last_sample_token'])
        scene_intervals.append({
            'start': first['timestamp'],
            'end': last['timestamp'],
            'token': scene['token'],
            'index': i
        })
        valid_scenes.append(scene)

    if not scene_intervals:
        if args.quiet: print(args.range_start)
        return

    # 3. Load PKLs (Only what's needed)
    scenes_map = {s['token']: [] for s in valid_scenes}
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    
    if not args.quiet: log("Loading PKL metadata...")
    
    for split in splits:
        pkl_path = Path(args.pkl_root) / f"nuscenes_interp_12Hz_infos_{split}_with_bid.pkl"
        if not pkl_path.exists(): continue
        
        data_dict = mmengine.load(str(pkl_path))
        raw_infos = data_dict['data_list'] if 'data_list' in data_dict else data_dict.get('infos', data_dict)
        
        # Filter infos to our target scenes
        # Optimization: Global timestamp check
        min_ts = scene_intervals[0]['start'] - 1e6
        max_ts = scene_intervals[-1]['end'] + 1e6
        
        for info in raw_infos:
            ts = info['timestamp']
            if ts < min_ts or ts > max_ts: continue
            
            # Match to specific scene
            for sc in scene_intervals:
                if sc['start'] - 1e6 <= ts <= sc['end'] + 1e6:
                    scenes_map[sc['token']].append(info)
                    break

    # 4. Scan existing files
    if not args.quiet: log(f"Scanning {save_root}...")
    existing_files = set(p.name for p in save_root.glob('*.png'))

    # 5. Determine Resume Index
    last_processed_idx = -1
    
    # Check sequences in order
    for idx_in_list, scene in enumerate(valid_scenes):
        global_idx = args.range_start + idx_in_list
        token = scene['token']
        frames = scenes_map.get(token, [])
        
        if not frames: continue # Empty scene, ignore
        
        found_any = False
        for info in frames:
            cam_dict = info.get('cams', info.get('images', {}))
            for cam_info in cam_dict.values():
                src_p = cam_info.get('data_path', cam_info.get('img_path', ''))
                fname = Path(src_p).name.replace('.jpg', '.png')
                if fname in existing_files:
                    found_any = True
                    break
            if found_any: break
        
        if found_any:
            last_processed_idx = global_idx

    # 6. Output Logic
    if last_processed_idx != -1:
        # If we found processed data at index X, resume from X (to be safe/complete)
        resume_at = last_processed_idx
    else:
        # Nothing processed in this range yet
        resume_at = args.range_start

    if args.quiet:
        print(resume_at)
    else:
        log(f"Range [{args.range_start}-{args.range_end}]: Found progress up to {last_processed_idx}. Resuming at {resume_at}")

if __name__ == "__main__":
    main()