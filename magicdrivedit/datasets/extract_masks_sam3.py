"""
@file   extract_masks_sam3.py
@brief  Extract semantic masks (Pedestrian) using SAM 3 Video Processing on NuScenes.
        Strictly aligned with NuScenes 12Hz interpolated data (nuscenes-mmdet3d-12Hz).
"""

import os
import sys
import torch
import cv2
import numpy as np
import imageio
import pickle
import gc
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import torch.multiprocessing as mp

from sam3.model_builder import build_sam3_video_predictor

# Try importing NuScenes
try:
    from nuscenes.nuscenes import NuScenes
except ImportError:
    print("Error: nuscenes-devkit not installed. Please install it.")
    sys.exit(1)

CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

def setup_model(device_id, verbose=True):
    """Load SAM 3 model on specific GPU"""
    device = f"cuda:{device_id}"
    if verbose:
        print(f"Loading SAM 3 on {device}...")
    
    # We use bfloat16 for efficiency as recommended in SAM 3 docs for video
    # Check if we should override the default (which is usually controlled by model internals or env vars)
    # The build_sam3_video_predictor usually loads to CUDA:0 by default unless configured
    # We need to manually move it or set device context.
    
    try:
        # According to doc: video_predictor = build_sam3_video_predictor()
        # This loads the model. 
        # CAUTION: The native builder might default to cuda:0. 
        # We might need to set CUDA_VISIBLE_DEVICESenv var per process if the builder doesn't support device arg.
        # However, we are inside a multiprocessing spawn. We can try setting the device via torch.
        
        torch.cuda.set_device(device_id)
        video_predictor = build_sam3_video_predictor()
        
        # Move internal components if they are not on right device
        # The predictor wraps the model.
        # video_predictor.model.to(device) # Try this if needed
        
    except Exception as e:
        print(f"Error loading SAM 3: {e}")
        print("Ensure you have access to facebook/sam3 and HUGGING_FACE_TOKEN is set.")
        raise e
        
    return video_predictor, device

def get_output_path(input_path, data_root, save_root):
    """
    Derive output path from input path.
    Input: .../data/nuscenes/samples/CAM_FRONT/file.jpg
    Output: .../data/nuscenes_masks/human/samples/CAM_FRONT/file.png
    """
    # Normalize paths
    abs_input = os.path.abspath(input_path)
    abs_data = os.path.abspath(data_root)
    
    if abs_input.startswith(abs_data):
        rel_path = abs_input[len(abs_data):].strip(os.sep)
    else:
        # Fallback heuristic if paths don't align nicely
        # assum input is .../folder_type/cam_name/filename
        parts = input_path.split(os.sep)
        # Find 'samples' or 'sweeps'
        idx = -1
        if 'samples' in parts: idx = parts.index('samples')
        elif 'sweeps' in parts: idx = parts.index('sweeps')
        
        if idx != -1:
            rel_path = os.path.join(*parts[idx:])
        else:
            # Last resort: just filename? No, we need structure.
            # Assume last 3 parts: folder_type/cam/clean_file
            rel_path = os.path.join(*parts[-3:])
            
    # Construct output
    out_rel = os.path.splitext(rel_path)[0] + ".png"
    out_full = os.path.join(save_root, "human", out_rel)
    return out_full

def process_chunk(rank, gpu_ids, task_chunk, data_root, save_root):
    """
    Worker function.
    task_chunk: List of tasks. Each task is {'id': str, 'frames': [path1, path2...]}
    """
    gpu_id = gpu_ids[rank]
    
    if not task_chunk:
        return

    # Moved model init inside loop to prevent memory leak
    #try:
    #    video_predictor, device = setup_model(gpu_id)
    #except Exception:
    #    return

def process_single_task(gpu_id, video_predictor, task, data_root, save_root):
    """
    Process one task with existing model.
    """
    frames = task['frames']
    if not frames: return

    # Workaround for "Video" composed of disparate frames
    temp_dir = Path(f"/tmp/sam3_gpu{gpu_id}_{task['id']}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean temp dir
    for f in temp_dir.glob("*"): f.unlink()
    
    try:
        valid_frames_indices = []
        for i, path in enumerate(frames):
            if not os.path.exists(path): 
                continue
            sym_name = f"{len(valid_frames_indices):05d}{os.path.splitext(path)[1]}"
            os.symlink(path, temp_dir / sym_name)
            valid_frames_indices.append(i)
            
        if not valid_frames_indices:
            tqdm.write(f"Skipping task {task['id']}: No valid frames found")
            return

        # Start Session
        response = video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=str(temp_dir),
            )
        )
        session_id = response['session_id']
        
        # Add Prompt
        num_frames = len(valid_frames_indices)
        text_prompts = ["pedestrian", "person", "human"]
        
        for f_idx in range(0, num_frames, 1):
            for txt in text_prompts:
                video_predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=f_idx,
                        text=txt,
                    )
                )
        
        if hasattr(video_predictor, "propagate_in_video"):
             propagator = video_predictor.propagate_in_video(
                 session_id,
                 propagation_direction="forward",
                 start_frame_idx=0,
                 max_frame_num_to_track=len(valid_frames_indices)
             )
             
             for out in propagator:
                 if 'masks' in out:
                     masks = out['masks']
                 elif 'outputs' in out and 'out_binary_masks' in out['outputs']:
                     masks = out['outputs']['out_binary_masks']
                 else:
                     continue

                 frame_idx = out['frame_index']
                 if len(masks) == 0: continue

                 if masks.ndim == 4:
                    masks = masks.squeeze(1)
                
                 union_mask = np.any(masks, axis=0)
                 if union_mask.ndim == 3:
                     union_mask = union_mask.squeeze(0)

                 if not np.any(union_mask):
                     continue

                 final_mask = (union_mask.astype(np.uint8) * 255)
                 
                 original_idx = valid_frames_indices[frame_idx]
                 original_path = frames[original_idx]
                 
                 out_path = get_output_path(original_path, data_root, save_root)
                 os.makedirs(os.path.dirname(out_path), exist_ok=True)
                 imageio.imwrite(out_path, final_mask)
            
             tqdm.write(f"GPU {gpu_id}: Finished {task['id']}")
        else:
             tqdm.write(f"Error: Native video_predictor does not match expected API.")
             
    finally:
        # Cleanup temp dir
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            
        # Try to reset session
        if 'session_id' in locals():
             try:
                 # Standard SAM2/3 API often uses reset_state or handle_request('reset_state')
                 if hasattr(video_predictor, "reset_state"):
                     video_predictor.reset_state(session_id)
                 else:
                     # Try generic cleanup requests which might free memory
                     try:
                        video_predictor.handle_request(dict(type="reset_state", session_id=session_id))
                     except: pass
                     
                     try:
                        video_predictor.handle_request(dict(type="close_session", session_id=session_id))
                     except: pass
                     
                     try:
                        video_predictor.handle_request(dict(type="finish_session", session_id=session_id))
                     except: pass
             except: pass
            

def process_chunk(rank, gpu_ids, task_chunk, data_root, save_root):
    """
    Worker function.
    """
    gpu_id = gpu_ids[rank]
    
    if not task_chunk:
        return

    try:
        video_predictor, device = setup_model(gpu_id)
    except Exception as e:
        print(f"GPU {gpu_id}: Model Init Failed: {e}")
        return

    iterator = tqdm(task_chunk, desc=f"GPU {gpu_id}", position=rank)
    
    for task in iterator:
        iterator.set_description(f"GPU {gpu_id}: {task['id']}")

        try:
            process_single_task(gpu_id, video_predictor, task, data_root, save_root)
        except torch.cuda.OutOfMemoryError:
            tqdm.write(f"GPU {gpu_id}: OOM on {task['id']}. Resetting model.")
            try:
                del video_predictor
            except: pass
            torch.cuda.empty_cache()
            gc.collect()
            video_predictor, device = setup_model(gpu_id, verbose=False)
            continue
            
        except Exception as e:
            tqdm.write(f"GPU {gpu_id}: Error on {task['id']}: {e}")

        # Aggressive Global Cleanup b/w tasks
        torch.cuda.empty_cache()
        gc.collect()
        
        # Memory Watchdog: Monitor TOTAL GPU Usage (global)
        # mem_info returns (free_mem, total_mem)
        free_mem, total_mem = torch.cuda.mem_get_info(gpu_id)
        used_mem = total_mem - free_mem
        
        if used_mem / total_mem > 0.5:
             tqdm.write(f"GPU {gpu_id}: GPU Memory > 50% ({used_mem/1024**3:.1f}/{total_mem/1024**3:.1f}GB). Reloading model...")
             del video_predictor
             torch.cuda.empty_cache()
             gc.collect()
             video_predictor, device = setup_model(gpu_id, verbose=False)

def load_12hz_scenes(pkl_root, image_root, splits=['train', 'val']):
    """
    Load scenes aligned with 12Hz interpolated data, similar to NuScenes12HzPipeline.
    """
    print("Initializing NuScenes SDK...")
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(image_root), verbose=False)
    
    scene_intervals = []
    scenes = {} # token -> list of 12hz frames
    
    # 1. Pre-calculate scene intervals from 2Hz data
    for i, scene in enumerate(nusc.scene):
        token = scene['token']
        first = nusc.get('sample', scene['first_sample_token'])
        last = nusc.get('sample', scene['last_sample_token'])
        scene_intervals.append({
            'start': first['timestamp'],
            'end': last['timestamp'],
            'token': token,
            'name': scene['name'],
            'index': i
        })
        scenes[token] = {'name': scene['name'], 'frames': [], 'index': i}

    # 2. Load 12Hz pickle data and map to scenes
    for split in splits:
        pkl_path = Path(pkl_root) / f"nuscenes_interp_12Hz_infos_{split}_with_bid.pkl"
        print(f"Loading {split} split from {pkl_path}...")
        if not pkl_path.exists(): 
            print(f"Warning: {pkl_path} not found.")
            continue
            
        with open(str(pkl_path), 'rb') as f:
            data_dict = pickle.load(f)
            
        raw_infos = data_dict['data_list'] if 'data_list' in data_dict else data_dict.get('infos', data_dict)
        
        print(f"  Mapping {len(raw_infos)} frames to scenes...")
        current_sc_idx = 0
        
        for info in tqdm(raw_infos, desc=f"Mapping {split}"):
            ts = info['timestamp']
            s_token = None
            
            # Optimization: Check current and next scene first
            intervals_to_check = [current_sc_idx]
            if current_sc_idx + 1 < len(scene_intervals):
                intervals_to_check.append(current_sc_idx + 1)
            
            found = False
            for idx in intervals_to_check:
                sc = scene_intervals[idx]
                if sc['start'] - 1e6 <= ts <= sc['end'] + 1e6:
                    s_token = sc['token']; current_sc_idx = idx; found = True; break
            
            # Fallback: Search all
            if not found:
                for idx, sc in enumerate(scene_intervals):
                    if sc['start'] - 1e6 <= ts <= sc['end'] + 1e6:
                        s_token = sc['token']; current_sc_idx = idx; found = True; break
                        
            if s_token and s_token in scenes:
                scenes[s_token]['frames'].append(info)

    # 3. Sort frames by timestamp for each scene
    final_scenes = []
    for token, data in scenes.items():
        if not data['frames']: continue
        data['frames'].sort(key=lambda x: x['timestamp'])
        final_scenes.append(data)
        
    print(f"Total: Populated {len(final_scenes)} scenes aligned with SDK.")
    return final_scenes

def check_task_started(task, data_root, save_root):
    """Check if at least one frame in the task has a generated mask."""
    for path in task['frames']:
        out_path = get_output_path(path, data_root, save_root)
        if os.path.exists(out_path):
            return True
    return False

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/nuscenes')
    parser.add_argument('--pkl_root', type=str, default='data/nuscenes_mmdet3d-12Hz')
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--splits', type=str, default='train,val', help='Comma separated splits')
    parser.add_argument("--save_root", type=str, default='data/nuscenes_masks')
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7', help='Comma separated GPU IDs')
    
    # Scene filtering arguments
    parser.add_argument('--scene_idx', type=int, default=None, help='Specific scene index to process')
    parser.add_argument('--scene_idx_start', type=int, default=None, help='Start scene index (inclusive)')
    parser.add_argument('--scene_idx_end', type=int, default=None, help='End scene index (inclusive)')

    parser.add_argument('--ignore_existing', action='store_true')
    args = parser.parse_args()
    
    gpu_ids = [int(x) for x in args.gpus.split(',') if x.strip()]
    n_gpus = len(gpu_ids)
    
    # 1. Load Data
    splits = [s.strip() for s in args.splits.split(',')]
    scenes_data = load_12hz_scenes(args.pkl_root, args.data_root, splits)
    
    # 2. Build Tasks (Scene-Camera Pairs)
    tasks = []
    print("Building inference tasks (grouping by Scene & Camera)...")
    
    for scene in scenes_data:
        # Check Scene Filtering
        scene_index = scene['index']
        
        # Exact Match
        if args.scene_idx is not None:
            if scene_index != args.scene_idx:
                continue
        
        # Range Match
        if args.scene_idx_start is not None:
             if scene_index < args.scene_idx_start:
                 continue
        
        if args.scene_idx_end is not None:
            if scene_index > args.scene_idx_end:
                continue

        scene_name = scene['name']
        frames = scene['frames']
        if not frames: continue
        
        for cam in CAMS:
            # Extract file paths for this camera across the scene
            # info['cams'][cam]['data_path'] is relative to data_root (often ./data/nuscenes/...)
            cam_paths = []
            for info in frames:
                if cam in info['cams']:
                     # Handle path variations depending on pkl generation
                     rel_path = info['cams'][cam]['data_path']
                     # Often pkl paths are literal relative paths "./data/nuscenes/..."
                     # We need to ensure we join them correctly with data_root
                     if rel_path.startswith("./"):
                         rel_path = rel_path[2:]
                     
                     # If rel_path already includes "data/nuscenes", we might be doubling up depending on data_root
                     # Let's assume data_root is the base folder.
                     # If data_root is "data/nuscenes" and path is "data/nuscenes/samples/...", we need only "samples/..."
                     
                     # Robust join:
                     full_path = os.path.abspath(os.path.join(os.getcwd(), rel_path))
                     if not os.path.exists(full_path):
                         # Try joining with data_root if absolute check failed
                         full_path = os.path.join(args.data_root, rel_path)
                     
                     cam_paths.append(full_path)
            
            if not cam_paths: continue

            # Previous simple check - replaced by heuristic below
            # if args.ignore_existing:
            #     last_out = get_output_path(cam_paths[-1], args.data_root, args.save_root)
            #     if os.path.exists(last_out):
            #         continue
            
            tasks.append({
                'id': f"{scene_name}_{cam}",
                'frames': cam_paths
            })
            
    total_tasks = len(tasks)
    print(f"Total video sequences to process: {total_tasks}")
    
    if args.ignore_existing:
        print("Applying skip heuristic...")
        # Pre-check "started" status for performance
        is_started = [check_task_started(t, args.data_root, args.save_root) for t in tqdm(tasks, desc="Checking status")]
        
        tasks_to_keep = []
        for i in range(len(tasks)):
            skipped = False
            # Check heuristic: if this task started AND next task started, this task is done
            if i + 1 < len(tasks):
                 if is_started[i]:
                     if is_started[i+1]:
                         skipped = True
                     else:
                         print(f"Task {tasks[i]['id']} started but next task {tasks[i+1]['id']} not started. Rerunning.")
            
            if not skipped:
                tasks_to_keep.append(tasks[i])

        tasks = tasks_to_keep
        total_tasks = len(tasks)
        print(f"Tasks after filtering: {total_tasks}")

    
    if total_tasks == 0:
        print("No tasks found.")
        return

    # 3. Distribute
    chunk_size = int(np.ceil(total_tasks / n_gpus))
    chunks = [tasks[i:i + chunk_size] for i in range(0, total_tasks, chunk_size)]
    while len(chunks) < n_gpus: chunks.append([])

    # 4. Run
    mp.set_start_method('spawn', force=True)
    processes = []
    
    print(f"Starting {n_gpus} workers...")
    for rank in range(n_gpus):
        p = mp.Process(
            target=process_chunk,
            args=(rank, gpu_ids, chunks[rank], args.data_root, args.save_root)
        )
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    print("Done.")

if __name__ == "__main__":
    main()
