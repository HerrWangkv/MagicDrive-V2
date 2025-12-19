import cv2
import numpy as np
import mmengine
import os
from pathlib import Path
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

class VideoGenerator:
    def __init__(self, pkl_root, image_root, output_root, splits=['train', 'val']):
        self.pkl_root = Path(pkl_root)
        self.image_root = Path(image_root)
        self.output_root = Path(output_root)
        
        print("Initializing NuScenes SDK...")
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=str(self.image_root), verbose=False)
        
        self.scene_tokens_ordered = [s['token'] for s in self.nusc.scene]
        self.scenes = {token: [] for token in self.scene_tokens_ordered}
        
        self.scene_intervals = []
        for i, scene in enumerate(self.nusc.scene):
            first = self.nusc.get('sample', scene['first_sample_token'])
            last = self.nusc.get('sample', scene['last_sample_token'])
            self.scene_intervals.append({
                'start': first['timestamp'],
                'end': last['timestamp'],
                'token': scene['token'],
                'index': i
            })
            
        for split in splits:
            pkl_path = self.pkl_root / f"nuscenes_interp_12Hz_infos_{split}_with_bid.pkl"
            print(f"Loading {split} split from {pkl_path}...")
            if not pkl_path.exists(): continue
            
            data_dict = mmengine.load(str(pkl_path))
            if 'data_list' in data_dict:
                raw_infos = data_dict['data_list']
            elif 'infos' in data_dict:
                raw_infos = data_dict['infos']
            else:
                raw_infos = data_dict
            
            print(f"  Mapping {len(raw_infos)} frames to scenes...")
            current_sc_idx = 0
            for info in raw_infos:
                ts = info['timestamp']
                s_token = None
                
                intervals_to_check = [current_sc_idx]
                if current_sc_idx + 1 < len(self.scene_intervals):
                    intervals_to_check.append(current_sc_idx + 1)
                
                found = False
                for idx in intervals_to_check:
                    sc = self.scene_intervals[idx]
                    if sc['start'] - 1e6 <= ts <= sc['end'] + 1e6:
                        s_token = sc['token']
                        current_sc_idx = idx
                        found = True
                        break
                
                if not found:
                    for idx, sc in enumerate(self.scene_intervals):
                        if sc['start'] - 1e6 <= ts <= sc['end'] + 1e6:
                            s_token = sc['token']
                            current_sc_idx = idx
                            found = True
                            break
                            
                if s_token and s_token in self.scenes:
                    self.scenes[s_token].append(info)

        for token in self.scenes:
            self.scenes[token].sort(key=lambda x: x['timestamp'])

    def make_video(self, scene_idx, fps=12):
        if not (0 <= scene_idx < len(self.scene_tokens_ordered)):
            print(f"Scene index {scene_idx} out of range.")
            return

        token = self.scene_tokens_ordered[scene_idx]
        frames = self.scenes[token]
        print(f"Generating video for Scene {scene_idx}: {len(frames)} frames")
        
        if not frames:
            print("No frames found.")
            return

        # Setup Video Writer
        # Grid: 
        # Orig: FL F FR
        # Orig: BL B BR
        # Ped:  FL F FR
        # Ped:  BL B BR
        
        # Resize to width 400 per image -> 1200 width
        # Aspect ratio 16:9 -> 400x225
        # Total height: 4 * 225 = 900
        
        W_img = 400
        H_img = 225
        W_grid = W_img * 3
        H_grid = H_img * 4
        
        video_path = f"video_scene_{scene_idx}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (W_grid, H_grid))
        
        cameras_top = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']
        cameras_bot = ['CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
        
        for info in tqdm(frames):
            canvas = np.zeros((H_grid, W_grid, 3), dtype=np.uint8)
            
            if 'cams' in info: cam_dict = info['cams']; use_cams=True
            elif 'images' in info: cam_dict = info['images']; use_cams=False
            else: continue
            
            # Helper to process a row
            def process_row(cams, row_idx, is_ped=False):
                for col_idx, cam_name in enumerate(cams):
                    if cam_name not in cam_dict: continue
                    cam_info = cam_dict[cam_name]
                    
                    if use_cams:
                        raw = cam_info['data_path']
                        rel = raw.split('nuscenes/')[-1] if 'nuscenes/' in raw else raw
                        img_path = self.image_root / rel
                    else:
                        img_path = self.image_root / cam_info['img_path']
                    
                    # Load Image
                    if is_ped:
                        # Pedestrian Image (PNG)
                        ped_name = Path(img_path).name.replace('.jpg', '.png')
                        ped_path = self.output_root / ped_name
                        if ped_path.exists():
                            img = cv2.imread(str(ped_path))
                        else:
                            img = np.zeros((900, 1600, 3), dtype=np.uint8) # Assume default size
                    else:
                        # Original Image (JPG)
                        if img_path.exists():
                            img = cv2.imread(str(img_path))
                        else:
                            img = np.zeros((900, 1600, 3), dtype=np.uint8)
                            
                    if img is None: continue
                    
                    # Resize
                    img_small = cv2.resize(img, (W_img, H_img))
                    
                    # Place
                    y = row_idx * H_img
                    x = col_idx * W_img
                    canvas[y:y+H_img, x:x+W_img] = img_small

            # Row 1: Orig Top
            process_row(cameras_top, 0, is_ped=False)
            # Row 2: Orig Bot
            process_row(cameras_bot, 1, is_ped=False)
            # Row 3: Ped Top
            process_row(cameras_top, 2, is_ped=True)
            # Row 4: Ped Bot
            process_row(cameras_bot, 3, is_ped=True)
            
            out.write(canvas)
            
        out.release()
        print(f"Video saved to {video_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate a video for a specific NuScenes scene index.")
    parser.add_argument('scene_idx', type=int, default=0, help='Scene index to generate video for (default: 0)')
    parser.add_argument('--pkl_root', type=str, default="data/nuscenes_mmdet3d-12Hz", help='Root directory for PKL files')
    parser.add_argument('--image_root', type=str, default="data/nuscenes", help='Root directory for images (NuScenes dataroot)')
    parser.add_argument('--output_root', type=str, default="output_12hz_aligned", help='Directory with pedestrian outputs')
    parser.add_argument('--fps', type=int, default=12, help='Frames per second for the video (default: 12)')
    parser.add_argument('--splits', type=str, default='train,val', help='Comma-separated list of splits (default: train,val)')

    args = parser.parse_args()
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    gen = VideoGenerator(
        pkl_root=args.pkl_root,
        image_root=args.image_root,
        output_root=args.output_root,
        splits=splits
    )
    gen.make_video(args.scene_idx, fps=args.fps)
