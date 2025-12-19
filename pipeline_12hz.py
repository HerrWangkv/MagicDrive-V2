"""
@file   pipeline_12hz.py
@brief  Generates pedestrian-only images using MMDet3D 12Hz Interpolated Data
        Strictly aligned with NuScenes SDK scene indices.
        *FIXED*: Converts GT Boxes from Lidar Coordinates to World Coordinates.
"""
import sys
import cv2
import numpy as np
import torch
import mmengine
from tqdm import tqdm
from pathlib import Path
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from nuscenes.nuscenes import NuScenes

# Import your processor
from pedestrian_processor import PedestrianProcessor, PoseProcessor

class NuScenes12HzPipeline:
    def __init__(self, pkl_root, image_root, splits=['train', 'val'], device='cuda'):
        self.pkl_root = Path(pkl_root)
        self.image_root = Path(image_root)
        
        # 1. Init NuScenes
        print("Initializing NuScenes SDK...")
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=str(self.image_root), verbose=False)
        
        self.token_to_index = {s['token']: i for i, s in enumerate(self.nusc.scene)}
        self.scene_tokens_ordered = [s['token'] for s in self.nusc.scene]
        self.scenes = {token: [] for token in self.scene_tokens_ordered}
        
        # Pre-calculate intervals
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
        
        # 2. Load Data
        for split in splits:
            pkl_path = self.pkl_root / f"nuscenes_interp_12Hz_infos_{split}_with_bid.pkl"
            print(f"Loading {split} split from {pkl_path}...")
            if not pkl_path.exists(): continue
                
            data_dict = mmengine.load(str(pkl_path))
            raw_infos = data_dict['data_list'] if 'data_list' in data_dict else data_dict.get('infos', data_dict)
            
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
                        s_token = sc['token']; current_sc_idx = idx; found = True; break
                
                if not found:
                    for idx, sc in enumerate(self.scene_intervals):
                        if sc['start'] - 1e6 <= ts <= sc['end'] + 1e6:
                            s_token = sc['token']; current_sc_idx = idx; found = True; break
                            
                if s_token and s_token in self.scenes:
                    self.scenes[s_token].append(info)

        for token in self.scenes:
            self.scenes[token].sort(key=lambda x: x['timestamp'])
            
        print(f"Total: Populated {len(self.scenes)} scenes aligned with SDK.")
        self.processor = PedestrianProcessor(device=device)
        self.pose_processor = PoseProcessor()

    def project_lidar_to_img(self, box_3d, lidar2img):
        center = box_3d[:3]
        size = box_3d[3:6] 
        heading = box_3d[6]
        quat = Quaternion(axis=[0, 0, 1], radians=heading)
        box = Box(center, size, quat) 
        corners_3d = box.corners()
        corners_hom = np.vstack((corners_3d, np.ones((1, 8))))
        corners_img_hom = lidar2img @ corners_hom
        if np.any(corners_img_hom[2, :] <= 0): return None 
        corners_img = corners_img_hom[:2, :] / corners_img_hom[2, :]
        return np.array([np.min(corners_img[0]), np.min(corners_img[1]), np.max(corners_img[0]), np.max(corners_img[1])])

    def run(self, save_root, scene_idx=None):
        save_root = Path(save_root)
        save_root.mkdir(exist_ok=True, parents=True)
        
        if scene_idx is not None:
            if 0 <= scene_idx < len(self.scene_tokens_ordered):
                target_token = self.scene_tokens_ordered[scene_idx]
                scene_name = self.nusc.scene[scene_idx]['name']
                print(f"Processing Scene #{scene_idx}: {scene_name} ({target_token})")
                frames = self.scenes[target_token]
                if not frames: return
                scene_list = [(target_token, frames)]
            else: return
        else:
            scene_list = [(t, self.scenes[t]) for t in self.scene_tokens_ordered]

        for scene_token, frames in tqdm(scene_list, desc="Scenes"):
            if not frames: continue
            
            scene_textures = {}
            smpl_cache = {}
            gt_center_cache = {} # Key: (f_idx, cam_name, inst_tok) -> center_world

            print(f"  Pass 1: Harvesting Textures from {len(frames)} frames...")
            
            for f_idx, info in enumerate(tqdm(frames, desc="Pass 1", leave=False)):
                cameras = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
                if 'cams' in info: cam_dict = info['cams']; use_cams=True
                elif 'images' in info: cam_dict = info['images']; use_cams=False
                else: continue

                # --- 1. Compute Lidar to World Transform (L2W) ---
                # P_world = E2G @ L2E @ P_lidar
                R_l2e = Quaternion(info['lidar2ego_rotation']).rotation_matrix
                t_l2e = np.array(info['lidar2ego_translation']).reshape(3, 1)
                L2E = np.eye(4); L2E[:3, :3] = R_l2e; L2E[:3, 3] = t_l2e.flatten()

                if 'ego2global_rotation' in info:
                    R_e2g = Quaternion(info['ego2global_rotation']).rotation_matrix
                    t_e2g = np.array(info['ego2global_translation']).reshape(3, 1)
                    E2G = np.eye(4); E2G[:3, :3] = R_e2g; E2G[:3, 3] = t_e2g.flatten()
                    L2W = E2G @ L2E
                else:
                    L2W = L2E # Fallback

                for cam_name in cameras:
                    if cam_name not in cam_dict: continue
                    cam_info = cam_dict[cam_name]
                    
                    if use_cams:
                        raw = cam_info['data_path']
                        rel = raw.split('nuscenes/')[-1] if 'nuscenes/' in raw else raw
                        img_path = self.image_root / rel
                    else:
                        img_path = self.image_root / cam_info['img_path']
                    
                    if not img_path.exists(): continue
                    image = cv2.imread(str(img_path))
                    if image is None: continue
                    H, W = image.shape[:2]

                    if use_cams:
                        R_s2e = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix
                        t_s2e = np.array(cam_info['sensor2ego_translation'])
                        R_l2e = Quaternion(info['lidar2ego_rotation']).rotation_matrix
                        t_l2e = np.array(info['lidar2ego_translation'])
                        R_l2s = R_s2e.T @ R_l2e
                        t_l2s = R_s2e.T @ (t_l2e - t_s2e)
                        view = np.eye(4); view[:3, :3] = R_l2s; view[:3, 3] = t_l2s
                        K = np.eye(4); K[:3,:3]=np.array(cam_info['camera_intrinsics'])
                        lidar2img = (K@view)[:3,:]
                    else:
                        lidar2img = np.array(cam_info['lidar2img'])
                    
                    global_mask = self.processor.get_global_human_mask(image)
                    ped_data = [] 
                    
                    def is_ped(lbl, nm=None):
                        return (nm and 'pedestrian' in nm) or (lbl == 6)

                    if 'gt_boxes' in info:
                        for i, b in enumerate(info['gt_boxes']):
                            if is_ped(-1, info['gt_names'][i]):
                                res = self.project_lidar_to_img(b[:7], lidar2img)
                                if res is not None: 
                                    inst_tok = info.get('gt_box_ids', [None]*len(info['gt_boxes']))[i]
                                    if inst_tok is None: raise ValueError("Missing instance token")
                                    
                                    # --- CONVERT LIDAR -> WORLD ---
                                    center_lidar = b[:3]
                                    center_hom = np.append(center_lidar, 1.0)
                                    center_world = (L2W @ center_hom)[:3]
                                    
                                    ped_data.append((res, inst_tok, center_world))
                    elif 'instances' in info:
                        for inst in info['instances']:
                            if is_ped(inst['bbox_label']):
                                res = self.project_lidar_to_img(np.array(inst['bbox_3d']), lidar2img)
                                if res is not None: 
                                    # MMDetection3D instances are typically in Lidar coords too
                                    center_lidar = np.array(inst['bbox_3d'])[:3]
                                    center_hom = np.append(center_lidar, 1.0)
                                    center_world = (L2W @ center_hom)[:3]
                                    ped_data.append((res, inst['instance_token'], center_world))
                    
                    if not ped_data: continue
                    
                    smpl_outputs = []
                    ped_ids = []
                    valid_ped_data = []
                    
                    for bbox, inst_tok, center_world in ped_data:
                        x1, y1, x2, y2 = bbox
                        cx1, cy1 = max(0, x1), max(0, y1)
                        cx2, cy2 = min(W, x2), min(H, y2)
                        
                        if (cx2 - cx1) < 10 or (cy2 - cy1) < 20: continue
                        
                        smpl = self.processor.estimate_smpl(image, bbox)
                        if not self.processor.is_mesh_valid(smpl): continue

                        smpl_outputs.append(smpl)
                        ped_ids.append(len(valid_ped_data) + 1)
                        valid_ped_data.append((inst_tok, smpl))
                        
                        smpl_cache[(f_idx, cam_name, inst_tok)] = smpl
                        gt_center_cache[(f_idx, cam_name, inst_tok)] = center_world
                            
                    if not smpl_outputs: continue
                    
                    id_map, depth_map = self.processor.render_instance_id_map(smpl_outputs, ped_ids, (H, W))
                    
                    for i, (inst_tok, smpl) in enumerate(valid_ped_data):
                        current_id = ped_ids[i]
                        if inst_tok not in scene_textures:
                            scene_textures[inst_tok] = {'sum': np.zeros((6890, 3), np.float32), 'count': np.zeros((6890, 1), np.float32)}
                            
                        v_colors, v_weights = self.processor.project_and_sample_vertices(smpl, image, global_mask, id_map, depth_map, current_id)
                        scene_textures[inst_tok]['sum'] += v_colors
                        scene_textures[inst_tok]['count'] += v_weights

            print(f"  Smoothing poses for {len(scene_textures)} pedestrians...")
            all_c2ws = {}; all_intrinsics = {}
            
            for f_idx, info in enumerate(frames):
                all_c2ws[f_idx] = {}; all_intrinsics[f_idx] = {}
                if 'cams' in info: cam_dict = info['cams']
                elif 'images' in info: cam_dict = info['images']
                else: continue
                
                for cname, cinfo in cam_dict.items():
                    R_s2e = Quaternion(cinfo['sensor2ego_rotation']).rotation_matrix
                    t_s2e = np.array(cinfo['sensor2ego_translation']).reshape(3, 1)
                    S2E = np.eye(4); S2E[:3, :3] = R_s2e; S2E[:3, 3] = t_s2e.flatten()
                    
                    if 'ego2global_rotation' in info:
                        R_e2g = Quaternion(info['ego2global_rotation']).rotation_matrix
                        t_e2g = np.array(info['ego2global_translation']).reshape(3, 1)
                        E2G = np.eye(4); E2G[:3, :3] = R_e2g; E2G[:3, 3] = t_e2g.flatten()
                        C2W = E2G @ S2E
                    else: C2W = np.eye(4)
                    all_c2ws[f_idx][cname] = C2W
                    
                    if 'cam_intrinsic' in cinfo: K = np.array(cinfo['cam_intrinsic'])
                    elif 'intrinsic' in cinfo: K = np.array(cinfo['intrinsic'])
                    elif 'intrinsics' in cinfo: K = np.array(cinfo['intrinsics'])
                    else: K = np.array(cinfo['camera_intrinsics'])
                    all_intrinsics[f_idx][cname] = K

            sparse_poses = {}
            for (f_idx, cam_name, inst_tok), smpl_out in smpl_cache.items():
                key = inst_tok
                if key not in sparse_poses:
                    sparse_poses[key] = {'frame_indices': [], 'pose': [], 'betas': [], 'cam': [], 'tform': []}
                
                root_orient = smpl_out['global_orient'].squeeze(0)
                body_pose = smpl_out['smpl_pose'].squeeze(0)
                full_pose = torch.cat([root_orient, body_pose], dim=0).cpu().numpy()
                betas = smpl_out['betas'].squeeze(0).cpu().numpy()
                
                # --- FIX: USE PRE-CALCULATED WORLD CENTER ---
                center_world = gt_center_cache[(f_idx, cam_name, inst_tok)]
                pos_world = center_world.copy()
                pos_world[2] -= 0.1 # Shift down to pelvis
                
                tform = smpl_out['crop_info']['tform']
                C2W = all_c2ws[f_idx][cam_name]
                
                R_c2w = C2W[:3, :3]
                full_pose[0] = R_c2w @ full_pose[0] # Cam Rot -> World Rot
                
                sparse_poses[key]['frame_indices'].append(f_idx)
                sparse_poses[key]['pose'].append(full_pose)
                sparse_poses[key]['betas'].append(betas)
                sparse_poses[key]['cam'].append(pos_world)
                sparse_poses[key]['tform'].append(tform)

            smoothed_results = {}
            for inst_tok, data in sparse_poses.items():
                indices = np.array(data['frame_indices'])
                sort_idx = np.argsort(indices)
                for k in data: data[k] = np.array(data[k])[sort_idx]
                
                smoothed = self.pose_processor.process_sequence(data, len(frames))
                if smoothed: smoothed_results[inst_tok] = smoothed

            print(f"  Inpainting textures...")
            final_textures = {}
            for inst_tok, data in scene_textures.items():
                if (data['count'] > 0).sum() / 6890.0 < 0.1: continue
                final_textures[inst_tok] = self.processor.inpaint_missing_colors(data['sum'], data['count'])

            print(f"  Pass 2: Rendering...")            
            for f_idx, info in enumerate(tqdm(frames, desc="Pass 2", leave=False)):
                cameras = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
                if 'cams' in info: cam_dict = info['cams']; use_cams=True
                else: cam_dict = info['images']; use_cams=False

                for cam_name in cameras:
                    if cam_name not in cam_dict: continue
                    cam_info = cam_dict[cam_name]
                    
                    if use_cams:
                        raw = cam_info['data_path']
                        rel = raw.split('nuscenes/')[-1] if 'nuscenes/' in raw else raw
                        img_path = self.image_root / rel
                    else:
                        img_path = self.image_root / cam_info['img_path']
                    
                    if not img_path.exists(): continue
                    ref_image = cv2.imread(str(img_path))
                    if ref_image is None: continue
                    H, W = ref_image.shape[:2]
                    
                    final_canvas = np.zeros((H, W, 3), dtype=np.uint8)
                    global_depth = np.full((H, W), np.inf, dtype=np.float32)
                    render_list = []
                    
                    for inst_tok, texture in final_textures.items():
                        if inst_tok in smoothed_results:
                            smoothed = smoothed_results[inst_tok]
                            min_f, max_f = smoothed['valid_range']
                            if f_idx < min_f or f_idx > max_f: continue

                            pose_world = smoothed['pose'][f_idx]
                            betas = smoothed['betas'][f_idx]
                            pos_world = smoothed['cam'][f_idx]
                            
                            if f_idx not in all_c2ws or cam_name not in all_c2ws[f_idx]: continue
                            C2W = all_c2ws[f_idx][cam_name]
                            K = all_intrinsics[f_idx][cam_name]
                            
                            R_c2w = C2W[:3, :3]; T_c2w = C2W[:3, 3]; R_w2c = R_c2w.T
                            pos_cam = R_w2c @ (pos_world - T_c2w)
                            
                            if pos_cam[2] < 0.5: 
                                    continue 
                                
                            f_x = K[0, 0]; f_y = K[1, 1]
                            c_x = K[0, 2]; c_y = K[1, 2]
                            u_img = f_x * pos_cam[0] / pos_cam[2] + c_x
                            v_img = f_y * pos_cam[1] / pos_cam[2] + c_y
                            
                            pixel_height = f_x * 2.0 / pos_cam[2]
                            bbox_size = pixel_height / 0.8
                            
                            # Calculate the 2D box edges on the screen
                            min_x = u_img - bbox_size / 2
                            max_x = u_img + bbox_size / 2
                            min_y = v_img - bbox_size / 2
                            max_y = v_img + bbox_size / 2
                            
                            # INTERSECTION CHECK:
                            # Does this box touch the image canvas [0, W] x [0, H]?
                            # If it's completely to the left, right, above, or below, SKIP it.
                            if max_x < 0 or min_x > W or max_y < 0 or min_y > H:
                                continue 
                                
                            # If we pass this check, the person is at least partially visible.
                            # Clamp size to prevent memory issues with outlier glitches
                            bbox_size = min(bbox_size, max(H, W) * 2.0)

                            src_pts = np.array([[u_img - bbox_size/2, v_img - bbox_size/2], [u_img - bbox_size/2, v_img + bbox_size/2], [u_img + bbox_size/2, v_img - bbox_size/2]], dtype=np.float32)
                            dst_pts = np.array([[0, 0], [0, 255], [255, 0]], dtype=np.float32)
                            tform = cv2.getAffineTransform(src_pts, dst_pts)
                            
                            cam_t_crop = self.processor.convert_world_to_crop_cam(pos_world, {'tform': tform}, K, C2W)
                            z_crop = cam_t_crop[2]
                            z_real = pos_cam[2]
                            depth_scale = z_real / (z_crop + 1e-6)
                            root_rot_cam = R_w2c @ pose_world[0]
                            
                            smpl_params = {
                                'global_orient': torch.tensor(root_rot_cam, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.processor.device),
                                'body_pose': torch.tensor(pose_world[1:], dtype=torch.float32).unsqueeze(0).to(self.processor.device),
                                'betas': torch.tensor(betas, dtype=torch.float32).unsqueeze(0).to(self.processor.device)
                            }
                            
                            cam_t_t = torch.tensor(cam_t_crop, dtype=torch.float32).unsqueeze(0).to(self.processor.device)
                            vertices = self.processor.compute_vertices(smpl_params)
                            render_list.append(({'vertices': vertices, 'cam_t': cam_t_t, 'crop_info': {'tform': tform}, 'depth_scale': depth_scale}, texture))
                    
                    if render_list:
                        for r_data, tex in render_list:
                            render, mask, depth = self.processor.render_colored_mesh(r_data, tex, (H, W))
                            scale = r_data['depth_scale']
                            real_depth = depth * scale
                            foreground_candidate = mask & (real_depth > 0)
                            
                            better_depth_mask = np.zeros_like(mask)
                            # Compare REAL depth against GLOBAL depth (which is now also in real meters)
                            better_depth_mask[foreground_candidate] = real_depth[foreground_candidate] < global_depth[foreground_candidate]
                            
                            update_indices = foreground_candidate & better_depth_mask
                            
                            final_canvas[update_indices] = render[update_indices]
                            global_depth[update_indices] = real_depth[update_indices] # Store Real Depth
                        
                        out_name = Path(img_path).name.replace('.jpg', '.png')
                        cv2.imwrite(str(Path(save_root)/out_name), final_canvas)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run NuScenes12HzPipeline for a specific scene index or range.")
    parser.add_argument('--image_root', type=str, default="data/nuscenes", help='Root directory for images (NuScenes dataroot)')
    parser.add_argument('--pkl_root', type=str, default="data/nuscenes_mmdet3d-12Hz", help='Root directory for PKL files')
    parser.add_argument('--save_root', type=str, default="output_12hz_aligned", help='Directory to save outputs')
    parser.add_argument('--scene_idx', type=int, default=None, help='Scene index to process (default: all)')
    parser.add_argument('--scene_idx_start', type=int, default=None, help='Start scene index (inclusive) for range processing')
    parser.add_argument('--scene_idx_end', type=int, default=None, help='End scene index (inclusive) for range processing')
    parser.add_argument('--splits', type=str, default='train,val', help='Comma-separated list of splits (default: train,val)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (default: cuda)')

    args = parser.parse_args()
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    pipeline = NuScenes12HzPipeline(args.pkl_root, args.image_root, splits=splits, device=args.device)

    # If a range is specified, process that range
    if args.scene_idx_start is not None and args.scene_idx_end is not None:
        for idx in range(args.scene_idx_start, args.scene_idx_end + 1):
            print(f"\n==== Processing scene_idx {idx} ====")
            pipeline.run(args.save_root, scene_idx=idx)
    elif args.scene_idx is not None:
        pipeline.run(args.save_root, scene_idx=args.scene_idx)
    else:
        pipeline.run(args.save_root, scene_idx=None)