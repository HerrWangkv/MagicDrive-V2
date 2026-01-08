import os
# Set EGL platform for pyrender before importing it (or libraries that use it)
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Mock pyrender to avoid EGL errors if we don't need visualization
from unittest.mock import MagicMock
import sys
sys.modules["pyrender"] = MagicMock()
sys.modules["pyrender.OffscreenRenderer"] = MagicMock()

import torch
import numpy as np
import cv2
from pathlib import Path
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, PerspectiveCameras,
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    TexturesVertex, PointLights, HardPhongShader
)
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import gaussian_filter1d

# HMR2 Imports
from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT

# SegFormer Imports
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class UnlitShader(torch.nn.Module):
    def __init__(self, device="cpu", **kwargs):
        super().__init__()
        
    def forward(self, fragments, meshes, **kwargs):
        # Sample textures (interpolated vertex colors)
        texels = meshes.sample_textures(fragments) # (N, H, W, K, C)
        rgb = texels[..., 0, :] # (N, H, W, C) assuming K=1
        
        # Alpha from mask
        mask = fragments.pix_to_face[..., 0] >= 0 # (N, H, W)
        alpha = mask.float().unsqueeze(-1)
        
        return torch.cat([rgb, alpha], dim=-1)

class PedestrianProcessor:
    def __init__(self, device='cuda'):
        self.device = device
        
        # 1. Load HMR2
        print(f"Loading HMR2...")
        hmr2_checkpoint = "pedestrian_checkpoints/hmr2_data/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"
        self.hmr2_model, _ = load_hmr2(hmr2_checkpoint)
        self.hmr2_model = self.hmr2_model.to(device)
        self.hmr2_model.eval()

        # 2. Load SegFormer
        print(f"Loading SegFormer...")
        segformer_path = "pedestrian_checkpoints/segformer"
        
        # Check if the directory exists AND contains model weights
        has_weights = False
        if os.path.exists(segformer_path):
            for fname in os.listdir(segformer_path):
                if fname in ["pytorch_model.bin", "model.safetensors", "tf_model.h5"]:
                    has_weights = True
                    break
        
        if not has_weights:
            print(f"Local SegFormer weights not found in {segformer_path}, downloading from Hub...")
            segformer_path = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
        else:
            print(f"Loading local SegFormer from {segformer_path}...")
            
        self.seg_processor = SegformerImageProcessor.from_pretrained(segformer_path)
        self.seg_model = SegformerForSemanticSegmentation.from_pretrained(segformer_path, use_safetensors=True)
        self.seg_model.to(device)
        self.seg_model.eval()
        
        # 3. Setup Renderer Basics
        # We don't init the full renderer here because cameras change per person
        self.raster_settings = RasterizationSettings(
            image_size=256, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
            bin_size=0,
        )
        self.lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

        # 4. Compute SMPL Symmetry Indices
        print("Computing SMPL symmetry indices...")
        template_verts = self.hmr2_model.smpl.v_template.cpu().numpy() # (1, 6890, 3)
        if template_verts.ndim == 3: template_verts = template_verts[0]
        
        # Flip X coordinate
        flipped_verts = template_verts.copy()
        flipped_verts[:, 0] *= -1
        
        # Find nearest neighbor for each flipped vertex
        # This maps index i to the index of its symmetric counterpart
        tree = cKDTree(template_verts)
        _, self.symmetry_idx = tree.query(flipped_verts, k=1)

    def get_global_human_mask(self, image_bgr):
        """
        Run SegFormer on the full image and return a boolean mask for 'Person' (Class 11).
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.seg_processor(images=image_rgb, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.seg_model(**inputs)
            logits = outputs.logits  # (B, NumClasses, H/4, W/4)
            
        # Upsample logits to original image size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image_rgb.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        
        pred_seg = upsampled_logits.argmax(dim=1)[0] # (H, W)
        
        # Class 11 is Person in Cityscapes
        person_mask = (pred_seg == 11).cpu().numpy().astype(bool)
        return person_mask

    def estimate_smpl(self, image, bbox):
        """
        Run HMR2 to get SMPL parameters.
        """
        # 1. Get Center and Scale
        center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        scale = max(width, height) / 200.0
        
        # 2. Crop and Resize to 256x256
        img_h, img_w = image.shape[:2]
        src_w = scale * 200.0
        src_h = scale * 200.0
        
        src_pts = np.array([
            [center[0] - src_w/2, center[1] - src_h/2],
            [center[0] - src_w/2, center[1] + src_h/2],
            [center[0] + src_w/2, center[1] - src_h/2],
        ], dtype=np.float32)
        
        dst_size = 256
        dst_pts = np.array([
            [0, 0], [0, dst_size - 1], [dst_size - 1, 0],
        ], dtype=np.float32)
        
        tform = cv2.getAffineTransform(src_pts, dst_pts)
        img_crop = cv2.warpAffine(image, tform, (dst_size, dst_size), flags=cv2.INTER_LINEAR)
        
        # 3. Normalize
        img_crop = img_crop.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_crop = (img_crop - mean) / std
        img_crop = img_crop.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img_crop).float().unsqueeze(0).to(self.device)
        
        # 4. Run Model
        with torch.no_grad():
            batch = {'img': img_tensor}
            out = self.hmr2_model(batch)
        
        return {
            'vertices': out['pred_vertices'], 
            'cam_t': out['pred_cam_t'],
            'smpl_pose': out['pred_smpl_params']['body_pose'], # (1, 23, 3, 3)
            'global_orient': out['pred_smpl_params']['global_orient'], # (1, 1, 3, 3)
            'betas': out['pred_smpl_params']['betas'], # (1, 10)
            'crop_info': {'tform': tform},
            'bbox_height': height
        }

    def compute_vertices(self, smpl_params):
        """
        Run SMPL layer to get vertices from params.
        smpl_params: dict with global_orient, body_pose, betas
        """
        # smpl_params tensors should be on device
        # Ensure shapes are correct for SMPL call
        # global_orient: (N, 1, 3, 3)
        # body_pose: (N, 23, 3, 3)
        # betas: (N, 10)
        
        output = self.hmr2_model.smpl(
            global_orient=smpl_params['global_orient'],
            body_pose=smpl_params['body_pose'],
            betas=smpl_params['betas'],
            pose2rot=False # We are passing rotation matrices
        )
        return output.vertices # (N, 6890, 3)

    def render_smpl(self, smpl_data, image_shape):
        """
        Legacy method for single-pass rendering (Gray).
        """
        # Just call render_colored_mesh with gray colors and lighting
        V = smpl_data['vertices'].shape[1]
        gray_colors = np.ones((V, 3), dtype=np.float32) * 0.6
        return self.render_colored_mesh(smpl_data, gray_colors, image_shape, use_lighting=True)

    def render_instance_id_map(self, smpl_outputs, ped_ids, image_shape):
        """
        Render an ID map for occlusion handling.
        """
        if not smpl_outputs:
            return np.zeros(image_shape[:2], dtype=np.int32), np.full(image_shape[:2], np.inf, dtype=np.float32)

        H, W = image_shape[:2]
        
        # 1. Collect all vertices and faces
        faces_template = self.hmr2_model.smpl.faces.astype(np.int64)
        faces_tensor = torch.from_numpy(faces_template).to(self.device)
        
        # Camera setup (Same as render_smpl)
        focal_length_ndc = 5000.0 / (256.0 / 2.0)
        R_fix = torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], device=self.device)
        
        full_id_map = np.zeros((H, W), dtype=np.int32)
        full_depth_map = np.full((H, W), np.inf, dtype=np.float32)
        
        for i, (smpl_out, pid) in enumerate(zip(smpl_outputs, ped_ids)):
            # Render depth for this person
            vertices = smpl_out['vertices']
            cam_t = smpl_out['cam_t']
            crop_info = smpl_out['crop_info']
            
            cameras = FoVPerspectiveCameras(
                device=self.device,
                R=R_fix.unsqueeze(0),
                T=torch.zeros_like(cam_t), 
                fov=2.0 * np.arctan(1.0 / focal_length_ndc) * 180.0 / np.pi
            )
            
            verts_cam = vertices + cam_t.unsqueeze(1)
            
            # Create mesh
            mesh = Meshes(verts=verts_cam, faces=faces_tensor.unsqueeze(0))
            
            # Rasterize to get fragments (depth)
            raster_settings = RasterizationSettings(
                image_size=256, blur_radius=0.0, faces_per_pixel=1, bin_size=0
            )
            rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
            fragments = rasterizer(mesh)
            depth_crop = fragments.zbuf[0, ..., 0].cpu().numpy() # (256, 256)
            mask_crop = fragments.pix_to_face[0, ..., 0].cpu().numpy() >= 0
            
            # Warp back
            tform = crop_info['tform']
            tform_inv = cv2.invertAffineTransform(tform)
            
            # Warp depth (nearest neighbor to preserve values)
            depth_full = cv2.warpAffine(
                depth_crop, tform_inv, (W, H), 
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=np.inf
            )
            mask_full = cv2.warpAffine(
                mask_crop.astype(np.uint8), tform_inv, (W, H), 
                flags=cv2.INTER_NEAREST
            ).astype(bool)
            
            # Update global buffers
            # Where this person is visible AND closer than current buffer
            update_mask = mask_full & (depth_full < full_depth_map)
            
            full_depth_map[update_mask] = depth_full[update_mask]
            full_id_map[update_mask] = pid
            
        return full_id_map, full_depth_map

    def project_and_sample_vertices(self, smpl_out, image, seg_mask, id_map, depth_map, current_id):
        """
        Project vertices to image, filter by mask and ID, and sample colors.
        FIXED: Better color sampling with bilinear interpolation and proper weighting.
        """
        vertices = smpl_out['vertices'][0] # (V, 3)
        cam_t = smpl_out['cam_t'][0]       # (3,)
        crop_info = smpl_out['crop_info']
        tform = crop_info['tform']         # (2, 3)
        
        H, W = image.shape[:2]
        
        # 1. Project to Crop Coordinates (256x256)
        focal_length_px = 5000.0 
        center_px = 128.0
        
        # Vertices in camera frame
        v_cam = vertices + cam_t
        x = v_cam[:, 0]
        y = v_cam[:, 1]
        z = v_cam[:, 2]
        
        # Perspective projection
        u = focal_length_px * x / z + center_px
        v = focal_length_px * y / z + center_px
        
        # 2. Transform to Full Image Coordinates
        tform_inv = cv2.invertAffineTransform(tform)
        
        # Stack to (N, 3)
        ones = torch.ones_like(u)
        uv_hom = torch.stack([u, v, ones], dim=1).cpu().numpy().T # (3, V)
        
        uv_full = tform_inv @ uv_hom # (2, V)
        u_full = uv_full[0, :]
        v_full = uv_full[1, :]
        
        # 3. Filter
        u_int = np.round(u_full).astype(np.int32)
        v_int = np.round(v_full).astype(np.int32)
        
        valid_mask = (u_int >= 1) & (u_int < W-1) & (v_int >= 1) & (v_int < H-1)  # Keep 1px margin for bilinear
        
        # Filter by SegMask and ID Map
        final_mask = np.zeros(vertices.shape[0], dtype=bool)
        
        # Only check indices that are inside image bounds
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 0:
            us = u_int[valid_indices]
            vs = v_int[valid_indices]
            
            # Check SegMask (is Person?)
            is_person = seg_mask[vs, us]
            
            # Check ID Map (is Visible/Not Occluded?)
            is_visible_id = (id_map[vs, us] == current_id) | (id_map[vs, us] == 0)
            
            # Check Depth (Self-Occlusion)
            z_vertex = z[valid_indices].cpu().numpy()
            z_buffer = depth_map[vs, us]
            depth_threshold = 0.05 # 5cm tolerance
            is_visible_depth = (z_vertex - z_buffer) < depth_threshold

            keep = is_person & is_visible_id & is_visible_depth
            final_mask[valid_indices[keep]] = True
            
        # 4. Sample Colors with BILINEAR INTERPOLATION
        vertex_colors = np.zeros((vertices.shape[0], 3), dtype=np.float32)
        vertex_weights = np.zeros((vertices.shape[0], 1), dtype=np.float32)
        
        if np.any(final_mask):
            # Get sub-pixel coordinates
            u_subpix = u_full[final_mask]
            v_subpix = v_full[final_mask]
            
            # Bilinear interpolation
            u0 = np.floor(u_subpix).astype(np.int32)
            v0 = np.floor(v_subpix).astype(np.int32)
            u1 = u0 + 1
            v1 = v0 + 1
            
            # Clip to valid range
            u0 = np.clip(u0, 0, W-1)
            u1 = np.clip(u1, 0, W-1)
            v0 = np.clip(v0, 0, H-1)
            v1 = np.clip(v1, 0, H-1)
            
            # Compute interpolation weights
            wu = u_subpix - u0
            wv = v_subpix - v0
            wu = np.clip(wu, 0, 1)
            wv = np.clip(wv, 0, 1)
            
            # Sample 4 corners (BGR -> RGB)
            img_rgb = image[:, :, ::-1].astype(np.float32) / 255.0
            
            c00 = img_rgb[v0, u0]  # top-left
            c01 = img_rgb[v0, u1]  # top-right
            c10 = img_rgb[v1, u0]  # bottom-left
            c11 = img_rgb[v1, u1]  # bottom-right
            
            # Bilinear interpolation
            wu = wu[:, np.newaxis]
            wv = wv[:, np.newaxis]
            colors = (c00 * (1 - wu) * (1 - wv) +
                    c01 * wu * (1 - wv) +
                    c10 * (1 - wu) * wv +
                    c11 * wu * wv)
            
            # Calculate Weight based on BBox Height (Resolution)
            # FIXED: Use height^2 instead of height^4 (less extreme)
            # FIXED: Add minimum resolution threshold
            bbox_height = max(smpl_out.get('bbox_height', 100.0), 50.0)  # At least 50px
            w = bbox_height ** 2
            
            # FIXED: Don't pre-multiply colors by weight
            # Store raw weighted sum and weights separately
            num_visible = np.sum(final_mask)
            weights = np.full((num_visible, 1), w, dtype=np.float32)
            
            vertex_colors[final_mask] = colors * weights  # Weighted sum
            vertex_weights[final_mask] = weights
            
        return vertex_colors, vertex_weights


    def inpaint_missing_colors(self, vertex_sums, vertex_counts):
        """
        Fill missing colors using Symmetry then KNN.
        FIXED: Added median filtering for outliers.
        """
        # 1. Basic Average
        counts_safe = vertex_counts.copy()
        counts_safe[counts_safe == 0] = 1.0
        avg_colors = vertex_sums / counts_safe
        
        valid_mask = (vertex_counts[:, 0] > 0)
        missing_mask = ~valid_mask
        
        if not np.any(valid_mask):
            return np.ones_like(avg_colors) * 0.5
        if not np.any(missing_mask):
            # FIXED: Apply median filter to remove color outliers
            return self._median_filter_colors(avg_colors, valid_mask)

        # 2. Symmetry Inpainting
        missing_indices = np.where(missing_mask)[0]
        sym_indices = self.symmetry_idx[missing_indices]
        sym_valid = valid_mask[sym_indices]
        
        fill_indices = missing_indices[sym_valid]
        source_indices = sym_indices[sym_valid]
        
        if len(fill_indices) > 0:
            avg_colors[fill_indices] = avg_colors[source_indices]
            valid_mask[fill_indices] = True
            missing_mask[fill_indices] = False
        
        if not np.any(missing_mask):
            return self._median_filter_colors(avg_colors, valid_mask)

        # 3. KNN Inpainting for remaining holes
        template_verts = self.hmr2_model.smpl.v_template.cpu().numpy()
        if template_verts.ndim == 3: template_verts = template_verts[0]
        
        valid_verts = template_verts[valid_mask]
        valid_colors = avg_colors[valid_mask]
        
        remaining_missing_indices = np.where(missing_mask)[0]
        missing_verts = template_verts[remaining_missing_indices]
        
        tree = cKDTree(valid_verts)
        dists, idxs = tree.query(missing_verts, k=3)
        
        neighbor_colors = valid_colors[idxs] # (N_missing, 3, 3)
        filled_colors = np.mean(neighbor_colors, axis=1)
        
        avg_colors[remaining_missing_indices] = filled_colors
        
        # FIXED: Apply median filter to final result
        full_valid_mask = np.ones(len(avg_colors), dtype=bool)
        return self._median_filter_colors(avg_colors, full_valid_mask)


    def _median_filter_colors(self, colors, valid_mask):
        """
        Apply spatial median filtering to remove color outliers.
        Uses mesh connectivity for spatial neighborhood.
        """
        if np.sum(valid_mask) < 10:
            return colors
        
        # Get mesh faces
        faces = self.hmr2_model.smpl.faces
        
        # Build vertex neighbor list
        neighbors = [set() for _ in range(len(colors))]
        for face in faces:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        neighbors[face[i]].add(face[j])
        
        # Apply median filter only to valid vertices
        filtered_colors = colors.copy()
        
        for i in np.where(valid_mask)[0]:
            if len(neighbors[i]) < 3:
                continue
            
            # Get neighbor colors
            neighbor_list = list(neighbors[i])
            neighbor_colors = colors[neighbor_list]
            neighbor_valid = valid_mask[neighbor_list]
            
            if np.sum(neighbor_valid) >= 3:
                valid_neighbor_colors = neighbor_colors[neighbor_valid]
                # Include self
                all_colors = np.vstack([colors[i:i+1], valid_neighbor_colors])
                # Take median for each channel
                filtered_colors[i] = np.median(all_colors, axis=0)
        
        return filtered_colors

    def render_colored_mesh(self, smpl_out, vertex_colors, image_shape, use_lighting=False, intrinsics=None):
        vertices = smpl_out['vertices'] # (1, V, 3)
        H, W = image_shape[:2]
        
        # === FIX: USE TRAJECTORY POSITION ===
        # If using Real Intrinsics, we must use the Real Position (pos_cam),
        # not the Crop Position (cam_t).
        if intrinsics is not None and 'pos_cam' in smpl_out:
            T_mesh = smpl_out['pos_cam'] # (3,) Tensor
            
            # Use Real Intrinsics
            f_x, f_y = intrinsics[0, 0], intrinsics[1, 1]
            c_x, c_y = intrinsics[0, 2], intrinsics[1, 2]
            
            # Recalculate ROI using the real projection
            # We project the vertices roughly to guess bounds
            # Or use the crop info as a heuristic
            tform = smpl_out['crop_info']['tform']
            tform_inv = cv2.invertAffineTransform(tform)
            corners_crop = np.array([[0, 0], [256, 0], [256, 256], [0, 256]], dtype=np.float32)
            corners_crop_hom = np.hstack([corners_crop, np.ones((4, 1))])
            corners_full = (tform_inv @ corners_crop_hom.T).T
            min_x, max_x = np.min(corners_full[:, 0]), np.max(corners_full[:, 0])
            min_y, max_y = np.min(corners_full[:, 1]), np.max(corners_full[:, 1])
            
        else:
            # Fallback: Legacy HMR Crop Logic
            T_mesh = smpl_out['cam_t'][0] # (3,)
            
            crop_info = smpl_out['crop_info']
            tform = crop_info['tform']
            tform_inv = cv2.invertAffineTransform(tform)
            s_x, s_y = tform_inv[0, 0], tform_inv[1, 1]
            t_x, t_y = tform_inv[0, 2], tform_inv[1, 2]
            f_crop, c_crop = 5000.0, 128.0
            
            f_x, f_y = s_x * f_crop, s_y * f_crop
            c_x, c_y = s_x * c_crop + t_x, s_y * c_crop + t_y
            
            corners_crop = np.array([[0, 0], [256, 0], [256, 256], [0, 256]], dtype=np.float32)
            corners_crop_hom = np.hstack([corners_crop, np.ones((4, 1))])
            corners_full = (tform_inv @ corners_crop_hom.T).T
            min_x, max_x = np.min(corners_full[:, 0]), np.max(corners_full[:, 0])
            min_y, max_y = np.min(corners_full[:, 1]), np.max(corners_full[:, 1])

        # === APPLY TRANSLATION ===
        # Move mesh to Camera Space
        if vertices.ndim == 3:
             verts_cam = vertices + T_mesh.view(1, 1, 3)
        else:
             verts_cam = vertices + T_mesh.view(1, 3)

        # ROI Padding & Bounds
        pad_x = (max_x - min_x) * 0.5
        pad_y = (max_y - min_y) * 0.5
        roi_min_x = int(max(0, min_x - pad_x))
        roi_min_y = int(max(0, min_y - pad_y))
        roi_max_x = int(min(W, max_x + pad_x))
        roi_max_y = int(min(H, max_y + pad_y))
        
        roi_w = roi_max_x - roi_min_x
        roi_h = roi_max_y - roi_min_y
        
        if roi_w <= 0 or roi_h <= 0:
             return np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W), dtype=bool), np.full((H, W), np.inf, dtype=np.float32)

        # Adjust Principal Point for ROI
        c_x_roi = c_x - roi_min_x
        c_y_roi = c_y - roi_min_y
        
        # Setup Camera
        R_fix = torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], device=self.device)
        
        cameras = PerspectiveCameras(
            device=self.device,
            focal_length=((f_x, f_y),),
            principal_point=((c_x_roi, c_y_roi),),
            image_size=((roi_h, roi_w),),
            in_ndc=False,
            R=R_fix.unsqueeze(0),
            T=torch.zeros((1, 3), device=self.device) # T is already applied to verts_cam
        )
        
        faces_tensor = torch.from_numpy(self.hmr2_model.smpl.faces.astype(np.int64)).to(self.device).unsqueeze(0)
        
        # Texture
        verts_rgb = torch.from_numpy(vertex_colors).float().to(self.device).unsqueeze(0)
        textures = TexturesVertex(verts_features=verts_rgb)
        
        mesh = Meshes(verts=verts_cam, faces=faces_tensor, textures=textures)
        
        if use_lighting:
            shader = HardPhongShader(device=self.device, cameras=cameras, lights=self.lights)
        else:
            shader = UnlitShader(device=self.device)

        # 3. Render to ROI
        raster_settings = RasterizationSettings(
            image_size=(roi_h, roi_w), 
            blur_radius=0.0, 
            faces_per_pixel=1, 
            bin_size=0,
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=shader
        )
        
        # === UPDATE: SEPARATE RASTERIZATION FOR DEPTH ===
        fragments = renderer.rasterizer(mesh)
        images = renderer.shader(fragments, mesh)
        
        render_roi = images[0, ..., :3].cpu().numpy() * 255
        render_roi = render_roi[..., ::-1] # RGB -> BGR
        mask_roi = images[0, ..., 3].cpu().numpy() > 0
        depth_roi = fragments.zbuf[0, ..., 0].cpu().numpy() # Get depth
        
        # 4. Paste ROI into Full Image
        render_full = np.zeros((H, W, 3), dtype=np.uint8)
        mask_full = np.zeros((H, W), dtype=bool)
        depth_full = np.full((H, W), np.inf, dtype=np.float32) # Init Depth with Infinity
        
        render_full[roi_min_y:roi_max_y, roi_min_x:roi_max_x] = render_roi.astype(np.uint8)
        mask_full[roi_min_y:roi_max_y, roi_min_x:roi_max_x] = mask_roi
        
        # Paste depth carefully using masks
        # Slices in depth_full
        d_slice = depth_full[roi_min_y:roi_max_y, roi_min_x:roi_max_x]
        
        # We only update depth where the object actually is (mask_roi)
        # However, since we are pasting into an empty buffer (locally), we can just paste.
        # But wait, roi size might match.
        d_slice[mask_roi] = depth_roi[mask_roi]
        
        # Note: If ROI size mismatches slightly due to int casting, standard slicing handles it.
        # If mask_roi shape != d_slice shape, standard python error. 
        # But here they are constructed from same coords.
        
        return render_full, mask_full, depth_full

    def convert_crop_cam_to_world(self, cam_t, crop_info, cam_intrinsics, c2w):
        """
        Convert HMR crop-camera translation to World Space translation.
        """
        tform = crop_info['tform']
        s_x = np.linalg.norm(tform[0, :2])
        s_y = np.linalg.norm(tform[1, :2])
        s = (s_x + s_y) / 2.0
        
        f_real = (cam_intrinsics[0, 0] + cam_intrinsics[1, 1]) / 2.0
        f_hmr = 5000.0
        
        z_crop = cam_t[2]
        z_real = z_crop * (s * f_real / f_hmr)
        
        cx_hmr = 128.0
        cy_hmr = 128.0
        u_crop = f_hmr * cam_t[0] / z_crop + cx_hmr
        v_crop = f_hmr * cam_t[1] / z_crop + cy_hmr
        
        pt_crop = np.array([u_crop, v_crop, 1.0])
        tform_inv = cv2.invertAffineTransform(tform)
        pt_img = tform_inv @ pt_crop # (2,)
        
        cx_real = cam_intrinsics[0, 2]
        cy_real = cam_intrinsics[1, 2]
        
        x_real = (pt_img[0] - cx_real) * z_real / f_real
        y_real = (pt_img[1] - cy_real) * z_real / f_real
        
        pos_cam = np.array([x_real, y_real, z_real])
        
        R_c2w = c2w[:3, :3]
        T_c2w = c2w[:3, 3]
        
        pos_world = R_c2w @ pos_cam + T_c2w
        return pos_world

    def convert_world_to_crop_cam(self, pos_world, crop_info, cam_intrinsics, c2w):
        """
        Convert World Space translation back to HMR crop-camera translation.
        """
        R_c2w = c2w[:3, :3]
        T_c2w = c2w[:3, 3]
        R_w2c = R_c2w.T
        
        pos_cam = R_w2c @ (pos_world - T_c2w)
        x_real, y_real, z_real = pos_cam
        
        if z_real <= 0.1: z_real = 0.1 
        
        f_real = (cam_intrinsics[0, 0] + cam_intrinsics[1, 1]) / 2.0
        cx_real = cam_intrinsics[0, 2]
        cy_real = cam_intrinsics[1, 2]
        
        u_img = f_real * x_real / z_real + cx_real
        v_img = f_real * y_real / z_real + cy_real
        
        tform = crop_info['tform']
        pt_img = np.array([u_img, v_img, 1.0])
        pt_crop = tform @ pt_img
        u_crop, v_crop = pt_crop
        
        s_x = np.linalg.norm(tform[0, :2])
        s_y = np.linalg.norm(tform[1, :2])
        s = (s_x + s_y) / 2.0
        f_hmr = 5000.0
        
        z_crop = z_real * (f_hmr / (s * f_real))
        
        cx_hmr = 128.0
        cy_hmr = 128.0
        
        x_crop = (u_crop - cx_hmr) * z_crop / f_hmr
        y_crop = (v_crop - cy_hmr) * z_crop / f_hmr
        
        return np.array([x_crop, y_crop, z_crop])

    def is_mesh_valid(self, smpl_out):
        """
        Check if the mesh is plausible.
        """
        vertices = smpl_out['vertices'][0] # (V, 3)
        cam_t = smpl_out['cam_t'][0]       # (3,)
        
        focal_length_px = 5000.0 
        center_px = 128.0
        
        v_cam = vertices + cam_t
        x = v_cam[:, 0]
        y = v_cam[:, 1]
        z = v_cam[:, 2]
        
        u = focal_length_px * x / z + center_px
        v = focal_length_px * y / z + center_px
        
        if (u.max() - u.min()).item() > 300 or (v.max() - v.min()).item() > 300:
            return False
            
        return True

class PoseProcessor:
    def __init__(self):
        pass

    def matrix_to_rotation_6d(self, matrix):
        batch_dim = matrix.shape[:-2]
        m = matrix.reshape(-1, 3, 3)
        r6d = np.concatenate([m[:, :, 0], m[:, :, 1]], axis=1)
        return r6d.reshape(*batch_dim, 6)

    def rotation_6d_to_matrix(self, d6):
        batch_dim = d6.shape[:-1]
        d6 = d6.reshape(-1, 6)
        a1 = d6[:, :3]
        a2 = d6[:, 3:]
        
        b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-8)
        b2 = a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1
        b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8)
        b3 = np.cross(b1, b2)
        
        matrix = np.stack((b1, b2, b3), axis=-1)
        return matrix.reshape(*batch_dim, 3, 3)
    
    def correct_outliers_with_trend(self, pose_mat, cam, window_size=5, thresh_trans=0.5, thresh_rot=0.5):
        """
        Calculates a robust trend (median filter) for Position and Root Rotation.
        Overwrites outliers with the trend value to prevent glitches.
        """
        n = len(cam)
        if n < 5: return pose_mat, cam

        if window_size % 2 == 0: window_size += 1
        pad_w = window_size // 2

        # 1. Position Trend (Median Filter)
        cam_pad = np.pad(cam, ((pad_w, pad_w), (0, 0)), mode='edge')
        cam_trend = np.zeros_like(cam)
        for i in range(3):
            cam_trend[:, i] = medfilt(cam_pad[:, i], kernel_size=window_size)[pad_w:-pad_w]

        # 2. Root Rotation Trend (6D Median Filter)
        root_rot = pose_mat[:, 0] # (N, 3, 3)
        root_6d = self.matrix_to_rotation_6d(root_rot.reshape(n, 1, 3, 3)).reshape(n, 6)
        
        root_pad = np.pad(root_6d, ((pad_w, pad_w), (0, 0)), mode='edge')
        root_trend_6d = np.zeros_like(root_6d)
        for i in range(6):
            root_trend_6d[:, i] = medfilt(root_pad[:, i], kernel_size=window_size)[pad_w:-pad_w]

        # 3. Overwrite Outliers
        # Position
        dist_cam = np.linalg.norm(cam - cam_trend, axis=1)
        bad_cam_mask = dist_cam > thresh_trans
        if np.any(bad_cam_mask):
            cam[bad_cam_mask] = cam_trend[bad_cam_mask]
            
        # Rotation
        dist_rot = np.linalg.norm(root_6d - root_trend_6d, axis=1)
        bad_rot_mask = dist_rot > thresh_rot
        if np.any(bad_rot_mask):
            fixed_roots = self.rotation_6d_to_matrix(root_trend_6d[bad_rot_mask])
            pose_mat[bad_rot_mask, 0] = fixed_roots

        return pose_mat, cam

    def process_sequence(self, sparse_data, total_frames, full_cam2world=None):
        indices = np.array(sparse_data['frame_indices'])
        pose = np.array(sparse_data['pose']) 
        betas = np.array(sparse_data['betas']) 
        cam = np.array(sparse_data['cam']) 
        tform = np.array(sparse_data['tform']) 
        
        # EARLY EXIT
        if len(indices) < 2: return None

        # 1. Standard Packing & Sorting
        orig_min_idx = indices.min()
        orig_max_idx = indices.max()
            
        if pose.ndim == 2 and pose.shape[1] == 72:
            pose_flat = pose.reshape(-1, 3)
            r = R.from_rotvec(pose_flat)
            pose_mat = r.as_matrix().reshape(-1, 24, 3, 3)
        elif pose.ndim == 4 and pose.shape[-2:] == (3, 3):
            pose_mat = pose
        else:
            raise ValueError(f"Unknown pose shape: {pose.shape}")

        sort_order = np.argsort(indices)
        indices = indices[sort_order]
        pose_mat = pose_mat[sort_order]
        betas = betas[sort_order]
        cam = cam[sort_order]
        tform = tform[sort_order]

        # Deduplicate
        unique_indices, counts = np.unique(indices, return_counts=True)
        if len(unique_indices) < len(indices):
            new_pose, new_betas, new_cam, new_tform = [], [], [], []
            for u_idx in unique_indices:
                mask = (indices == u_idx)
                new_betas.append(np.mean(betas[mask], axis=0))
                new_cam.append(np.mean(cam[mask], axis=0))
                new_tform.append(np.mean(tform[mask], axis=0))
                
                p_subset = pose_mat[mask]
                p_6d = self.matrix_to_rotation_6d(p_subset)
                p_6d_avg = np.mean(p_6d, axis=0)
                new_pose.append(self.rotation_6d_to_matrix(p_6d_avg))
            
            indices = unique_indices
            pose_mat = np.array(new_pose)
            betas = np.array(new_betas)
            cam = np.array(new_cam)
            tform = np.array(new_tform)

        # --- [STEP 1]: Trend-based Correction (Outlier Fix) ---
        # We keep this to fix "teleporting" glitches, but use a small window
        pose_mat, cam = self.correct_outliers_with_trend(
            pose_mat, cam, window_size=5, thresh_trans=0.5, thresh_rot=0.5
        )

        # 2. Interpolation
        all_indices = np.arange(total_frames)
        full_pose = np.zeros((total_frames, 24, 3, 3))
        full_betas = np.zeros((total_frames, betas.shape[1]))
        full_cam = np.zeros((total_frames, cam.shape[1]))
        full_tform = np.zeros((total_frames, 2, 3))
        
        # Handle Single Frame Case
        if len(indices) == 1:
            full_pose[:] = pose_mat[0]
            full_betas[:] = betas[0]
            full_cam[:] = cam[0]
            full_tform[:] = tform[0]
            
            if full_cam2world is not None:
                 idx = indices[0]
                 R_c2w = full_cam2world[idx, :3, :3]
                 T_c2w = full_cam2world[idx, :3, 3]
                 R_w2c = R_c2w.T
                 full_pose[0, 0] = R_w2c @ full_pose[0, 0]
                 full_cam[0] = R_w2c @ (full_cam[0] - T_c2w)
                 
            return {
                'pose': full_pose, 
                'betas': full_betas, 
                'cam': full_cam, 
                'tform': full_tform,
                'valid_range': (orig_min_idx, orig_max_idx)
            }

        # Linear Interp for Vectors
        for i in range(betas.shape[1]):
            full_betas[:, i] = np.interp(all_indices, indices, betas[:, i])
        for i in range(cam.shape[1]):
            full_cam[:, i] = np.interp(all_indices, indices, cam[:, i])
            
        tform_flat = tform.reshape(-1, 6)
        full_tform_flat = np.zeros((total_frames, 6))
        for i in range(6):
            full_tform_flat[:, i] = np.interp(all_indices, indices, tform_flat[:, i])
        full_tform = full_tform_flat.reshape(total_frames, 2, 3)
            
        # SLERP for Rotations
        min_idx, max_idx = orig_min_idx, orig_max_idx
        valid_range_mask = (all_indices >= min_idx) & (all_indices <= max_idx)
        valid_range_indices = all_indices[valid_range_mask]
        
        for j in range(24):
            rots_j = R.from_matrix(pose_mat[:, j])
            slerp = Slerp(indices, rots_j)
            
            slerp_min, slerp_max = indices[0], indices[-1]
            slerp_mask = (valid_range_indices >= slerp_min) & (valid_range_indices <= slerp_max)
            slerp_indices = valid_range_indices[slerp_mask]
            
            if len(slerp_indices) > 0:
                interp_rots = slerp(slerp_indices)
                full_pose[slerp_indices, j] = interp_rots.as_matrix()
            
            if min_idx < slerp_min:
                full_pose[min_idx:slerp_min, j] = pose_mat[0, j]
            if max_idx > slerp_max:
                full_pose[slerp_max+1:max_idx+1, j] = pose_mat[-1, j]
            
        # --- [STEP 2]: Final Smoothing ---
        # [MODIFICATION]: We smooth BODY POSE (jittery HMR) but NOT POSITION (LAG)
        
        pose_6d = self.matrix_to_rotation_6d(full_pose)
        pose_6d_flat = pose_6d.reshape(total_frames, -1)
        
        target_rot_window = 31
        target_body_window = 7
        
        def get_valid_window(target, total):
            w = target if total >= target else total
            if w % 2 == 0: w -= 1
            if w < 3: w = 3 
            return w
            
        traj_w = get_valid_window(target_rot_window, total_frames)
        pose_w = get_valid_window(target_body_window, total_frames)
        
        if total_frames >= 3:

            # 1. Smooth Body Pose (HMR is noisy, needs smoothing)
            pose_6d_reshaped = pose_6d_flat.reshape(total_frames, 24, 6)
            
            root_6d = pose_6d_reshaped[:, 0, :] 
            root_smooth = savgol_filter(root_6d, traj_w, 2, axis=0) # Smooth root rotation slightly
            
            body_6d = pose_6d_reshaped[:, 1:, :] 
            body_smooth = savgol_filter(body_6d, pose_w, 2, axis=0)

            pose_6d_smooth = np.concatenate([root_smooth[:, None, :], body_smooth], axis=1)
            pose_smooth_mat = self.rotation_6d_to_matrix(pose_6d_smooth.reshape(total_frames, 24, 6))

            # 2. Smooth Shape & Crop info
            betas_smooth = savgol_filter(full_betas, traj_w, 2, axis=0) 
            tform_smooth = savgol_filter(full_tform.reshape(total_frames, 6), traj_w, 2, axis=0).reshape(total_frames, 2, 3)

            # 3. [CHANGE]: DO NOT SMOOTH POSITION
            # Use interpolated GT directly to remove lag.
            cam_smooth = full_cam 

        else:
            pose_smooth_mat = full_pose
            betas_smooth = full_betas
            cam_smooth = full_cam
            tform_smooth = full_tform
            
        if full_cam2world is not None:
            R_c2w_full = full_cam2world[:, :3, :3]
            R_w2c_full = np.transpose(R_c2w_full, (0, 2, 1))
            root_rot_world = pose_smooth_mat[:, 0]
            root_rot_cam = np.matmul(R_w2c_full, root_rot_world)
            pose_smooth_mat[:, 0] = root_rot_cam
        
        return {
            'pose': pose_smooth_mat,
            'betas': betas_smooth,
            'cam': cam_smooth,
            'tform': tform_smooth,
            'valid_range': (orig_min_idx, orig_max_idx)
        }