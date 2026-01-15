import os
import cv2
import numpy as np
import secrets

class SHHQManager:
    def __init__(self, 
                 shhq_root, 
                 min_height=30, 
                 clean_artifacts=True,
                 cache_limit=1000):
        """
        Adapts to the official SHHQ directory structure:
        shhq_root/
          ├── no_segment/  (Contains original RGB images, e.g., 00000.png)
          └── segments/    (Contains Masks, e.g., 00000.png)
        """
        self.shhq_root = shhq_root
        self.img_dir = os.path.join(shhq_root, "no_segment")
        self.mask_dir = os.path.join(shhq_root, "segments")
        
        self.min_height = min_height
        self.clean_artifacts = clean_artifacts
        
        # 1. Index all valid images
        if not os.path.exists(self.img_dir) or not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"SHHQ root must contain 'no_segment' and 'segments' folders. Checked: {shhq_root}")

        valid_exts = {'.png', '.jpg', '.jpeg'}
        # Get list of filenames (assuming img and mask filenames match)
        self.filenames = sorted([
            f for f in os.listdir(self.img_dir) 
            if os.path.splitext(f)[1].lower() in valid_exts
        ])
        
        if len(self.filenames) == 0:
            raise FileNotFoundError(f"No images found in {self.img_dir}")
            
        print(f"[SHHQManager] Indexed {len(self.filenames)} pedestrians.")
        
        self.cache = {} 
        self.cache_limit = cache_limit

    def _load_and_clean_image(self, index):
        if index in self.cache:
            return self.cache[index]

        filename = self.filenames[index]
        img_path = os.path.join(self.img_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)

        # 1. Read RGB and Mask separately
        rgb = cv2.imread(img_path) # BGR
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # Gray
        
        if rgb is None or mask is None:
            return None, None
            
        # Ensure dimensions match (some dataset masks might differ slightly from images)
        if rgb.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 2. Clean Mask (Remove artifacts + Soften edges)
        if self.clean_artifacts:
            # Binarize
            _, alpha = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            # Erosion (Remove white edges/halos)
            kernel = np.ones((3, 3), np.uint8)
            alpha = cv2.erode(alpha, kernel, iterations=1)
            
            # Largest Connected Component (Remove floating noise points)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(alpha, connectivity=8)
            if num_labels > 1:
                # stats[1:, 4] is area (index 0 is background). Find largest object.
                largest_label = 1 + np.argmax(stats[1:, 4]) 
                alpha = np.where(labels == largest_label, 255, 0).astype(np.uint8)
            
            # Gaussian Blur (Soften edges for natural blending - Critical!)
            alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        else:
            alpha = mask

        # Update cache
        if len(self.cache) > self.cache_limit:
            self.cache.pop(next(iter(self.cache)))
        self.cache[index] = (rgb, alpha)
        
        return rgb, alpha

    def _generate_random_box(self, img_h, img_w, aspect_ratio):
        """
        生成随机框 (使用 secrets 避免 DataLoader 随机种子重复问题)
        """
        horizon = img_h // 2
        
        # 1. 随机脚底位置 (y_bottom)
        # secrets.randbelow(N) 生成 [0, N-1]
        min_y = horizon + 50
        max_y = int(img_h * 1.1)
        if max_y > min_y:
            y_bottom = min_y + secrets.randbelow(max_y - min_y)
        else:
            y_bottom = max_y
        
        # 2. 计算高度 (透视关系)
        dist_from_horizon = y_bottom - horizon
        max_dist = img_h - horizon
        max_person_height = img_h * 0.7 
        scale = dist_from_horizon / max_dist
        
        # 随机抖动 (0.8 ~ 1.2)
        # secrets 生成浮点数技巧: randbelow(1000) / 1000.0
        jitter = 0.8 + (secrets.randbelow(400) / 1000.0)
        
        height = int(max_person_height * scale * jitter)
        height = max(height, self.min_height)
        
        width = int(height / aspect_ratio)
        
        # 3. 随机 X 中心 (允许部分出界)
        margin = 20
        min_center = -width // 2 + margin
        max_center = img_w + width // 2 - margin
        
        if min_center >= max_center:
            x_center = img_w // 2
        else:
            x_center = min_center + secrets.randbelow(max_center - min_center)
        
        # 4. 构建坐标
        x1 = x_center - width // 2
        x2 = x_center + width // 2
        y2 = y_bottom
        y1 = y2 - height
        
        return [x1, y1, x2, y2]

    def paste_pedestrians(self, target_img, bboxes_2d=None, num_paste=None):
        """
        使用 secrets 进行随机采样的粘贴函数。
        """
        canvas = target_img.copy()
        img_h, img_w = canvas.shape[:2]
        pasted_mask = np.zeros((img_h, img_w), dtype=np.float32)
        
        # 1. 确定位置
        final_bboxes = []
        
        if bboxes_2d is None:
            # === 随机模式 ===
            if num_paste is None: 
                # 随机生成 1-4 个
                num_paste = 1 + secrets.randbelow(4)
            
            for _ in range(num_paste):
                # [关键] 使用 secrets 随机选图
                idx = secrets.randbelow(len(self.filenames))
                
                rgb, _ = self._load_and_clean_image(idx)
                if rgb is None: continue
                
                h, w = rgb.shape[:2]
                if w == 0: continue
                aspect_ratio = h / w
                
                bbox = self._generate_random_box(img_h, img_w, aspect_ratio)
                final_bboxes.append(bbox)
        else:
            # === 指定框模式 ===
            if isinstance(bboxes_2d, np.ndarray):
                final_bboxes = bboxes_2d.tolist()
            else:
                final_bboxes = list(bboxes_2d)

        # 2. 排序 (Painter's Algorithm: 远->近)
        final_bboxes.sort(key=lambda b: b[3]) 

        # 3. 粘贴循环
        for i, bbox in enumerate(final_bboxes):
            x1, y1, x2, y2 = map(int, bbox[:4])
            
            if x1 >= img_w or y1 >= img_h or x2 <= 0 or y2 <= 0: continue
            box_w, box_h = x2 - x1, y2 - y1
            if box_w <= 0 or box_h <= 0: continue

            # [关键] 再次使用 secrets 随机选图用于粘贴
            idx = secrets.randbelow(len(self.filenames))
            person_rgb, person_alpha = self._load_and_clean_image(idx)
            if person_rgb is None: continue

            try:
                person_rgb_res = cv2.resize(person_rgb, (box_w, box_h))
                person_alpha_res = cv2.resize(person_alpha, (box_w, box_h))
            except: continue

            # 坐标计算 & 混合 (保持不变)
            paste_x1, paste_y1 = max(0, x1), max(0, y1)
            paste_x2, paste_y2 = min(img_w, x2), min(img_h, y2)
            
            src_x1 = paste_x1 - x1
            src_y1 = paste_y1 - y1
            src_x2 = src_x1 + (paste_x2 - paste_x1)
            src_y2 = src_y1 + (paste_y2 - paste_y1)

            if paste_x2 <= paste_x1 or paste_y2 <= paste_y1: continue

            alpha_roi = person_alpha_res[src_y1:src_y2, src_x1:src_x2].astype(float) / 255.0
            alpha_roi = np.expand_dims(alpha_roi, axis=2)
            
            fg = person_rgb_res[src_y1:src_y2, src_x1:src_x2]
            bg = canvas[paste_y1:paste_y2, paste_x1:paste_x2]
            
            blended = (fg * alpha_roi) + (bg * (1.0 - alpha_roi))
            
            canvas[paste_y1:paste_y2, paste_x1:paste_x2] = blended.astype(np.uint8)
            pasted_mask[paste_y1:paste_y2, paste_x1:paste_x2] = np.maximum(
                pasted_mask[paste_y1:paste_y2, paste_x1:paste_x2], 
                alpha_roi.squeeze()
            )

        return canvas, pasted_mask, final_bboxes