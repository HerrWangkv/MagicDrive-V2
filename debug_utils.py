import cv2
import numpy as np
import os
from PIL import Image

def vis_grid(x, save_prefix="debug_grid", fps=4):
    """
    Visualizes batch inputs. Saves as .gif for videos (VSCode friendly) 
    and .jpg for single images.
    """
    # 1. Prepare Data
    # [B*NC, C, T, H, W] -> numpy -> uint8
    data = ((x.detach().cpu().float().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    
    total_imgs, C, T, H, W = data.shape
    batch_size = total_imgs // 6
    
    print(f"[Debug] Processing {batch_size} samples. Time dim: {T}")

    for b in range(batch_size):
        # === 视频模式 (T > 1) -> 保存为 GIF ===
        if T > 1:
            filename = f"{save_prefix}_batch{b}.gif"
            frames = []
            
            for t in range(T):
                # 收集 6 个相机画面
                imgs = []
                for i in range(6):
                    idx = b * 6 + i
                    # [C, H, W] -> [H, W, C] (RGB)
                    # 注意：PIL 需要 RGB，所以我们这里不需要转 BGR
                    img = data[idx, :, t].transpose(1, 2, 0)
                    imgs.append(img)
                
                # 拼图 (2x3 Grid)
                row1 = np.hstack(imgs[:3])
                row2 = np.hstack(imgs[3:])
                grid_rgb = np.vstack((row1, row2))
                
                # 转为 PIL Image
                frames.append(Image.fromarray(grid_rgb))
            
            # 保存 GIF
            # duration = 毫秒每帧 (1000/fps)
            frames[0].save(filename, save_all=True, append_images=frames[1:], duration=int(1000/fps), loop=0)
            print(f"Saved GIF: {filename}")

        # === 单帧模式 (T = 1) -> 保存为 JPG ===
        else:
            imgs = []
            for i in range(6):
                idx = b * 6 + i
                # 单帧依然需要转 BGR 给 cv2 保存
                img_rgb = data[idx, :, 0].transpose(1, 2, 0)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                imgs.append(img_bgr)
                
            row1 = np.hstack(imgs[:3])
            row2 = np.hstack(imgs[3:])
            grid_bgr = np.vstack((row1, row2))
            
            filename = f"{save_prefix}_batch{b}.jpg"
            cv2.imwrite(filename, grid_bgr)
            print(f"Saved Image: {filename}")