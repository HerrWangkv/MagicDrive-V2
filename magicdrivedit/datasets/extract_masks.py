"""
@file   extract_masks.py
@brief  Extract semantic masks (Human) using Hugging Face Transformers
        Replaces legacy mmseg/SegFormer dependencies.
"""

import os
import cv2
import torch
import imageio
import numpy as np
import torch.multiprocessing as mp
from glob import glob
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# Standard Cityscapes Class Mapping (11: Person)
DATASET_CLASSES = {'human': [11]}

def setup_model(device_id):
    """Load model on specific GPU"""
    device = f"cuda:{device_id}"
    model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name, use_safetensors=True)
    model.to(device)
    model.eval()
    return processor, model, device

def process_chunk(rank, gpu_ids, files_chunk, save_root, base_root):
    """
    Worker function running on a specific GPU.
    """
    gpu_id = gpu_ids[rank]
    try:
        processor, model, device = setup_model(gpu_id)
    except Exception as e:
        print(f"[GPU {gpu_id}] Failed to load model: {e}")
        return

    # Progress bar only for rank 0 to avoid console spam, or simple print for others
    iterator = tqdm(files_chunk, desc=f"GPU {gpu_id}", position=rank)

    for meta in iterator:
        # meta contains: (folder_type, cam_name, filename)
        folder_type, cam, filename = meta
        
        # Source Path
        img_path = os.path.join(base_root, folder_type, cam, filename)
        
        # Output Path
        human_mask_dir = os.path.join(save_root, "human", folder_type, cam)
        out_name = os.path.splitext(filename)[0] + ".png"
        human_mask_path = os.path.join(human_mask_dir, out_name)

        # Skip if exists (logic moved inside worker to minimize communication)
        # Note: Directory creation should be thread-safe or pre-created, 
        # but os.makedirs(exist_ok=True) is generally safe.
        if not os.path.exists(human_mask_dir):
            try:
                os.makedirs(human_mask_dir, exist_ok=True)
            except FileExistsError:
                pass

        image = cv2.imread(img_path)
        if image is None: continue

        # Inference
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = processor(images=image_rgb, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Interpolate to original size
            upsampled_logits = torch.nn.functional.interpolate(
                outputs.logits,
                size=image_rgb.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
        
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        
        # Save Human Mask
        human_mask = np.isin(pred_seg, DATASET_CLASSES['human'])
        if np.any(human_mask):
            imageio.imwrite(human_mask_path, human_mask.astype(np.uint8) * 255)

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/nuscenes')
    parser.add_argument("--save_root", type=str, default='data/nuscenes_masks')
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7', help='Comma separated GPU IDs')
    parser.add_argument('--ignore_existing', action='store_true')
    args = parser.parse_args()

    # 1. Parse GPUs
    gpu_ids = [int(x) for x in args.gpus.split(',') if x.strip()]
    n_gpus = len(gpu_ids)
    print(f"Initializing extraction on {n_gpus} GPUs: {gpu_ids}")

    # 2. Collect ALL files first (Main Thread)
    print("Collecting file list... (this may take a moment)")
    cams = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    folder_types = ["samples", "sweeps"]
    
    all_tasks = []
    
    for folder_type in folder_types:
        for cam in cams:
            cam_dir = os.path.join(args.data_root, folder_type, cam)
            if not os.path.exists(cam_dir): continue
            
            # Target dir check for "ignore_existing"
            target_dir = os.path.join(args.save_root, "human", folder_type, cam)
            
            files = [f for f in os.listdir(cam_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            for f in files:
                if args.ignore_existing:
                    tgt_path = os.path.join(target_dir, os.path.splitext(f)[0] + ".png")
                    if os.path.exists(tgt_path):
                        continue
                
                # Task: (folder_type, cam_name, filename)
                all_tasks.append((folder_type, cam, f))

    total_files = len(all_tasks)
    print(f"Total files to process: {total_files}")
    if total_files == 0:
        print("Nothing to do.")
        return

    # 3. Split tasks into chunks
    chunk_size = int(np.ceil(total_files / n_gpus))
    chunks = [all_tasks[i:i + chunk_size] for i in range(0, total_files, chunk_size)]
    
    # Handle edge case where we have more GPUs than chunks
    while len(chunks) < n_gpus:
        chunks.append([])

    # 4. Spawn Processes
    mp.set_start_method('spawn', force=True)
    processes = []
    
    for rank in range(n_gpus):
        p = mp.Process(
            target=process_chunk, 
            args=(rank, gpu_ids, chunks[rank], args.save_root, args.data_root)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\nMulti-GPU Extraction Complete.")

if __name__ == "__main__":
    main()