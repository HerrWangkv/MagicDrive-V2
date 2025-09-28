"""
@file   extract_masks.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Extract semantic mask

Using SegFormer, 2021. Cityscapes 83.2%
Relies on timm==0.3.2 & pytorch 1.8.1 (buggy on pytorch >= 1.9)

Installation:
    NOTE: mmcv-full==1.2.7 requires another pytorch version & conda env.
        Currently mmcv-full==1.2.7 does not support pytorch>=1.9; 
            will raise AttributeError: 'super' object has no attribute '_specify_ddp_gpu_num'
        Hence, a seperate conda env is needed.

    git clone https://github.com/NVlabs/SegFormer

    conda create -n segformer python=3.8
    conda activate segformer
    # conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

    pip install timm==0.3.2 pylint debugpy opencv-python attrs ipython tqdm imageio scikit-image omegaconf
    pip install mmcv-full==1.2.7 --no-cache-dir
    
    cd SegFormer
    pip install .

Usage:
    Direct run this script in the newly set conda env.
"""


from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

semantic_classes = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle'
]
dataset_classes_in_sematic = {
    'Vehicle': [13, 14, 15],   # 'car', 'truck', 'bus'
    'human': [11, 12, 17, 18], # 'person', 'rider', 'motorcycle', 'bicycle'
}

if __name__ == "__main__":
    import os
    import imageio
    import numpy as np
    from glob import glob
    from tqdm import tqdm
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # Custom configs
    parser.add_argument('--data_root', type=str, default='data/nuscenes')
    parser.add_argument("--save_root", type=str, default='data/nuscenes_masks', help="Where to save the masks")
    parser.add_argument('--ignore_existing', action='store_true')

    # Algorithm configs
    parser.add_argument('--segformer_path', type=str, default='third_party/SegFormer')
    parser.add_argument('--config', help='Config file', type=str, default=None)
    parser.add_argument('--checkpoint', help='Checkpoint file', type=str, default=None)
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
  
    args = parser.parse_args()
    if args.config is None:
        args.config = os.path.join(args.segformer_path, 'local_configs', 'segformer', 'B5', 'segformer.b5.1024x1024.city.160k.py')
    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.segformer_path, 'pretrained', 'segformer.b5.1024x1024.city.160k.pth')

    cams = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    save_dir = args.save_root
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for cam in tqdm(cams, f'Extracting Masks ...'):
        cam_dir = os.path.join(args.data_root, "samples", cam)
        for img_file in os.listdir(cam_dir):
            img_path = os.path.join(cam_dir, img_file)
        human_mask_dir = os.path.join(save_dir, "human", cam)
        if not os.path.exists(human_mask_dir):
            os.makedirs(human_mask_dir)
        vehicle_mask_dir = os.path.join(save_dir, "vehicle", cam)
        if not os.path.exists(vehicle_mask_dir):
            os.makedirs(vehicle_mask_dir)
        
        for filename in tqdm(os.listdir(cam_dir), f'Extracting Masks from {cam} ...'):
            human_mask_path = os.path.join(human_mask_dir, filename[:-4]+".png")
            vehicle_mask_path = os.path.join(vehicle_mask_dir, filename[:-4]+".png")
            if args.ignore_existing and os.path.exists(human_mask_path) and os.path.exists(vehicle_mask_path):
                continue
            fpath = os.path.join(cam_dir, filename)
            #---- Inference and save outputs
            result = inference_segmentor(model, fpath)
            mask = result[0].astype(np.uint8)   # NOTE: in the settings of "cityscapes", there are 19 classes at most.
            # save human masks
            human_mask = np.isin(mask, dataset_classes_in_sematic['human'])
            imageio.imwrite(human_mask_path, human_mask.astype(np.uint8)*255)

            # save vehicle 
            vehicle_mask = np.isin(mask, dataset_classes_in_sematic['Vehicle'])
            imageio.imwrite(vehicle_mask_path, vehicle_mask.astype(np.uint8)*255)
