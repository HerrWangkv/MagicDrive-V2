#! /bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 scripts/inference_magicdrive_repaint.py configs/magicdrive/inference/fullx424x800_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_repaint.py \
    --cfg-options model.from_pretrained=./ckpts/MagicDriveDiT-stage3-40k-ft/ema.pt \
    num_frames=9 cpu_offload=true scheduler.type=rflow-slice-repaint