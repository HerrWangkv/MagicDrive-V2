#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nproc_per_node 1 scripts/inference_magicdrive.py configs/magicdrive/inference/fullx424x800_stdit3_CogVAE_boxTDS_wCT_xCE_wSST.py \
    --cfg-options model.from_pretrained=ckpts/MagicDriveDiT-stage3-40k-ft \
    cpu_offload=true scheduler.type=rflow-slice