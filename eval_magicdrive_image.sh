#!/bin/bash

export NCCL_P2P_DISABLE=1
PATH_TO_MODEL="ckpts/MagicDriveDiT-stage3-40k-ft/ema.pt"

torchrun --standalone --nproc_per_node 4 scripts/test_magicdrive.py \
    configs/magicdrive/test/1x848x1600_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_map0_cfg2.0.py \
    --cfg-options model.from_pretrained=${PATH_TO_MODEL}