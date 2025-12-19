#!/bin/bash
# Script: run_pipeline_12hz_parallel.sh
# Description: Run pipeline_12hz.py on 8 GPUs in parallel, splitting 850 scenes evenly.
# Each process gets a range of scene indices to minimize NuScenes SDK reloads.

NUM_GPUS=8
TOTAL_SCENES=850
SCENES_PER_GPU=$(( (TOTAL_SCENES + NUM_GPUS - 1) / NUM_GPUS ))

IMAGE_ROOT="data/nuscenes"
PKL_ROOT="data/nuscenes_mmdet3d-12Hz"
SAVE_ROOT="output_12hz_aligned"
SPLITS="train,val"


for ((i=0; i<NUM_GPUS; i++)); do
    START_SCENE=$((i * SCENES_PER_GPU))
    END_SCENE=$(( (i+1) * SCENES_PER_GPU - 1 ))
    if [ $END_SCENE -ge $TOTAL_SCENES ]; then
        END_SCENE=$((TOTAL_SCENES - 1))
    fi
    echo "Launching GPU $i for scenes $START_SCENE to $END_SCENE"
    CUDA_VISIBLE_DEVICES=$i \
    python3 pipeline_12hz.py \
        --image_root "$IMAGE_ROOT" \
        --pkl_root "$PKL_ROOT" \
        --save_root "$SAVE_ROOT" \
        --splits "$SPLITS" \
        --device cuda \
        --scene_idx_start $START_SCENE --scene_idx_end $END_SCENE &
done

wait
echo "All parallel jobs finished."
