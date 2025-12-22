#!/bin/bash
# Script: run_pipeline_12hz_parallel.sh
# Description: Run pipeline_12hz.py on 8 GPUs in parallel with auto-resume capability.

NUM_GPUS=8
TOTAL_SCENES=850
SCENES_PER_GPU=$(( (TOTAL_SCENES + NUM_GPUS - 1) / NUM_GPUS ))

IMAGE_ROOT="data/nuscenes"
PKL_ROOT="data/nuscenes_mmdet3d-12Hz"
SAVE_ROOT="output_12hz_aligned"
SPLITS="train,val"

echo "=========================================================="
echo "Starting Parallel Generation on $NUM_GPUS GPUs"
echo "Total Scenes: $TOTAL_SCENES | Scenes per GPU: $SCENES_PER_GPU"
echo "Output Dir: $SAVE_ROOT"
echo "=========================================================="

# Create pids array to store background process IDs
pids=()

for ((i=0; i<NUM_GPUS; i++)); do
    # 1. Calculate assigned range
    START_SCENE=$((i * SCENES_PER_GPU))
    END_SCENE=$(( (i+1) * SCENES_PER_GPU - 1 ))
    
    # Clamp end scene
    if [ $END_SCENE -ge $TOTAL_SCENES ]; then
        END_SCENE=$((TOTAL_SCENES - 1))
    fi

    echo "[GPU $i] Checking progress for scenes $START_SCENE to $END_SCENE..."

    # 2. Check Progress (Get Resume Index)
    # We use the python script to determine where to start for this specific chunk.
    # The --quiet flag ensures we only get the integer number back.
    RESUME_IDX=$(python3 check_progress.py \
        --image_root "$IMAGE_ROOT" \
        --pkl_root "$PKL_ROOT" \
        --save_root "$SAVE_ROOT" \
        --splits "$SPLITS" \
        --range_start $START_SCENE \
        --range_end $END_SCENE \
        --quiet)

    # 3. Launch Job
    if [ "$RESUME_IDX" -gt "$END_SCENE" ]; then
        echo "[GPU $i] Chunk fully completed ($START_SCENE-$END_SCENE). Skipping."
    else
        echo "[GPU $i] Launching job. Range: $START_SCENE-$END_SCENE | Resuming from: $RESUME_IDX"
        
        CUDA_VISIBLE_DEVICES=$i \
        python3 pipeline_12hz.py \
            --image_root "$IMAGE_ROOT" \
            --pkl_root "$PKL_ROOT" \
            --save_root "$SAVE_ROOT" \
            --splits "$SPLITS" \
            --device cuda \
            --scene_idx_start $RESUME_IDX \
            --scene_idx_end $END_SCENE > "logs/gpu_${i}.log" 2>&1 &
            
        # Store PID
        pids+=($!)
    fi
done

# Wait for all processes
echo "All jobs launched. Waiting for completion..."
for pid in "${pids[@]}"; do
    wait $pid
done

echo "All parallel jobs finished."