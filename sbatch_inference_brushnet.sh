#!/bin/bash
#SBATCH --job-name=inference_brushnet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --partition=accelerated-h100
#SBATCH --output=logs/inference_brushnet_%j.out
#SBATCH --error=logs/inference_brushnet_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500

# SquashFS configuration
export DATA_PATH="/hkfs/work/workspace/scratch/xw2723-nuscenes"
export NUSCENES_SQFS="$DATA_PATH/nuscenes.sqfs"
# Use shared memory with unique path per job to avoid conflicts
export MOUNT_PATH_RAW="/dev/shm/$(whoami)/sqsh_${SLURM_JOB_ID}_nuscenes"

# SquashFS mount/unmount functions
unmount_squashfuse() {
    # Do nothing on tasks with node-local rank other than 0
    ((SLURM_LOCALID)) && return 0
    
    # Force unmount if mounted
    if mountpoint -q "$MOUNT_PATH_RAW" 2>/dev/null; then
        fusermount3 -u "$MOUNT_PATH_RAW" 2>/dev/null || fusermount3 -uz "$MOUNT_PATH_RAW" 2>/dev/null || true
    fi
    
    # Clean up the directory
    rm -rf "$MOUNT_PATH_RAW" 2>/dev/null || true
    
    # Clean up parent directories if empty
    [ -d "$(dirname "$MOUNT_PATH_RAW")" ] && rmdir "$(dirname "$MOUNT_PATH_RAW")" 2>/dev/null || true
}
export -f unmount_squashfuse

# Function to aggressively clean stale FUSE mount points
clean_stale_mounts() {
    # Do nothing on tasks with node-local rank other than 0
    ((SLURM_LOCALID)) && return 0
    
    local base_dir="/dev/shm/$(whoami)"
    
    # Kill any squashfuse processes for this user
    pkill -u "$(whoami)" squashfuse_ll 2>/dev/null || true
    sleep 1
    
    # Find and clean any stale mount points in our user directory
    if [ -d "$base_dir" ]; then
        # Try to unmount anything that looks mounted
        find "$base_dir" -type d -name "*sqsh*" 2>/dev/null | while read -r dir; do
            if mountpoint -q "$dir" 2>/dev/null; then
                echo "Found mounted directory: $dir, attempting unmount..."
                fusermount3 -u "$dir" 2>/dev/null || fusermount3 -uz "$dir" 2>/dev/null || true
            fi
        done
        
        # Now try to remove stale directories (including broken mount points)
        find "$base_dir" -type d -name "*sqsh*" 2>/dev/null | while read -r dir; do
            # Use lazy unmount for stubborn mount points
            umount -l "$dir" 2>/dev/null || true
            # Try to remove the directory
            rm -rf "$dir" 2>/dev/null || true
        done
        
        # Clean up empty parent directories
        rmdir "$base_dir" 2>/dev/null || true
    fi
}
export -f clean_stale_mounts

mount_squashfuse() {
    # Do nothing on tasks with node-local rank other than 0
    ((SLURM_LOCALID)) && return 0

    # Clean up any existing mounts first
    unmount_squashfuse
    
    # Create mount directory structure
    mkdir -p "$(dirname "$MOUNT_PATH_RAW")"
    mkdir -p "$MOUNT_PATH_RAW"
    chmod 700 "$MOUNT_PATH_RAW"

    # Register cleanup handler
    trap 'bash -c unmount_squashfuse' EXIT SIGINT SIGTERM SIGCONT

    # Mount SquashFS file
    if [ -f "$NUSCENES_SQFS" ]; then
        echo "Attempting to mount $NUSCENES_SQFS at $MOUNT_PATH_RAW"
        if squashfuse_ll "$NUSCENES_SQFS" "$MOUNT_PATH_RAW"; then
            echo "Successfully mounted $NUSCENES_SQFS at $MOUNT_PATH_RAW"
        else
            echo "Error: Failed to mount SquashFS"
            exit 1
        fi
    else
        echo "Warning: $NUSCENES_SQFS not found"
        exit 1
    fi

    # Keep the mount process alive with better error handling
    while true; do
        # Check if mount is still active
        if ! mountpoint -q "$MOUNT_PATH_RAW" 2>/dev/null; then
            echo "Warning: Mount lost, attempting remount..."
            if [ -f "$NUSCENES_SQFS" ] && squashfuse_ll "$NUSCENES_SQFS" "$MOUNT_PATH_RAW" 2>/dev/null; then
                echo "Success: Remounted at $MOUNT_PATH_RAW"
            else
                echo "Error: Failed to remount SquashFS"
            fi
        fi
        sleep 300  # Check every 5 minutes
    done
}
export -f mount_squashfuse

wait_for_mount() {
    # Get the process ID of the most recent mount process
    mount_pid="$(pgrep -n -f -u "$(whoami)" -- ' -c mount_squashfuse$' 2>/dev/null || echo "")"
    if [ -n "$mount_pid" ]; then
        # Wait for mount to complete
        while ps -p "$mount_pid" > /dev/null 2>&1 && ! mountpoint -q "$MOUNT_PATH_RAW" 2>/dev/null; do
            sleep 1
        done
    fi
}
export -f wait_for_mount

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_PER_NODE"
echo "Start time: $(date)"

# Pre-cleanup: Remove any stale mount points aggressively
echo "Cleaning up any existing mount points..."
# First run aggressive stale mount cleanup
srun bash -c clean_stale_mounts
sleep 2
# Then run normal cleanup
srun bash -c unmount_squashfuse

# Mount the SquashFS file (in background, with resource overlap)
echo "Mounting SquashFS..."
srun --overlap bash -c mount_squashfuse &

# Wait for mount to complete
srun bash -c wait_for_mount

echo "SquashFS mount is ready"

# Run the training script with Apptainer
apptainer exec --nv --writable-tmpfs \
    --env MASTER_ADDR="${MASTER_ADDR}" \
    --env MASTER_PORT="${MASTER_PORT}" \
    --env SLURM_PROCID="${SLURM_PROCID}" \
    --env SLURM_LOCALID="${SLURM_LOCALID}" \
    --env SLURM_NODEID="${SLURM_NODEID}" \
    --env SLURM_JOB_NUM_NODES="${SLURM_JOB_NUM_NODES}" \
    --env MOUNT_PATH_RAW="${MOUNT_PATH_RAW}" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --bind /home/hk-project-p0023969/xw2723/test/MagicDrive-V2:/MagicDrive-V2 \
    --bind "${MOUNT_PATH_RAW}:/data/nuscenes" \
    --bind /hkfs/work/workspace/scratch/xw2723-nuscenes/nuscenes_pedestrian:/data/nuscenes_pedestrian \
    --bind /hkfs/work/workspace/scratch/xw2723-nuscenes/interp_12Hz_trainval:/MagicDrive-V2/data/nuscenes/interp_12Hz_trainval \
    --bind /hkfs/work/workspace/scratch/xw2723-nuscenes/nuscenes_map_aux_12Hz:/MagicDrive-V2/data/nuscenes_map_aux_12Hz \
    --bind /hkfs/work/workspace/scratch/xw2723-nuscenes/nuscenes_mmdet3d-12Hz:/MagicDrive-V2/data/nuscenes_mmdet3d-12Hz \
    --workdir /MagicDrive-V2 \
    magicdrive2.sif \
    bash -c "
        # Override CUDA_HOME to use container's CUDA installation
        export CUDA_HOME=/usr/local/cuda
        export CXX=/usr/bin/g++
        export CC=/usr/bin/gcc
        export TMPDIR=$HOME/tmp
        mkdir -p $TMPDIR
        
        # CRITICAL: Force change to container path
        cd /MagicDrive-V2
        echo 'Forced working directory change to:'
        pwd
        
        # Debug: Check working directory and Python paths
        echo 'Current working directory:'
        pwd
        echo 'Python sys.path:'
        python3 -c 'import sys; [print(p) for p in sys.path]'
        echo 'Contents of current directory:'
        ls -la . | head -5
        
        # Verify CUDA setup (container's built-in CUDA)
        echo \"CUDA_HOME: \$CUDA_HOME\"
        echo \"nvcc location: \$(which nvcc 2>/dev/null || echo 'nvcc not found')\"
        if which nvcc >/dev/null 2>&1; then
            echo \"nvcc version: \$(nvcc --version | grep release)\"
        fi
        echo \"PyTorch CUDA version: \$(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'Could not detect PyTorch CUDA version')\"
        echo \"CUDA available in PyTorch: \$(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Could not check CUDA availability')\"
        
        # Create nuscenes data directory structure
        mkdir -p /MagicDrive-V2/data/nuscenes
        
        # Debug: Check what's available in the mount
        echo 'Contents of /data/nuscenes:'
        ls -la /data/nuscenes/ | head -10
        
        # Create symbolic links for all items from the squashfs mount
        # BUT skip items that are already bind-mounted
        for item in /data/nuscenes/*; do
            if [ -e \"\$item\" ]; then
                item_name=\$(basename \"\$item\")
                target=\"/MagicDrive-V2/data/nuscenes/\$item_name\"
                
                # Skip if target is already a bind mount
                if mountpoint -q \"\$target\" 2>/dev/null; then
                    echo \"Skipping \$item_name (already bind-mounted)\"
                    continue
                fi
                
                # Remove if exists, then create symlink
                if [ -L \"\$target\" ] || [ -f \"\$target\" ]; then
                    rm -f \"\$target\"
                fi
                ln -s \"\$item\" \"\$target\"
                echo \"Created symlink: \$target -> \$item\"
            fi
        done
        
        # Debug: Check final structure
        echo 'Contents of /MagicDrive-V2/data/nuscenes:'
        ls -la /MagicDrive-V2/data/nuscenes/ | head -10
        
        # Verify a specific file path
        test_file=\"/MagicDrive-V2/data/nuscenes/samples/CAM_FRONT/n008-2018-08-30-15-52-26-0400__CAM_FRONT__1535659487012449.jpg\"
        if [ -f \"\$test_file\" ]; then
            echo \"SUCCESS: Test file found at \$test_file\"
        else
            echo \"ERROR: Test file NOT found at \$test_file\"
            echo \"Looking for samples directory:\"
            find /MagicDrive-V2/data/nuscenes/ -name \"samples\" -type d 2>/dev/null || echo \"No samples directory found\"
        fi
        
        # Run the training command with relative paths from /MagicDrive-V2
        echo 'About to run training from:' \$(pwd)
        echo 'Training script location:' \$(ls -la scripts/train_brushnet.py)

        python3 -m pip install --no-cache-dir numpy==1.24.2
        torchrun --nproc-per-node=4 --nnodes=$SLURM_JOB_NUM_NODES --node_rank=$SLURM_NODEID \
            --master_addr=\$MASTER_ADDR --master_port=\$MASTER_PORT \
            scripts/inference_magicdrive_brushnet.py configs/magicdrive/inference/fullx848x1600_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_brushnet.py \
            --cfg-options cpu_offload=true
    "

echo "End time: $(date)"
echo "Job completed successfully"