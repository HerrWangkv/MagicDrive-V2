#!/bin/bash
#SBATCH --job-name=magicdrive_sde_brushnet
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --partition=accelerated-h100
#SBATCH --output=logs/train_sde_brushnet_%j.out
#SBATCH --error=logs/train_sde_brushnet_%j.err
#SBATCH --constraint=BEEOND
#SBATCH --exclusive

# Create logs directory if it doesn't exist
mkdir -p logs

# Set BeeOND mountpoint
export BEEOND_MOUNTPOINT="/mnt/odfs/${SLURM_JOB_ID}/stripe_default"

# Set environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
# Use job ID to generate unique port between 29500-32767 to avoid conflicts
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 3268))

# Print distributed training info
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"

# SquashFS configuration
export DATA_PATH="/hkfs/work/workspace/scratch/xw2723-nuscenes"
export NUSCENES_SQFS="$DATA_PATH/nuscenes.sqfs"
# CRITICAL: Must use local filesystem for FUSE mounts, NOT BeeOND
# BeeGFS (0x19830326) forbids mounting FUSE filesystems on top of it
# Use /tmp (local node SSD) - faster than /dev/shm and no RAM limit
export MOUNT_PATH_RAW="/tmp/$(whoami)/sqsh_${SLURM_JOB_ID}_nuscenes"

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
}
export -f unmount_squashfuse

mount_squashfuse() {
    # Do nothing on tasks with node-local rank other than 0
    ((SLURM_LOCALID)) && return 0

    # Clean up any existing mounts first
    unmount_squashfuse
    
    # Create parent directory and mount directory
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
echo "Start time: $(date)"

# Try to verify SLURM BeeOND setup
echo "=== SLURM BeeOND Environment ==="
env | grep -i beeond || echo "No BEEOND environment variables set"
echo ""

# Debug: Check BeeOND on ALL nodes and verify no stale mounts
echo "=== BeeOND Debug Information (All Nodes) ==="
srun --label bash -c '
    echo "Node $(hostname): Checking /mnt/odfs/ for stale mounts..."
    odfs_contents=$(ls /mnt/odfs/ 2>/dev/null || echo "")
    if [ -n "$odfs_contents" ]; then
        echo "Node $(hostname): Contents of /mnt/odfs/:"
        ls -lah /mnt/odfs/ 2>&1
        
        # Check for stale mounts (directories other than current job ID)
        for dir in /mnt/odfs/*; do
            if [ -d "$dir" ]; then
                dir_name=$(basename "$dir")
                if [ "$dir_name" != "${SLURM_JOB_ID}" ]; then
                    echo "Node $(hostname): WARNING - Stale BeeOND mount detected: $dir_name"
                    echo "Node $(hostname): This may cause issues. Please contact support to clean up."
                fi
            fi
        done
    fi
    
    if [ -d "/mnt/odfs/${SLURM_JOB_ID}" ]; then
        echo "Node $(hostname): SUCCESS - /mnt/odfs/${SLURM_JOB_ID} exists"
        echo "Node $(hostname): Contents of /mnt/odfs/${SLURM_JOB_ID}:"
        ls -lah "/mnt/odfs/${SLURM_JOB_ID}/" 2>&1 | head -20
        echo "Node $(hostname): Disk usage:"
        du -sh "/mnt/odfs/${SLURM_JOB_ID}/"* 2>/dev/null || echo "Node $(hostname): No subdirectories yet"
        echo "Node $(hostname): Available space:"
        df -h "/mnt/odfs/${SLURM_JOB_ID}" 2>&1
    else
        echo "Node $(hostname): ERROR - /mnt/odfs/${SLURM_JOB_ID} does NOT exist"
        echo "Node $(hostname): /mnt/odfs/ directory not found or empty"
    fi
'
echo "=== End BeeOND Debug ==="
echo ""

echo "Temporary storage configured at: $BEEOND_MOUNTPOINT"

# Wait for BeeOND to be ready on ALL nodes with retries
echo "Waiting for BeeOND to be accessible on all nodes..."
srun --label bash -c '
    max_retries=30
    retry=0
    while [ $retry -lt $max_retries ]; do
        if [ -d "/mnt/odfs/${SLURM_JOB_ID}" ] && [ -w "/mnt/odfs/${SLURM_JOB_ID}" ]; then
            echo "Node $(hostname): BeeOND ready (attempt $((retry+1)))"
            exit 0
        fi
        echo "Node $(hostname): Waiting for BeeOND... (attempt $((retry+1))/$max_retries)"
        sleep 2
        retry=$((retry+1))
    done
    echo "Node $(hostname): ERROR - BeeOND not ready after $max_retries attempts"
    exit 1
'

echo "SUCCESS: BeeOND is ready on all nodes"
echo ""

# Create directories on BeeOND storage for temporary runtime data
echo "Creating directories on BeeOND storage..."
srun --label bash -c "mkdir -p $BEEOND_MOUNTPOINT/tmp $BEEOND_MOUNTPOINT/torch_dist $BEEOND_MOUNTPOINT/hf_cache && echo 'Node $(hostname): Directories created'"

# Mount the SquashFS file (in background, with resource overlap)
echo "Mounting SquashFS on each node..."
srun --overlap bash -c mount_squashfuse &

# Wait for mount to complete
srun bash -c wait_for_mount

echo "SquashFS mount is ready, BeeOND storage is ready for use"

# Run the training script with Apptainer on ALL nodes (one container per node)
srun --overlap --label --export=ALL --ntasks-per-node=1 --gres=gpu:4 \
    apptainer exec --nv --writable-tmpfs \
    --bind /home/hk-project-p0023969/xw2723/test/MagicDrive-V2:/MagicDrive-V2 \
    --bind "${BEEOND_MOUNTPOINT}:${BEEOND_MOUNTPOINT}" \
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
        # Use BeeOND for temporary files and outputs (multi-node SSD storage)
        export TMPDIR=$BEEOND_MOUNTPOINT/tmp
        export TORCH_DISTRIBUTED_STORE_DIR=$BEEOND_MOUNTPOINT/torch_dist
        export OMP_NUM_THREADS=1
        export PYTHONDONTWRITEBYTECODE=1  # Prevent .pyc race conditions
        # Get the node rank early
        NODE_RANK=\${SLURM_NODEID:-\${SLURM_PROCID:-0}}
        # Set HuggingFace cache to BeeOND to avoid HOME I/O
        # CRITICAL: Give every node its own private folder so they don't fight
        export HF_HOME=\"$BEEOND_MOUNTPOINT/node_\${NODE_RANK}/hf_cache\"
        export TRANSFORMERS_CACHE=\"\$HF_HOME\"
        # CRITICAL: Each node/rank gets its own torch extensions directory
        export TORCH_EXTENSIONS_DIR=\"/tmp/$(whoami)/torch_extensions_${SLURM_JOB_ID}_node\${NODE_RANK}\"
        # CRITICAL: Clear any stale torch extensions cache (see: github.com/hpcaitech/Open-Sora/issues/629)
        # rm -rf ~/.cache/colossalai/torch_extensions/ ~/.cache/torch_extensions/ 2>/dev/null || true
        mkdir -p \"\$HF_HOME\" \"\$TORCH_EXTENSIONS_DIR\"
        
        # CRITICAL: Force change to container path
        cd /MagicDrive-V2
        echo 'Forced working directory change to:'
        pwd
        
        # Create nuscenes data directory structure
        mkdir -p /MagicDrive-V2/data/nuscenes
        
        # We use a LOCK FILE on the shared filesystem instead of sleep
        SETUP_LOCK='$BEEOND_MOUNTPOINT/setup_complete.lock'
        # Only create symlinks on node rank 0 to avoid race conditions
        # All nodes share the same /MagicDrive-V2 via bind mount
        # NODE_RANK already set above
        
        if [ \"\$NODE_RANK\" -eq 0 ]; then
            # Remove any existing symlinks first
            echo 'Node 0: Cleaning up old symlinks in /MagicDrive-V2/data/nuscenes...'
            find /MagicDrive-V2/data/nuscenes -maxdepth 1 -type l -delete
            
            # Create symbolic links for all items from the squashfs mount
            # Skip items that are already bind-mounted (directories)
            for item in /data/nuscenes/*; do
                if [ -e \"\$item\" ]; then
                    item_name=\$(basename \"\$item\")
                    target=\"/MagicDrive-V2/data/nuscenes/\$item_name\"
                    
                    # Skip if exists
                    if ! mountpoint -q \"\$target\" 2>/dev/null && [ ! -d \"\$target\" ]; then
                        ln -s \"\$item\" \"\$target\" 2>/dev/null
                    fi
                fi
            done
            
            # Signal symlinks are ready
            touch \"\$SETUP_LOCK\"
            echo 'Node 0: Symlinks ready.'
        else
            echo \"Node \$NODE_RANK: Waiting for Setup...\"
            while [ ! -f \"\$SETUP_LOCK\" ]; do sleep 1; done
            echo \"Node \$NODE_RANK: Symlinks detected.\"
        fi
        
        TRANSFORMERS_INIT=\"/usr/local/lib/python3.10/dist-packages/transformers/models/__init__.py\"
        if [ -f \"\$TRANSFORMERS_INIT\" ]; then
            echo \"Node \${NODE_RANK}: Patching transformers...\"
            sed -i '/superpoint/d' \"\$TRANSFORMERS_INIT\"
        fi

        # NODE_RANK already set at the top - use it directly for torchrun
        echo 'Launching torchrun with nnodes='\$SLURM_JOB_NUM_NODES' node_rank='\$NODE_RANK' master='\$MASTER_ADDR':'\$MASTER_PORT; \
        torchrun --nproc-per-node=4 --nnodes=\$SLURM_JOB_NUM_NODES --node_rank=\$NODE_RANK \
          --master_addr=\$MASTER_ADDR --master_port=\$MASTER_PORT \
          scripts/train_sde_brushnet.py configs/magicdrive/train/sde_brushnet.py \
          --cfg-options num_workers=2 prefetch_factor=2"

echo "End time: $(date)"
echo "Job completed successfully"
echo "BeeOND temporary files will be automatically cleaned up by SLURM"