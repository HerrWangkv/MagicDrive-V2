#!/bin/bash

# Ensure HUGGING_FACE_TOKEN is set
if [ -z "$HUGGING_FACE_TOKEN" ]; then
    echo "Error: HUGGING_FACE_TOKEN environment variable is not set."
    echo "Please export your token: export HUGGING_FACE_TOKEN=hf_..."
    exit 1
fi


# Run Docker container with:
# - GPU support
# - Privileged mode and fuse/squashfuse support for mounting datasets
# - Host IPC for memory performance
# - Mounts for code, dataset (images), and cache
# - Envrionment variable for HF Token
docker run -it --rm --gpus all --name SAM3_Container \
  --privileged \
  --ipc=host \
  --device /dev/fuse \
  --cap-add SYS_ADMIN \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HUGGING_FACE_TOKEN=$HUGGING_FACE_TOKEN \
  -v /storage_local/kwang/repos/MagicDrive-V2:/MagicDrive-V2 \
  -v /mrtstorage/datasets/public/nuscenes.sqfs:/data/nuscenes.sqfs \
  -w /MagicDrive-V2 \
  --entrypoint /bin/bash \
  sam3 -c "
    # Create mount point
    mkdir -p /data/nuscenes
    
    # Mount the squashfs to temporary location
    squashfuse /data/nuscenes.sqfs /data/nuscenes
    
    # Create target directory for symbolic links
    mkdir -p /MagicDrive-V2/data/nuscenes
    
    # Create symbolic links for everything under /data/nuscenes to data/nuscenes
    for item in /data/nuscenes/*; do
        if [ -e \"\$item\" ]; then
            item_name=\$(basename \"\$item\")
            target=\"/MagicDrive-V2/data/nuscenes/\$item_name\"
            if [ -L \"\$target\" ] || [ -f \"\$target\" ]; then
                rm -f \"\$target\"
            fi
            ln -s \"\$item\" \"\$target\"
        fi
    done
    
    # Run the SAM3 checkpoint preparation script
    echo 'Running script to download SAM3 checkpoints...'
    python3 scripts/prepare_sam3_checkpoints.py
    
    # Start interactive bash session
    echo 'Starting bash session...'
    exec /bin/bash
  "
