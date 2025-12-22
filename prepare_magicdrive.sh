#!/bin/bash

docker run -it --rm --gpus all --name MagicDrive2 \
  --privileged \
  --ipc=host \
  --device /dev/fuse \
  --cap-add SYS_ADMIN \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /storage_local/kwang/repos/MagicDrive-V2:/MagicDrive-V2 \
  -v /mrtstorage/datasets/public/nuscenes.sqfs:/data/nuscenes.sqfs \
  -w /MagicDrive-V2 \
  --entrypoint /bin/bash \
  magicdrive2:latest -c "
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
    
    # Create symbolic link for data/nuscenes_pedestrian to /data/nuscenes_pedestrian
    if [ -L \"/data/nuscenes_pedestrian\" ] || [ -e \"/data/nuscenes_pedestrian\" ]; then
        rm -f \"/data/nuscenes_pedestrian\"
    fi
    ln -s /MagicDrive-V2/data/nuscenes_pedestrian /data/
    
    # Start interactive bash session
    exec /bin/bash
  "