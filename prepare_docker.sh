#!/bin/bash

docker run -it --rm --gpus all --name MagicDrive2 \
  --privileged \
  --shm-size=8g \
  --device /dev/fuse \
  --cap-add SYS_ADMIN \
  -v /storage_local/kwang/repos/MagicDrive-V2:/MagicDrive-V2 \
  -v /mrtstorage/users/kwang/MagicDrive-V2/data:/MagicDrive-V2/data \
  -v /mrtstorage/users/kwang/MagicDrive-V2/pretrained:/MagicDrive-V2/pretrained \
  -v /mrtstorage/users/kwang/MagicDrive-V2/ckpts:/MagicDrive-V2/ckpts \
  -v /mrtstorage/datasets/public/nuscenes.sqfs:/data/nuscenes.sqfs \
  -w /MagicDrive-V2 \
  --entrypoint /bin/bash \
  magicdrive2:latest -c "
    # Create mount point
    mkdir -p /data/nuscenes
    
    # Mount the squashfs to temporary location
    squashfuse /data/nuscenes.sqfs /data/nuscenes
       
    # Start interactive bash session
    exec /bin/bash
  "