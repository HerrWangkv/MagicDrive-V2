# Dockerfile for MagicDrive-V2
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ca-certificates \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN python3 -m pip install --upgrade pip

# Create working directory
WORKDIR /workspace/MagicDrive-V2

# Install torch, torchvision, packaging and apex in one command to ensure torch is available for apex build
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir packaging wheel setuptools
RUN pip install -v --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers --no-deps
RUN pip install flash-attn==2.8.2 --no-build-isolation --no-deps
RUN APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation git+https://github.com/NVIDIA/apex.git
RUN BUILD_EXT=1 pip install --no-cache-dir git+https://github.com/flymin/ColossalAI.git@pt2.4  --no-build-isolation

# Copy source code just before installing requirements
COPY . /workspace/MagicDrive-V2
RUN pip install --no-cache-dir -r requirements/requirements.txt  --no-build-isolation
RUN pip install peft==0.10.0
# Install squashfuse and ffmpeg
RUN apt-get update && apt-get install -y squashfuse fuse ffmpeg

# Set entrypoint
CMD ["/bin/bash"]
