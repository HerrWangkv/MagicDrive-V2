# Dockerfile — CUDA 12.4, Ubuntu 22.04, System Python 3.10 (Hopper-ready)
#
# REASON: The -devel image is needed to build Apex, ColossalAI, etc.
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# -------------------------------------------------------------------------
# Environment settings (All defined at the top)
# -------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# BEST PRACTICE: Support A100, A6000, Ada (40xx), and Hopper (H100/H200)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;9.0a"
# The base image sets these, but being explicit is good practice.
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# -------------------------------------------------------------------------
# System dependencies (Consolidated into a single layer)
# -------------------------------------------------------------------------
# REASON: Installing all system dependencies once at the beginning.
# The `squashfuse`, `fuse`, and `ffmpeg` are already included here.
RUN apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
    # For building Python extensions
    build-essential python3-dev python3-pip python3-venv \
    # General tools
    ca-certificates git wget curl unzip \
    # Project & Library dependencies
    libglib2.0-0 libgl1-mesa-glx libjpeg-dev libpng-dev \
    libsndfile1 ffmpeg pkg-config squashfuse fuse \
    cmake ninja-build libyaml-dev libboost-all-dev \
    # BEST PRACTICE: Clean up apt cache in the same layer to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Update pip using the system's python3
RUN python3 -m pip install --no-cache-dir --upgrade pip wheel setuptools packaging

# -------------------------------------------------------------------------
# Install all Python dependencies BEFORE copying source code
# BEST PRACTICE: This ensures this slow layer is cached effectively.
# -------------------------------------------------------------------------
WORKDIR /workspace/MagicDrive-V2

# Build backends
RUN python3 -m pip install --no-cache-dir flit_core hatchling pdm-pep517

# PyTorch (CUDA 12.4 wheels)
# REASON: Removed '+cu124' from version. The --index-url handles CUDA version selection.
RUN python3 -m pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Core Python dependencies
RUN python3 -m pip install --no-cache-dir \
    absl-py==1.4.0 accelerate==0.29.3 addict==2.4.0 aiosignal==1.3.1 \
    annotated-types==0.7.0 antlr4-python3-runtime==4.9.3 anyio==4.4.0 \
    attrs==23.2.0 bcrypt==4.1.3 beartype==0.18.5 beautifulsoup4==4.12.3 \
    blinker==1.9.0 cachetools==5.3.1 certifi==2023.7.22 cffi==1.17.1 \
    cfgv==3.4.0 charset-normalizer==3.2.0 click==8.1.6 contexttimer==0.3.3 \
    contourpy==1.1.0 cryptography==43.0.3 cycler==0.11.0 debugpy==1.6.7 \
    decorator==4.4.2 deprecated==1.2.14 diffusers==0.30.0 distlib==0.3.8 \
    einops==0.8.0 exceptiongroup==1.2.2 fabric==3.0.0 fastapi==0.111.1 \
    filelock==3.12.2 fire==0.6.0 flask==2.0.1 fonttools==4.41.1 \
    frozenlist==1.4.1 fsspec==2024.6.0 ftfy==6.2.0 google==3.0.0 \
    grpcio==1.56.2 h11==0.14.0 h5py==3.11.0 huggingface-hub==0.23.4 \
    hydra-core==1.3.0 identify==2.5.36 idna==3.4 imageio==2.34.1 \
    imageio-ffmpeg==0.5.1 importlib-metadata==7.1.0 importlib-resources==6.4.5 \
    invoke==2.2.0 itsdangerous==2.1.2 jinja2==3.0.1 joblib==1.3.1 \
    jsonschema==4.22.0 jsonschema-specifications==2023.12.1 kiwisolver==1.4.4 \
    llvmlite==0.43.0rc1 markdown==3.4.4 markdown-it-py==3.0.0 \
    markupsafe==2.1.3 matplotlib==3.5.3 mdurl==0.1.2 moviepy==1.0.3 \
    mpmath==1.3.0 msgpack==1.0.8 networkx==3.1 ninja==1.11.1.1 \
    nodeenv==1.9.1 numba==0.60.0 numpy==1.24.2 nuscenes-devkit==1.1.11 \
    omegaconf==2.3.0 packaging==24.1 paramiko==3.4.0 pillow==10.3.0 \
    platformdirs==4.2.2 plumbum==1.8.3 proglog==0.1.10 protobuf==3.20.1 \
    psutil==5.9.4 pycocotools pycparser==2.22 pydantic==2.7.4 \
    pydantic-core==2.18.4 pygments==2.15.1 pynacl==1.5.0 pyparsing==3.1.0 \
    pyquaternion==0.9.9 python-dateutil==2.8.2 pyyaml==6.0.1 ray==2.30.0 \
    regex==2024.5.15 requests==2.28.2 rich==13.9.4 rotary-embedding-torch==0.5.3 \
    rpds-py==0.18.1 rpyc==6.0.0 safetensors==0.4.3 scikit-learn==1.1.3 \
    scikit-image scipy==1.9.3 sentencepiece==0.2.0 shapely==1.8.5.post1 six==1.16.0 \
    sniffio==1.3.1 soupsieve==2.5 starlette==0.37.2 sympy==1.12 \
    tensorboard==2.11.2 tensorboard-data-server==0.6.1 termcolor==2.4.0 \
    terminaltables==3.1.10 threadpoolctl==3.2.0 timm==0.9.16 tokenizers==0.15.2 \
    tomli==2.0.1 tqdm==4.65.0 transformers==4.39.3 triton==3.0.0 \
    typing-extensions==4.12.2 urllib3==1.26.16 uvicorn==0.29.0 virtualenv==20.26.3 \
    wcwidth==0.2.13 werkzeug==2.3.6 wrapt==1.16.0 yapf==0.40.2 zipp==3.19.2 \
    bitsandbytes==0.43.2

# High-performance libraries (FlashAttention, xFormers, Apex, etc.)
RUN python3 -m pip install --no-cache-dir --no-build-isolation flash-attn==2.6.3
RUN python3 -m pip install xformers==0.0.28.post1 --no-deps --index-url https://download.pytorch.org/whl/cu124
RUN APEX_CPP_EXT=1 APEX_CUDA_EXT=1 python3 -m pip install --no-cache-dir --no-build-isolation -v git+https://github.com/NVIDIA/apex.git
RUN python3 -m pip install --no-cache-dir --no-build-isolation -v git+https://github.com/flymin/ColossalAI.git@pt2.4
RUN python3 -m pip install peft==0.10.0
RUN python3 -m pip install git+https://github.com/zengxianyu/structured-noise.git

# -------------------------------------------------------------------------
# Copy project source code and install project-specific requirements
# BEST PRACTICE: Do this last. Changes to your code will only re-run from here.
# -------------------------------------------------------------------------
COPY requirements/requirements.txt .
RUN python3 -m pip install --no-cache-dir --no-build-isolation -r requirements.txt

COPY . /workspace/MagicDrive-V2

# Clean up any remaining cache
RUN rm -rf /root/.cache/pip

CMD ["/bin/bash"]