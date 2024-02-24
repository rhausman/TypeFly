# Stage 1: Base
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as base

ARG LLAVA_COMMIT=fd3f3d29c418ccfca618cc96a8c3f63302b3bda7

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on \
    SHELL=/bin/bash

# Create workspace working directory
WORKDIR /

# Install Ubuntu packages
RUN apt update && \
    apt -y upgrade && \
    apt install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        python3.10-venv \
        python3-pip \
        python3-tk \
        python3-dev \
        nginx \
        bash \
        dos2unix \
        git \
        ncdu \
        net-tools \
        openssh-server \
        libglib2.0-0 \
        libsm6 \
        libgl1 \
        libxrender1 \
        libxext6 \
        ffmpeg \
        wget \
        curl \
        psmisc \
        rsync \
        vim \
        zip \
        unzip \
        htop \
        screen \
        tmux \
        pkg-config \
        libcairo2-dev \
        libgoogle-perftools4 \
        libtcmalloc-minimal4 \
        apt-transport-https \
        ca-certificates && \
    update-ca-certificates && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Set Python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Stage 2: Install LLaVA and python modules
FROM base as setup

# Create and use the Python venv
#RUN python3 -m venv /venv

# Install Torch
#RUN source /venv/bin/activate && \
RUN    pip3 install --no-cache-dir torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install grpcio-tools transformers
RUN pip3 install accelerate
#    pip3 install --no-cache-dir xformers==0.0.22 && \
#deactivate

COPY ../.. /workspace
WORKDIR /workspace
# Make sure the LLaVA source code can be accessed
ENV PYTHONPATH "${PYTHONPATH}:/workspace/serving/llava/LLaVA"

# generate the python files from the proto files
RUN cd /workspace/proto && bash ./generate.sh
# run the llava service
CMD ["python3", "./serving/llava/llava_service.py"]