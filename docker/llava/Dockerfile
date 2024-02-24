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
RUN python3 -m venv /venv

# Install Torch
RUN source /venv/bin/activate && \
    pip3 install --no-cache-dir torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
#    pip3 install --no-cache-dir xformers==0.0.22 && \
    deactivate

# Clone the git repo of LLaVA and set version
WORKDIR /
RUN git clone https://github.com/ashleykleynhans/LLaVA.git && \
    cd /LLaVA && \
    git checkout ${LLAVA_COMMIT}

# Install the dependencies for LLaVA
WORKDIR /LLaVA
RUN source /venv/bin/activate && \
    pip3 install --upgrade pip && \
    pip3 install wheel && \
    pip3 install -e . && \
    pip3 install ninja && \
    pip3 install flash-attn --no-build-isolation && \
    pip3 install transformers==4.37.2 && \
    pip3 install protobuf && \
    deactivate

# Download the default model
ENV MODEL="liuhaotian/llava-v1.6-mistral-7b"
ENV HF_HOME="/"
COPY --chmod=755 docker/llava/llava-docker/scripts/download_models.py /download_models.py
RUN source /venv/bin/activate && \
    pip3 install huggingface_hub && \
    python3 /download_models.py && \
    deactivate

# Install Jupyter, gdown and OhMyRunPod
RUN pip3 install -U --no-cache-dir jupyterlab \
        jupyterlab_widgets \
        ipykernel \
        ipywidgets \
        gdown \
        OhMyRunPod

# Install RunPod File Uploader
RUN curl -sSL https://github.com/kodxana/RunPod-FilleUploader/raw/main/scripts/installer.sh -o installer.sh && \
    chmod +x installer.sh && \
    ./installer.sh

# Install rclone
RUN curl https://rclone.org/install.sh | bash

# Install runpodctl
RUN wget https://github.com/runpod/runpodctl/releases/download/v1.13.0/runpodctl-linux-amd64 -O runpodctl && \
    chmod a+x runpodctl && \
    mv runpodctl /usr/local/bin

# Install croc
RUN curl https://getcroc.schollz.com | bash

# Install speedtest CLI
RUN curl -s https://packagecloud.io/install/repositories/ookla/speedtest-cli/script.deb.sh | bash && \
    apt -y install speedtest

# Remove existing SSH host keys
RUN rm -f /etc/ssh/ssh_host_*

# NGINX Proxy
COPY docker/llava/llava-docker/nginx/nginx.conf /etc/nginx/nginx.conf
COPY docker/llava/llava-docker/nginx/502.html /usr/share/nginx/html/502.html

# Set template version
ENV TEMPLATE_VERSION=1.4.4

# Copy the scripts
WORKDIR /
COPY --chmod=755 docker/llava/llava-docker/scripts/* ./

# Start the container
SHELL ["/bin/bash", "--login", "-c"]
CMD [ "/start.sh" ]
#RUN /start.sh
#CMD ["python", "-m", "llava.serve.api", "-H", "0.0.0.0", "-p", "5000"]