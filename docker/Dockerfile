FROM python:3.11-slim

ENV DEBIAN_FRONTEND noninteractive

# Install system dependencies for convinient development inside container
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    curl \
    git \
    wget \
    vim \
    sudo \
    tar \
    unzip \
    openssh-server \
    python3-pip \
    build-essential \
    ninja-build \
    cmake \
    swig \
    libopenblas-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev\
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    python3-venv \
    git-lfs \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /home

RUN mkdir country_recognition && chown -R ${UID}:${GID} /home

ENV PATH="/home/.local/bin:$PATH"

# create venv
ENV VIRTUAL_ENV="/home/.venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python -m venv $VIRTUAL_ENV

# Ensure pip, wheel and setuptools are up to date
RUN python3 -m pip install --upgrade pip wheel setuptools
# RUN python3 -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install -r /tmp/requirements.txt

# COPY . .

# EXPOSE 5000

# CMD ["python", "app.py"]
