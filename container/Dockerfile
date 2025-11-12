# CARLA Build Container for HPC
FROM ubuntu:18.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    clang-8 \
    lld-8 \
    g++-7 \
    cmake \
    ninja-build \
    libvulkan1 \
    python3-dev \
    python3-pip \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    tzdata \
    sed \
    curl \
    wget \
    unzip \
    autoconf \
    libtool \
    rsync \
    libxml2-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set clang as default compiler
ENV CC=clang-8
ENV CXX=clang++-8

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install numpy

# Create workspace
WORKDIR /workspace

# Copy CARLA source (will be mounted)
# Build will happen via mounted volume

# Default command
CMD ["/bin/bash"]
