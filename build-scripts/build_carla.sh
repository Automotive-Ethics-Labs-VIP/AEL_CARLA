#!/bin/bash
# CARLA Build Script for HPC Container

set -e

echo "=== CARLA AEL Build Script ==="

# Configuration
CARLA_DIR="${CARLA_DIR:-/workspace/carla}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"

# Check if CARLA directory exists
if [ ! -d "$CARLA_DIR" ]; then
    echo "ERROR: CARLA directory not found at $CARLA_DIR"
    echo "Please mount CARLA source code to /workspace/carla"
    exit 1
fi

cd "$CARLA_DIR"

echo "Building CARLA in $BUILD_TYPE mode with $JOBS jobs..."

# Update CARLA content (downloads assets if needed)
# Skip if already done
if [ ! -d "Unreal/CarlaUE4/Content/Carla" ]; then
    echo "Downloading CARLA content..."
    ./Update.sh
fi

# Build CARLA
echo "Compiling CARLA..."
make PythonAPI -j$JOBS
make LibCarla -j$JOBS
make launch -j$JOBS

# Package server-only build
echo "Packaging server..."
make package ARGS="--packages=CarlaUE4"

echo "Build complete!"
echo "Package location: $(pwd)/Dist"
