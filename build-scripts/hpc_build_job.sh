#!/bin/bash
#SBATCH --job-name=carla-build
#SBATCH --output=carla-build-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --partition=standard

# Load Singularity module (adjust for your HPC)
module load singularity

# Paths (UPDATE THESE FOR YOUR SETUP)
CARLA_SIF="/gpfs/projects/$USER/carla-build.sif"
CARLA_SOURCE="$(pwd)/AEL_CARLA"
OUTPUT_DIR="$(pwd)/deploy"

# Build CARLA in container
singularity exec \
    --bind ${CARLA_SOURCE}:/workspace/carla \
    --bind ${OUTPUT_DIR}:/workspace/deploy \
    ${CARLA_SIF} \
    /workspace/carla/build-scripts/build_carla.sh

echo "Build job completed"
