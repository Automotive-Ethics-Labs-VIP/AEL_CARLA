# Team B - CARLA Custom Build for AEL

## Overview
Custom CARLA build for RLHF-based ethical decision-making with modified walker blueprints.

**Branch:** `team-b-custom-pedestrians`

## Workflow

### Local Development
1. Open `AEL_CARLA/Unreal/CarlaUE4/CarlaUE4.uproject` in Unreal Editor
2. Modify walker blueprints (add custom attributes)
3. Edit C++ spawn controller code
4. Commit and push changes

### HPC Build & Deploy
1. Pull latest changes on HPC
2. Build in Singularity container (2-4 hours)
3. Run headless CARLA server
4. Extract 40D state vectors via Python API

## Directory Structure
```
TeamB/
├── AEL_CARLA/           # Forked CARLA repo (Unreal + C++)
├── container/           # Docker/Singularity for HPC builds
├── build-scripts/       # Automated build scripts
├── python-api/          # State vector extraction
└── docs/                # Documentation
```

## State Vector (40D)
```
[velocity_ego, num_passengers, lane_position, velocity_delta,
 num_ped_if_straight, num_ped_if_left, num_ped_if_right,
 obstacle_type[3][11]]
```

## Quick Start

### 1. Modify Blueprints Locally
```bash
# Open Unreal Editor
cd AEL_CARLA
# Edit walker blueprints - see docs/walker_modifications.md
```

### 2. Build on HPC
```bash
# On HPC
cd build-scripts
sbatch hpc_build_job.sh
```

### 3. Run & Extract Data
```bash
# Run server
singularity exec carla-build.sif ./deploy/CarlaUE4Server

# Python client
python python-api/generate_training_data.py
```

## Documentation
- `docs/workflow.md` - Complete development workflow
- `docs/walker_modifications.md` - Blueprint modification guide
- `docs/hpc_quickstart.md` - HPC setup and usage

## Integration with Team A
Team B generates 40D state vectors → Team A trains RLHF model
