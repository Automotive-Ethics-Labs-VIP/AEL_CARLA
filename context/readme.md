# Sprint Progress - Team B Custom Pedestrians

## Current Status: Week 1

### Sprint Deliverables

#### Sprint 1: Working Codespace with Portability to HPC & Prebuilt Ethical Params
**Status:** In Progress

**Goals:**
- Establish portable development environment for local and HPC deployment
- Implement prebuilt ethical parameter schemas for walker attributes
- Ensure seamless transition between local development and HPC builds

**Key Components:**
- Container-based build system (Docker/Singularity)
- Ethical attribute schema implementation (see `/python_api/ethical_attributes.py`)
- Actor registry system for managing walker metadata (see `/python_api/actor_registry.py`)

**Resources:**
- `/docs/api/ethical_attributes_api.md` - Complete API documentation
- `/test/test_ethical_attributes.py` - Unit tests for ethical attributes
- `/python_api/spawn_walkers.py` - Walker spawning with ethical metadata

---

#### Sprint 2: Built Out Ethical Fork of CARLA
**Status:** Planned

**Goals:**
- Complete modification of CARLA walker blueprints with ethical attributes
- Implement vulnerability scoring system
- Extend walker spawning to support ethical decision-making scenarios

**Key Features:**
- **Ethical Attributes:**
  - Age Group (Child, Teen, Adult, Elderly)
  - Disability (None, Wheelchair, Cane, Blind)
  - Pregnancy (Boolean)
  - Group Size (1-5)
  - Social Role (Civilian, Emergency, Healthcare)
  - Vulnerability Score (0.0-1.0, computed)

**Resources:**
- `/docs/walker_modifications.md` - Blueprint modification guide
- `/python_api/ethical_attributes.py` - Core attribute schema

---

#### Sprint 3: Forked CARLA API with On-Demand Data Transfer for Training
**Status:** Planned

**Goals:**
- Develop state vector extraction system for RLHF training
- Implement on-demand data export capabilities
- Create integration layer for Team A's training pipeline

**Key Features:**
- 40D state vector extraction
- Scenario persistence (JSON export/import)
- Real-time walker attribute querying
- Thread-safe registry for concurrent access

**Resources:**
- `/python_api/state_vector_extractor.py` - State vector extraction
- `/python_api/generate_training_data.py` - Training data generation
- `/python_api/actor_registry.py` - Thread-safe actor management

---

## Development Workflow

### Local Development
1. Modify walker blueprints in Unreal Editor (`AEL_CARLA/Unreal/CarlaUE4/`)
2. Test ethical attributes via Python API
3. Run unit tests (`/test/`)
4. Commit and push changes

### HPC Deployment
1. Build CARLA in Singularity container (2-4 hours)
2. Deploy headless server
3. Extract training data via Python API
4. Transfer to Team A for RLHF training

---

## Integration Points

### Team A Integration
- **Output:** 40D state vectors with ethical attributes
- **Format:** JSON scenario files with walker metadata
- **Use Case:** RLHF-based ethical decision-making model training

### State Vector Format (40D)
```
[velocity_ego, num_passengers, lane_position, velocity_delta,
 num_ped_if_straight, num_ped_if_left, num_ped_if_right,
 obstacle_type[3][11]]
```

---

## Quick Links

### Documentation
- [Ethical Attributes API](/docs/api/ethical_attributes_api.md)
- [Walker Modifications](/docs/walker_modifications.md)
- [Main README](/README.md)

### Python API
- [Ethical Attributes Schema](/python_api/ethical_attributes.py)
- [Actor Registry](/python_api/actor_registry.py)
- [Walker Spawner](/python_api/spawn_walkers.py)
- [State Vector Extractor](/python_api/state_vector_extractor.py)
- [Training Data Generator](/python_api/generate_training_data.py)

### Testing
- [Ethical Attributes Tests](/test/test_ethical_attributes.py)

---

## Notes

**Branch:** `team-b-custom-pedestrians`

This project extends CARLA with ethical metadata for pedestrian walkers without modifying CARLA's core source code. The ethical attribute system enables AI training for scenarios involving vulnerable populations and ethical decision-making.
