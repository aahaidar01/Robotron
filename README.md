# Robotron: Sim2Real Autonomous Navigation using VFH-QL

A physics-based extension of the VFH-QL algorithm (Abdalmanan et al., 2023) from 2D pixel simulation to Gazebo/ROS 2, with sim-to-real deployment on a physical DAGU Wild Thumper 4WD robot.

**Imperial College London — MSc Applied Machine Learning / AML Devices Module Project**
Authors: Ali Haidar, Andria Kyriacou, Mahmoud El Etreby
Reference: Abdalmanan et al., *IEEE Access* (2023). DOI: [10.1109/ACCESS.2023.3265207](https://doi.org/10.1109/ACCESS.2023.3265207)

---

## Overview

Robotron trains a tabular Q-learning agent to navigate unknown environments using only LiDAR and a target bearing, then deploys the learned policy onto real hardware with no fine-tuning. The project reproduces the core VFH-QL formulation from the reference paper and introduces a set of targeted modifications that unlock hard-spawn performance and enable sim-to-real transfer.

## Novel Contributions

Relative to Abdalmanan et al. (2023), this work introduces:

- **Forward-facing sector reduction** — 8 omnidirectional LiDAR sectors reduced to 5 forward-facing sectors, eliminating decision-irrelevant rear observations and compressing the state space to 512 states.
- **Distance-to-target zone encoding** — 4 discrete distance zones added to the state to resolve positional aliasing, expanding the space to 2,048 states.
- **Composite reward function** — zone-transition bonuses, anti-oscillation penalty, and a visibility reward term, designed to shape exploration under sparse goal feedback.
- **Curriculum learning** — 11 spawn positions across 4 difficulty levels, progressing from easy to full-maze starts.
- **Sim-to-real deployment** — physical hardware rollout on a DAGU Wild Thumper 4WD platform (the reference paper used only a Python grid simulation).
- **Custom URDF digital twin** — DAGU Wild Thumper model replacing the standard TurtleBot3 for physics-accurate training.

## Results

| Configuration | Overall Success | Hard / Full-Maze Spawns |
|---|---|---|
| 2,048-state + curriculum (final) | ~91% | 87–91% |
| 512-state baseline | ~55% | <7% |
| Fixed-spawn (no curriculum) baseline | 0% | 0% |

The jump from 512 → 2,048 states (distance-zone encoding) and the use of curriculum learning were each individually necessary; removing either collapses hard-spawn performance.

## System Architecture

### Simulation & Training (ROS 2 / Gazebo / Python)
- Physics-based Gazebo environment with Gaussian sensor noise
- `rl_agent.py` — tabular Q-learning loop with curriculum scheduler
- Custom Wild Thumper URDF with calibrated inertial and friction parameters
- Q-table exported as a static C++ header (`q_table.h`) for firmware use

### Hardware Firmware (C++ / Arduino Portenta H7)
Modular architecture:
- `lidar_sensor.h/.cpp` — RPLidar A1 driver and sector aggregation
- `chassis.h/.cpp` — skid-steer motor control with feedforward + PI
- `config.h` — pin mappings, tuning constants, thresholds
- `q_table.h` — exported policy (static)
- `main.ino` — control loop and state-machine

## Hardware Stack

- **Chassis:** DAGU Wild Thumper 4WD
- **MCU:** Arduino Portenta H7 + Hat Carrier (mbed pin naming: `PD_4`, `PG_10`, `PJ_7`, etc.)
- **LiDAR:** RPLidar A1 (on `Serial1`)
- **IMU:** BNO055 (onboard Kalman filter handles short-episode drift)
- **Motor driver:** Cytron MDD20A
- **Motors:** Pololu 34:1 HP 6V 25D gearmotors

---

## Repository Layout

```
Robotron/
├── arduino/                              # Portenta H7 firmware
│   ├── main.ino                          # Entry point + Ethernet control state machine
│   ├── chassis.{h,cpp}                   # Skid-steer control, encoder odom, IMU fusion, PI + FF
│   ├── lidar_sensor.{h,cpp}              # RPLidar A1 driver, sector aggregation, state index
│   ├── config.h                          # Target, spawn, logging constants
│   ├── q_table.h                         # Exported policy (2048 × 3 floats)
│   └── tests/
│       ├── test_motors/                  # Basic PWM sweep
│       ├── test_motors_bt/               # HC-05 Bluetooth feedforward calibration
│       ├── test_motors_eth/              # Ethernet-controlled motor test
│       ├── test_encoders/                # Encoder tick + odometry sanity
│       ├── test_imu/                     # BNO055 sanity
│       ├── test_lidar/                   # RPLidar A1 sanity
│       └── test_eth_connection/          # Portenta Ethernet link check
├── robot_simulation/                     # ROS 2 (ament_python) package
│   ├── package.xml
│   ├── setup.py                          # Entry points: rl_agent, lidar_debug
│   ├── launch/
│   │   └── simulation.launch.py          # Cleans Gazebo, spawns URDF, loads world
│   ├── urdf/
│   │   ├── dagu.urdf                     # Wild Thumper digital twin (active)
│   │   └── my_robot.urdf                 # Legacy TurtleBot-derivative
│   ├── worlds/
│   │   ├── maze.world                    # Full 4-splitter curriculum maze
│   │   ├── maze_zig_zag.world            # Default world in the launch file
│   │   ├── maze_simple.world
│   │   ├── maze_easy.world
│   │   ├── u_shape.world
│   │   └── my_world.world
│   ├── meshes/                           # URDF visual / collision meshes
│   └── robot_simulation/
│       ├── rl_agent.py                   # Q-learning trainer (main)
│       └── lidar.py                      # Terminal LiDAR sector dashboard
└── utils/
    ├── export_qtable.py                  # .npy → q_table.h
    ├── plot_training_fixed_spawn.py
    ├── plot_training_curriculum_spawn.py
    └── plot_training_curriculum_spawn_classify.py
```

---

## Main Scripts

### `robot_simulation/robot_simulation/rl_agent.py` — Q-learning trainer

The main training script. Runs a ROS 2 node (`rl_agent`) that consumes `/scan` and `/odom`, publishes `/cmd_vel`, and resets the robot each episode via either `/reset_simulation` (fixed spawn) or `/set_entity_state` (curriculum teleport).

**CLI**
```bash
# Fresh run
ros2 run robot_simulation rl_agent --fresh

# Resume from a checkpoint (relative to ./q_tables/)
ros2 run robot_simulation rl_agent --checkpoint run_20250128_143022/q_table_ep500.npy
```

**Outputs**
- `q_tables/run_<timestamp>/q_table_ep<N>.npy` — checkpoint every 100 episodes
- `q_tables/run_<timestamp>/q_table_latest.npy` — rolling latest
- `training_logs/training_log_<timestamp>.txt` — CSV with reward, steps, success/collision/timeout, ε, α, action distribution, spawn index

### `robot_simulation/robot_simulation/lidar.py` — LiDAR dashboard (entry point `lidar_debug`)

A debugging node that clears the terminal at 5 Hz and prints the robot pose, the 5-sector occupancy grid (Front/Left/Right/FarLeft/FarRight), target sector, target visibility, and the derived state index. Useful when tuning `safe_distance_threshold` or verifying sector geometry.

```bash
ros2 run robot_simulation lidar_debug
```

### `robot_simulation/launch/simulation.launch.py`

Kills any stale Gazebo processes, sets `GAZEBO_MODEL_PATH`, launches Gazebo, publishes the robot description, and spawns the robot. To switch maze or robot, edit the variables at the top:

- `world_file` — path in `worlds/`; currently `maze_zig_zag.world`
- `urdf_file` — path in `urdf/`; currently `dagu.urdf`
- `robot_pose_maze` — `[x, y, z, yaw]` spawn for the spawner (only used when `rl_agent` is in fixed-spawn mode)

### `utils/export_qtable.py` — policy export

Converts a trained `.npy` Q-table into a C++ header for the firmware. **Edit the two hardcoded paths at the bottom** (`input_file`, `output_file`) before running:

```bash
python3 utils/export_qtable.py
```

The generated `q_table.h` writes `const float Q_TABLE[2048][3]`; copy it into `arduino/` before compiling the firmware.

### `utils/plot_training_*.py` — training-curve plotting

- `plot_training_fixed_spawn.py` — reward/steps/success curves for a fixed-spawn run
- `plot_training_curriculum_spawn.py` — curriculum run with per-spawn success breakdown
- `plot_training_curriculum_spawn_classify.py` — same as above with spawn-difficulty classification

All three read a `training_logs/*.txt` CSV — point them at your run.

### `arduino/main.ino` — firmware entry point

State machine: `WAIT_CLIENT → WAIT_TRIGGER → INITIALIZING → RUNNING`.

The Portenta serves Telnet on `192.168.1.177:23`. Connect from a host on the same subnet, send any character to arm, and the RL loop starts inferring from `Q_TABLE` at 5.5 Hz. Sending another character after an episode ends (success or collision) restarts.

### `arduino/tests/` — hardware bring-up sketches

| Sketch | Purpose |
|---|---|
| `test_motors` | PWM sweep to verify direction and deadzone |
| `test_motors_bt` | HC-05 Bluetooth PWM calibration (feedforward constants) |
| `test_motors_eth` | Ethernet-commanded motor test |
| `test_encoders` | Encoder ticks + derived odometry |
| `test_imu` | BNO055 orientation |
| `test_lidar` | RPLidar A1 scan |
| `test_eth_connection` | Portenta Ethernet link-up |

---

## RL Agent Parameters

All parameters live in `robot_simulation/robot_simulation/rl_agent.py`. File:line references point to the default values.

### Environment (rl_agent.py:25–58)

| Parameter | Default | Meaning |
|---|---|---|
| `target_coords` | `(0.0, 1.8)` | Goal position in world frame |
| `action_space` | `[0, 1, 2]` | Forward / Left / Right |
| `random_spawn_enabled` | `True` | `True` = curriculum (random from `spawns`), `False` = fixed spawn |
| `fixed_spawn` | `(-0.5, -2.1, 0.0)` | Used only when curriculum is off |
| `spawns[]` | 11 entries | Curriculum spawns across 4 difficulty tiers (see below) |

### State Space (rl_agent.py:66–70)

| Parameter | Default | Meaning |
|---|---|---|
| `num_lidar_bits` | `5` | Sectors: FarLeft, Left, Front, Right, FarRight |
| `num_vis_bits` | `1` | Binary target-visibility flag |
| `num_target_sectors` | `8` | Relative bearing buckets (45° each) |
| `num_distance_zones` | `4` | Distance-to-target buckets |
| `distance_zone_boundaries` | `[1.0, 2.0, 3.0]` m | Zone thresholds; zone 3 is anything ≥ 3 m |

Total states: `2⁵ × 2 × 4 × 8 = 2048`.

### Q-Learning Hyperparameters (rl_agent.py:73–80)

| Parameter | Default | Meaning |
|---|---|---|
| `alpha` | `0.1` | Initial learning rate |
| `alpha_decay` | `0.9998` | Per-episode α decay (α > 0.05 until ≈ ep 3500) |
| `alpha_min` | `0.02` | Floor |
| `gamma` | `0.95` | Discount factor |
| `epsilon` | `1.0` | Initial exploration rate |
| `epsilon_decay` | `0.9995` | Per-episode ε decay (ε > 0.1 until ≈ ep 4600) |
| `epsilon_min` | `0.05` | Floor |

### Sector & Safety Thresholds (rl_agent.py:119–121)

| Parameter | Default | Meaning |
|---|---|---|
| `safe_distance_threshold` | `0.6` m | Sector flips to BLOCKED below this minimum range |
| `collision_threshold` | `0.20` m | Any ray below this ends the episode |
| `target_radius` | `0.30` m | Success radius around `target_coords` |

### Reward Shaping (rl_agent.py:123–130, 516–578)

| Parameter | Default | Meaning |
|---|---|---|
| `reward_success` | `+2500.0` | Terminal success (must exceed `max_steps × per-step penalty`) |
| `reward_collision` | `-200.0` | Terminal collision |
| `reward_mode` | `'hybrid'` | Enables zone-transition + visibility + oscillation shaping |
| Zone transition | `±15.0` | Per-episode reward for moving to a closer/farther distance zone |
| Target-visible bonus | `+1.0` | Added when `target_vis == 1` |
| Oscillation penalty | `-3.0` | Triggered on L–R–L or R–L–R patterns (`max_action_history = 3`) |
| VFH penalty | `-1.0` to `-5.0` | Scales with angular deviation from the optimal open action; `-5.0` if the chosen action is blocked |

### Episode / Control Loop (rl_agent.py:138–181)

| Parameter | Default | Meaning |
|---|---|---|
| `max_steps` | `1000` | Episode-length cap → `timeout` |
| `max_episodes` | `10000` | Soft cap; training keeps running while `rclpy.spin` is alive |
| Control timer | `0.18` s | ~5.5 Hz — matched to firmware `CONTROL_LOOP_INTERVAL_MS` |

### Action Velocities (rl_agent.py:587–598)

| Action | `linear.x` | `angular.z` |
|---|---|---|
| 0 — Forward | `0.18` | `0.0` |
| 1 — Left (in-place) | `0.03` | `+0.5` |
| 2 — Right (in-place) | `0.03` | `-0.5` |

### Curriculum Spawns (rl_agent.py:35–58)

Sampled uniformly at random every episode. Easy spawns produce quick successes whose Q-values propagate backward to bootstrap harder starts.

| Level | Description | Count |
|---|---|---|
| 1 — Easy | Past Splitter-3, short straight path | 3 |
| 2 — Medium | Between Splitter-2 and Splitter-3 (one turn) | 2 |
| 3 — Hard | Between Splitter-1 and Splitter-2 (two turns) | 3 |
| 4 — Full maze | Start zone (three turns) | 3 |

---

## Firmware Parameters

### `arduino/config.h`

| Constant | Default | Meaning |
|---|---|---|
| `LOG_LEVEL` | `2` | `0` silent / `1` one-line-per-step / `2` detailed per-module |
| `TARGET_X`, `TARGET_Y` | `0.0`, `1.8` | Goal in world frame (must match training) |
| `TARGET_RADIUS` | `0.30` | Success radius (must match training) |
| `SPAWN_X`, `SPAWN_Y`, `SPAWN_YAW` | `-0.5`, `-2.1`, `0.0` | Physical robot placement; offsets odometry so it shares the Q-table's frame |

### `arduino/main.ino`

| Constant | Default | Meaning |
|---|---|---|
| `mac[]` | `DE:AD:BE:EF:FE:ED` | Portenta MAC |
| `ip` | `192.168.1.177` | Static IP (host PC must be on same subnet) |
| `ETH_PORT` | `23` | Telnet-style control port |
| `CONTROL_LOOP_INTERVAL_MS` | `180` | RL step period — keep synced with Python `self.timer` |
| `COLLISION_COOLDOWN_MS` | `1000` | Motor quiescence after each episode reset |

---

## Usage / Reproduction

### Prerequisites

- **Simulation host:** Ubuntu 22.04, ROS 2 Humble (or newer), Gazebo 11, `ros-<distro>-gazebo-ros-pkgs`, Python 3 with `numpy`, `matplotlib`.
- **Firmware host:** Arduino IDE or `arduino-cli` with the `arduino:mbed_portenta` core; Portenta-specific Ethernet library plus the LiDAR/IMU drivers referenced in `chassis.cpp` / `lidar_sensor.cpp`.

### 1. Build the ROS 2 package

From your colcon workspace root (with `Robotron/robot_simulation` inside `src/`):

```bash
colcon build --symlink-install --packages-select robot_simulation
source install/setup.bash
```

### 2. Launch the simulation

```bash
ros2 launch robot_simulation simulation.launch.py
```

This boots Gazebo with `maze_zig_zag.world` and spawns the DAGU URDF. To change either, edit `robot_simulation/launch/simulation.launch.py` and rebuild.

### 3. Train a Q-table

In a second terminal (with the workspace sourced):

```bash
# Fresh training run
ros2 run robot_simulation rl_agent --fresh

# Resume — path is relative to ./q_tables/
ros2 run robot_simulation rl_agent --checkpoint run_<timestamp>/q_table_ep500.npy
```

ε and α are auto-restored from the episode number encoded in the checkpoint filename.

### 4. Debug the LiDAR pipeline (optional)

With the simulation running:

```bash
ros2 run robot_simulation lidar_debug
```

### 5. Plot training curves

```bash
python3 utils/plot_training_curriculum_spawn.py   # or the fixed-spawn variant
# Edit the log path inside the script to point at training_logs/*.txt
```

### 6. Export the policy for firmware

```bash
# Edit input_file and output_file at the bottom of utils/export_qtable.py first
python3 utils/export_qtable.py
cp <output_path>/q_table.h arduino/q_table.h
```

### 7. Flash the firmware

```bash
arduino-cli compile --fqbn arduino:mbed_portenta:envie_m7 arduino/
arduino-cli upload  -p /dev/ttyACM0 --fqbn arduino:mbed_portenta:envie_m7 arduino/
```

### 8. Run on hardware

1. Place the robot physically at `(SPAWN_X, SPAWN_Y, SPAWN_YAW)` from `config.h`.
2. Connect your PC to the Portenta over Ethernet on the same subnet as `192.168.1.177`.
3. Open a TCP session:
   ```bash
   telnet 192.168.1.177 23       # or: nc 192.168.1.177 23
   ```
4. Send any character to arm. Telemetry streams back at `LOG_LEVEL` verbosity. After a `SUCCESS` or `COLLISION`, send another character to start the next episode.

---

## Key Engineering Learnings

- **Distance-zone state encoding unlocked hard-spawn performance.** A pure 5-sector + visibility + target-bearing state collapsed spatially distinct regions of the maze onto identical observations. Adding 4 distance zones expanded the space from 512 → 2,048 states, broke the aliasing, and lifted hard-spawn success from single digits to ~90%.
- **Explicit coordinate-frame handling across the sim-to-real boundary.** The RPLidar reports angles clockwise, while the Q-table is indexed in Gazebo's counter-clockwise frame. Keeping `lidar_target_angle` and `target_sector_angle` as separate variables (rather than a single shared bearing) made the frame conversion trivial and kept left/right target sectors consistent between simulation and hardware.
- **Feedforward + PI outperformed pure PID for discrete-action control.** Per-action feedforward PWM constants deliver the bulk of each command on cycle one; a low-gain PI loop handles the residual; integrator state resets on every action change. The result is crisp, jitter-free transitions that match the 5.5 Hz RL cadence.
- **Ordered motor-command pipeline for clean transitions.** Deadzone compensation runs *before* the slew limiter so the two stages compose correctly — the deadzone boost never gets clipped back out, and current draw stays within the PMIC's headroom across sharp action changes.
- **Curriculum learning was the key to generalisation.** Sampling uniformly across 11 spawns spanning four difficulty tiers let easy-spawn successes seed the Q-table early; those values then propagated backward to bootstrap the hard and full-maze starts, producing a single policy that generalises across the whole maze.

## Citation

If this work is useful, please cite the original VFH-QL paper:

> Abdalmanan et al., "VFH-QL: A Hybrid Approach for Autonomous Navigation," *IEEE Access*, 2023. DOI: 10.1109/ACCESS.2023.3265207
