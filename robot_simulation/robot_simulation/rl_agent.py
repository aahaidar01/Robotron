import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan 
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty 

from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState

import numpy as np
import math
import os
from datetime import datetime
import argparse
import random

class RLAgent(Node):

    def __init__(self, checkpoint=None):
        super().__init__('rl_agent')
        
        
        self.target_coords = (0.0, 1.8) # target position
        self.action_space = [0, 1, 2]  # Forward, Left, Right
        
        # --- SPAWN CONFIGURATION ---
        self.random_spawn_enabled = False  # Toggle: False = fixed spawn, True = random
        self.fixed_spawn = (0.0, 0.0, 0.0)  # Default fixed spawn (updated per world)
        self.spawns = [
            # Easy
            (0.5, 1.15, 1.57),
            (0.5, 1.8, 3.14),
            # Medium
            (0.5, -0.5, 1.57),
            (0.0, -0.5, 1.57),
            # Hard
            (-0.5, -2.1, 1.57),
        ]
        
        self.set_state_client = self.create_client(SetEntityState, '/set_entity_state')
        
        
        # --- STATE SPACE (512 States) ---
        # Paper uses 8 Lidar + 1 Visibility + 1 Target Angle (Total 10 elements).
        # We optimize Lidar to 5 sectors (Front, FL, FR, L, R) because Turtlebot can't strafe/reverse.
        # Structure: 5 Bits (Lidar) + 1 Bit (Visible) + 3 Bits (Target Sector 0-7)
        self.num_lidar_bits = 5  # 5 sectors: F, FL, L, R, FR
        self.num_vis_bits = 1
        self.num_target_sectors = 8
        
        # --- HYPERPARAMETERS ---
        self.alpha = 0.1       # Learning Rate
        self.gamma = 0.95      # Discount Factor
        self.epsilon = 1.0     # Exploration Rate
        self.epsilon_decay = 0.9990 # Slower decay for better learning
        self.epsilon_min = 0.05
        
        self.alpha_decay = 0.9995
        self.alpha_min = 0.01
        
        # --- Q-TABLE ---
        self.num_states = (2**self.num_lidar_bits) * (2**self.num_vis_bits) * self.num_target_sectors
        self.q_table = np.zeros((self.num_states, len(self.action_space)))
        
        self.get_logger().info(f"Q-Table Initialized with {self.num_states} states.")
        
        # --- ROS INFRASTRUCTURE ---
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.reset_client = self.create_client(Empty, '/reset_simulation')

        # --- STATE VARIABLES ---
        self.lidar_state = [0] * self.num_lidar_bits
        self.target_vis = 0
        self.robot_pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.target_sector = 0
        
        self.current_state_idx = 0
        self.previous_state_idx = 0
        self.previous_action = 0
        self.done = False
        self.termination_reason = "running"
        
        # --- LIDAR PROCESSING VARIABLES ---
        self.scan_ranges = None
        self.scan_ranges_smoothed = None
        self.angle_min = None
        self.angle_max = None
        self.angle_increment = None
        self.range_min = 0.15
        self.range_max = 12.0
        
        self.scans_since_reset = 0
        
        # --- SECTOR DEFINITIONS ---
        self.sectors = None
        self.safe_distance_threshold = 0.4  # distance to obstacle threshold
        self.collision_threshold = 0.20 # distance to obstacle for immediate collision
        self.target_radius = 0.30  # distance to target for success

        self.reward_success = 2500.0  # Must be > (max_steps * penalty)
        self.reward_collision = -200.0
        
        # --- REWARD SHAPING ---
        self.prev_dist = None
        
        # --- VFH-QL REWARD CONFIGURATION ---
        self.reward_mode = 'paper'  # 'paper' or 'hybrid'
        self.action_history = []
        self.max_action_history = 3
        
        # --- EPISODE TRACKING ---
        self.episode_num = 0
        self.episode_reward = 0
        self.step_count = 0
        self.max_steps = 1000
        
        self.max_episodes = 10000
        
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_successes = []
        self.episode_collisions = []
        self.episode_timeouts = []
        self.action_counts = {0: 0, 1: 0, 2: 0}
    
        
        # --- Q-TABLE PERSISTENCE ---
        self.log_dir = 'training_logs'
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.log_dir = 'training_logs'
        os.makedirs(self.log_dir, exist_ok=True)

        self.checkpoint = checkpoint
        if self.checkpoint == 'fresh':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.q_table_dir = f'q_tables/run_{timestamp}'
            os.makedirs(self.q_table_dir, exist_ok=True)
            self.get_logger().info(f"Starting fresh: {self.q_table_dir}")
        else:
            # checkpoint is full path like "run_20250128_143022/q_table_ep500.npy"
            self.q_table_dir = os.path.join('q_tables', os.path.dirname(checkpoint))
            if not os.path.exists(self.q_table_dir):
                raise FileNotFoundError(f"Run folder not found: {self.q_table_dir}")
        
        self.load_q_table()
        
        
        # --- LOGGING ---
        self.setup_logging()
        
        # --- CONTROL LOOP ---
        self.timer = self.create_timer(0.18, self.control_loop)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("RL Agent Started")
        self.get_logger().info(f"Target: {self.target_coords}")
        self.get_logger().info(f"State Space: {self.num_states} states")
        self.get_logger().info(f"Random Spawn: {'ENABLED' if self.random_spawn_enabled else 'DISABLED'}")
        self.get_logger().info("=" * 60)

    def setup_logging(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f'training_logs/training_log_{timestamp}.txt'
        
        with open(self.log_filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("VFH-QL Training Log\n")
            f.write(f"Started: {timestamp}\n")
            f.write(f"Target: {self.target_coords}\n")
            f.write(f"Hyperparameters: α={self.alpha}, γ={self.gamma}, ε_decay={self.epsilon_decay}\n")
            f.write("=" * 80 + "\n\n")
            f.write("Episode,Steps,Reward,Success,Collision,Timeout,Epsilon,Alpha,Fwd,Left,Right\n")

    def load_q_table(self):
        if self.checkpoint == 'fresh':
            self.get_logger().info("Starting fresh training (no checkpoint loaded)")
            return
        
        filepath = os.path.join('q_tables', self.checkpoint)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        try:
            loaded_table = np.load(filepath)
            
            if loaded_table.shape != self.q_table.shape:
                raise ValueError(f"Shape mismatch: {loaded_table.shape} vs {self.q_table.shape}")
            
            self.q_table = loaded_table
            episode_num = int(os.path.basename(filepath).split('ep')[1].split('.')[0])
            self.episode_num = episode_num
            self.epsilon = max(self.epsilon_min, 1.0 * (self.epsilon_decay ** episode_num))
            self.alpha = max(self.alpha_min, 0.1 * (self.alpha_decay ** episode_num))
            
            self.get_logger().info(f"✓ Loaded: {self.checkpoint}")
            self.get_logger().info(f"✓ Episode {episode_num}, ε={self.epsilon:.3f}, α={self.alpha:.4f}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Q-table: {e}")

    def _initialize_sectors(self):
        """
        Initialize sector definitions (Corrected).
        Fixes:
        1. Continuous boundaries (No 1-degree gaps).
        2. Exact 180-degree FOV (36 degrees per sector).
        3. Standard Spatially-Contiguous Indexing for easier debugging.
        
        Indices:
        0: Far Left   (+54 to +90)
        1: Left       (+18 to +54)
        2: Front      (-18 to +18)  <-- CENTER
        3: Right      (-54 to -18)
        4: Far Right  (-90 to -54)
        """
        # We use strict contiguous floats to ensure 22.5 falls into one bin.
        # Ranges are in degrees (start, end)
        sector_definitions_deg = {
            0: [(54, 90)],       # Far Left
            1: [(18, 54)],       # Left
            2: [(342, 18)],      # Front (Handles wrap: 342->360 & 0->18)
            3: [(306, 342)],     # Right
            4: [(270, 306)]      # Far Right
        }
        
        self.sectors = {}
        for idx, ranges in sector_definitions_deg.items():
            self.sectors[idx] = []
            for start_deg, end_deg in ranges:
                start_rad = math.radians(start_deg)
                end_rad = math.radians(end_deg)
                
                # Handle Wraparound (e.g. Front Sector: 342 to 18)
                if start_deg > end_deg:  
                    self.sectors[idx].append((start_rad, 2 * math.pi))
                    self.sectors[idx].append((0.0, end_rad))
                else:
                    self.sectors[idx].append((start_rad, end_rad))
                    
        self.get_logger().info(f"✓ Sectors initialized. Safe distance: {self.safe_distance_threshold}m")

    def scan_callback(self, msg):
        """Paper's VFH Peak Finder Algorithm"""
        if self.angle_min is None:
            self.angle_min = msg.angle_min
            self.angle_max = msg.angle_max
            self.angle_increment = msg.angle_increment
            self.range_min = msg.range_min
            self.range_max = msg.range_max
            self._initialize_sectors()
        
        # Reset state
        self.lidar_state = [0] * self.num_lidar_bits
        self.target_vis = 0
        
        # Clean data
        self.scan_ranges = np.array(msg.ranges, dtype=np.float32)
        self.scan_ranges[~np.isfinite(self.scan_ranges)] = self.range_max
        self.scan_ranges = np.clip(self.scan_ranges, self.range_min, self.range_max)
        
    
        # Pipeline
        self._check_collision()           
        self._smooth_range_vals()                   
        self._update_lidar_state()        
        self._target_visible()            

    def _check_collision(self):
        
        #Disregard the first 3 readings to make sure the sensor is stable        
        if self.scans_since_reset < 3:
            self.scans_since_reset += 1
            return
        
        # Don't flag collision if we're at the target (success takes priority)
        dist_to_target = math.sqrt(
            (self.target_coords[0] - self.robot_pose['x'])**2 + 
            (self.target_coords[1] - self.robot_pose['y'])**2
        )
        if dist_to_target < self.target_radius:
            return
        
        if np.min(self.scan_ranges) < self.collision_threshold:
            self.done = True
            stop_msg = Twist()
            self.cmd_pub.publish(stop_msg)

    def _smooth_range_vals(self):
        """11-point moving average with circular padding"""
        r = self.scan_ranges
        r_pad = np.r_[r[-5:], r, r[:5]]
        kernel = np.ones(11, dtype=np.float32) / 11.0
        self.scan_ranges_smoothed = np.convolve(r_pad, kernel, mode='same')[5:-5]

    def _normalize_angle(self, angle):
        """Normalize angle to [0, 2π)"""
        angle = angle % (2 * math.pi)
        if angle < 0:
            angle += 2 * math.pi
        return angle

    def _check_interval(self, angle, intervals):
        """Check if angle falls within any interval"""
        for lo, hi in intervals:
            if lo <= angle <= hi:
                return True
        return False

    def _update_lidar_state(self):
        """
        Corrected Logic:
        Instead of finding peaks (which fails on flat walls/open space),
        we check the MINIMUM distance in each sector.
        
        If min_dist < threshold -> Sector is BLOCKED (0)
        If min_dist > threshold -> Sector is FREE (1)
        """
        # 1. Reset state
        self.lidar_state = [0] * self.num_lidar_bits
        
        # 2. Map every laser ray to a sector
        # We create a dictionary to hold all valid ranges for each sector
        sector_ranges = {i: [] for i in range(self.num_lidar_bits)}
        
        N = len(self.scan_ranges_smoothed)
        
        for i in range(N):
            # Calculate angle of this specific ray
            angle = self.angle_min + i * self.angle_increment
            norm_angle = self._normalize_angle(angle)
            dist = self.scan_ranges_smoothed[i]
            
            # Check which sector this ray belongs to
            for sector_idx, intervals in self.sectors.items():
                if self._check_interval(norm_angle, intervals):
                    sector_ranges[sector_idx].append(dist)
                    break 
        
        # 3. Determine Binary State based on MINIMUM distance (Safety First)
        for idx in range(self.num_lidar_bits):
            readings = sector_ranges[idx]
            
            if not readings:
                # No readings in this sector (rare, but possible with strange FOV)
                # Assume clear or maintain previous state
                self.lidar_state[idx] = 0 
                continue
                
            # CRITICAL FIX: The Paper's "Occupancy" check.
            # If the closest object is far away, the sector is FREE (1).
            # If there is ANY object close by, it is OCCUPIED (0).
            min_dist = np.min(readings)
            
            if min_dist > self.safe_distance_threshold:
                self.lidar_state[idx] = 1  # Free
            else:
                self.lidar_state[idx] = 0  # Occupied

    def _target_visible(self):
        """
        Ray-trace visibility check.
        FIX: Proper angle frame conversion.
        """
        dx = self.target_coords[0] - self.robot_pose['x']
        dy = self.target_coords[1] - self.robot_pose['y']
        dT = np.hypot(dx, dy)
        
        # Angle to target (global frame)
        theta_global = math.atan2(dy, dx)
        
        # Convert to robot frame
        theta_robot = theta_global - self.robot_pose['yaw']
        
        # Normalize to LiDAR's angle range (not [0, 2π)!)
        while theta_robot < self.angle_min:
            theta_robot += 2 * math.pi
        while theta_robot >= self.angle_max:
            theta_robot -= 2 * math.pi
        
        # Calculate index
        idx_target = int(round((theta_robot - self.angle_min) / self.angle_increment))
        idx_target = np.clip(idx_target, 0, len(self.scan_ranges) - 1)
        
        lidar_reading = self.scan_ranges[idx_target]
        self.target_vis = 1 if (lidar_reading > dT) else 0

    def odom_callback(self, msg):
        self.robot_pose['x'] = msg.pose.pose.position.x
        self.robot_pose['y'] = msg.pose.pose.position.y
        
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_pose['yaw'] = math.atan2(siny_cosp, cosy_cosp)
        
        self._calculate_target_sector()

    def _calculate_target_sector(self):
        dx = self.target_coords[0] - self.robot_pose['x']
        dy = self.target_coords[1] - self.robot_pose['y']
        global_target_angle = math.atan2(dy, dx)
        relative_angle = global_target_angle - self.robot_pose['yaw']
        
        while relative_angle > math.pi:
            relative_angle -= 2 * math.pi
        while relative_angle < -math.pi:
            relative_angle += 2 * math.pi
        
        deg = math.degrees(relative_angle)
        if deg < 0:
            deg += 360
        
        self.target_sector = int((deg + 22.5) / 45.0) % 8

    def get_state_index(self):
        lidar_int = 0
        for i, val in enumerate(self.lidar_state):
            lidar_int += (val * (2**i))
        
        state_idx = (lidar_int * 16) + (self.target_vis * 8) + self.target_sector
        return state_idx
    

    def _is_action_open(self, action):
        """Check if action direction leads to an open way (VFH Criterion 1)."""
        if action == 0:  # Forward
            return self.lidar_state[2] == 1
        elif action == 1:  # Turn Left
            return self.lidar_state[1] == 1 or self.lidar_state[0] == 1
        elif action == 2:  # Turn Right
            return self.lidar_state[3] == 1 or self.lidar_state[4] == 1
        return False

    def _get_optimal_open_action(self):
        """Find the open action closest to target direction (VFH Criterion 2)."""
        sector_action_priority = {
            0: [0, 1, 2],  # Target ahead
            1: [1, 0, 2],  # Target left-front
            2: [1, 0, 2],  # Target left
            3: [1, 2, 0],  # Target left-back
            4: [1, 2, 0],  # Target behind
            5: [2, 1, 0],  # Target right-back
            6: [2, 0, 1],  # Target right
            7: [0, 2, 1],  # Target right-front
        }
        
        priority_list = sector_action_priority.get(self.target_sector, [0, 1, 2])
        
        for action in priority_list:
            if self._is_action_open(action):
                return action
        return 0

    def _get_action_angle_difference(self, action1, action2):
        """Get angle difference between two actions in degrees."""
        action_angles = {0: 0, 1: 60, 2: -60}
        diff = abs(action_angles.get(action1, 0) - action_angles.get(action2, 0))
        return min(diff, 360 - diff)

    def _check_oscillation(self):
        """Check if robot is oscillating (L-R-L or R-L-R pattern)."""
        if len(self.action_history) < 3:
            return False
        last_3 = self.action_history[-3:]
        if last_3[0] in [1, 2] and last_3[1] in [1, 2] and last_3[2] in [1, 2]:
            if last_3[0] != last_3[1] and last_3[1] != last_3[2] and last_3[0] == last_3[2]:
                return True
        return False


    def get_reward(self):
        """VFH-QL Reward Function (Modified for Trap Safety)"""
        dist = math.sqrt(
            (self.target_coords[0] - self.robot_pose['x'])**2 + 
            (self.target_coords[1] - self.robot_pose['y'])**2
        )
        
        # --- TERMINAL CONDITIONS ---
        if dist < self.target_radius:
            self.termination_reason = "success"
            return self.reward_success , True
            
        if self.done: # This flag comes from collision check in scan_callback
            self.termination_reason = "collision"
            return self.reward_collision, True
        
        action = self.previous_action
        reward = 0.0
        
        optimal_action = self._get_optimal_open_action()
        action_is_open = self._is_action_open(action)
        
        # VFH Logic
        if not action_is_open:
            reward += -5.0
        else:
            if action == optimal_action:
                reward += -1.0 # Small time penalty
            else:
                angle_diff = self._get_action_angle_difference(action, optimal_action)
                penalty = -2.0 - (angle_diff / 90.0) * 3.0
                reward += max(-5.0, penalty)
        
        # Hybrid Logic
        if self.reward_mode == 'hybrid':
            if self.prev_dist is not None:
                delta_dist = self.prev_dist - dist
                if action == optimal_action and delta_dist < 0:
                     reward += 0.0 
                else:
                     reward += 20.0 * delta_dist
            
            if self.target_vis == 1:
                reward += 1.0
            
            if self._check_oscillation():
                reward += -3.0
        
        self.prev_dist = dist
        return reward, False

    def choose_action(self, state_idx):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.q_table[state_idx])

    
    def execute_action(self, action):
        msg = Twist()

        if action == 0:    # Forward
            msg.linear.x = 0.18
            msg.angular.z = 0.0
        elif action == 1:  # Left (in-place rotation)
            msg.linear.x = 0.03
            msg.angular.z = 0.5
        elif action == 2:  # Right (in-place rotation)
            msg.linear.x = 0.03
            msg.angular.z = -0.5
        
        # Track action history for anti-oscillation
        self.action_history.append(action)
        if len(self.action_history) > self.max_action_history:
            self.action_history.pop(0)
            
            
        self.cmd_pub.publish(msg)
        self.action_counts[action] += 1

    def control_loop(self):
        if self.scan_ranges is None or self.sectors is None:
            return
        
        if self.scans_since_reset < 3:
            return
        
        current_state_idx = self.get_state_index()
        reward, done = self.get_reward()
        self.episode_reward += reward
        
        # Q-Learning Update
        if self.step_count > 0:
            old_q = self.q_table[self.previous_state_idx, self.previous_action]
            if done:
                target = reward
            else:
                next_max = np.max(self.q_table[current_state_idx])
                target = reward + self.gamma * next_max
            
            new_q = old_q + self.alpha * (target - old_q)
            self.q_table[self.previous_state_idx, self.previous_action] = new_q
            
        # Check termination
        if done or self.step_count >= self.max_steps:
            if self.step_count >= self.max_steps and not done:
                self.termination_reason = "timeout"
            self.end_episode(done, self.step_count >= self.max_steps)
            return
        
        action = self.choose_action(current_state_idx)
        self.execute_action(action)
        
        self.previous_state_idx = current_state_idx
        self.previous_action = action
        self.step_count += 1



    def end_episode(self, collision_or_success, timeout):
        self.episode_num += 1

        # Priority Logic: Success > Collision > Timeout
        if self.termination_reason == "success":
            outcome = "SUCCESS ✓"
            is_success = 1
            is_collision = 0
            is_timeout = 0
        elif self.termination_reason == "collision":
            outcome = "COLLISION ✗"
            is_success = 0
            is_collision = 1
            is_timeout = 0
        else:
            # If done is True but neither success nor collision, or timeout happened
            outcome = "TIMEOUT ⏱"
            is_success = 0
            is_collision = 0
            is_timeout = 1
        
        self.episode_rewards.append(self.episode_reward)
        self.episode_steps.append(self.step_count)
        self.episode_successes.append(is_success)
        self.episode_collisions.append(is_collision)
        self.episode_timeouts.append(is_timeout)
        
        self.get_logger().info(
            f"Episode {self.episode_num:4d} | "
            f"{outcome:12s} | "
            f"Steps: {self.step_count:4d} | "
            f"Reward: {self.episode_reward:7.1f} | "
            f"ε: {self.epsilon:.3f}"
        )

        self.log_to_file(is_success, is_collision, is_timeout)
        
        if self.episode_num % 10 == 0:
            self.log_statistics()
        
        if self.episode_num % 100 == 0:
            self.save_q_table()
        
        self.reset_simulation()

    def log_statistics(self):
        last_10_rewards = self.episode_rewards[-10:]
        last_10_steps = self.episode_steps[-10:]
        last_10_successes = self.episode_successes[-10:]
        
        avg_reward = np.mean(last_10_rewards)
        avg_steps = np.mean(last_10_steps)
        success_rate = np.sum(last_10_successes) / 10.0 * 100
        
        self.get_logger().info("=" * 80)
        self.get_logger().info(f"📊 STATISTICS (Episodes {self.episode_num-9} - {self.episode_num})")
        self.get_logger().info(f"   Avg Reward:    {avg_reward:7.1f}")
        self.get_logger().info(f"   Avg Steps:     {avg_steps:7.1f}")
        self.get_logger().info(f"   Success Rate:  {success_rate:5.1f}%")
        
        total_actions = sum(self.action_counts.values())
        if total_actions > 0:
            fwd_pct = self.action_counts[0] / total_actions * 100
            left_pct = self.action_counts[1] / total_actions * 100
            right_pct = self.action_counts[2] / total_actions * 100
            self.get_logger().info(
                f"   Actions:       Fwd={fwd_pct:.1f}%, Left={left_pct:.1f}%, Right={right_pct:.1f}%"
            )
        
        self.get_logger().info("=" * 80)
        # FIX: Don't reset here - move to reset_simulation()

    def log_to_file(self, success, collision, timeout):
        total_actions = sum(self.action_counts.values())
        fwd_pct = self.action_counts[0] / max(total_actions, 1) * 100
        left_pct = self.action_counts[1] / max(total_actions, 1) * 100
        right_pct = self.action_counts[2] / max(total_actions, 1) * 100
        
        with open(self.log_filename, 'a') as f:
            f.write(
                f"{self.episode_num},"
                f"{self.step_count},"
                f"{self.episode_reward:.2f},"
                f"{int(success)},"
                f"{int(collision)},"
                f"{int(timeout)},"
                f"{self.epsilon:.4f},"
                f"{self.alpha:.4f},"
                f"{fwd_pct:.1f},"
                f"{left_pct:.1f},"
                f"{right_pct:.1f}\n"
            )

    def save_q_table(self):
        filename = f'q_table_ep{self.episode_num}.npy'
        filepath = os.path.join(self.q_table_dir, filename)
        np.save(filepath, self.q_table)
        self.get_logger().info(f"💾 Q-table saved: {filename}")
        np.save(os.path.join(self.q_table_dir, 'q_table_latest.npy'), self.q_table)

    def teleport_robot(self, x, y, yaw):
        """Teleport robot to specified pose using Gazebo SetEntityState."""
        if not self.set_state_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('Set entity state service not available')
            return False
        
        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name = 'my_turtlebot'  # Must match entity name in launch file
        req.state.pose.position.x = x
        req.state.pose.position.y = y
        req.state.pose.position.z = 0.1
        
        # Yaw to quaternion
        req.state.pose.orientation.x = 0.0
        req.state.pose.orientation.y = 0.0
        req.state.pose.orientation.z = math.sin(yaw / 2.0)
        req.state.pose.orientation.w = math.cos(yaw / 2.0)
        
        # Zero velocity
        req.state.twist.linear.x = 0.0
        req.state.twist.angular.z = 0.0
        
        future = self.set_state_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        
        return True
    
    def reset_simulation(self):
        # Stop the robot first
        stop_msg = Twist()
        self.cmd_pub.publish(stop_msg)
        
        if self.random_spawn_enabled:
            # Random spawn: use teleport (no /reset_simulation to preserve odom frame)
            spawn = random.choice(self.spawns)
            self.teleport_robot(spawn[0], spawn[1], spawn[2])
            time.sleep(0.3)
            self.get_logger().info(f"Spawned at ({spawn[0]:.1f}, {spawn[1]:.1f}, yaw={spawn[2]:.2f})")
        else:
            # Fixed spawn: use /reset_simulation (resets odom cleanly to origin)
            req = Empty.Request()
            while not self.reset_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn('Reset service not available, waiting...')
            future = self.reset_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            time.sleep(0.5)
        
        # Reset state variables
        self.scan_ranges = None
        self.scans_since_reset = 0
        self.done = False
        self.termination_reason = "running"
        self.step_count = 0
        self.episode_reward = 0
        self.previous_state_idx = 0
        self.previous_action = 0
        self.prev_dist = None
        self.action_counts = {0: 0, 1: 0, 2: 0}
        self.action_history = []
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if self.alpha > self.alpha_min:
            self.alpha *= self.alpha_decay

def main(args=None):
    parser = argparse.ArgumentParser(description='VFH-QL Training')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--fresh', action='store_true', help='Start fresh training')
    group.add_argument('--checkpoint', type=str, help='Checkpoint to load (filename or "latest")')
    
    parsed_args, ros_args = parser.parse_known_args()
    
    rclpy.init(args=ros_args)
    
    if parsed_args.fresh:
        checkpoint = 'fresh'
    elif parsed_args.checkpoint == 'latest':
        checkpoint = None  # triggers latest loading
    else:
        checkpoint = parsed_args.checkpoint
    
    try:
        node = RLAgent(checkpoint=checkpoint)
        rclpy.spin(node)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        rclpy.shutdown()
        return
    except KeyboardInterrupt:
        node.get_logger().info("Training interrupted by user.")
    finally:
        if 'node' in locals():
            node.save_q_table()
            node.get_logger().info("Final Q-table saved. Shutting down...")
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()