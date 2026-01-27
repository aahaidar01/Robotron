import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan 
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty 
import numpy as np
import math
import os
from datetime import datetime

class RLAgent(Node):

    def __init__(self):
        super().__init__('rl_agent')
        
        # --- CONFIGURATION ---
        self.target_coords = (2.0, 0.0)
        self.action_space = [0, 1, 2]  # Forward, Left, Right
        
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
        self.epsilon_decay = 0.998 # Slower decay for better learning
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
        
        # --- LIDAR PROCESSING VARIABLES ---
        self.scan_ranges = None
        self.scan_ranges_smoothed = None
        self.angle_min = None
        self.angle_max = None
        self.angle_increment = None
        self.range_min = 0.15
        self.range_max = 12.0
        self.peak_pairs = []  # Stores (angle, distance) of peaks
        
        # --- SECTOR DEFINITIONS ---
        self.sectors = None
        self.safe_distance_threshold = 0.5  # Peaks closer than this are walls
        
        # --- REWARD SHAPING ---
        self.prev_dist = None
        
        # --- EPISODE TRACKING ---
        self.episode_num = 0
        self.episode_reward = 0
        self.step_count = 0
        self.max_steps = 1000
        
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_successes = []
        self.episode_collisions = []
        self.episode_timeouts = []
        self.action_counts = {0: 0, 1: 0, 2: 0}
        
        # --- LOGGING ---
        self.setup_logging()
        
        # --- Q-TABLE PERSISTENCE ---
        self.q_table_dir = 'q_tables'
        self.log_dir = 'training_logs'
        os.makedirs(self.q_table_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.load_q_table()
        
        # --- CONTROL LOOP ---
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("RL Agent Started (Paper's VFH Peak Finder)")
        self.get_logger().info(f"Target: {self.target_coords}")
        self.get_logger().info(f"State Space: {self.num_states} states")
        self.get_logger().info("=" * 60)

    def setup_logging(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f'training_logs/training_log_{timestamp}.txt'
        
        with open(self.log_filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("VFH-QL Training Log (Paper Implementation)\n")
            f.write(f"Started: {timestamp}\n")
            f.write(f"Target: {self.target_coords}\n")
            f.write(f"Hyperparameters: α={self.alpha}, γ={self.gamma}, ε_decay={self.epsilon_decay}\n")
            f.write("=" * 80 + "\n\n")
            f.write("Episode,Steps,Reward,Success,Collision,Timeout,Epsilon,Alpha,Fwd,Left,Right\n")

    def load_q_table(self):
        q_table_files = [f for f in os.listdir(self.q_table_dir) if f.startswith('q_table_ep')]
        
        if q_table_files:
            latest_file = sorted(q_table_files)[-1]
            filepath = os.path.join(self.q_table_dir, latest_file)
            
            try:
                self.q_table = np.load(filepath)
                episode_num = int(latest_file.split('ep')[1].split('.')[0])
                self.episode_num = episode_num
                self.epsilon = max(self.epsilon_min, 1.0 * (self.epsilon_decay ** episode_num))
                
                self.get_logger().info(f"✓ Loaded Q-table from {latest_file}")
                self.get_logger().info(f"✓ Resuming from Episode {episode_num}, ε={self.epsilon:.3f}")
            except Exception as e:
                self.get_logger().warn(f"Failed to load Q-table: {e}")

    def _initialize_sectors(self):
        """
        Initialize sector definitions.
        Sectors: Front, Front-Left, Left, Right, Front-Right
        """
        sector_definitions_deg = {
            0: [(338, 22)],      # Front: -22° to +22°
            1: [(23, 67)],       # Front-Left: 23° to 67°
            2: [(68, 112)],      # Left: 68° to 112°
            3: [(248, 292)],     # Right: 248° to 292°
            4: [(293, 337)]      # Front-Right: 293° to 337°
        }
        
        self.sectors = {}
        for idx, ranges in sector_definitions_deg.items():
            self.sectors[idx] = []
            for start_deg, end_deg in ranges:
                start_rad = math.radians(start_deg)
                end_rad = math.radians(end_deg)
                
                if start_deg > end_deg:  # Wraparound (e.g., 338 to 22)
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
        self._peak_finder()               
        self._update_lidar_state()        
        self._target_visible()            

    def _check_collision(self):
        if np.min(self.scan_ranges) < 0.20:
            self.done = True
            stop_msg = Twist()
            self.cmd_pub.publish(stop_msg)

    def _smooth_range_vals(self):
        """11-point moving average with circular padding"""
        r = self.scan_ranges
        r_pad = np.r_[r[-5:], r, r[:5]]
        kernel = np.ones(11, dtype=np.float32) / 11.0
        self.scan_ranges_smoothed = np.convolve(r_pad, kernel, mode='same')[5:-5]

    def _peak_finder(self):
        """
        Finds local maxima (peaks) in smoothed data.
        Stores both angle AND distance for each peak.
        """
        scan = self.scan_ranges_smoothed
        N = len(scan)
        self.peak_pairs = []  # Reset
        
        for i in range(N):
            if scan[i] > scan[(i+1) % N] and scan[i] > scan[(i-1) % N]:
                angle = self.angle_min + i * self.angle_increment
                norm_angle = self._normalize_angle(angle)
                self.peak_pairs.append((norm_angle, scan[i]))

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
        Map peaks to 5 sectors.
        CRITICAL: Only peaks beyond safe_distance_threshold mark sectors as open.
        """
        if not self.peak_pairs:
            return  # All sectors remain blocked (0)
        
        for angle, distance in self.peak_pairs:
            # Only far peaks represent navigable space
            if distance > self.safe_distance_threshold:
                for idx, intervals in self.sectors.items():
                    if self._check_interval(angle, intervals):
                        self.lidar_state[idx] = 1
                        break  # FIX: One peak belongs to one sector only

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

    def get_reward(self):
        dist = math.sqrt(
            (self.target_coords[0] - self.robot_pose['x'])**2 + 
            (self.target_coords[1] - self.robot_pose['y'])**2
        )
        
        if self.done:
            return -100, True
        if dist < 0.3:
            return 300, True
        
        if self.prev_dist is None:
            self.prev_dist = dist
        
        delta_dist = self.prev_dist - dist
        self.prev_dist = dist
        
        shaped_reward = -1 + (10 * delta_dist)
        return shaped_reward, False

    def choose_action(self, state_idx):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.q_table[state_idx])

    def execute_action(self, action):
        msg = Twist()
        if action == 0:   # Forward
            msg.linear.x = 0.22
            msg.angular.z = 0.0
        elif action == 1: # Left
            msg.linear.x = 0.1
            msg.angular.z = 0.8
        elif action == 2: # Right
            msg.linear.x = 0.1
            msg.angular.z = -0.8
        
        self.cmd_pub.publish(msg)
        self.action_counts[action] += 1

    def control_loop(self):
        if self.scan_ranges is None or self.sectors is None:
            return
        
        current_state_idx = self.get_state_index()
        reward, done = self.get_reward()
        self.episode_reward += reward
        
        if self.step_count > 0:
            old_q = self.q_table[self.previous_state_idx, self.previous_action]
            
            if done:
                target = reward
            else:
                next_max = np.max(self.q_table[current_state_idx])
                target = reward + self.gamma * next_max
            
            new_q = old_q + self.alpha * (target - old_q)
            self.q_table[self.previous_state_idx, self.previous_action] = new_q
            
            if done or self.step_count >= self.max_steps:
                self.end_episode(done, self.step_count >= self.max_steps)
                return
        
        action = self.choose_action(current_state_idx)
        self.execute_action(action)
        
        self.previous_state_idx = current_state_idx
        self.previous_action = action
        self.step_count += 1

    def end_episode(self, collision_or_success, timeout):
        self.episode_num += 1
        success = (self.episode_reward > 200)
        collision = self.done and not success
        
        self.episode_rewards.append(self.episode_reward)
        self.episode_steps.append(self.step_count)
        self.episode_successes.append(1 if success else 0)
        self.episode_collisions.append(1 if collision else 0)
        self.episode_timeouts.append(1 if timeout else 0)
        
        outcome = "SUCCESS ✓" if success else ("COLLISION ✗" if collision else "TIMEOUT ⏱")
        self.get_logger().info(
            f"Episode {self.episode_num:4d} | "
            f"{outcome:12s} | "
            f"Steps: {self.step_count:4d} | "
            f"Reward: {self.episode_reward:7.1f} | "
            f"ε: {self.epsilon:.3f}"
        )

        # Log to file BEFORE stats (uses action_counts)
        self.log_to_file(success, collision, timeout)
        
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

    def reset_simulation(self):
        req = Empty.Request()
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Reset service not available, waiting...')
        
        self.reset_client.call_async(req)
        
        self.done = False
        self.step_count = 0
        self.episode_reward = 0
        self.previous_state_idx = 0
        self.previous_action = 0
        self.prev_dist = None
        
        # FIX: Reset action counts EVERY episode
        self.action_counts = {0: 0, 1: 0, 2: 0}
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if self.alpha > self.alpha_min:
            self.alpha *= self.alpha_decay

def main(args=None):
    rclpy.init(args=args)
    node = RLAgent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Training interrupted by user.")
    finally:
        node.save_q_table()
        node.get_logger().info("Final Q-table saved. Shutting down...")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()