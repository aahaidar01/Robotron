import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan 
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty 
import numpy as np
import math

class RLAgent(Node):

    def __init__(self):
        super().__init__('rl_agent')
        
        # --- CONFIGURATION ---
        self.target_coords = (2.0, 0.0) # Target Coordinates
        self.action_space = [0, 1, 2]   # Forward, Left, Right
        
        # --- OPTIMIZATION: STATE SPACE REDUCTION ---
        # Paper uses 8 Lidar + 1 Visibility + 1 Target Angle (Total 10 elements).
        # We optimize Lidar to 5 sectors (Front, FL, FR, L, R) because Turtlebot can't strafe/reverse.
        # Structure: 5 Bits (Lidar) + 1 Bit (Visible) + 3 Bits (Target Sector 0-7)
        self.num_lidar_bits = 5 
        self.num_vis_bits = 1
        self.num_target_sectors = 8
        
        # --- 1. HYPERPARAMETERS ---
        self.alpha = 0.1       # Learning Rate
        self.gamma = 0.95      # Discount Factor
        self.epsilon = 1.0     # Exploration Rate
        self.epsilon_decay = 0.998 # Slower decay for better learning
        self.epsilon_min = 0.05
        
        # --- 2. Q-TABLE SETUP ---
        # Size = 2^5 (32) * 2 (Vis) * 8 (Target) = 512 States
        self.num_states = (2**self.num_lidar_bits) * (2**self.num_vis_bits) * self.num_target_sectors
        self.q_table = np.zeros((self.num_states, len(self.action_space)))
        
        self.get_logger().info(f"Q-Table Initialized with {self.num_states} states.")
        
        # --- 3. ROS INFRASTRUCTURE ---
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.reset_client = self.create_client(Empty, '/reset_simulation')

        # Variables
        self.lidar_state = [0] * self.num_lidar_bits 
        self.target_vis = 0
        self.robot_pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.target_sector = 0
        
        self.current_state_idx = 0
        self.previous_state_idx = 0
        self.previous_action = 0
        self.done = False
        
        # Sectors 
        self.sectors = {
            0: [(math.radians(330), math.radians(360)), (0.0, math.radians(30))], # Front
            1:[(math.radians(30), math.radians(90))],                             # FL  
            2: [(math.radians(270), math.radians(330))],                          # FR
            3:[(math.radians(90), math.radians(150))],                            # L
            4:[(math.radians(210), math.radians(270))],                           # R
        }
        
        # Episode Tracking
        self.episode_reward = 0
        self.step_count = 0
        self.max_steps = 1000  # Can be increased for longer episodes
        
        # Control Loop (10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("RL Agent Started. Training...")

    def calculate_target_sector(self):
        dx = self.target_coords[0] - self.robot_pose['x']
        dy = self.target_coords[1] - self.robot_pose['y']
        global_target_angle = math.atan2(dy, dx)
        relative_angle = global_target_angle - self.robot_pose['yaw']
        
        while relative_angle > math.pi: relative_angle -= 2 * math.pi
        while relative_angle < -math.pi: relative_angle += 2 * math.pi
        
        deg = math.degrees(relative_angle)
        if deg < 0: deg += 360
        
        # Target Sector (0-7)
        self.target_sector = int((deg + 22.5) / 45.0) % 8
        return relative_angle
    
    # Peak Finder Algorithm - Paper
    def peak_finder(self):
        scan = self.scan_ranges_smoothed
        N = len(scan)
        peak_idx = []
        for i in range(N):
            # use %N not to exceed 
            if scan[i] > scan[(i+1)%N] and scan[i] > scan[(i-1)%N]:
                peak_idx.append(i)
        
        peak_idx = np.array(peak_idx, dtype=np.int32)
        self.peak_angles = self.angle_min + peak_idx.astype(np.float32) * self.angle_increment
        
    
    # Smooth the data 
    def smooth_range_vals(self):
        r = self.scan_ranges
        
        # Need to pad the data - Laser scan is circular (Angle 0 next to angle 359)
        r_pad = np.r_[r[-5:], r, r[:5]]
        
        # Use a kernel with convolution - Moving average filter
        kernel = np.ones(11, np.float32) / 11.0
                    
        self.scan_ranges_smoothed = np.convolve(r_pad, kernel, mode='same')[5:-5]
    
    def check_interval(self, angle, interval):
        for lo, hi in interval:
            if lo <= angle < hi:
                return True
        return False
    
    def update_lidar_state(self):
        for angle in self.peak_angles:
            angle = float(angle) % (2*np.pi)  # keep in [0, 2pi)
            for idx, intervals in self.sectors.items():
                if self.check_interval(angle, intervals):
                    self.lidar_state[idx] = 1
                
    def target_visible(self):
        theta = self.calculate_target_sector()
        
        dx = self.target_coords[0] - self.robot_pose['x']
        dy = self.target_coords[1] - self.robot_pose['y']
        
        dT = np.hypot(dx, dy)
        if theta < 0:
            theta += 2*math.pi  # now in [0, 2pi)
        
        idx_target = int(round((theta - self.angle_min) / self.angle_increment))
        lidar_reading = self.scan_ranges[idx_target]
        
        self.target_vis = 1 if (lidar_reading > dT) else 0
        
    def scan_callback(self, msg):
        """
        Process LiDAR for:
        1. Collision Detection
        2. Visibility Check (Paper Element 3)
        3. VFH Peaks / Open Sectors (Paper Element 1)
        """
        self.lidar_state = [0] * self.num_lidar_bits # reset state before each scan
        self.target_vis = 0 # reset target visible
        self.scan_ranges = np.array(msg.ranges,  np.float32)
        self.range_min = msg.range_min
        self.range_max = msg.range_max
        self.angle_increment = msg.angle_increment
        
        # Replace 'inf' values with range_max for peak finder algorithm
        self.scan_ranges[~np.isfinite(self.scan_ranges)] = self.range_max
        
        # Clamp to sensor limits
        self.scan_ranges = np.clip(self.scan_ranges, self.range_min, self.range_max)

        self.smooth_range_vals()
        self.peak_finder()
        self.update_lidar_state()
        self.target_visible()
        
    def odom_callback(self, msg):
        self.robot_pose['x'] = msg.pose.pose.position.x
        self.robot_pose['y'] = msg.pose.pose.position.y
        
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_pose['yaw'] = math.atan2(siny_cosp, cosy_cosp)
        
        self.calculate_target_sector()

    def get_state_index(self):
        """
        Combines 3 Elements into Unique Integer:
        1. Lidar (5 bits -> 0-31)
        2. Visibility (1 bit -> 0-1)
        3. Target Sector (3 bits -> 0-7)
        Formula: (Lidar * 16) + (Vis * 8) + Target
        """
        # 1. Convert Lidar List to Int
        lidar_int = 0
        for i, val in enumerate(self.lidar_state):
            lidar_int += (val * (2**i))
            
        # 2. Combine
        # Multipliers ensure bits don't overlap
        # Target (0-7) occupies bits 0-2 (size 8)
        # Vis (0-1) occupies bit 3 (size 2 -> offset 8)
        # Lidar (0-31) occupies bits 4-8 (size 32 -> offset 16)
        state_idx = (lidar_int * 16) + (self.target_vis * 8) + self.target_sector
        
        return state_idx
    
    def get_reward(self):
        dist = math.sqrt(
            (self.target_coords[0] - self.robot_pose['x'])**2 + 
            (self.target_coords[1] - self.robot_pose['y'])**2
        )
        
        if self.done: 
            return -100, True # Stronger penalty for crashing
            
        if dist < 0.3: 
            return 300, True # Strong reward for success
            
        return -1, False 
    
    def choose_action(self, state_idx):
        
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_space) #Explore
        else:
            return np.argmax(self.q_table[state_idx]) #Exploit
        
    def execute_action(self, action):
    #Update velocity based on action chosen
        pass

    def control_loop(self):
        # 1. Get Current State
        current_state_idx = self.get_state_index()
        
        # 2. Learn from PREVIOUS Step (If not first step)
        if self.step_count > 0:
            reward, done = self.get_reward()
            self.episode_reward += reward
            
            # Q-Learning Formula
            old_q = self.q_table[self.previous_state_idx, self.previous_action]
            next_max = np.max(self.q_table[current_state_idx])
            
            new_q = old_q + self.alpha * (reward + self.gamma * next_max - old_q)
            self.q_table[self.previous_state_idx, self.previous_action] = new_q
            
            # Check Termination
            if done or self.step_count >= self.max_steps:
                self.get_logger().info(f"End Ep. Reward: {self.episode_reward}, Epsilon: {self.epsilon:.3f}")
                self.reset_simulation()
                return

        # 3. Choose NEXT Action
        action = self.choose_action(current_state_idx)
        
        # 4. Act
        self.execute_action(action)
        
        # 5. Update History
        self.previous_state_idx = current_state_idx
        self.previous_action = action
        self.step_count += 1
    
    def reset_simulation(self):
        req = Empty.Request()
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Reset service not available, waiting...')
        self.reset_client.call_async(req)
        
        # Reset internal variables
        self.done = False
        self.step_count = 0
        self.episode_reward = 0
        self.previous_state_idx = 0
        self.previous_action = 0
        
        # Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def main(args=None):
    rclpy.init(args=args)
    node = RLAgent()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()