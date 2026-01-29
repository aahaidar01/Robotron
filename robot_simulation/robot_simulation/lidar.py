import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan 
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import os
import math

class LidarDebug(Node):
    def __init__(self):
        super().__init__('lidar_debug')
        
        # --- CONFIGURATION ---
        self.target_coords = (2.5, 0.5)
        self.action_space = [0, 1, 2]  # Forward, Left, Right
        
        self.get_logger().info("Starting LiDAR Debugger...")
        
        # --- ROS INFRASTRUCTURE ---
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # --- LIDAR PROCESSING VARIABLES ---
        self.scan_ranges = None
        self.scan_ranges_smoothed = None
        self.angle_min = None
        self.angle_max = None
        self.angle_increment = None
        
        self.num_lidar_bits = 5
        self.num_vis_bits = 1
        self.num_target_sectors = 8
        
        # --- STATE VARIABLES ---
        self.lidar_state = [0] * self.num_lidar_bits #FL, L, Front, R, FR
        self.target_vis = 0
        self.robot_pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.target_sector = 0
        self.scans_since_reset = 0
        
        self.current_state_idx = 0
        self.previous_state_idx = 0
        self.previous_action = 0
        self.done = False
    
        # --- SECTOR DEFINITIONS ---
        self.sectors = None
        self.safe_distance_threshold = 0.5  # distance to obstacle threshold
        
        # --- DASHBOARD TIMER (5Hz) ---
        self.timer = self.create_timer(1/5, self.lidar_dashboard)
    
    def _initialize_sectors(self):
        """Standard Spatially-Contiguous Indexing for 180 FOV."""
        
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
         # Reset state
        self.lidar_state = [0] * self.num_lidar_bits
        self.target_vis = 0
        
        if self.angle_min is None:
            self.angle_min = msg.angle_min
            self.angle_increment = msg.angle_increment
            self.range_min = msg.range_min
            self.range_max = msg.range_max
            self._initialize_sectors()
        
        # Get data
        self.scan_ranges = np.array(msg.ranges, dtype=np.float32)
        self.angle_min = msg.angle_min 
        self.angle_max = msg.angle_max
        self.range_min = msg.range_min
        self.range_max = msg.range_max
        
        # Clean Data
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


        
    def _invalid_ranges(self):
        scan = self.scan_ranges
        invalid = (scan < self.range_min) | (scan > self.range_max) | (~np.isfinite(scan))
        invalid_scans = scan[invalid]
        
        if len(invalid_scans) == 0:
            self.get_logger().info(" ✅ No invallid scan ranges recorded...")
        else:
            self.get_logger().warn("❌ Invalid LiDAR Ranges Recorded...")
            self.get_logger().info(f"Invalid Ranges= {invalid_scans.size/scan.size:.3f} %")
            
            sample = invalid_scans[:10]
            self.get_logger().warn(f"Sample invalid values: {sample}")

    def lidar_dashboard(self):
        """Prints a visual representation to the terminal"""
        os.system('clear') # Clear terminal
        self._invalid_ranges()
        print("\n=== LiDAR Q-Learning Debugger ===")
        print(f"Robot Pose: x={self.robot_pose['x']:.2f}, y={self.robot_pose['y']:.2f}")
        print(f"Target: {self.target_coords}")
        print("-" * 30)
        
        # Visualize Lidar Sectors (1 = Clear, 0 = Blocked)
        # Displaying sectors in a circle-like layout
        print(f"1: Clear   🚀  | 0: Blocked  ❌")
        s = self.lidar_state
        print("\n          [FRONT]")
        print(f"          {[s[2]]}")
        print("\n   [Left]           [Right]")
        print(f"   {[s[1]]}              {[s[3]]}")
        print("\n[Far Left]            [Far Right]")
        print(f"{[s[0]]}                   {[s[4]]}")
        
        print("-" * 30)
        print(f"Target Sector: {self.target_sector} | Target Visible: {self.target_vis}")
        
        # Binary State calculation check
        lidar_int = sum(v * (2**i) for i, v in enumerate(self.lidar_state))
        state_idx = (lidar_int * 16) + (self.target_vis * 8) + self.target_sector
        print(f"CURRENT STATE INDEX: {state_idx}")      



def main(args=None):
    rclpy.init(args=args)
    try:
        node = LidarDebug()
        rclpy.spin(node)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        rclpy.shutdown()
        return
    except KeyboardInterrupt:
        node.get_logger().info("Debug Info Interrupted by User")
    
    