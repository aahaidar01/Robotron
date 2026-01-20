import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan 
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty  #reset simulation
import numpy as np
import math

class RLAgent(Node):

    def __init__(self):
        super().__init__('rl_agent')
        
        # --- 1. HYPERPARAMETERS ---
        self.target_coords = (2.0, 0.0) # Target Coordinates
        self.alpha = 0.1 # Learning Rate
        self.gamma = 0.95 # Discount Factor
        self.epsilon = 1.0 # Exploration Rate (experimenting with high values first)
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        
        # --- 2. Q-TABLE SETUP ---
        # 4096 States x 3 Actions (Forward, Left, Right)
        # We can also load a file if it exists, otherwise start fresh
        self.q_table = np.zeros((4096, 3)) 
        
        # --- 3. ROS INFRASTRUCTURE ---
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10) #Laser scans as input
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10) #Odometry as input
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10) #Send command velocities as output
        
        # Service Client to reset Gazebo
        self.reset_client = self.create_client(Empty, '/reset_simulation')

        # Variables
        self.lidar_state = [0]*8 #8 lidar sectors calculated from peak finder
        self.target_vis = 0
        self.robot_pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        
        # Control Loop (10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)

    def scan_callback(self, msg):
        # --- VFH PROCESSING HERE ---
        pass

    def odom_callback(self, msg):
        # --- ODOMETRY PROCESSING HERE ---
        pass

    def get_state_index(self):
        # Convert binary list + target sector into one integer (0 - 4095)
        # Logic: Binary to Int conversion + Target Offset
        pass

    def control_loop(self):
        # 1. Get current state index
        # 2. Choose Action (Epsilon Greedy)
        # 3. Publish Cmd_Vel
        # 4. Wait/Sleep (implicitly handled by timer)
        # 5. Observe Reward (Did I crash?)
        # 6. Update Q-Table
        # 7. Reset if crash
        pass

def main(args=None):
    rclpy.init(args=args)
    node = RLAgent()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()