#!/usr/bin/env python3
"""
test_gripper.py
-------------------------
A simple loop that:
1. Wakes up the arm (Home position).
2. Opens and Closes the Gripper continuously.
3. Publishes joint states to RViz so the robot appears "Solid".
"""

import time
import math
import sys

# --- ROS 2 Imports ---
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Header
except ImportError:
    print("[ERROR] ROS 2 python libraries not found. Source your ROS setup!")
    sys.exit(1)

# ============================================================
# CONFIGURATION
# ============================================================

# These MUST match the 'name' fields in your URDF <joint> tags
JOINT_NAMES = [
    "joint2_to_joint1",       # Arm Axis 1
    "joint3_to_joint2",       # Arm Axis 2
    "joint4_to_joint3",       # Arm Axis 3
    "joint5_to_joint4",       # Arm Axis 4
    "joint6_to_joint5",       # Arm Axis 5
    "joint6output_to_joint6", # Arm Axis 6 (Flange)
    "gripper_controller"      # GRIPPER (Driver Joint)
]

# Gripper Limits (from your URDF)
# lower = -0.7 (Open/Wide)
# upper =  0.15 (Closed/Tight)
GRIPPER_OPEN = -0.7
GRIPPER_CLOSED = 0.15

class GripperDemo(Node):
    def __init__(self):
        super().__init__('gripper_tester_node')
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        self.timer = self.create_timer(0.05, self.update_loop) # 20Hz
        
        self.start_time = time.time()
        print(f"--- Publishing Joint States for {len(JOINT_NAMES)} joints ---")
        print(f"Joints: {JOINT_NAMES}")

    def update_loop(self):
        t = time.time() - self.start_time
        
        # 1. Calculate Gripper Motion (Sine wave)
        # Oscillate between Open and Closed every 2 seconds
        # Sine moves from -1 to 1. We map that to our limits.
        sin_val = math.sin(t * 2.0) # Speed multiplier
        
        # Map sine (-1 to 1) to gripper range
        # simple linear interpolation
        gripper_range = GRIPPER_CLOSED - GRIPPER_OPEN
        normalized_sin = (sin_val + 1) / 2.0 # 0.0 to 1.0
        current_gripper_pos = GRIPPER_OPEN + (normalized_sin * gripper_range)

        # 2. Calculate Arm Motion (Just a gentle sway to prove it's alive)
        # We keep the arm mostly home (0.0) but sway joint 2 slightly
        arm_sway = math.sin(t) * 0.1

        # 3. Build the Message
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        
        # The Order MUST match JOINT_NAMES above
        msg.position = [
            0.0,              # J1 (Fixed/Base)
            arm_sway,         # J2 (Swaying)
            0.0,              # J3
            0.0,              # J4
            0.0,              # J5
            0.0,              # J6
            current_gripper_pos # <--- THE GRIPPER!
        ]
        
        # 4. Publish
        self.publisher_.publish(msg)

def main():
    rclpy.init()
    node = GripperDemo()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
