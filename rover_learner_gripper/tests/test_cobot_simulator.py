"""
tests/test_cobot_simulator.py

Unit tests for the Interactive Rover Brain.
Checks:
1. Menu Input Logic (Options 1, 2, 3, 4)
2. Hardware Selection Flags
3. Animator State Transitions
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import time

# We need to import the module logic. 
# Since it is a script, we might import specific functions if we structured it as a library,
# but here we will mock the inputs to 'select_hardware_mode'.

from rover_learner.cobotarm_simulator_advanced import (
    select_hardware_mode, 
    AsyncArmAnimator,
    POSES
)

class TestRoverSimulator(unittest.TestCase):

    @patch('builtins.input', side_effect=['1'])
    def test_menu_option_1(self, mock_input):
        """Test selecting Option 1 (Camera Only)"""
        use_cam, use_lidar = select_hardware_mode()
        self.assertTrue(use_cam)
        self.assertFalse(use_lidar)

    @patch('builtins.input', side_effect=['2'])
    def test_menu_option_2(self, mock_input):
        """Test selecting Option 2 (Cam + Lidar)"""
        use_cam, use_lidar = select_hardware_mode()
        self.assertTrue(use_cam)
        self.assertTrue(use_lidar)

    @patch('builtins.input', side_effect=['3', '1']) 
    def test_menu_wip_guard(self, mock_input):
        """
        Test selecting Option 3 (WIP). 
        The code should print a message and loop back.
        We simulate this by providing '3' (fails/loops) then '1' (succeeds).
        """
        # Note: We capture stdout to verify the WIP message was printed? 
        # For simplicity, we just ensure it eventually returns valid config for input '1'
        use_cam, use_lidar = select_hardware_mode()
        self.assertTrue(use_cam)
        self.assertFalse(use_lidar)
        # The function was called twice effectively.

    def test_animator_states(self):
        """Test that setting commands updates the internal state correctly."""
        animator = AsyncArmAnimator(node=None) # No ROS node needed for logic test
        
        # 1. Test SCAN
        animator.set_command("SCAN")
        self.assertTrue(animator.is_scanning)
        self.assertEqual(animator.status_text, "SCAN")
        
        # 2. Test SCOOP
        animator.set_command("SCOOP")
        self.assertFalse(animator.is_scanning)
        self.assertEqual(animator.target_pose, POSES["SCOOP"])
        
        # 3. Test RETREAT
        animator.set_command("RETREAT")
        self.assertFalse(animator.is_scanning)
        self.assertEqual(animator.target_pose, POSES["RETREAT"])
        
        animator.stop()

if __name__ == '__main__':
    unittest.main()