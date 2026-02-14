import unittest
from unittest.mock import MagicMock, patch
import time
""" to run: python3 -m unittest tests/test_scenario_logic.py"""

# Import the class we are testing
from rover_learner.rover_trainer.scenario_logic import ScenarioManager
from rover_learner.rover_trainer.hardware_manager import SensorPacket

class TestScenarioLogic(unittest.TestCase):

    def setUp(self):
        self.logic = ScenarioManager()

    def create_mock_yolo(self, class_name="rock"):
        """Helper to create a fake YOLO result object without loading the heavy model."""
        mock_result = MagicMock()
        mock_box = MagicMock()
        # Mock the class ID and Name lookup
        mock_box.cls = [0] 
        mock_result.names = {0: class_name}
        mock_result.boxes = [mock_box]
        return [mock_result]

    # --- TEST 1: The Human Safety Override ---
    def test_human_override(self):
        """
        Situation: Lidar detects object VERY CLOSE (0.3m).
        Result: Should be RETREAT... UNLESS it's a human.
        """
        # Case A: A Rock is close -> RETREAT
        packet_rock = SensorPacket(None, None, 0.3, None, 0.3, 100)
        res_rock = self.create_mock_yolo("rock")
        action = self.logic.evaluate(packet_rock, res_rock)
        self.assertEqual(action, "RETREAT", "Should retreat from rock")

        # Case B: A Human is close -> STOP (Don't run over grandma, but don't run away)
        packet_human = SensorPacket(None, None, 0.3, None, 0.3, 100)
        res_human = self.create_mock_yolo("person")
        action = self.logic.evaluate(packet_human, res_human)
        self.assertEqual(action, "STOP", "Should STOP for human, not retreat")

    # --- TEST 2: The "Sticky" State Machine (Time Travel) ---
    def test_sticky_sequence(self):
        """
        Situation: We see 'regolith'.
        Result: Should cycle SCOOP -> DUMP -> WIGGLE -> RETREAT over 12 seconds.
        We use 'patch' to fake the system time.
        """
        packet = SensorPacket(None, None, 1.0, None, 1.0, 100) # Safe distance
        res_sand = self.create_mock_yolo("regolith")

        with patch('time.time') as mock_time:
            # T=0: Start
            mock_time.return_value = 1000.0 
            action = self.logic.evaluate(packet, res_sand)
            self.assertEqual(action, "SCOOP", "Start with Scoop")

            # T+2s: Still Scooping
            mock_time.return_value = 1002.0
            action = self.logic.evaluate(packet, res_sand)
            self.assertEqual(action, "SCOOP", "Still Scooping")

            # T+4s: Should be Dumping (Threshold is 3s)
            mock_time.return_value = 1004.0
            action = self.logic.evaluate(packet, res_sand)
            self.assertEqual(action, "DUMP", "Time to Dump")

            # T+10s: Should be Wiggling (Threshold is 9s)
            mock_time.return_value = 1010.0
            action = self.logic.evaluate(packet, res_sand)
            self.assertEqual(action, "WIGGLE", "Time to Wiggle")

    # --- TEST 3: Adaptive Panic (Darkness) ---
    def test_darkness_panic(self):
        """Situation: Camera brightness drops to 10 (Pitch black)."""
        # Brightness is the last arg in SensorPacket
        packet_dark = SensorPacket(None, None, 1.0, None, 1.0, 10.0) 
        action = self.logic.evaluate(packet_dark, None)
        self.assertEqual(action, "ADAPTIVE_PANIC", "Should panic in the dark")

if __name__ == '__main__':
    unittest.main()