import unittest
from unittest.mock import MagicMock, patch
import sys

"""To run: python3 -m unittest tests/test_cobot_integration.py"""

# We need to test HardwareManager, which is in the trainer folder
from rover_learner.rover_trainer.hardware_manager import HardwareManager

class TestHardwareManager(unittest.TestCase):

    @patch('rover_learner.rover_trainer.hardware_manager.CSICameraProvider')
    @patch('rover_learner.rover_trainer.hardware_manager.USBCameraProvider')
    @patch('rover_learner.rover_trainer.hardware_manager.SerialRPLidarProvider')
    def test_mode_4_full_loadout(self, mock_lidar, mock_usb_cam, mock_csi_cam):
        """
        Test selecting Mode 4 (2 Cams, 2 Lidars).
        Verifies that the class attempts to instantiate the correct sensors.
        """
        # Initialize Manager in Mode 4
        hw = HardwareManager(mode=4)

        # Check Primary Camera (Should try CSI first)
        mock_csi_cam.assert_called() 
        
        # Check Secondary Camera (Should try USB)
        mock_usb_cam.assert_called()

        # Check Lidars (Should be called twice, once for USB0, once for USB1)
        self.assertEqual(mock_lidar.call_count, 2)
        
        # Verify specific ports were requested
        # We check the calls to ensure one was /dev/ttyUSB0 and one was /dev/ttyUSB1
        call_args_list = mock_lidar.call_args_list
        # Note: Depending on order, we just check both exist
        ports_requested = [call.kwargs.get('port') for call in call_args_list]
        self.assertIn('/dev/ttyUSB0', ports_requested)
        self.assertIn('/dev/ttyUSB1', ports_requested)

    @patch('rover_learner.rover_trainer.hardware_manager.CSICameraProvider')
    def test_sensor_fusion_logic(self, mock_cam):
        """
        Test that the .read() method correctly calculates the MINIMUM distance
        from two different lidars.
        """
        hw = HardwareManager(mode=4)
        
        # Mock the sensors to return specific values
        hw.lidar1 = MagicMock()
        hw.lidar2 = MagicMock()
        hw.cam1 = MagicMock()

        # Case: Lidar 1 sees 10m, Lidar 2 sees 0.5m
        hw.lidar1.get_distance_m.return_value = 10.0
        hw.lidar2.get_distance_m.return_value = 0.5
        # Mock camera frame for brightness calc
        hw.cam1.read.return_value = (MagicMock(), 0) 

        # Perform Read
        packet = hw.read()

        # The packet.min_dist should be 0.5 (The closer danger)
        self.assertEqual(packet.min_dist, 0.5)
        self.assertEqual(packet.dist_primary, 10.0)

    def test_graceful_lidar_fail(self):
        """
        Test that if Serial Lidar fails to init, it falls back to MockLidarProvider
        so the code doesn't crash.
        """
        # We patch SerialRPLidarProvider to raise an ImportError
        with patch('rover_learner.rover_trainer.hardware_manager.SerialRPLidarProvider', side_effect=ImportError("No driver")):
            with patch('rover_learner.rover_trainer.hardware_manager.MockLidarProvider') as mock_mock_lidar:
                
                # Init hardware
                hw = HardwareManager(mode=2)
                
                # Serial should have failed, so Mock should have been called
                mock_mock_lidar.assert_called()
                # The internal lidar object should be the mock
                self.assertNotEqual(hw.lidar1, None)

if __name__ == '__main__':
    unittest.main()