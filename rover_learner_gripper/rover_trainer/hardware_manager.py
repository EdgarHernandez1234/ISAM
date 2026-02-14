import time
from dataclasses import dataclass
from typing import Optional, Tuple, Any

# Import our hardware drivers
# We use absolute imports to ensure they work from anywhere in the package
from rover_learner.camera_provider import (
    CSICameraProvider, USBCameraProvider, MockCameraProvider
)
from rover_learner.lidar_provider import (
    ROS2LaserScanProvider, SerialRPLidarProvider, MockLidarProvider
)

@dataclass
class SensorPacket:
    """
    Standard data packet passed from Hardware -> Logic
    """
    image_primary: Any          # OpenCV image from main camera
    image_secondary: Any        # OpenCV image from secondary camera (optional)
    dist_primary: float         # Distance from main Lidar
    dist_secondary: float       # Distance from secondary Lidar (optional)
    min_dist: float             # The smallest distance seen by ANY sensor (Safety Critical)
    brightness: float           # Average brightness of primary image (for panic detection)


class HardwareManager:
    def __init__(self, mode: int = 2):
        """
        Initialize hardware based on the selected mode.
        
        Modes:
        0 = Full Simulation (Mock Cam, Mock Lidar)
        1 = Vision Only (CSI Cam, Mock Lidar)
        2 = Standard Robot (CSI Cam, Serial Lidar) - MOST COMMON
        3 = ROS2 Mode (CSI Cam, ROS2 Lidar)
        4 = Advanced/Test (Dual Cam, Dual Lidar)
        """
        self.mode = mode
        self.cam1 = None
        self.cam2 = None
        self.lidar1 = None
        self.lidar2 = None

        print(f"[HW] Initializing Hardware Manager in Mode {mode}...")

        # --- CAMERA SETUP ---
        if mode == 0:
            self.cam1 = MockCameraProvider()
        elif mode in [1, 2, 3]:
            try:
                self.cam1 = CSICameraProvider(width=640, height=480, fps=20)
            except Exception as e:
                print(f"[HW] CSI Camera missing ({e}). Switching to Mock.")
                self.cam1 = MockCameraProvider()
        elif mode == 4:
            # Dual Camera Mode
            self.cam1 = CSICameraProvider()
            try:
                self.cam2 = USBCameraProvider()
            except Exception:
                self.cam2 = MockCameraProvider()

        # --- LIDAR SETUP (The part that failed the test!) ---
        if mode in [0, 1]:
            self.lidar1 = MockLidarProvider()
        
        elif mode == 2:
            # Standard Mode: Try Serial, Fail to Mock
            try:
                self.lidar1 = SerialRPLidarProvider(port="/dev/ttyUSB0", baudrate=460800)
            except (ImportError, RuntimeError, Exception) as e:
                print(f"[HW] CRITICAL: Lidar Driver Failed ({e}). Safe-mode (Mock) engaged.")
                self.lidar1 = MockLidarProvider()

        elif mode == 3:
            # ROS2 Mode
            try:
                self.lidar1 = ROS2LaserScanProvider()
            except Exception as e:
                print(f"[HW] ROS2 Lidar failed ({e}). Switching to Mock.")
                self.lidar1 = MockLidarProvider()

        elif mode == 4:
            # Dual Lidar Mode (Sensor Fusion)
            # 1. Try Serial 1
            try:
                self.lidar1 = SerialRPLidarProvider(port="/dev/ttyUSB0")
            except Exception:
                self.lidar1 = MockLidarProvider()
            
            # 2. Try Serial 2 (or USB Lidar)
            try:
                self.lidar2 = SerialRPLidarProvider(port="/dev/ttyUSB1")
            except Exception:
                self.lidar2 = MockLidarProvider()

        # Give sensors time to warm up
        time.sleep(1.0)

    def read(self) -> SensorPacket:
        """
        Read all sensors and fuse the data into a single packet.
        """
        # 1. Read Cameras
        frame1, _ = self.cam1.read() if self.cam1 else (None, 0)
        frame2, _ = self.cam2.read() if self.cam2 else (None, 0)

        # Calculate brightness (simple average of green channel)
        brightness = 100.0
        if frame1 is not None:
            try:
                brightness = frame1[:, :, 1].mean()
            except: pass

        # 2. Read Lidars
        d1 = self.lidar1.get_distance_m() if self.lidar1 else None
        d2 = self.lidar2.get_distance_m() if self.lidar2 else None

        # Sanitize inputs (None -> Infinity)
        val1 = d1 if d1 is not None else 999.0
        val2 = d2 if d2 is not None else 999.0

        # FUSION: The "Safety" distance is the minimum of all sensors
        min_dist = min(val1, val2)

        return SensorPacket(
            image_primary=frame1,
            image_secondary=frame2,
            dist_primary=val1,
            dist_secondary=val2,
            min_dist=min_dist,
            brightness=brightness
        )

    def close(self):
        """Clean shutdown of all hardware."""
        if hasattr(self.cam1, 'close'): self.cam1.close()
        if hasattr(self.cam2, 'close'): self.cam2.close()
        if hasattr(self.lidar1, 'close'): self.lidar1.close()
        if hasattr(self.lidar2, 'close'): self.lidar2.close()