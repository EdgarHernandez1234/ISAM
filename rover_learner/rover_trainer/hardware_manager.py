import time
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Import your existing providers
from rover_learner.camera_provider import CSICameraProvider, USBCameraProvider, MockCameraProvider
from rover_learner.lidar_provider import SerialRPLidarProvider, MockLidarProvider

@dataclass
class SensorPacket:
    frame_primary: Optional[np.ndarray]
    frame_secondary: Optional[np.ndarray]
    dist_primary: Optional[float]
    dist_secondary: Optional[float]
    min_dist: Optional[float]
    brightness: float  # For "Adaptive" darkness check

class HardwareManager:
    def __init__(self, mode: int):
        self.mode = mode
        self.cam1 = None
        self.cam2 = None
        self.lidar1 = None
        self.lidar2 = None

        print(f"[Hardware] Initializing Mode {mode}...")
        
        # --- CAMERA SETUP ---
        if mode in [1, 2, 3, 4]:
            try:
                self.cam1 = CSICameraProvider(0, 640, 480)
            except:
                self.cam1 = USBCameraProvider(0)
        
        if mode in [3, 4]:
            # Secondary Camera (USB usually)
            print("[Hardware] Attempting secondary camera...")
            try:
                self.cam2 = USBCameraProvider(1)
            except:
                print("[Warn] Cam 2 failed. Using Mock.")
                self.cam2 = MockCameraProvider()

        # --- LIDAR SETUP ---
        if mode in [2, 4]:
            # Primary Lidar
            self.lidar1 = SerialRPLidarProvider(port="/dev/ttyUSB0", baudrate=460800)
        
        if mode == 4:
            # Secondary Lidar (Hypothetical second USB port)
            print("[Hardware] Attempting secondary Lidar...")
            try:
                self.lidar2 = SerialRPLidarProvider(port="/dev/ttyUSB1", baudrate=460800)
            except:
                print("[Warn] Lidar 2 failed. Using Mock.")
                self.lidar2 = MockLidarProvider()

    def read(self) -> SensorPacket:
        # Read Cams
        f1, _ = self.cam1.read() if self.cam1 else (None, 0)
        f2, _ = self.cam2.read() if self.cam2 else (None, 0)
        
        # Read Lidars
        d1 = self.lidar1.get_distance_m() if self.lidar1 else None
        d2 = self.lidar2.get_distance_m() if self.lidar2 else None
        
        # Data Fusion: Minimum distance is the safety constraints
        dists = [d for d in [d1, d2] if d is not None]
        min_d = min(dists) if dists else None

        # Calc Brightness (for Adaptive Scenario)
        bright = 0.0
        if f1 is not None:
            bright = np.mean(f1) # Average pixel intensity (0-255)

        return SensorPacket(f1, f2, d1, d2, min_d, bright)

    def close(self):
        if self.cam1: self.cam1.close()
        if self.cam2: self.cam2.close()
        if self.lidar1: self.lidar1.close()
        if self.lidar2: self.lidar2.close()