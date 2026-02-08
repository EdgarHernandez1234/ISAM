#!/usr/bin/env python3
"""
camera_provider.py (rover_learner)

Provides camera frames for the live decider.
Uses GStreamer (nvarguscamerasrc) for CSI cameras.

UPDATED FIX: Uses a simplified pipeline matching the user's successful 
CLI command 'gst-launch-1.0 ...' to avoid strict resolution/framerate negotiation errors.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Protocol

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None  # type: ignore


class CameraProvider(Protocol):
    def read(self) -> Tuple[Any, float]:
        """Return (frame_bgr, timestamp). Raise RuntimeError if not available."""
        ...
    
    def close(self) -> None:
        """Release resources."""
        ...


def build_csi_gstreamer_pipeline(
    sensor_id: int = 0,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    flip_method: int = 0,
) -> str:
    """
    Constructs a SIMPLIFIED GStreamer pipeline matching the working CLI command.
    We ask nvarguscamerasrc to just 'work' without forcing strict caps immediately,
    then convert to BGR for OpenCV.
    """
    # Note: We use 'sensor-id' (standard) but arguably 'sensor_id' works in some shells.
    # We stick to the standard GStreamer property name 'sensor-id'.
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"nvvidconv flip-method={flip_method} ! "
        "video/x-raw, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink drop=1"
    )


@dataclass
class CSICameraProvider:
    """
    CSI Camera provider for Jetson using GStreamer (nvarguscamerasrc).
    """
    sensor_id: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 30
    flip_method: int = 0

    def __post_init__(self) -> None:
        if cv2 is None:
            raise RuntimeError("cv2 not available. Install python3-opencv.")

        self.pipeline_str = build_csi_gstreamer_pipeline(
            self.sensor_id, self.width, self.height, self.fps, self.flip_method
        )
        
        # Debug print to help verify what we are running
        print(f"[diag] GStreamer Pipeline: {self.pipeline_str}")

        # Must use CAP_GSTREAMER
        self._cap = cv2.VideoCapture(self.pipeline_str, cv2.CAP_GSTREAMER)
        
        if not self._cap.isOpened():
            # If this fails, it might be that OpenCV was installed without GStreamer support.
            # The __main__ block below checks for this.
            raise RuntimeError(
                f"Failed to open CSI camera (sensor_id={self.sensor_id}). \n"
                "1. Check if 'sudo systemctl restart nvargus-daemon' helps.\n"
                "2. Ensure your OpenCV supports GStreamer (run this file directly to check)."
            )

        # Warm-up
        for _ in range(5):
            if self._cap.isOpened():
                self._cap.read()

    def read(self) -> Tuple[Any, float]:
        if not self._cap.isOpened():
            raise RuntimeError("Camera is not open")
            
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read frame from CSI camera")
            
        return frame, time.time()

    def close(self) -> None:
        if hasattr(self, "_cap") and self._cap.isOpened():
            self._cap.release()


@dataclass
class USBCameraProvider:
    """Standard USB Camera provider (V4L2/GStreamer auto)."""
    index: int = 0
    width: int = 1920
    height: int = 1080

    def __post_init__(self) -> None:
        if cv2 is None:
            raise RuntimeError("cv2 not available.")
            
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open USB camera index={self.index}")
            
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self) -> Tuple[Any, float]:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read frame from USB camera")
        return frame, time.time()

    def close(self) -> None:
        if hasattr(self, "_cap"):
            self._cap.release()


class MockCameraProvider:
    """Test-only camera provider returning black frames."""
    def __init__(self):
        pass

    def read(self) -> Tuple[Any, float]:
        import numpy as np
        return np.zeros((1080, 1920, 3), dtype=np.uint8), time.time()

    def close(self) -> None:
        pass


# -------------------------------
# Standalone bring-up runner
# -------------------------------
if __name__ == "__main__":
    import argparse
    import sys

    # 1. Verify OpenCV has GStreamer support
    if cv2 is None:
        print("[error] cv2 (OpenCV) not found.")
        sys.exit(1)
    
    build_info = cv2.getBuildInformation()
    if "GStreamer:                   NO" in build_info:
        print("\n[CRITICAL ERROR] Your OpenCV does NOT support GStreamer!")
        print("You must install the Jetson version of OpenCV, usually via apt:")
        print("  sudo apt-get install python3-opencv")
        print("Do NOT use 'pip install opencv-python' on Jetson (it lacks GStreamer).\n")
        sys.exit(1)
    else:
        print("[diag] OpenCV GStreamer support: DETECTED (Good)")

    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor-id", type=int, default=0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(sys.argv[1:])

    print(f"[diag] Starting CSICameraProvider(sensor_id={args.sensor_id})...")
    
    try:
        cam = CSICameraProvider(
            sensor_id=args.sensor_id, 
            width=args.width, 
            height=args.height
        )
        print("[diag] Camera opened successfully.")
        
        while True:
            frame, ts = cam.read()
            if args.debug:
                print(f"[debug] ts={ts:.3f} shape={frame.shape}")
                
            cv2.imshow("CSICameraProvider", frame)
            # Quit on ESC or q
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
    except Exception as e:
        print(f"\n[error] {e}")
    finally:
        try:
            if 'cam' in locals():
                cam.close()
            cv2.destroyAllWindows()
        except:
            pass