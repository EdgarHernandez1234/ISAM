#!/usr/bin/env python3
"""
two_camera_provider.py (rover_learner)

Dual Camera Driver for Jetson Nano.
- Camera A (Primary): Port A (sensor-id=0)
- Camera B (Secondary): Port C (sensor-id=1)

UPDATES:
- Uses 'Loose' Pipeline (Auto-negotiation) to prevent hanging.
- Added Debug prints to isolate freezes.
"""

from __future__ import annotations

import cv2
import time
import sys
import argparse
from typing import Optional, Tuple, Any

def build_loose_pipeline(sensor_id=0):
    """
    A permissive pipeline that lets the camera choose its native resolution,
    then resizes it for OpenCV. This prevents hangs caused by invalid caps.
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        "video/x-raw(memory:NVMM) ! "  # Let camera pick best mode
        "nvvidconv flip-method=0 ! "   # Convert to raw
        "video/x-raw, width=640, height=360, format=(string)BGRx ! " # Resize HERE
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )

class TwoCameraProvider:
    def __init__(self):
        self.cam_a = None
        self.cam_b = None
        self.cam_a_ok = False
        self.cam_b_ok = False

        print("[DualCam] Initializing Cameras (Loose Mode)...")
        
        # --- INIT CAMERA A ---
        try:
            print("[DualCam] Opening Primary (ID=0)...")
            self.cam_a = cv2.VideoCapture(build_loose_pipeline(0), cv2.CAP_GSTREAMER)
            if self.cam_a.isOpened():
                self.cam_a_ok = True
                print("[DualCam] Primary: SUCCESS")
            else:
                print("[DualCam] Primary: FAILED")
        except Exception as e:
            print(f"[DualCam] Primary Error: {e}")

        # --- INIT CAMERA B ---
        try:
            print("[DualCam] Opening Secondary (ID=1)...")
            self.cam_b = cv2.VideoCapture(build_loose_pipeline(1), cv2.CAP_GSTREAMER)
            if self.cam_b.isOpened():
                self.cam_b_ok = True
                print("[DualCam] Secondary: SUCCESS")
            else:
                print("[DualCam] Secondary: FAILED (Check hardware)")
        except Exception as e:
            print(f"[DualCam] Secondary Error: {e}")

    def read(self) -> Tuple[Optional[Any], Optional[Any], float]:
        ts = time.time()
        frame_a = None
        frame_b = None

        # We add a print here only if running in debug mode (handled by caller)
        # but to prevent console spam, we keep it silent in production.
        
        if self.cam_a_ok:
            grabbed, img = self.cam_a.read()
            if grabbed: frame_a = img
        
        if self.cam_b_ok:
            grabbed, img = self.cam_b.read()
            if grabbed: frame_b = img

        return frame_a, frame_b, ts

    def close(self):
        if self.cam_a: self.cam_a.release()
        if self.cam_b: self.cam_b.release()


# =========================================================
# DEBUG TEST
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run visual check")
    args = parser.parse_args()

    print("--- STARTING DUAL CAMERA TEST ---")
    provider = TwoCameraProvider()
    
    if not provider.cam_a_ok and not provider.cam_b_ok:
        print("[FAIL] No cameras found.")
        sys.exit(1)

    print("\n[INFO] Starting Loop. If it freezes below, a camera is hanging.")
    print("Press 'q' to quit.")

    try:
        while True:
            # We print a dot to show the loop is alive. 
            # If dots stop appearing, we know exactly where it froze.
            print(".", end="", flush=True) 
            
            fa, fb, _ = provider.read()
            
            if fa is not None:
                cv2.imshow("Primary", fa)
            
            if fb is not None:
                cv2.imshow("Secondary", fb)
            elif provider.cam_b_ok:
                # If we expect B but it's empty, show a blank screen
                import numpy as np
                blank = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "WAITING FOR B...", (50, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("Secondary", blank)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[INFO] Closing cameras...")
        provider.close()
        cv2.destroyAllWindows()
