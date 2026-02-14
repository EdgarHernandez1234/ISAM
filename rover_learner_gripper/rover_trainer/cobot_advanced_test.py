#!/usr/bin/env python3
"""
cobot_advanced_test.py

LIVE VALIDATION SUITE
---------------------
Orchestrates hardware, AI, and scenarios for physical testing.
to run: python3 -m rover_learner.cobot_advanced_test --model models/regolith/cls/sand_cls_clean_vs_dirt.pt
"""

import argparse
import time
import cv2
import sys
from ultralytics import YOLO

# Import our new modules
from rover_learner.rover_trainer.hardware_manager import HardwareManager
from rover_learner.rover_trainer.scenario_logic import ScenarioManager
from rover_learner.cobot_advanced_demo_2 import AsyncArmAnimator # Reuse the animator

def get_menu_choice():
    print("\n" + "="*40)
    print("      ROVER LIVE TEST SUITE")
    print("="*40)
    print("1) 1 Camera (Monocular Vision)")
    print("2) 1 Cam + 1 Lidar (Standard)")
    print("3) 2 Cameras (Stereo Vision - Experimental)")
    print("4) 2 Cams + 2 Lidars (Full Surround)")
    print("="*40)
    return int(input("Select Hardware Configuration [1-4]: "))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="best.pt")
    args = parser.parse_args()

    # 1. Setup
    mode = get_menu_choice()
    hw = HardwareManager(mode)
    logic = ScenarioManager()
    
    # 2. Init AI
    print("[Init] Loading YOLO...")
    model = YOLO(args.model)

    # 3. Init Arm (Headless or ROS)
    animator = AsyncArmAnimator(None) # Pass ROS node if you have it
    animator.start()

    print("\n[TEST STARTED] presenting regolith or blocking sensors triggers scenarios.\n")

    try:
        while True:
            # A. Get Data (Handles 1, 2, or 4 sensors)
            packet = hw.read()
            
            # B. AI Inference
            results = None
            if packet.frame_primary is not None:
                # Use a small size for speed
                results = model(packet.frame_primary, verbose=False, imgsz=320)

            # C. Logic Engine (The "Brain")
            action = logic.evaluate(packet, results)
            
            # D. Execution (The "Body")
            # Map logical actions to animator states
            if action == "ADAPTIVE_PANIC":
                animator.set_target("HOME") # Cower
                animator.status_text = "SENSOR FAILURE - PANIC"
            elif action == "WIGGLE":
                animator.set_target("DUMP", mode="WIGGLE")
            else:
                animator.set_target(action, mode="NORMAL")

            # E. Feedback
            if packet.frame_primary is not None:
                img = cv2.resize(packet.frame_primary, (640, 480))
                
                # Dynamic HUD
                color = (0, 255, 0)
                if action == "RETREAT": color = (0, 0, 255)
                if action == "ADAPTIVE_PANIC": color = (0, 255, 255)
                
                cv2.putText(img, f"ACTION: {action}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(img, f"DIST: {packet.min_dist}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
                
                # Show Secondary Cam if available
                if packet.frame_secondary is not None:
                    # Picture-in-Picture
                    small = cv2.resize(packet.frame_secondary, (160, 120))
                    img[360:480, 480:640] = small
                    cv2.rectangle(img, (480, 360), (640, 480), (255,0,0), 2)

                cv2.imshow("Test Suite", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            time.sleep(0.05) # Loop at ~20Hz

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        hw.close()
        animator.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()