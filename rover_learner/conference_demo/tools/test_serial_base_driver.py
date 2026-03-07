import os
import sys
import time

# Add parent folder: rover_learner/conference_demo/
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFERENCE_DEMO_DIR = os.path.dirname(THIS_DIR)
if CONFERENCE_DEMO_DIR not in sys.path:
    sys.path.insert(0, CONFERENCE_DEMO_DIR)

from serial_base_driver import SerialBaseDriver, CommandRejected

PORT = "/dev/ttyACM0"


def main():
    with SerialBaseDriver(port=PORT, debug=True) as driver:
        print("\n--- ping ---")
        driver.ping()

        print("\n--- enable ---")
        driver.enable()

        print("\n--- drive forward for 1 second (resending at 10 Hz) ---")
        driver.drive_for(200, 200, duration_sec=1.0, resend_hz=10.0)

        print("\n--- read a few status packets ---")
        for _ in range(5):
            status = driver.poll_for_status(timeout_sec=0.3)
            if status:
                print("STATUS:", status)
            time.sleep(0.05)

        print("\n--- controlled stop ---")
        driver.stop()
        time.sleep(0.2)

        print("\n--- estop ---")
        driver.estop()
        time.sleep(0.2)

        print("\n--- try drive during estop (should fail) ---")
        try:
            driver.drive(100, 100)
        except CommandRejected as exc:
            print("EXPECTED REJECTION:", exc)

        print("\n--- clear estop ---")
        driver.clear()
        time.sleep(0.2)

        print("\n--- disable ---")
        driver.disable()

        print("\nDone.")


if __name__ == "__main__":
    main()
