import os
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFERENCE_DEMO_DIR = os.path.dirname(THIS_DIR)
if CONFERENCE_DEMO_DIR not in sys.path:
    sys.path.insert(0, CONFERENCE_DEMO_DIR)

from serial_base_driver import SerialBaseDriver

PORT = "/dev/ttyACM0"


def print_status(label, status):
    if status is None:
        print(f"{label}: <no status>")
        return
    print(
        f"{label}: seq={status.seq} "
        f"cmd=({status.left_cmd},{status.right_cmd}) "
        f"meas=({status.left_meas},{status.right_meas}) "
        f"enc=({status.enc_l},{status.enc_r}) "
        f"faults={status.faults}"
    )


def run_segment(driver, left_mmps, right_mmps, seconds, resend_hz=10.0):
    period = 1.0 / resend_hz
    end_time = time.time() + seconds
    while time.time() < end_time:
        driver.drive(left_mmps, right_mmps)
        status = driver.poll_for_status(timeout_sec=0.12)
        print_status("STATUS", status)
        time.sleep(period)


def main():
    with SerialBaseDriver(port=PORT, debug=True) as driver:
        print("\n--- ping ---")
        driver.ping()

        print("\n--- clear + enable ---")
        driver.clear()
        driver.enable()

        print("\n--- segment 1: forward 150 mm/s ---")
        run_segment(driver, 150, 150, seconds=2.0, resend_hz=10.0)

        print("\n--- segment 2: forward 250 mm/s ---")
        run_segment(driver, 250, 250, seconds=2.0, resend_hz=10.0)

        print("\n--- stop ---")
        driver.stop()
        time.sleep(0.3)

        print("\n--- final status ---")
        for _ in range(5):
            status = driver.poll_for_status(timeout_sec=0.2)
            print_status("FINAL", status)

        print("\n--- disable ---")
        driver.disable()


if __name__ == "__main__":
    main()
