import os
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFERENCE_DEMO_DIR = os.path.dirname(THIS_DIR)
if CONFERENCE_DEMO_DIR not in sys.path:
    sys.path.insert(0, CONFERENCE_DEMO_DIR)

from serial_base_driver import SerialBaseDriver

PORT = "/dev/ttyACM0"


def main():
    with SerialBaseDriver(port=PORT, debug=True) as driver:
        print("\n--- ping ---")
        driver.ping()

        print("\n--- clear + enable ---")
        driver.clear()
        driver.enable()

        print("\n--- drive forward for 2 seconds ---")
        start = time.time()
        while time.time() - start < 2.0:
            driver.drive(200, 200)
            status = driver.poll_for_status(timeout_sec=0.15)
            if status:
                print(
                    f"seq={status.seq} "
                    f"cmd=({status.left_cmd},{status.right_cmd}) "
                    f"meas=({status.left_meas},{status.right_meas}) "
                    f"enc=({status.enc_l},{status.enc_r}) "
                    f"faults={status.faults}"
                )
            time.sleep(0.1)

        print("\n--- stop ---")
        driver.stop()
        time.sleep(0.3)

        print("\n--- final status samples ---")
        for _ in range(5):
            status = driver.poll_for_status(timeout_sec=0.2)
            if status:
                print(
                    f"seq={status.seq} "
                    f"cmd=({status.left_cmd},{status.right_cmd}) "
                    f"meas=({status.left_meas},{status.right_meas}) "
                    f"enc=({status.enc_l},{status.enc_r}) "
                    f"faults={status.faults}"
                )

        print("\n--- disable ---")
        driver.disable()


if __name__ == "__main__":
    main()
