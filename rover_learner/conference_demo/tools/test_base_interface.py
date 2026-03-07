import os
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFERENCE_DEMO_DIR = os.path.dirname(THIS_DIR)
if CONFERENCE_DEMO_DIR not in sys.path:
    sys.path.insert(0, CONFERENCE_DEMO_DIR)

from base_interface import SerialBaseInterface

PORT = "/dev/ttyACM0"


def print_status(status):
    if status is None:
        print("STATUS: <none>")
        return
    print(
        f"STATUS: seq={status.seq} "
        f"cmd=({status.left_cmd},{status.right_cmd}) "
        f"meas=({status.left_meas},{status.right_meas}) "
        f"enc=({status.enc_l},{status.enc_r}) "
        f"faults={status.faults}"
    )


def main():
    base = SerialBaseInterface(port=PORT, debug=True, auto_open=True)

    try:
        print("\n--- ping ---")
        base.ping()

        print("\n--- clear + enable ---")
        base.clear()
        base.enable()

        print("\n--- drive via BaseInterface ---")
        for _ in range(10):
            base.drive_lr(180, 180)
            status = base.poll_status(timeout_sec=0.15)
            print_status(status)
            time.sleep(0.1)

        print("\n--- stop ---")
        base.stop()
        time.sleep(0.3)

        print("\n--- latest health ---")
        health = base.get_health()
        print(health)
        print("fault descriptions:", base.describe_faults())

        print("\n--- disable ---")
        base.disable()

    finally:
        base.close()


if __name__ == "__main__":
    main()
