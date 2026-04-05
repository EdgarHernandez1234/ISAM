import sys
import tty
import termios
import select
import subprocess
import time

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)

print("Press i for rover forward, k for rover stop, r for joint4 burst, x to exit.")

def pub_twist(lin: float, ang: float):
    result = subprocess.run(
        [
            "gz", "topic",
            "-t", "/cmd_vel",
            "-m", "gz.msgs.Twist",
            "-p", f"linear: {{x: {lin}, y: 0.0, z: 0.0}} angular: {{x: 0.0, y: 0.0, z: {ang}}}",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=5.0,
    )
    print(f"/cmd_vel lin={lin} ang={ang} returncode={result.returncode} stderr={repr(result.stderr)}")

def pub_joint4(value: float):
    result = subprocess.run(
        ["gz", "topic", "-t", "/arm/joint4/cmd_vel", "-m", "gz.msgs.Double", "-p", f"data: {value}"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=5.0,
    )
    print(f"/arm/joint4/cmd_vel value={value} returncode={result.returncode} stderr={repr(result.stderr)}")

try:
    tty.setcbreak(fd)
    while True:
        r, _, _ = select.select([sys.stdin], [], [], 0.1)
        if not r:
            continue

        ch = sys.stdin.read(1)
        print("key =", repr(ch))

        if ch == "x":
            break
        elif ch == "i":
            pub_twist(0.7, 0.0)
        elif ch == "k":
            pub_twist(0.0, 0.0)
        elif ch == "r":
            pub_joint4(0.15)
            time.sleep(0.18)
            pub_joint4(0.0)
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
