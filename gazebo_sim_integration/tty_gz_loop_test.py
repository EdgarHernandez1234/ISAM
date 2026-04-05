import sys
import tty
import termios
import select
import subprocess
import time

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)

print("Press r repeatedly to burst joint4. Press x to exit.")

def pub(value: float):
    result = subprocess.run(
        ["gz", "topic", "-t", "/arm/joint4/cmd_vel", "-m", "gz.msgs.Double", "-p", f"data: {value}"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=5.0,
    )
    print(f"value={value} returncode={result.returncode} stderr={repr(result.stderr)}")

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
        elif ch == "r":
            pub(0.15)
            time.sleep(0.18)
            pub(0.0)
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
