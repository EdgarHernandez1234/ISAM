import sys
import tty
import termios
import select
import subprocess
import time

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)

print("Press one key. After that, it will publish 0.15 to /arm/joint4/cmd_vel, wait 0.18s, then publish 0.0.")

try:
    tty.setcbreak(fd)
    while True:
        r, _, _ = select.select([sys.stdin], [], [], 0.1)
        if not r:
            continue

        ch = sys.stdin.read(1)
        print("key =", repr(ch))

        r1 = subprocess.run(
            ["gz", "topic", "-t", "/arm/joint4/cmd_vel", "-m", "gz.msgs.Double", "-p", "data: 0.15"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5.0,
        )
        print("first returncode =", r1.returncode)
        print("first stderr =", repr(r1.stderr))

        time.sleep(0.18)

        r2 = subprocess.run(
            ["gz", "topic", "-t", "/arm/joint4/cmd_vel", "-m", "gz.msgs.Double", "-p", "data: 0.0"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5.0,
        )
        print("second returncode =", r2.returncode)
        print("second stderr =", repr(r2.stderr))
        break
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
