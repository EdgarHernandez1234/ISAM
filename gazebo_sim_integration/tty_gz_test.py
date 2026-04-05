import sys
import tty
import termios
import select
import subprocess

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)

print("Press one key. After that, it will publish once to /arm/joint4/cmd_vel and print the result.")

try:
    tty.setcbreak(fd)
    while True:
        r, _, _ = select.select([sys.stdin], [], [], 0.1)
        if not r:
            continue

        ch = sys.stdin.read(1)
        print("key =", repr(ch))

        result = subprocess.run(
            ["gz", "topic", "-t", "/arm/joint4/cmd_vel", "-m", "gz.msgs.Double", "-p", "data: 0.15"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5.0,
        )

        print("returncode =", result.returncode)
        print("stderr =", repr(result.stderr))
        break
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
