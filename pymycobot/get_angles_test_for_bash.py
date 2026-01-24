python3 - << 'PY'

import time
from pymycobot.mycobot import MyCobot

mc = MyCobot('/dev/ttyAMA0', 1000000)
time.sleep(0.5)

angles = mc.get_angles()
print("Angles:", angles)
PY