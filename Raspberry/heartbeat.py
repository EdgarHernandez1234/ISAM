import serial
import time
import threading

# Initialize Serial to Arduino
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)

def send_heartbeat():
    while True:
        ser.write(b'H') # Send heartbeat character
        time.sleep(0.1) # 100ms interval

# Start the heartbeat thread
heartbeat_thread = threading.Thread(target=send_heartbeat, daemon=True)
heartbeat_thread.start()