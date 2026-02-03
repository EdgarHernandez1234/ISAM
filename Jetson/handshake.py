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

def send_packet(msg_type, val1=0, val2=0):
    payload = f"{msg_type},{val1},{val2}"
    
    # Simple XOR Checksum
    checksum = 0
    for char in payload:
        checksum ^= ord(char)
    
    full_packet = f"<{payload}|{checksum:02x}>"
    ser.write(full_packet.encode('utf-8'))

# Usage:
send_packet('H', 1) # Heartbeat with System State '1' (Active)

