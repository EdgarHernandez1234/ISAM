import serial # Requires pyserial package: pip install pyserial
import struct
import time
import threading

SOF1 = 0xAA
SOF2 = 0x55
VERSION = 0x01

MSG_HEARTBEAT = 0x01
MSG_CMD       = 0x02

CMD_STOP           = 1
CMD_HOLD           = 2
CMD_ENABLE_RETREAT = 3
CMD_POWER_CUT      = 4
CMD_RESET_FAULT    = 5

# CRC16 CCITT-FALSE (poly 0x1021, init 0xFFFF)
def crc16_ccitt_false(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= (b << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) & 0xFFFF) ^ 0x1021
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF

def build_frame(msg_type: int, seq: int, timestamp_ms: int, payload: bytes) -> bytes:
    fixed = struct.pack("<BBH I B", VERSION, msg_type, seq, timestamp_ms, len(payload))
    crc_input = fixed + payload
    crc = crc16_ccitt_false(crc_input)
    return bytes([SOF1, SOF2]) + fixed + payload + struct.pack("<H", crc)

def send_cmd(ser: serial.Serial, seq: int, cmd: int, arg0: int = 0) -> int:
    payload = struct.pack("<BB", cmd, arg0)
    frame = build_frame(MSG_CMD, seq, int(time.time() * 1000) & 0xFFFFFFFF, payload)
    ser.write(frame)
    return (seq + 1) & 0xFFFF

def send_heartbeat(ser: serial.Serial, seq: int) -> int:
    payload = b""
    frame = build_frame(MSG_HEARTBEAT, seq, int(time.time() * 1000) & 0xFFFFFFFF, payload)
    ser.write(frame)
    return (seq + 1) & 0xFFFF

def reader_thread(ser: serial.Serial, stop_evt: threading.Event):
    # Just print raw incoming bytes chunks (Arduino will emit ACK/NACK/STATE frames)
    while not stop_evt.is_set():
        data = ser.read(ser.in_waiting or 1)
        if data:
            print(f"RX ({len(data)} bytes): {data.hex()}")

def main():
    port = "/dev/ttyACM0"   # Windows example: "COM5"
    baud = 115200
    ser = serial.Serial(port, baud, timeout=0.1)

    stop_evt = threading.Event()
    t = threading.Thread(target=reader_thread, args=(ser, stop_evt), daemon=True)
    t.start()

    seq = 1

    print("Sending heartbeats for 2 seconds...")
    start = time.time()
    while time.time() - start < 2.0:
        seq = send_heartbeat(ser, seq)
        time.sleep(0.05)  # 20 Hz

    print("Command: HOLD")
    seq = send_cmd(ser, seq, CMD_HOLD)
    time.sleep(1.0)

    print("Command: ENABLE_RETREAT")
    seq = send_cmd(ser, seq, CMD_ENABLE_RETREAT)
    time.sleep(1.0)

    print("Command: STOP")
    seq = send_cmd(ser, seq, CMD_STOP)
    time.sleep(1.0)

    print("Command: RESET_FAULT (may require heartbeat streak and estop released)")
    seq = send_cmd(ser, seq, CMD_RESET_FAULT)
    time.sleep(1.0)

    print("Simulating heartbeat loss for 1 second (should trip watchdog)")
    time.sleep(1.0)

    print("Resuming heartbeats for 2 seconds...")
    start = time.time()
    while time.time() - start < 2.0:
        seq = send_heartbeat(ser, seq)
        time.sleep(0.05)

    print("Try RESET_FAULT again after watchdog trip")
    seq = send_cmd(ser, seq, CMD_RESET_FAULT)
    time.sleep(1.0)

    print("Command: POWER_CUT")
    seq = send_cmd(ser, seq, CMD_POWER_CUT)
    time.sleep(1.0)

    stop_evt.set()
    ser.close()

if __name__ == "__main__":
    main()
