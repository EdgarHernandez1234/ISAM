import serial
import time

PORT = "/dev/ttyACM0"   # change if needed
BAUD = 115200

def checksum(payload: str) -> str:
    x = 0
    for ch in payload:
        x ^= ord(ch)
    return f"{x:02X}"

def packet(payload: str) -> str:
    return f"@{payload}*{checksum(payload)}\n"

def send_and_read(ser, payload: str, pause: float = 0.3):
    msg = packet(payload)
    print(f"TX: {msg.strip()}")
    ser.write(msg.encode("utf-8"))
    ser.flush()
    time.sleep(pause)

    got_any = False
    while ser.in_waiting:
        got_any = True
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if line:
            print(f"RX: {line}")
    if not got_any:
        print("RX: <nothing>")

def main():
    with serial.Serial(PORT, BAUD, timeout=0.2) as ser:
        # Opening the port often resets the Uno
        time.sleep(2.0)

        print("\nReading boot output...")
        while ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="replace").strip()
            if line:
                print(f"BOOT: {line}")

        print("\nSending test packets...\n")
        send_and_read(ser, "PNG,1")
        send_and_read(ser, "ENA,2")
        send_and_read(ser, "DRV,3,200,200,0")
        time.sleep(0.4)  # let watchdog trigger if it is working
        while ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="replace").strip()
            if line:
                print(f"RX: {line}")

        send_and_read(ser, "STP,4")
        send_and_read(ser, "EST,5")
        send_and_read(ser, "CLR,6")

if __name__ == "__main__":
    main()
