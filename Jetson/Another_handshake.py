import serial, time # This one specifically goes with Arduino handshake py

PORT = "/dev/ttyACM0"
BAUD = 115200

def read_line(ser, timeout=1.0):
    end = time.time() + timeout
    buf = b""
    while time.time() < end:
        b = ser.read(1)
        if b:
            buf += b
            if buf.endswith(b"\n"):
                return buf.decode(errors="replace").strip()
    return None

with serial.Serial(PORT, BAUD, timeout=0.1) as ser:
    time.sleep(1.0)  # allow Arduino reset on USB connect

    # Read READY line (optional; some systems might miss it)
    line = read_line(ser, timeout=1.0)
    print("BOOT:", line)

    ser.write(b"PING\n")
    print("RESP:", read_line(ser, timeout=1.0))

    ser.write(b"WHOAMI\n")
    print("RESP:", read_line(ser, timeout=1.0))

    ser.write(b"LED 1\n")
    print("RESP:", read_line(ser, timeout=1.0))
    time.sleep(0.5)

    ser.write(b"LED 0\n")
    print("RESP:", read_line(ser, timeout=1.0))
