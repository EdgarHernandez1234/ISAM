Here’s a reliable **USB-serial handshake** you can use to prove Arduino Uno R3 ↔ Jetson Orin Nano communication works (and that both directions work). It avoids the full framing/CRC for now and is perfect as a “unit test” before you plug in the safety protocol.

---

## Arduino sketch: “HELLO / PING / LED” handshake

* Jetson sends: `PING\n`
* Arduino replies: `PONG <millis>\n`
* Jetson can also send: `LED 1\n` or `LED 0\n` to toggle Arduino’s built-in LED (pin 13)

```cpp
// Arduino UNO R3 USB-Serial Handshake Test
// Commands (newline-terminated):
//   PING        -> PONG <millis>
//   WHOAMI      -> UNO_R3
//   LED 1 / LED 0 -> sets builtin LED

#include <Arduino.h>

static const uint32_t BAUD = 115200;
static const uint8_t LED_PIN = 13;

String line;

void setup() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Serial.begin(BAUD);

  // Optional: wait a moment for serial connection stability on some hosts
  delay(500);

  Serial.println("READY UNO_R3");
}

static void handle_line(const String& s) {
  if (s == "PING") {
    Serial.print("PONG ");
    Serial.println(millis());
    return;
  }
  if (s == "WHOAMI") {
    Serial.println("UNO_R3");
    return;
  }
  if (s.startsWith("LED ")) {
    if (s.endsWith("1")) {
      digitalWrite(LED_PIN, HIGH);
      Serial.println("OK LED=1");
    } else if (s.endsWith("0")) {
      digitalWrite(LED_PIN, LOW);
      Serial.println("OK LED=0");
    } else {
      Serial.println("ERR LED expects 0 or 1");
    }
    return;
  }

  Serial.print("ERR unknown: ");
  Serial.println(s);
}

void loop() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    // line-based parser
    if (c == '\n') {
      line.trim(); // removes \r and spaces
      if (line.length() > 0) handle_line(line);
      line = "";
    } else {
      // ignore very long lines for safety
      if (line.length() < 80) line += c;
    }
  }
}
```

---

## Jetson test: quick terminal handshake (no Python needed)

1. Find the Arduino device:

```bash
ls -l /dev/ttyACM* /dev/ttyUSB*
dmesg | tail -n 30
```

You’ll likely see `/dev/ttyACM0`.

2. Watch Arduino output:

```bash
sudo apt-get update
sudo apt-get install -y screen
screen /dev/ttyACM0 115200
```

You should see:

```
READY UNO_R3
```

3. Type into screen:

* `PING` then press Enter → should respond `PONG <number>`
* `LED 1` → LED turns on, Arduino prints `OK LED=1`
* `LED 0` → off

Exit screen with: `Ctrl+A` then `K` then `Y`.

*(If screen doesn’t show input echo, that’s okay—Arduino should still reply.)*

---

## Jetson test: Python “unit test” (recommended)

This one does a proper request/response and fails loudly if anything’s wrong.

```python
import serial, time

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
```

Run:

```bash
pip3 install pyserial
python3 handshake_test.py
```

Expected output looks like:

```
BOOT: READY UNO_R3
RESP: PONG 1234
RESP: UNO_R3
RESP: OK LED=1
RESP: OK LED=0
```

---

## Common gotchas (and quick fixes)

* **No `/dev/ttyACM0`**: try a different USB cable (some are charge-only), different port, check `dmesg`.
* **Permission denied**: add your user to dialout:

  ```bash
  sudo usermod -a -G dialout $USER
  ```

  then log out/in.
* **Garbled text**: baud mismatch (use 115200 on both ends).
* **Arduino resets when opening serial**: normal on Uno; that’s why the Python waits 1 second.

---