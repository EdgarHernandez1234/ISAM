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
