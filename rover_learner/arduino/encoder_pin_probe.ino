#include <Arduino.h>

static const int PIN_ENC_LEFT = 2;
static const int PIN_ENC_RIGHT = 3;

volatile unsigned long leftEdges = 0;
volatile unsigned long rightEdges = 0;

void isrLeft() {
  leftEdges++;
}

void isrRight() {
  rightEdges++;
}

void setup() {
  Serial.begin(115200);

  pinMode(PIN_ENC_LEFT, INPUT_PULLUP);
  pinMode(PIN_ENC_RIGHT, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(PIN_ENC_LEFT), isrLeft, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_RIGHT), isrRight, CHANGE);

  Serial.println("encoder_pin_probe starting");
  Serial.println("Rotate wheels by hand and watch raw pin states + edge counts.");
}

void loop() {
  static unsigned long lastPrint = 0;
  unsigned long now = millis();

  if (now - lastPrint >= 100) {
    lastPrint = now;

    noInterrupts();
    unsigned long lEdges = leftEdges;
    unsigned long rEdges = rightEdges;
    interrupts();

    int lState = digitalRead(PIN_ENC_LEFT);
    int rState = digitalRead(PIN_ENC_RIGHT);

    Serial.print("Lstate=");
    Serial.print(lState);
    Serial.print(" Rstate=");
    Serial.print(rState);
    Serial.print(" Ledges=");
    Serial.print(lEdges);
    Serial.print(" Redges=");
    Serial.println(rEdges);
  }
}
