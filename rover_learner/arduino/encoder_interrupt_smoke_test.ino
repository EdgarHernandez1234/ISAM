#include <Arduino.h>

static const int PIN_ENC_LEFT = 2;
static const int PIN_ENC_RIGHT = 3;

volatile long leftTicks = 0;
volatile long rightTicks = 0;

void isrLeft() {
  leftTicks++;
}

void isrRight() {
  rightTicks++;
}

void setup() {
  Serial.begin(115200);

  pinMode(PIN_ENC_LEFT, INPUT_PULLUP);
  pinMode(PIN_ENC_RIGHT, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(PIN_ENC_LEFT), isrLeft, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_RIGHT), isrRight, RISING);

  Serial.println("encoder_interrupt_smoke_test starting");
}

void loop() {
  static unsigned long lastPrint = 0;
  unsigned long now = millis();

  if (now - lastPrint >= 200) {
    lastPrint = now;

    noInterrupts();
    long l = leftTicks;
    long r = rightTicks;
    interrupts();

    Serial.print("leftTicks=");
    Serial.print(l);
    Serial.print(" rightTicks=");
    Serial.println(r);
  }
}
