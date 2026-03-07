#include <Arduino.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

// =========================
// Protocol / timing config
// =========================
static const unsigned long SERIAL_BAUD = 115200;
static const unsigned long WATCHDOG_TIMEOUT_MS = 250;
static const unsigned long STATUS_PERIOD_MS = 100;   // 10 Hz status
static const unsigned long CONTROL_PERIOD_MS = 50;   // 20 Hz control loop
static const int MAX_LINE_LEN = 96;

// =========================
// Fault bits
// =========================
static const uint8_t FAULT_WATCHDOG      = 1 << 0;
static const uint8_t FAULT_ESTOP         = 1 << 1;
static const uint8_t FAULT_BAD_PACKET    = 1 << 2;
static const uint8_t FAULT_ENCODER       = 1 << 3;
static const uint8_t FAULT_LOW_BATTERY   = 1 << 4;
static const uint8_t FAULT_DRIVER        = 1 << 5;
static const uint8_t FAULT_OVERCURRENT   = 1 << 6;

// =========================
// Error codes
// =========================
static const uint8_t ERR_BAD_ENVELOPE    = 1;
static const uint8_t ERR_BAD_CHECKSUM    = 2;
static const uint8_t ERR_UNKNOWN_COMMAND = 3;
static const uint8_t ERR_FIELD_COUNT     = 4;
static const uint8_t ERR_FIELD_PARSE     = 5;
static const uint8_t ERR_OUT_OF_RANGE    = 6;
static const uint8_t ERR_BAD_STATE       = 7;
static const uint8_t ERR_ESTOP_ACTIVE    = 8;

// =========================
// Motion / target limits
// =========================
static const int16_t MAX_TARGET_MMPS = 1000;

// =========================
// Encoder / wheel placeholders
// UPDATE THESE WHEN HARDWARE ARRIVES
// =========================
static const int PIN_ENC_LEFT  = 2;
static const int PIN_ENC_RIGHT = 3;

// Example placeholders only.
// Replace with real wheel circumference and encoder counts.
static const float TICKS_PER_REV   = 20.0f;
static const float WHEEL_CIRCUM_MM = 345.6f;

// =========================
// Closed-loop control tuning
// These are conservative starter values.
// Tune later on the real chassis.
// =========================
static const bool ENABLE_CLOSED_LOOP = true;

// If encoder fault occurs while commanded to move:
// false = stop motors for safety
// true  = keep feedforward / open-loop fallback
static const bool ALLOW_OPEN_LOOP_WITH_ENCODER_FAULT = false;

// Ignore tiny commanded speeds for encoder-fault detection
static const int16_t MIN_MOVING_CMD_MMPS = 50;

// Number of consecutive control cycles with zero ticks before encoder fault
static const uint8_t ENCODER_STALL_CYCLES = 4;

// Feedforward + PI gains
static const int MIN_ACTIVE_PWM = 70;
static const float KP_PWM_PER_MMPS = 0.45f;
static const float KI_PWM_PER_MMPS_S = 0.80f;
static const float INTEGRAL_LIMIT = 300.0f;

// =========================
// Robot state
// =========================
enum ControllerState : uint8_t {
  STATE_DISABLED = 0,
  STATE_ENABLED  = 1,
  STATE_ESTOP    = 2
};

ControllerState controllerState = STATE_DISABLED;

// Latest accepted command seq
uint16_t lastAcceptedSeq = 0;

// Current target commands
int16_t leftCmdMmps = 0;
int16_t rightCmdMmps = 0;

// Measured speeds
int16_t leftMeasMmps = 0;
int16_t rightMeasMmps = 0;

// Encoder counts
volatile long encLeftTicksVolatile = 0;
volatile long encRightTicksVolatile = 0;

long encLeft = 0;
long encRight = 0;
long prevEncLeft = 0;
long prevEncRight = 0;

// Controller outputs / state
int leftPwmCmd = 0;
int rightPwmCmd = 0;
float leftIntegral = 0.0f;
float rightIntegral = 0.0f;
uint8_t leftNoTickCycles = 0;
uint8_t rightNoTickCycles = 0;

// Placeholder battery
uint16_t batteryMv = 12000;

// Fault bitmask
uint8_t faults = 0;

// Timing
unsigned long lastValidDriveMs = 0;
unsigned long lastStatusMs = 0;
unsigned long lastControlMs = 0;

// Serial line buffer
char lineBuf[MAX_LINE_LEN];
int lineLen = 0;

// ==========================================
// Hardware mapping placeholders - CHANGE ME
// ==========================================
static const int PIN_LEFT_PWM  = 5;
static const int PIN_LEFT_DIR  = 4;
static const int PIN_RIGHT_PWM = 6;
static const int PIN_RIGHT_DIR = 7;

// =========================
// Encoder ISRs
// =========================
void isrEncLeft() {
  encLeftTicksVolatile++;
}

void isrEncRight() {
  encRightTicksVolatile++;
}

// =========================
// Utility helpers
// =========================
uint8_t xorChecksum(const char* payload) {
  uint8_t cs = 0;
  while (*payload) {
    cs ^= (uint8_t)(*payload);
    payload++;
  }
  return cs;
}

bool parseHexByte(const char* s, uint8_t& out) {
  char* endPtr = nullptr;
  long v = strtol(s, &endPtr, 16);
  if (endPtr == s || *endPtr != '\0' || v < 0 || v > 255) {
    return false;
  }
  out = (uint8_t)v;
  return true;
}

bool parseUInt16(const char* s, uint16_t& out) {
  char* endPtr = nullptr;
  long v = strtol(s, &endPtr, 10);
  if (endPtr == s || *endPtr != '\0' || v < 0 || v > 65535) {
    return false;
  }
  out = (uint16_t)v;
  return true;
}

bool parseInt16(const char* s, int16_t& out) {
  char* endPtr = nullptr;
  long v = strtol(s, &endPtr, 10);
  if (endPtr == s || *endPtr != '\0' || v < -32768 || v > 32767) {
    return false;
  }
  out = (int16_t)v;
  return true;
}

float clampFloat(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

// =========================
// Packet send helpers
// =========================
void sendPacket(const char* payload) {
  uint8_t cs = xorChecksum(payload);
  char out[140];
  snprintf(out, sizeof(out), "@%s*%02X\n", payload, cs);
  Serial.print(out);
}

void sendAck(uint16_t seq) {
  char payload[32];
  snprintf(payload, sizeof(payload), "ACK,%u", seq);
  sendPacket(payload);
}

void sendErr(uint16_t seq, uint8_t code) {
  char payload[32];
  snprintf(payload, sizeof(payload), "ERR,%u,%u", seq, code);
  sendPacket(payload);
}

void sendFlt(uint16_t seq, uint8_t faultMask) {
  char payload[32];
  snprintf(payload, sizeof(payload), "FLT,%u,%u", seq, faultMask);
  sendPacket(payload);
}

void sendStatus() {
  char payload[128];
  snprintf(
    payload,
    sizeof(payload),
    "STA,%u,%d,%d,%d,%d,%ld,%ld,%u,%u",
    lastAcceptedSeq,
    leftCmdMmps,
    rightCmdMmps,
    leftMeasMmps,
    rightMeasMmps,
    encLeft,
    encRight,
    batteryMv,
    faults
  );
  sendPacket(payload);
}

void sendPowerOnBanner() {
  sendPacket("PON,1,uno_chassis_v1");
}

// =========================
// Fault helpers
// =========================
void setFault(uint8_t mask) {
  faults |= mask;
}

void clearFault(uint8_t mask) {
  faults &= ~mask;
}

// =========================
// Motor output helpers
// =========================
void stopMotorsImmediate() {
  analogWrite(PIN_LEFT_PWM, 0);
  analogWrite(PIN_RIGHT_PWM, 0);
}

void setSideMotorOutput(int pinDir, int pinPwm, int16_t targetMmps, int pwm) {
  pwm = constrain(pwm, 0, 255);

  if (targetMmps == 0 || pwm == 0) {
    analogWrite(pinPwm, 0);
    return;
  }

  digitalWrite(pinDir, (targetMmps >= 0) ? HIGH : LOW);
  analogWrite(pinPwm, pwm);
}

void applyCurrentMotorOutputs() {
  if (controllerState != STATE_ENABLED) {
    stopMotorsImmediate();
    return;
  }

  if ((faults & FAULT_ENCODER) && !ALLOW_OPEN_LOOP_WITH_ENCODER_FAULT) {
    stopMotorsImmediate();
    return;
  }

  setSideMotorOutput(PIN_LEFT_DIR, PIN_LEFT_PWM, leftCmdMmps, leftPwmCmd);
  setSideMotorOutput(PIN_RIGHT_DIR, PIN_RIGHT_PWM, rightCmdMmps, rightPwmCmd);
}

int targetToBasePwm(int16_t targetMmps) {
  int mag = abs(targetMmps);
  if (mag <= 0) {
    return 0;
  }

  long pwm = map(mag, 0, MAX_TARGET_MMPS, MIN_ACTIVE_PWM, 255);
  pwm = constrain(pwm, 0, 255);
  return (int)pwm;
}

void resetControllers() {
  leftIntegral = 0.0f;
  rightIntegral = 0.0f;
  leftPwmCmd = 0;
  rightPwmCmd = 0;
}

void applyFeedforwardTargets() {
  if (controllerState != STATE_ENABLED) {
    stopMotorsImmediate();
    return;
  }

  leftPwmCmd = targetToBasePwm(leftCmdMmps);
  rightPwmCmd = targetToBasePwm(rightCmdMmps);
  applyCurrentMotorOutputs();
}

// =========================
// Motion command helpers
// =========================
void controlledStop() {
  leftCmdMmps = 0;
  rightCmdMmps = 0;
  resetControllers();
  stopMotorsImmediate();
}

void latchEstop() {
  controllerState = STATE_ESTOP;
  setFault(FAULT_ESTOP);
  leftCmdMmps = 0;
  rightCmdMmps = 0;
  resetControllers();
  stopMotorsImmediate();
}

void clearEstopToDisabled() {
  controllerState = STATE_DISABLED;
  clearFault(FAULT_ESTOP);
  clearFault(FAULT_WATCHDOG);
  leftCmdMmps = 0;
  rightCmdMmps = 0;
  resetControllers();
  stopMotorsImmediate();
}

// =========================
// Encoder / measurement helpers
// =========================
void snapshotEncoders(long &leftTicks, long &rightTicks) {
  noInterrupts();
  leftTicks = encLeftTicksVolatile;
  rightTicks = encRightTicksVolatile;
  interrupts();
}

int16_t signedMmpsFromDelta(long deltaTicks, unsigned long dtMs, int16_t currentCmd) {
  if (dtMs == 0 || TICKS_PER_REV <= 0.0f || WHEEL_CIRCUM_MM <= 0.0f) {
    return 0;
  }

  float revs = ((float)deltaTicks) / TICKS_PER_REV;
  float mmMoved = revs * WHEEL_CIRCUM_MM;
  float mmps = (mmMoved * 1000.0f) / (float)dtMs;

  if (currentCmd < 0) {
    mmps = -mmps;
  } else if (currentCmd == 0) {
    mmps = 0.0f;
  }

  if (mmps > 32767.0f) mmps = 32767.0f;
  if (mmps < -32768.0f) mmps = -32768.0f;

  return (int16_t)mmps;
}

void updateEncoderFaultState(long dLeft, long dRight) {
  bool leftShouldMove = (controllerState == STATE_ENABLED && abs(leftCmdMmps) >= MIN_MOVING_CMD_MMPS);
  bool rightShouldMove = (controllerState == STATE_ENABLED && abs(rightCmdMmps) >= MIN_MOVING_CMD_MMPS);

  if (leftShouldMove) {
    if (dLeft == 0) {
      if (leftNoTickCycles < 255) leftNoTickCycles++;
    } else {
      leftNoTickCycles = 0;
    }
  } else {
    leftNoTickCycles = 0;
  }

  if (rightShouldMove) {
    if (dRight == 0) {
      if (rightNoTickCycles < 255) rightNoTickCycles++;
    } else {
      rightNoTickCycles = 0;
    }
  } else {
    rightNoTickCycles = 0;
  }

  if (leftNoTickCycles >= ENCODER_STALL_CYCLES || rightNoTickCycles >= ENCODER_STALL_CYCLES) {
    setFault(FAULT_ENCODER);
  } else {
    clearFault(FAULT_ENCODER);
  }
}

int computeClosedLoopPwm(int16_t targetMmps, int16_t measMmps, float &integral, float dtSec) {
  if (targetMmps == 0) {
    integral = 0.0f;
    return 0;
  }

  float targetMag = (float)abs(targetMmps);
  float measMag = (float)abs(measMmps);
  float error = targetMag - measMag;

  integral += error * dtSec;
  integral = clampFloat(integral, -INTEGRAL_LIMIT, INTEGRAL_LIMIT);

  float ff = (float)targetToBasePwm(targetMmps);
  float pwm = ff + (KP_PWM_PER_MMPS * error) + (KI_PWM_PER_MMPS_S * integral);

  if (pwm < 0.0f) pwm = 0.0f;
  if (pwm > 255.0f) pwm = 255.0f;

  return (int)pwm;
}

// =========================
// Closed-loop control step
// =========================
void runControlStep() {
  unsigned long now = millis();
  if (now - lastControlMs < CONTROL_PERIOD_MS) {
    return;
  }

  unsigned long dtMs = now - lastControlMs;
  lastControlMs = now;

  long leftTicksNow, rightTicksNow;
  snapshotEncoders(leftTicksNow, rightTicksNow);

  encLeft = leftTicksNow;
  encRight = rightTicksNow;

  long dLeft = leftTicksNow - prevEncLeft;
  long dRight = rightTicksNow - prevEncRight;

  prevEncLeft = leftTicksNow;
  prevEncRight = rightTicksNow;

  leftMeasMmps = signedMmpsFromDelta(dLeft, dtMs, leftCmdMmps);
  rightMeasMmps = signedMmpsFromDelta(dRight, dtMs, rightCmdMmps);

  updateEncoderFaultState(dLeft, dRight);

  // Placeholder until real analog battery measurement exists
  batteryMv = 12000;

  if (controllerState != STATE_ENABLED) {
    resetControllers();
    stopMotorsImmediate();
    return;
  }

  if (leftCmdMmps == 0 && rightCmdMmps == 0) {
    resetControllers();
    stopMotorsImmediate();
    return;
  }

  if ((faults & FAULT_ENCODER) && !ALLOW_OPEN_LOOP_WITH_ENCODER_FAULT) {
    resetControllers();
    stopMotorsImmediate();
    return;
  }

  if (!ENABLE_CLOSED_LOOP) {
    applyFeedforwardTargets();
    return;
  }

  float dtSec = ((float)dtMs) / 1000.0f;
  leftPwmCmd = computeClosedLoopPwm(leftCmdMmps, leftMeasMmps, leftIntegral, dtSec);
  rightPwmCmd = computeClosedLoopPwm(rightCmdMmps, rightMeasMmps, rightIntegral, dtSec);

  applyCurrentMotorOutputs();
}

// =========================
// Command handlers
// =========================
void handlePNG(char* tokens[], int count) {
  if (count != 2) {
    sendErr(0, ERR_FIELD_COUNT);
    return;
  }

  uint16_t seq;
  if (!parseUInt16(tokens[1], seq)) {
    sendErr(0, ERR_FIELD_PARSE);
    return;
  }

  lastAcceptedSeq = seq;
  sendAck(seq);
}

void handleENA(char* tokens[], int count) {
  if (count != 2) {
    sendErr(0, ERR_FIELD_COUNT);
    return;
  }

  uint16_t seq;
  if (!parseUInt16(tokens[1], seq)) {
    sendErr(0, ERR_FIELD_PARSE);
    return;
  }

  if (controllerState == STATE_ESTOP) {
    sendErr(seq, ERR_ESTOP_ACTIVE);
    return;
  }

  controllerState = STATE_ENABLED;
  leftCmdMmps = 0;
  rightCmdMmps = 0;
  resetControllers();
  clearFault(FAULT_WATCHDOG);
  lastValidDriveMs = millis();
  stopMotorsImmediate();

  lastAcceptedSeq = seq;
  sendAck(seq);
}

void handleDIS(char* tokens[], int count) {
  if (count != 2) {
    sendErr(0, ERR_FIELD_COUNT);
    return;
  }

  uint16_t seq;
  if (!parseUInt16(tokens[1], seq)) {
    sendErr(0, ERR_FIELD_PARSE);
    return;
  }

  controllerState = STATE_DISABLED;
  leftCmdMmps = 0;
  rightCmdMmps = 0;
  resetControllers();
  stopMotorsImmediate();

  lastAcceptedSeq = seq;
  sendAck(seq);
}

void handleDRV(char* tokens[], int count) {
  if (count != 5) {
    sendErr(0, ERR_FIELD_COUNT);
    return;
  }

  uint16_t seq;
  int16_t leftTarget;
  int16_t rightTarget;
  uint16_t flags;

  if (!parseUInt16(tokens[1], seq) ||
      !parseInt16(tokens[2], leftTarget) ||
      !parseInt16(tokens[3], rightTarget) ||
      !parseUInt16(tokens[4], flags)) {
    sendErr(0, ERR_FIELD_PARSE);
    return;
  }

  if (flags != 0) {
    sendErr(seq, ERR_OUT_OF_RANGE);
    return;
  }

  if (controllerState == STATE_ESTOP) {
    sendErr(seq, ERR_ESTOP_ACTIVE);
    return;
  }

  if (controllerState != STATE_ENABLED) {
    sendErr(seq, ERR_BAD_STATE);
    return;
  }

  if (leftTarget < -MAX_TARGET_MMPS || leftTarget > MAX_TARGET_MMPS ||
      rightTarget < -MAX_TARGET_MMPS || rightTarget > MAX_TARGET_MMPS) {
    sendErr(seq, ERR_OUT_OF_RANGE);
    return;
  }

  leftCmdMmps = leftTarget;
  rightCmdMmps = rightTarget;
  lastValidDriveMs = millis();
  clearFault(FAULT_WATCHDOG);

  // Immediate feedforward kick so the wheels begin moving before next control tick.
  applyFeedforwardTargets();

  lastAcceptedSeq = seq;
  sendAck(seq);
}

void handleSTP(char* tokens[], int count) {
  if (count != 2) {
    sendErr(0, ERR_FIELD_COUNT);
    return;
  }

  uint16_t seq;
  if (!parseUInt16(tokens[1], seq)) {
    sendErr(0, ERR_FIELD_PARSE);
    return;
  }

  if (controllerState == STATE_ESTOP) {
    sendErr(seq, ERR_ESTOP_ACTIVE);
    return;
  }

  controlledStop();

  lastAcceptedSeq = seq;
  sendAck(seq);
}

void handleEST(char* tokens[], int count) {
  if (count != 2) {
    sendErr(0, ERR_FIELD_COUNT);
    return;
  }

  uint16_t seq;
  if (!parseUInt16(tokens[1], seq)) {
    sendErr(0, ERR_FIELD_PARSE);
    return;
  }

  latchEstop();

  lastAcceptedSeq = seq;
  sendAck(seq);
  sendFlt(seq, faults);
}

void handleCLR(char* tokens[], int count) {
  if (count != 2) {
    sendErr(0, ERR_FIELD_COUNT);
    return;
  }

  uint16_t seq;
  if (!parseUInt16(tokens[1], seq)) {
    sendErr(0, ERR_FIELD_PARSE);
    return;
  }

  clearEstopToDisabled();

  lastAcceptedSeq = seq;
  sendAck(seq);
}

// =========================
// Payload tokenization
// =========================
int splitCsv(char* s, char* tokens[], int maxTokens) {
  int n = 0;
  char* tok = strtok(s, ",");
  while (tok != nullptr && n < maxTokens) {
    tokens[n++] = tok;
    tok = strtok(nullptr, ",");
  }
  return n;
}

void dispatchPayload(char* payload) {
  char work[96];
  strncpy(work, payload, sizeof(work) - 1);
  work[sizeof(work) - 1] = '\0';

  char* tokens[8];
  int count = splitCsv(work, tokens, 8);
  if (count < 1) {
    sendErr(0, ERR_BAD_ENVELOPE);
    return;
  }

  if (strcmp(tokens[0], "PNG") == 0) {
    handlePNG(tokens, count);
  } else if (strcmp(tokens[0], "ENA") == 0) {
    handleENA(tokens, count);
  } else if (strcmp(tokens[0], "DIS") == 0) {
    handleDIS(tokens, count);
  } else if (strcmp(tokens[0], "DRV") == 0) {
    handleDRV(tokens, count);
  } else if (strcmp(tokens[0], "STP") == 0) {
    handleSTP(tokens, count);
  } else if (strcmp(tokens[0], "EST") == 0) {
    handleEST(tokens, count);
  } else if (strcmp(tokens[0], "CLR") == 0) {
    handleCLR(tokens, count);
  } else {
    sendErr(0, ERR_UNKNOWN_COMMAND);
  }
}

// =========================
// Full packet processing
// =========================
void processLine(char* line) {
  size_t len = strlen(line);
  if (len > 0 && line[len - 1] == '\r') {
    line[len - 1] = '\0';
    len--;
  }

  if (len < 5 || line[0] != '@') {
    setFault(FAULT_BAD_PACKET);
    sendErr(0, ERR_BAD_ENVELOPE);
    return;
  }

  char* star = strchr(line, '*');
  if (star == nullptr) {
    setFault(FAULT_BAD_PACKET);
    sendErr(0, ERR_BAD_ENVELOPE);
    return;
  }

  *star = '\0';
  char* payload = line + 1;
  char* checksumStr = star + 1;

  if (strlen(checksumStr) != 2) {
    setFault(FAULT_BAD_PACKET);
    sendErr(0, ERR_BAD_CHECKSUM);
    return;
  }

  uint8_t gotCs;
  if (!parseHexByte(checksumStr, gotCs)) {
    setFault(FAULT_BAD_PACKET);
    sendErr(0, ERR_BAD_CHECKSUM);
    return;
  }

  uint8_t wantCs = xorChecksum(payload);
  if (gotCs != wantCs) {
    setFault(FAULT_BAD_PACKET);
    sendErr(0, ERR_BAD_CHECKSUM);
    return;
  }

  dispatchPayload(payload);
}

// =========================
// Serial receive loop
// =========================
void pollSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '\n') {
      lineBuf[lineLen] = '\0';
      if (lineLen > 0) {
        processLine(lineBuf);
      }
      lineLen = 0;
      continue;
    }

    if (c == '\r') {
      continue;
    }

    if (lineLen < MAX_LINE_LEN - 1) {
      lineBuf[lineLen++] = c;
    } else {
      lineLen = 0;
      setFault(FAULT_BAD_PACKET);
      sendErr(0, ERR_BAD_ENVELOPE);
    }
  }
}

// =========================
// Watchdog
// =========================
void checkWatchdog() {
  if (controllerState != STATE_ENABLED) {
    return;
  }

  unsigned long now = millis();
  if (now - lastValidDriveMs > WATCHDOG_TIMEOUT_MS) {
    if ((faults & FAULT_WATCHDOG) == 0) {
      setFault(FAULT_WATCHDOG);
      controlledStop();
      sendFlt(lastAcceptedSeq, faults);
    } else {
      controlledStop();
    }
  }
}

// =========================
// Setup / loop
// =========================
void setup() {
  pinMode(PIN_LEFT_PWM, OUTPUT);
  pinMode(PIN_LEFT_DIR, OUTPUT);
  pinMode(PIN_RIGHT_PWM, OUTPUT);
  pinMode(PIN_RIGHT_DIR, OUTPUT);

  pinMode(PIN_ENC_LEFT, INPUT_PULLUP);
  pinMode(PIN_ENC_RIGHT, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(PIN_ENC_LEFT), isrEncLeft, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_RIGHT), isrEncRight, RISING);

  stopMotorsImmediate();

  Serial.begin(SERIAL_BAUD);
  while (!Serial) {
    ;
  }

  unsigned long now = millis();
  lastValidDriveMs = now;
  lastStatusMs = now;
  lastControlMs = now;

  sendPowerOnBanner();
}

void loop() {
  pollSerial();
  runControlStep();
  checkWatchdog();

  unsigned long now = millis();
  if (now - lastStatusMs >= STATUS_PERIOD_MS) {
    lastStatusMs = now;
    sendStatus();
  }
}
