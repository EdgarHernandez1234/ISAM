/*
  Arduino UNO Safety Controller
  - Safety FSM: BOOT_SAFE, ARMED_IDLE, SAFE_HOLD, SAFE_STOP, RETREAT_ENABLED, POWER_CUT
  - Priority: ESTOP > WATCHDOG > POWER_CUT cmd > STOP cmd > HOLD cmd > ENABLE_RETREAT cmd > RESET_FAULT
  - Serial protocol: SOF(0xAA55), version, msg_type, seq, timestamp_ms, len, payload, crc16
  - msg types: HEARTBEAT(0x01), CMD(0x02), ACK(0x10), NACK(0x11), STATE(0x20)
  - CRC16: CCITT-FALSE (poly 0x1021, init 0xFFFF)
*/

#include <Arduino.h>

// ---------------- Pins ----------------
static const uint8_t PIN_ESTOP_IN = 2;        // INPUT_PULLUP; pressed -> LOW
static const uint8_t PIN_MOTOR_EN = 8;        // motor/driver enable gate
static const uint8_t PIN_POWER_RELAY = 9;     // relay/contactor driver input
static const uint8_t PIN_RETREAT_ALLOWED = 10;

// ---------------- Protocol ----------------
static const uint8_t SOF1 = 0xAA;
static const uint8_t SOF2 = 0x55;
static const uint8_t PROTO_VERSION = 0x01;

enum MsgType : uint8_t {
  MSG_HEARTBEAT = 0x01,
  MSG_CMD       = 0x02,
  MSG_ACK       = 0x10,
  MSG_NACK      = 0x11,
  MSG_STATE     = 0x20
};

enum Cmd : uint8_t {
  CMD_STOP          = 1,
  CMD_HOLD          = 2,
  CMD_ENABLE_RETREAT= 3,
  CMD_POWER_CUT     = 4,
  CMD_RESET_FAULT   = 5,
  CMD_SET_MODE      = 6
};

enum FaultCode : uint8_t {
  FAULT_OK               = 0x00,
  FAULT_ESTOP_ASSERTED   = 0x01,
  FAULT_WATCHDOG_TIMEOUT = 0x02,
  FAULT_COMMS_CRC_FAIL   = 0x03,
  FAULT_COMMS_SEQ_ERROR  = 0x04,
  FAULT_CMD_UNKNOWN      = 0x05
};

enum State : uint8_t {
  S_BOOT_SAFE       = 0,
  S_ARMED_IDLE      = 1,
  S_SAFE_HOLD       = 2,
  S_SAFE_STOP       = 3,
  S_RETREAT_ENABLED = 4,
  S_POWER_CUT       = 5
};

// ---------------- Watchdog timing ----------------
// Expect heartbeat ~20 Hz (50 ms). Timeout 300 ms is demo-friendly.
static const uint32_t WATCHDOG_TIMEOUT_MS = 300;
static const uint8_t  HEARTBEAT_STREAK_REQUIRED = 10; // require N good heartbeats before clearing faults (policy)

// ---------------- State + Fault Latching ----------------
static State g_state = S_BOOT_SAFE;

static bool g_fault_latched = false;
static FaultCode g_fault_code = FAULT_OK;

static uint16_t g_crc_fail_count = 0;
static uint16_t g_seq_error_count = 0;
static uint16_t g_watchdog_trip_count = 0;

static uint16_t g_last_seq_seen = 0;
static bool g_have_seq = false;

static uint32_t g_last_heartbeat_ms = 0;
static uint8_t  g_good_heartbeat_streak = 0;

// For command handling
static bool g_cmd_stop = false;
static bool g_cmd_hold = false;
static bool g_cmd_enable_retreat = false;
static bool g_cmd_power_cut = false;
static bool g_cmd_reset_fault = false;

// ---------------- CRC16 (CCITT-FALSE) ----------------
static uint16_t crc16_ccitt_false(const uint8_t* data, size_t len) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < len; i++) {
    crc ^= (uint16_t)data[i] << 8;
    for (uint8_t b = 0; b < 8; b++) {
      if (crc & 0x8000) crc = (crc << 1) ^ 0x1021;
      else crc <<= 1;
    }
  }
  return crc;
}

// ---------------- Helpers ----------------
static bool estop_asserted() {
  return digitalRead(PIN_ESTOP_IN) == LOW;
}

static void apply_outputs_for_state(State s) {
  // Safe defaults
  uint8_t motor_en = LOW;
  uint8_t power_en = LOW;
  uint8_t retreat_allowed = LOW;

  switch (s) {
    case S_BOOT_SAFE:
      // Policy choice: keep power off until arming conditions met
      motor_en = LOW;
      power_en = LOW;
      retreat_allowed = LOW;
      break;

    case S_ARMED_IDLE:
      motor_en = HIGH;
      power_en = HIGH;
      retreat_allowed = LOW;
      break;

    case S_SAFE_HOLD:
      motor_en = LOW;   // conservative
      power_en = HIGH;
      retreat_allowed = LOW;
      break;

    case S_SAFE_STOP:
      motor_en = LOW;
      power_en = HIGH;  // stop motion but keep rail energized
      retreat_allowed = LOW;
      break;

    case S_RETREAT_ENABLED:
      motor_en = HIGH;
      power_en = HIGH;
      retreat_allowed = HIGH;
      break;

    case S_POWER_CUT:
      motor_en = LOW;
      power_en = LOW;
      retreat_allowed = LOW;
      break;
  }

  digitalWrite(PIN_MOTOR_EN, motor_en);
  digitalWrite(PIN_POWER_RELAY, power_en);
  digitalWrite(PIN_RETREAT_ALLOWED, retreat_allowed);
}

static void latch_fault(FaultCode code) {
  g_fault_latched = true;
  g_fault_code = code;
}

static void clear_fault() {
  g_fault_latched = false;
  g_fault_code = FAULT_OK;
}

// ---------------- TX builders ----------------
static void send_frame(uint8_t msg_type, uint16_t seq, uint32_t timestamp_ms, const uint8_t* payload, uint8_t len) {
  // Frame: SOF1 SOF2 version type seq(u16) ts(u32) len payload crc(u16)
  uint8_t header[2 + 1 + 1 + 2 + 4 + 1]; // SOF1 SOF2 + version + type + seq + ts + len
  size_t idx = 0;
  header[idx++] = SOF1;
  header[idx++] = SOF2;
  header[idx++] = PROTO_VERSION;
  header[idx++] = msg_type;
  header[idx++] = (uint8_t)(seq & 0xFF);
  header[idx++] = (uint8_t)((seq >> 8) & 0xFF);
  header[idx++] = (uint8_t)(timestamp_ms & 0xFF);
  header[idx++] = (uint8_t)((timestamp_ms >> 8) & 0xFF);
  header[idx++] = (uint8_t)((timestamp_ms >> 16) & 0xFF);
  header[idx++] = (uint8_t)((timestamp_ms >> 24) & 0xFF);
  header[idx++] = len;

  // CRC is over [version..payload]
  // So compute over header from version (index 2) to end of payload.
  // We'll assemble into a temp buffer for CRC.
  const size_t crc_buf_len = (1 + 1 + 2 + 4 + 1) + len; // version..len + payload
  uint8_t crcbuf[32]; // enough for our small payloads; expand if needed
  if (crc_buf_len > sizeof(crcbuf)) return;

  size_t c = 0;
  for (size_t i = 2; i < sizeof(header); i++) crcbuf[c++] = header[i];
  for (uint8_t i = 0; i < len; i++) crcbuf[c++] = payload[i];

  uint16_t crc = crc16_ccitt_false(crcbuf, crc_buf_len);

  Serial.write(header, sizeof(header));
  if (len > 0) Serial.write(payload, len);
  uint8_t crc_le[2] = {(uint8_t)(crc & 0xFF), (uint8_t)((crc >> 8) & 0xFF)};
  Serial.write(crc_le, 2);
}

static void send_ack(uint16_t ack_seq) {
  uint8_t payload[2] = {(uint8_t)(ack_seq & 0xFF), (uint8_t)((ack_seq >> 8) & 0xFF)};
  send_frame(MSG_ACK, ack_seq, millis(), payload, 2);
}

static void send_nack(uint16_t nack_seq, FaultCode reason) {
  uint8_t payload[3] = {
    (uint8_t)(nack_seq & 0xFF), (uint8_t)((nack_seq >> 8) & 0xFF),
    (uint8_t)reason
  };
  send_frame(MSG_NACK, nack_seq, millis(), payload, 3);
}

static void send_state(uint16_t seq) {
  uint8_t payload[12];
  payload[0] = (uint8_t)g_state;
  payload[1] = (uint8_t)g_fault_code;
  payload[2] = (uint8_t)(g_fault_latched ? 1 : 0);
  payload[3] = (uint8_t)(estop_asserted() ? 1 : 0);
  const bool watchdog_alive = (millis() - g_last_heartbeat_ms) <= WATCHDOG_TIMEOUT_MS;
  payload[4] = (uint8_t)(watchdog_alive ? 1 : 0);
  payload[5] = (uint8_t)(g_crc_fail_count & 0xFF);
  payload[6] = (uint8_t)((g_crc_fail_count >> 8) & 0xFF);
  payload[7] = (uint8_t)(g_seq_error_count & 0xFF);
  payload[8] = (uint8_t)((g_seq_error_count >> 8) & 0xFF);
  payload[9]  = (uint8_t)(g_watchdog_trip_count & 0xFF);
  payload[10] = (uint8_t)((g_watchdog_trip_count >> 8) & 0xFF);
  payload[11] = (uint8_t)g_good_heartbeat_streak;

  send_frame(MSG_STATE, seq, millis(), payload, sizeof(payload));
}

// ---------------- RX parser ----------------
// Small stateful parser (byte-by-byte)
enum RxParseState : uint8_t {
  RX_WAIT_SOF1,
  RX_WAIT_SOF2,
  RX_READ_FIXED,   // version,type,seq(2),ts(4),len(1)
  RX_READ_PAYLOAD,
  RX_READ_CRC1,
  RX_READ_CRC2
};

static RxParseState rx_state = RX_WAIT_SOF1;
static uint8_t rx_fixed[1 + 1 + 2 + 4 + 1];
static uint8_t rx_fixed_idx = 0;
static uint8_t rx_payload[16];
static uint8_t rx_payload_len = 0;
static uint8_t rx_payload_idx = 0;
static uint16_t rx_crc_recv = 0;

static void reset_rx() {
  rx_state = RX_WAIT_SOF1;
  rx_fixed_idx = 0;
  rx_payload_len = 0;
  rx_payload_idx = 0;
  rx_crc_recv = 0;
}

static void handle_frame(uint8_t version, uint8_t type, uint16_t seq, uint32_t ts, const uint8_t* payload, uint8_t len) {
  // Seq checks (optional but useful)
  if (g_have_seq) {
    uint16_t expected = (uint16_t)(g_last_seq_seen + 1);
    if (seq != expected) {
      g_seq_error_count++;
      // Do NOT latch; just telemetry + optional NACK.
      // We'll still process the message to be robust.
    }
  }
  g_last_seq_seen = seq;
  g_have_seq = true;

  if (version != PROTO_VERSION) {
    send_nack(seq, FAULT_CMD_UNKNOWN);
    return;
  }

  if (type == MSG_HEARTBEAT) {
    g_last_heartbeat_ms = millis();
    if (g_good_heartbeat_streak < 255) g_good_heartbeat_streak++;
    send_ack(seq);
    return;
  }

  if (type == MSG_CMD) {
    if (len < 1) { send_nack(seq, FAULT_CMD_UNKNOWN); return; }
    uint8_t cmd = payload[0];

    // Clear one-shot command flags; we’ll set the one received.
    g_cmd_stop = g_cmd_hold = g_cmd_enable_retreat = g_cmd_power_cut = g_cmd_reset_fault = false;

    switch (cmd) {
      case CMD_STOP: g_cmd_stop = true; break;
      case CMD_HOLD: g_cmd_hold = true; break;
      case CMD_ENABLE_RETREAT: g_cmd_enable_retreat = true; break;
      case CMD_POWER_CUT: g_cmd_power_cut = true; break;
      case CMD_RESET_FAULT: g_cmd_reset_fault = true; break;
      case CMD_SET_MODE:
        // Not enforced here; acknowledge for now
        break;
      default:
        send_nack(seq, FAULT_CMD_UNKNOWN);
        return;
    }

    send_ack(seq);
    return;
  }

  // Unknown type: NACK
  send_nack(seq, FAULT_CMD_UNKNOWN);
}

static void poll_serial() {
  while (Serial.available() > 0) {
    uint8_t b = (uint8_t)Serial.read();
    switch (rx_state) {
      case RX_WAIT_SOF1:
        if (b == SOF1) rx_state = RX_WAIT_SOF2;
        break;

      case RX_WAIT_SOF2:
        if (b == SOF2) {
          rx_state = RX_READ_FIXED;
          rx_fixed_idx = 0;
        } else {
          rx_state = RX_WAIT_SOF1;
        }
        break;

      case RX_READ_FIXED:
        rx_fixed[rx_fixed_idx++] = b;
        if (rx_fixed_idx >= sizeof(rx_fixed)) {
          // parse len now
          rx_payload_len = rx_fixed[sizeof(rx_fixed) - 1];
          if (rx_payload_len > sizeof(rx_payload)) {
            // Too big -> drop
            reset_rx();
          } else if (rx_payload_len == 0) {
            rx_state = RX_READ_CRC1;
          } else {
            rx_payload_idx = 0;
            rx_state = RX_READ_PAYLOAD;
          }
        }
        break;

      case RX_READ_PAYLOAD:
        rx_payload[rx_payload_idx++] = b;
        if (rx_payload_idx >= rx_payload_len) {
          rx_state = RX_READ_CRC1;
        }
        break;

      case RX_READ_CRC1:
        rx_crc_recv = b; // low byte
        rx_state = RX_READ_CRC2;
        break;

      case RX_READ_CRC2: {
        rx_crc_recv |= (uint16_t)b << 8;

        // compute crc over [fixed(version..len) + payload]
        uint8_t crcbuf[32];
        size_t c = 0;
        if (sizeof(rx_fixed) + rx_payload_len > sizeof(crcbuf)) { reset_rx(); break; }
        for (size_t i = 0; i < sizeof(rx_fixed); i++) crcbuf[c++] = rx_fixed[i];
        for (uint8_t i = 0; i < rx_payload_len; i++) crcbuf[c++] = rx_payload[i];
        uint16_t crc_calc = crc16_ccitt_false(crcbuf, c);

        if (crc_calc != rx_crc_recv) {
          g_crc_fail_count++;
          // Don't latch comms CRC by default; but do NACK using seq if we can parse it.
          uint16_t seq = (uint16_t)rx_fixed[2] | ((uint16_t)rx_fixed[3] << 8);
          send_nack(seq, FAULT_COMMS_CRC_FAIL);
          reset_rx();
          break;
        }

        // Extract fields
        uint8_t version = rx_fixed[0];
        uint8_t type    = rx_fixed[1];
        uint16_t seq = (uint16_t)rx_fixed[2] | ((uint16_t)rx_fixed[3] << 8);
        uint32_t ts =  (uint32_t)rx_fixed[4]
                     | ((uint32_t)rx_fixed[5] << 8)
                     | ((uint32_t)rx_fixed[6] << 16)
                     | ((uint32_t)rx_fixed[7] << 24);

        handle_frame(version, type, seq, ts, rx_payload, rx_payload_len);

        // Optionally send STATE periodically from host request; for now, just emit state on every valid msg
        send_state(seq);

        reset_rx();
        break;
      }
    }
  }
}

// ---------------- Safety logic ----------------
static void enforce_safety_fsm() {
  const bool estop = estop_asserted();
  const uint32_t now = millis();
  const bool watchdog_alive = (now - g_last_heartbeat_ms) <= WATCHDOG_TIMEOUT_MS;

  // Priority 1: E-stop asserted => POWER_CUT latched
  if (estop) {
    latch_fault(FAULT_ESTOP_ASSERTED);
    g_state = S_POWER_CUT;
    apply_outputs_for_state(g_state);
    return;
  }

  // Priority 2: Watchdog timeout => STOP (or POWER_CUT if you prefer)
  if (!watchdog_alive) {
    if (!g_fault_latched || g_fault_code != FAULT_WATCHDOG_TIMEOUT) {
      g_watchdog_trip_count++;
      latch_fault(FAULT_WATCHDOG_TIMEOUT);
    }
    g_state = S_SAFE_STOP; // change to S_POWER_CUT if your policy is power-cut on WD
    apply_outputs_for_state(g_state);
    return;
  }

  // If heartbeat is alive, we can consider clearing streak
  // (streak increments on HEARTBEAT frames; we can also slowly decay if desired)
  if (g_good_heartbeat_streak > 0 && (now - g_last_heartbeat_ms) > 150) {
    // light decay so it reflects recent health
    g_good_heartbeat_streak--;
  }

  // Priority 3: POWER_CUT command
  if (g_cmd_power_cut) {
    latch_fault(FAULT_OK); // not really a fault, but we want latched power cut behavior
    g_state = S_POWER_CUT;
    apply_outputs_for_state(g_state);
    return;
  }

  // Priority 4: STOP command
  if (g_cmd_stop) {
    // Recommend latching STOP (policy); keep it latched under fault_latched for simplicity
    // If you want STOP non-latched, remove latch_fault() and handle clearing separately.
    latch_fault(FAULT_OK);
    g_state = S_SAFE_STOP;
    apply_outputs_for_state(g_state);
    return;
  }

  // Priority 5: HOLD command
  if (g_cmd_hold) {
    // HOLD can be non-latched; do not set fault_latched
    if (!g_fault_latched) {
      g_state = S_SAFE_HOLD;
      apply_outputs_for_state(g_state);
    }
    return;
  }

  // Priority 6: ENABLE_RETREAT
  if (g_cmd_enable_retreat) {
    if (!g_fault_latched) {
      g_state = S_RETREAT_ENABLED;
      apply_outputs_for_state(g_state);
    }
    return;
  }

  // Priority 7: RESET_FAULT
  if (g_cmd_reset_fault) {
    // Only clear if safe conditions met
    // - estop released (already true here)
    // - watchdog alive (already true here)
    // - sufficient good heartbeat streak
    if (g_good_heartbeat_streak >= HEARTBEAT_STREAK_REQUIRED) {
      clear_fault();
      g_state = S_BOOT_SAFE;
      apply_outputs_for_state(g_state);
    }
    return;
  }

  // Default progression:
  // If no faults latched, once we have enough heartbeats, go ARMED_IDLE.
  if (!g_fault_latched) {
    if (g_state == S_BOOT_SAFE && g_good_heartbeat_streak >= HEARTBEAT_STREAK_REQUIRED) {
      g_state = S_ARMED_IDLE;
      apply_outputs_for_state(g_state);
    } else if (g_state == S_SAFE_HOLD || g_state == S_RETREAT_ENABLED) {
      // If no command persists, you might want to return to ARMED_IDLE
      // We'll do it only if heartbeat is healthy and no hold/retreat command is active.
      g_state = S_ARMED_IDLE;
      apply_outputs_for_state(g_state);
    } else if (g_state == S_SAFE_STOP) {
      // Remain stopped until explicit RESET_FAULT (policy)
      // If you want STOP to auto-clear, change this.
    }
  }
}

void setup() {
  pinMode(PIN_ESTOP_IN, INPUT_PULLUP);
  pinMode(PIN_MOTOR_EN, OUTPUT);
  pinMode(PIN_POWER_RELAY, OUTPUT);
  pinMode(PIN_RETREAT_ALLOWED, OUTPUT);

  Serial.begin(115200);

  g_state = S_BOOT_SAFE;
  apply_outputs_for_state(g_state);

  g_last_heartbeat_ms = 0;
  g_good_heartbeat_streak = 0;
  clear_fault();
  reset_rx();
}

void loop() {
  poll_serial();
  enforce_safety_fsm();
}
