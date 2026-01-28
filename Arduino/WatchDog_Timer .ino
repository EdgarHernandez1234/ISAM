const int RELAY_PIN = 7; // Pin connected to the Power Relay
const unsigned long TIMEOUT = 500; // MS to wait before cutting power
unsigned long lastHeartbeat = 0;

void setup() {
  Serial.begin(9600);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, HIGH); // Start with power ON
}

void loop() {
  if (Serial.available() > 0) {
    char c = Serial.read();
    if (c == 'H') {
      lastHeartbeat = millis(); // Reset the timer
      digitalWrite(RELAY_PIN, HIGH); // Ensure power is ON
    }
  }

  // Check for timeout
  if (millis() - lastHeartbeat > TIMEOUT) {
    digitalWrite(RELAY_PIN, LOW); // CUT POWER - Safety Trip
  }
}