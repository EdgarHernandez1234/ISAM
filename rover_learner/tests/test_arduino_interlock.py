#!/usr/bin/env python3
"""
tests/test_arduino_interlock.py

Unit tests for rover_learner.arduino_interlock without requiring real hardware.

Also includes an optional hardware smoke test gated by:
  RUN_ARDUINO_HW=1

Run:
  cd rover_learner
  python -m unittest -q tests.test_arduino_interlock
"""

import os
import sys
import time
import unittest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rover_learner.arduino_interlock import ArduinoInterlock, ArduinoStatus, parse_stat_line


class _FakeSerial:
    def __init__(self, lines=None):
        self._lines = list(lines or [])
        self.writes = []
        self.closed = False

    def readline(self):
        if not self._lines:
            time.sleep(0.01)
            return b""
        return self._lines.pop(0)

    def write(self, b):
        self.writes.append(b)
        return len(b)

    def flush(self):
        return None

    def close(self):
        self.closed = True


class TestParse(unittest.TestCase):
    def test_parse_valid(self):
        st = parse_stat_line("STAT,ms=123,safe=1,estop=0,wd=0,armed=1,accel_g=-1.00,knock=2,ping_cm=-1", rx_ts=10.0)
        self.assertIsNotNone(st)
        assert st is not None
        self.assertEqual(st.ms, 123)
        self.assertTrue(st.safe)
        self.assertFalse(st.estop)
        self.assertFalse(st.wd)
        self.assertTrue(st.armed)
        self.assertEqual(st.knock, 2)
        self.assertAlmostEqual(st.accel_g, -1.0, places=2)
        self.assertEqual(st.rx_ts, 10.0)
        self.assertTrue(st.is_interlock_safe)

    def test_parse_ignores_non_stat(self):
        self.assertIsNone(parse_stat_line("HELLO"))
        self.assertIsNone(parse_stat_line(b"XYZ\n"))


class TestInterlockClient(unittest.TestCase):
    def test_set_armed_writes(self):
        fake = _FakeSerial()
        client = ArduinoInterlock(serial_factory=lambda *a, **k: fake, autostart=True)
        client.set_armed(True)
        client.set_armed(False)
        # Two ARM commands written
        self.assertTrue(any(b"ARM 1" in w for w in fake.writes))
        self.assertTrue(any(b"ARM 0" in w for w in fake.writes))
        client.close()

    def test_alive_and_safe(self):
        # Provide one valid STAT line; client should become alive and safe
        line = b"STAT,ms=1,safe=1,estop=0,wd=0,armed=1,accel_g=-1,knock=0,ping_cm=-1\n"
        fake = _FakeSerial(lines=[line])
        client = ArduinoInterlock(serial_factory=lambda *a, **k: fake, stat_timeout_s=0.5, ping_interval_s=10.0, autostart=True)

        # Wait up to 0.5s for the reader thread to ingest the STAT line
        t0 = time.time()
        while time.time() - t0 < 0.5:
            if client.is_alive():
                break
            time.sleep(0.01)

        self.assertTrue(client.is_alive())
        self.assertTrue(client.is_safe(required=True))
        client.close()

    def test_stale_becomes_not_alive(self):
        line = b"STAT,ms=1,safe=1,estop=0,wd=0,armed=1\n"
        fake = _FakeSerial(lines=[line])
        client = ArduinoInterlock(serial_factory=lambda *a, **k: fake, stat_timeout_s=0.05, ping_interval_s=10.0, autostart=True)

        # Wait until we see at least one STAT, then verify it goes stale
        t0 = time.time()
        while time.time() - t0 < 0.5:
            if client.is_alive():
                break
            time.sleep(0.01)

        self.assertTrue(client.is_alive())
        time.sleep(0.08)  # > 0.05s timeout
        self.assertFalse(client.is_alive())
        self.assertFalse(client.is_safe(required=True))
        client.close()


class TestHardwareSmoke(unittest.TestCase):
    def test_hw_smoke_optional(self):
        if os.environ.get("RUN_ARDUINO_HW", "").strip() != "1":
            self.skipTest("Set RUN_ARDUINO_HW=1 to run on real hardware")

        client = ArduinoInterlock(port="/dev/ttyACM0", autostart=True)
        client.set_armed(True)
        # Wait up to 2 seconds for status
        t0 = time.time()
        while time.time() - t0 < 2.0:
            if client.is_alive():
                break
            time.sleep(0.05)

        self.assertTrue(client.is_alive(), "No STAT messages received from Arduino")
        st = client.get_status()
        self.assertIsNotNone(st)
        client.close()


if __name__ == "__main__":
    unittest.main()
