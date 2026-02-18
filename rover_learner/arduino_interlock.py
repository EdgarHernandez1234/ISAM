"""
arduino_interlock.py

Arduino Uno R3 interlock client for ALAM:
- E-stop (D2, NC wiring recommended)
- Watchdog (Arduino expects periodic "PING" from Jetson)

Serial protocol (line-based):
  Arduino -> Jetson (10 Hz):
    STAT,ms=123456,safe=1,estop=0,wd=0,armed=1,accel_g=-1.00,knock=0,ping_cm=-1

  Jetson -> Arduino:
    PING\n
    ARM 1\n / ARM 0\n
    WDMS <ms>\n (optional)

Design goals:
- Fail-safe: if status is stale or missing, treat as unsafe when required.
- Testable without pyserial by injecting a serial_factory.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Dict, Optional, Union

try:
    import serial  # type: ignore
except Exception:  # pragma: no cover
    serial = None  # allows unit tests to run without pyserial


# -----------------------------
# Data model
# -----------------------------

@dataclass(frozen=True)
class ArduinoStatus:
    ms: int
    safe: bool
    estop: bool
    wd: bool
    armed: bool
    accel_g: float = -1.0
    knock: int = 0
    ping_cm: int = -1
    rx_ts: float = 0.0

    @property
    def is_interlock_safe(self) -> bool:
        # "safe" is the Arduino's computed safe bit; we also require armed and no flags.
        return bool(self.safe) and bool(self.armed) and (not self.estop) and (not self.wd)

    @property
    def reason(self) -> str:
        if self.estop:
            return "ESTOP"
        if self.wd:
            return "WATCHDOG"
        if not self.armed:
            return "NOT_ARMED"
        if not self.safe:
            return "UNSAFE"
        return "OK"


def parse_stat_line(line: Union[str, bytes], *, rx_ts: Optional[float] = None) -> Optional[ArduinoStatus]:
    """
    Parse a single STAT line.

    Returns:
      ArduinoStatus on success, None if line is not parseable.
    """
    if isinstance(line, bytes):
        try:
            line = line.decode("utf-8", errors="ignore")
        except Exception:
            return None

    s = str(line).strip()
    if not s.startswith("STAT"):
        return None

    # Example: STAT,ms=123,safe=1,estop=0,wd=0,armed=1,...
    parts = s.split(",")
    kv: Dict[str, str] = {}
    for p in parts[1:]:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        kv[k.strip()] = v.strip()

    def _int(key: str, default: int = 0) -> int:
        try:
            return int(float(kv.get(key, str(default))))
        except Exception:
            return default

    def _float(key: str, default: float = 0.0) -> float:
        try:
            return float(kv.get(key, str(default)))
        except Exception:
            return default

    ms = _int("ms", 0)
    safe = bool(_int("safe", 0))
    estop = bool(_int("estop", 0))
    wd = bool(_int("wd", 0))
    armed = bool(_int("armed", 0))
    accel_g = _float("accel_g", -1.0)
    knock = _int("knock", 0)
    ping_cm = _int("ping_cm", -1)
    rxt = time.time() if rx_ts is None else float(rx_ts)

    return ArduinoStatus(
        ms=ms,
        safe=safe,
        estop=estop,
        wd=wd,
        armed=armed,
        accel_g=accel_g,
        knock=knock,
        ping_cm=ping_cm,
        rx_ts=rxt,
    )


# -----------------------------
# Client
# -----------------------------

class ArduinoInterlock:
    """
    Maintains a background reader for Arduino STAT messages and a background PING sender.

    Typical usage:
        interlock = ArduinoInterlock("/dev/ttyACM0")
        interlock.set_armed(True)
        ...
        status = interlock.get_status()
        if interlock.is_safe(required=True):
            ...
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        *,
        baudrate: int = 115200,
        ping_interval_s: float = 0.2,   # 5 Hz
        stat_timeout_s: float = 1.0,    # consider stale if no STAT in 1s
        serial_factory: Optional[Callable[..., Any]] = None,
        autostart: bool = True,
    ):
        self.port = str(port)
        self.baudrate = int(baudrate)
        self.ping_interval_s = float(ping_interval_s)
        self.stat_timeout_s = float(stat_timeout_s)

        if serial_factory is None:
            if serial is None:
                raise ImportError("pyserial is required (pip install pyserial) or pass serial_factory for tests")
            serial_factory = serial.Serial  # type: ignore[attr-defined]

        self._serial_factory = serial_factory
        self._ser = None  # set in start()

        self._lock = threading.Lock()
        self._last_status: Optional[ArduinoStatus] = None
        self._last_rx_ts: float = 0.0

        self._stop = threading.Event()
        self._reader_t: Optional[threading.Thread] = None
        self._pinger_t: Optional[threading.Thread] = None

        if autostart:
            self.start()

    # --- lifecycle -------------------------------------------------

    def start(self) -> None:
        if self._ser is not None:
            return

        self._ser = self._serial_factory(
            self.port,
            self.baudrate,
            timeout=0.2,
            write_timeout=0.2,
        )

        self._stop.clear()

        self._reader_t = threading.Thread(target=self._reader_loop, name="ArduinoInterlockReader", daemon=True)
        self._pinger_t = threading.Thread(target=self._pinger_loop, name="ArduinoInterlockPinger", daemon=True)
        self._reader_t.start()
        self._pinger_t.start()

    def close(self) -> None:
        self._stop.set()

        # Join briefly; daemon threads won't block exit, but this cleans up for tests.
        for t in (self._reader_t, self._pinger_t):
            if t and t.is_alive():
                t.join(timeout=0.5)

        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    # --- commands --------------------------------------------------

    def send_ping(self) -> None:
        self._write_line("PING")

    def set_armed(self, armed: bool) -> None:
        self._write_line(f"ARM {1 if armed else 0}")

    def set_watchdog_ms(self, ms: int) -> None:
        self._write_line(f"WDMS {int(ms)}")

    # --- status ----------------------------------------------------

    def get_status(self) -> Optional[ArduinoStatus]:
        with self._lock:
            return self._last_status

    def rx_age_s(self, now: Optional[float] = None) -> float:
        n = time.time() if now is None else float(now)
        with self._lock:
            if self._last_rx_ts <= 0.0:
                return float("inf")
            return max(0.0, n - self._last_rx_ts)

    def is_alive(self, *, now: Optional[float] = None) -> bool:
        age = self.rx_age_s(now=now)
        return age <= self.stat_timeout_s

    def is_safe(self, *, required: bool = True) -> bool:
        """
        required=True => missing/stale status is unsafe
        required=False => missing/stale status returns False but caller may ignore
        """
        if not self.is_alive():
            return False if required else False
        st = self.get_status()
        if st is None:
            return False if required else False
        return bool(st.is_interlock_safe)

    # --- internals -------------------------------------------------

    def _write_line(self, line: str) -> None:
        if self._ser is None:
            return
        try:
            data = (str(line).strip() + "\n").encode("utf-8")
            self._ser.write(data)
            try:
                self._ser.flush()
            except Exception:
                pass
        except Exception:
            # swallow write errors; safety layer will detect stale status/unsafe and act
            pass

    def _reader_loop(self) -> None:
        assert self._ser is not None
        while not self._stop.is_set():
            try:
                raw = self._ser.readline()
            except Exception:
                time.sleep(0.05)
                continue

            if not raw:
                continue

            st = parse_stat_line(raw, rx_ts=time.time())
            if st is None:
                continue

            with self._lock:
                self._last_status = st
                self._last_rx_ts = st.rx_ts

    def _pinger_loop(self) -> None:
        # Wait a moment for Arduino bootloader reset when serial opens.
        time.sleep(0.25)
        while not self._stop.is_set():
            self.send_ping()
            time.sleep(max(0.05, self.ping_interval_s))
