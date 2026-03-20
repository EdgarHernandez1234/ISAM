from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import serial


# =========================
# Protocol constants
# =========================

FAULT_WATCHDOG = 1 << 0
FAULT_ESTOP = 1 << 1
FAULT_BAD_PACKET = 1 << 2
FAULT_ENCODER = 1 << 3
FAULT_LOW_BATTERY = 1 << 4
FAULT_DRIVER = 1 << 5
FAULT_OVERCURRENT = 1 << 6

ERR_BAD_ENVELOPE = 1
ERR_BAD_CHECKSUM = 2
ERR_UNKNOWN_COMMAND = 3
ERR_FIELD_COUNT = 4
ERR_FIELD_PARSE = 5
ERR_OUT_OF_RANGE = 6
ERR_BAD_STATE = 7
ERR_ESTOP_ACTIVE = 8

MAX_TARGET_MMPS = 1000


class SerialProtocolError(Exception):
    pass


class SerialPacketTimeout(Exception):
    pass


class CommandRejected(Exception):
    pass


@dataclass
class AckPacket:
    seq: int


@dataclass
class ErrPacket:
    seq: int
    code: int


@dataclass
class FltPacket:
    seq: int
    faults: int


@dataclass
class PonPacket:
    proto_version: int
    firmware_id: str


@dataclass
class StaPacket:
    seq: int
    left_cmd: int
    right_cmd: int
    left_meas: int
    right_meas: int
    enc_l: int
    enc_r: int
    batt_mv: int
    faults: int


class SerialBaseDriver:
    """
    Jetson-side serial driver for ALAM chassis protocol v1.
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baudrate: int = 115200,
        timeout: float = 0.1,
        write_timeout: float = 0.1,
        auto_open: bool = True,
        boot_wait_sec: float = 2.0,
        debug: bool = False,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.write_timeout = write_timeout
        self.boot_wait_sec = boot_wait_sec
        self.debug = debug

        self._ser: Optional[serial.Serial] = None
        self._seq: int = 0
        self.latest_status: Optional[StaPacket] = None
        self.last_fault_packet: Optional[FltPacket] = None
        self.last_poweron: Optional[PonPacket] = None

        if auto_open:
            self.open()

    # =========================
    # Lifecycle
    # =========================

    def open(self) -> None:
        if self._ser is not None and self._ser.is_open:
            return

        self._ser = serial.Serial(
            self.port,
            self.baudrate,
            timeout=self.timeout,
            write_timeout=self.write_timeout,
        )

        # Opening the Uno serial port often resets the board.
        # Wait for reboot, then clear partial boot noise.
        time.sleep(self.boot_wait_sec)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

        # Small extra settle time so the next reads are more likely to be whole lines.
        time.sleep(0.15)
        self.drain_input()

    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None

    def __enter__(self) -> "SerialBaseDriver":
        if self._ser is None or not self._ser.is_open:
            self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    # =========================
    # Sequence
    # =========================

    def next_seq(self) -> int:
        self._seq = (self._seq + 1) % 65536
        return self._seq

    # =========================
    # Packet helpers
    # =========================

    @staticmethod
    def checksum(payload: str) -> str:
        value = 0
        for ch in payload:
            value ^= ord(ch)
        return f"{value:02X}"

    @classmethod
    def build_packet(cls, payload: str) -> str:
        return f"@{payload}*{cls.checksum(payload)}\n"

    @classmethod
    def parse_packet_line(cls, line: str) -> Dict[str, Any]:
        line = line.strip()
        if not line:
            raise SerialProtocolError("Empty line")
        if not line.startswith("@"):
            raise SerialProtocolError(f"Missing @ prefix: {line}")
        if "*" not in line:
            raise SerialProtocolError(f"Missing checksum separator: {line}")

        payload_part, cs_part = line[1:].split("*", 1)
        if len(cs_part) != 2:
            raise SerialProtocolError(f"Bad checksum field length: {line}")

        want = cls.checksum(payload_part)
        if cs_part.upper() != want:
            raise SerialProtocolError(
                f"Checksum mismatch for line '{line}': got {cs_part}, want {want}"
            )

        fields = payload_part.split(",")
        if not fields or not fields[0]:
            raise SerialProtocolError(f"Empty payload: {line}")

        return {
            "raw": line,
            "payload": payload_part,
            "kind": fields[0],
            "fields": fields,
        }

    def _write_payload(self, payload: str) -> None:
        if self._ser is None:
            raise RuntimeError("Serial port not open")

        packet = self.build_packet(payload)
        if self.debug:
            print(f"TX: {packet.strip()}")
        self._ser.write(packet.encode("utf-8"))
        self._ser.flush()

    def _readline(self) -> Optional[str]:
        if self._ser is None:
            raise RuntimeError("Serial port not open")

        raw = self._ser.readline()
        if not raw:
            return None

        line = raw.decode("utf-8", errors="replace").strip()
        if self.debug and line:
            print(f"RX: {line}")
        return line if line else None

    def drain_input(self) -> None:
        if self._ser is None:
            raise RuntimeError("Serial port not open")

        end_time = time.time() + 0.4
        while time.time() < end_time:
            line = self._readline()
            if not line:
                break
            try:
                self._handle_packet_line(line, allow_unknown=True)
            except SerialProtocolError as exc:
                if self.debug:
                    print(f"IGNORING STARTUP NOISE: {exc}")
                continue

    # =========================
    # Packet decoding
    # =========================

    def _handle_packet_line(self, line: str, allow_unknown: bool = False) -> Optional[object]:
        pkt = self.parse_packet_line(line)
        kind = pkt["kind"]
        fields = pkt["fields"]

        if kind == "PON":
            if len(fields) != 3:
                raise SerialProtocolError(f"Bad PON field count: {line}")
            obj = PonPacket(proto_version=int(fields[1]), firmware_id=fields[2])
            self.last_poweron = obj
            return obj

        if kind == "ACK":
            if len(fields) != 2:
                raise SerialProtocolError(f"Bad ACK field count: {line}")
            return AckPacket(seq=int(fields[1]))

        if kind == "ERR":
            if len(fields) != 3:
                raise SerialProtocolError(f"Bad ERR field count: {line}")
            return ErrPacket(seq=int(fields[1]), code=int(fields[2]))

        if kind == "FLT":
            if len(fields) != 3:
                raise SerialProtocolError(f"Bad FLT field count: {line}")
            obj = FltPacket(seq=int(fields[1]), faults=int(fields[2]))
            self.last_fault_packet = obj
            return obj

        if kind == "STA":
            if len(fields) != 10:
                raise SerialProtocolError(f"Bad STA field count: {line}")
            obj = StaPacket(
                seq=int(fields[1]),
                left_cmd=int(fields[2]),
                right_cmd=int(fields[3]),
                left_meas=int(fields[4]),
                right_meas=int(fields[5]),
                enc_l=int(fields[6]),
                enc_r=int(fields[7]),
                batt_mv=int(fields[8]),
                faults=int(fields[9]),
            )
            self.latest_status = obj
            return obj

        if allow_unknown:
            return None

        raise SerialProtocolError(f"Unknown packet kind: {kind}")

    # =========================
    # Wait helpers
    # =========================

    def wait_for_response(
        self,
        expected_seq: int,
        timeout_sec: float = 0.75,
    ) -> object:
        """
        Wait for ACK or ERR for a specific command sequence.
        Status and fault packets are consumed and cached while waiting.
        """
        deadline = time.time() + timeout_sec

        while time.time() < deadline:
            line = self._readline()
            if not line:
                continue

            try:
                obj = self._handle_packet_line(line)
            except SerialProtocolError as exc:
                if self.debug:
                    print(f"Protocol warning: {exc}")
                continue

            if isinstance(obj, AckPacket) and obj.seq == expected_seq:
                return obj

            if isinstance(obj, ErrPacket) and obj.seq == expected_seq:
                return obj

        raise SerialPacketTimeout(
            f"Timed out waiting for ACK/ERR for seq {expected_seq}"
        )

    # =========================
    # Command API
    # =========================

    def _send_simple(self, cmd: str, timeout_sec: float = 0.75) -> AckPacket:
        seq = self.next_seq()
        self._write_payload(f"{cmd},{seq}")
        resp = self.wait_for_response(expected_seq=seq, timeout_sec=timeout_sec)

        if isinstance(resp, ErrPacket):
            raise CommandRejected(f"{cmd} rejected with error code {resp.code}")
        if not isinstance(resp, AckPacket):
            raise SerialProtocolError(f"Unexpected response type for {cmd}: {resp}")
        return resp

    def ping(self) -> AckPacket:
        return self._send_simple("PNG")

    def enable(self) -> AckPacket:
        return self._send_simple("ENA")

    def disable(self) -> AckPacket:
        return self._send_simple("DIS")

    def stop(self) -> AckPacket:
        return self._send_simple("STP")

    def estop(self) -> AckPacket:
        return self._send_simple("EST")

    def clear(self) -> AckPacket:
        return self._send_simple("CLR")

    def drive(
        self,
        left_mmps: int,
        right_mmps: int,
        flags: int = 0,
        timeout_sec: float = 0.75,
    ) -> AckPacket:
        if not (-MAX_TARGET_MMPS <= left_mmps <= MAX_TARGET_MMPS):
            raise ValueError(f"left_mmps out of range: {left_mmps}")
        if not (-MAX_TARGET_MMPS <= right_mmps <= MAX_TARGET_MMPS):
            raise ValueError(f"right_mmps out of range: {right_mmps}")
        if flags != 0:
            raise ValueError("flags must be 0 in protocol v1")

        seq = self.next_seq()
        payload = f"DRV,{seq},{left_mmps},{right_mmps},{flags}"
        self._write_payload(payload)
        resp = self.wait_for_response(expected_seq=seq, timeout_sec=timeout_sec)

        if isinstance(resp, ErrPacket):
            raise CommandRejected(f"DRV rejected with error code {resp.code}")
        if not isinstance(resp, AckPacket):
            raise SerialProtocolError(f"Unexpected response type for DRV: {resp}")
        return resp

    # =========================
    # Status polling
    # =========================

    def poll_once(self) -> Optional[object]:
        line = self._readline()
        if not line:
            return None

        try:
            return self._handle_packet_line(line, allow_unknown=True)
        except SerialProtocolError as exc:
            if self.debug:
                print(f"IGNORING MALFORMED LINE: {exc}")
            return None

    def poll_for_status(self, timeout_sec: float = 0.5) -> Optional[StaPacket]:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            obj = self.poll_once()
            if isinstance(obj, StaPacket):
                return obj
        return None

    def get_latest_status(self) -> Optional[StaPacket]:
        return self.latest_status

    # =========================
    # Higher-level helper
    # =========================

    def drive_for(
        self,
        left_mmps: int,
        right_mmps: int,
        duration_sec: float,
        resend_hz: float = 10.0,
    ) -> None:
        """
        Hold a drive command long enough to avoid watchdog timeout.
        """
        if resend_hz <= 0:
            raise ValueError("resend_hz must be > 0")

        period = 1.0 / resend_hz
        deadline = time.time() + duration_sec

        while time.time() < deadline:
            self.drive(left_mmps, right_mmps)
            time.sleep(period)

    def describe_faults(self, fault_mask: int) -> list[str]:
        names = []
        if fault_mask & FAULT_WATCHDOG:
            names.append("watchdog")
        if fault_mask & FAULT_ESTOP:
            names.append("estop")
        if fault_mask & FAULT_BAD_PACKET:
            names.append("bad_packet")
        if fault_mask & FAULT_ENCODER:
            names.append("encoder")
        if fault_mask & FAULT_LOW_BATTERY:
            names.append("low_battery")
        if fault_mask & FAULT_DRIVER:
            names.append("driver")
        if fault_mask & FAULT_OVERCURRENT:
            names.append("overcurrent")
        return names
