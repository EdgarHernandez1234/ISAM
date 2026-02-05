"""rover_decider.logger

CSV logger for per-decision-frame records.

Records minimum columns (plus extras that help later):
- timestamp
- min_distance_m
- yolo_pred_conf
- yolo_looks_dirty (0/1)
- yolo_class_id (string)
- action (SCOOP/BYPASS)
- safety_state (NORMAL/SAFE_HOLD/STOP)
- label (0/1 or blank)

Also logs:
- lidar_valid (0/1)
- x1..x4 features

Stdlib only.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from .core import now_iso_utc, lidar_is_valid, extract_features


@dataclass
class DecisionFrame:
    timestamp: str
    min_distance_m: str
    yolo_pred_conf: float
    yolo_looks_dirty: int
    yolo_class_id: str
    action: str
    safety_state: str
    label: str
    lidar_valid: int
    x1: float
    x2: float
    x3: float
    x4: float

    @staticmethod
    def build(
        min_distance_m: Optional[float],
        pred_conf: float,
        looks_dirty: int,
        class_id: str,
        action: str,
        safety_state: str,
        label: Optional[int] = None,
    ) -> "DecisionFrame":
        lv = 1 if lidar_is_valid(min_distance_m) else 0
        feats = extract_features(min_distance_m, pred_conf, looks_dirty, include_lidar_valid=True)

        return DecisionFrame(
            timestamp=now_iso_utc(),
            min_distance_m="" if min_distance_m is None else f"{float(min_distance_m):.4f}",
            yolo_pred_conf=float(pred_conf),
            yolo_looks_dirty=int(looks_dirty),
            yolo_class_id=str(class_id),
            action=str(action),
            safety_state=str(safety_state),
            label="" if label is None else str(int(label)),
            lidar_valid=lv,
            x1=float(feats["x1"]),
            x2=float(feats["x2"]),
            x3=float(feats["x3"]),
            x4=float(feats["x4"]),
        )


class CSVDecisionLogger:
    """Append-only CSV logger with auto-header."""

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        file_exists = self.csv_path.exists()
        self._fp = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fp, fieldnames=self._fieldnames())

        if (not file_exists) or (self.csv_path.stat().st_size == 0):
            self._writer.writeheader()
            self._fp.flush()

    @staticmethod
    def _fieldnames() -> list[str]:
        return [
            "timestamp",
            "min_distance_m",
            "yolo_pred_conf",
            "yolo_looks_dirty",
            "yolo_class_id",
            "action",
            "safety_state",
            "label",
            "lidar_valid",
            "x1",
            "x2",
            "x3",
            "x4",
        ]

    def log(self, frame: DecisionFrame) -> None:
        self._writer.writerow(asdict(frame))
        self._fp.flush()

    def close(self) -> None:
        try:
            self._fp.flush()
        finally:
            self._fp.close()

    def __enter__(self) -> "CSVDecisionLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
