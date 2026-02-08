#!/usr/bin/env python3
"""logger.py (rover_learner)

CSV logging for live decision steps.
Schema is intentionally close to rover_decider's offline logs so you can compare runs.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class DecisionFrame:
    ts: float
    frame_id: int

    pred_class: str
    pred_conf: float
    distance_m: Optional[float]

    proposed_action: str
    final_action: str
    reason: str

    # safety meta
    safety_json: str = ""

    # optional labeling (for later datasets)
    human_label: str = "u"


class CSVDecisionLogger:
    def __init__(self, out_csv: Path):
        self.out_csv = Path(out_csv)
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.out_csv.open("w", newline="", encoding="utf-8")
        self._writer = None

    def log(self, frame: DecisionFrame) -> None:
        row = asdict(frame)
        if self._writer is None:
            self._writer = csv.DictWriter(self._fp, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._fp.flush()

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass

    @staticmethod
    def safety_to_json(safety_obj: Dict[str, Any]) -> str:
        try:
            return json.dumps(safety_obj, ensure_ascii=False, sort_keys=True)
        except Exception:
            return ""
