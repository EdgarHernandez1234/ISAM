# rover_learner/conference_demo/replay_camera_provider.py
from __future__ import annotations
import time
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class ReplayFramePacket:
    frame: np.ndarray
    timestamp: float
    cam_ok: bool = True

class ReplayCameraProvider:
    """
    Minimal provider compatible with your demo loop:
      - read() -> ReplayFramePacket or None
    """
    def __init__(self, video_path: str, loop: bool = True, realtime: bool = True, target_fps: Optional[float] = None):
        self.video_path = video_path
        self.loop = loop
        self.realtime = realtime
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open replay video: {video_path}")
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.fps = float(target_fps) if target_fps else float(self.src_fps)
        self._t0 = time.time()
        self._frame_i = 0
        self._last_emit = time.time()

    def read(self) -> Optional[ReplayFramePacket]:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            if not self.loop:
                return None
            # loop
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok or frame is None:
                return None

        # optional realtime pacing
        if self.realtime and self.fps > 1:
            self._frame_i += 1
            target_t = self._t0 + (self._frame_i / self.fps)
            now = time.time()
            sleep_s = target_t - now
            if sleep_s > 0:
                time.sleep(min(sleep_s, 0.050))

        return ReplayFramePacket(frame=frame, timestamp=time.time(), cam_ok=True)

    def close(self):
        try:
            self.cap.release()
        except Exception:
            pass
