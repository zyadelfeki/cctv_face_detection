from typing import Optional

import numpy as np
import cv2
from loguru import logger


class VideoStream:
    def __init__(self, url: str, *, buffer_size: int = 1, reconnect_interval: int = 5, timeout: int = 10):
        self.url = url
        self.buffer_size = buffer_size
        self.reconnect_interval = reconnect_interval
        self.timeout = timeout
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self):
        self.cap = cv2.VideoCapture(self.url)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video stream: {self.url}")
        # Reduce internal buffering for low-latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

    def read(self) -> Optional[np.ndarray]:
        if self.cap is None:
            self.open()
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
