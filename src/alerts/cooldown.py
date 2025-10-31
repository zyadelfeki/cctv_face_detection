from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import time


@dataclass
class CooldownKey:
    criminal_id: int
    camera_id: int


class CooldownManager:
    def __init__(self, cooldown_seconds: int = 300):
        self.cooldown_seconds = cooldown_seconds
        self._store: Dict[Tuple[int, int], float] = {}

    def allow(self, criminal_id: int, camera_id: int) -> bool:
        key = (criminal_id, camera_id)
        now = time.time()
        last = self._store.get(key, 0)
        if now - last >= self.cooldown_seconds:
            self._store[key] = now
            return True
        return False
