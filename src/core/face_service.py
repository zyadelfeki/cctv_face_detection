import numpy as np
from typing import Any, Dict

from .pipeline import FacePipeline
from .matching import Matcher


class FaceService:
    def __init__(self, config):
        self.pipeline = FacePipeline(config)
        self.matcher = Matcher(config)

    def detect_and_match(self, frame_bgr) -> Dict[str, Any]:
        result = self.pipeline.process_frame(frame_bgr)
        if result["count"]:
            embs = np.array([f["embedding"] for f in result["faces"]], dtype=np.float32)
            matches = self.matcher.match(embs)
        else:
            matches = []
        result["matches"] = matches
        return result
