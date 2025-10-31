from typing import List, Tuple

import numpy as np
from mtcnn import MTCNN


class MTCNNDetector:
    """MTCNN-based face detector returning bboxes, keypoints, and confidences."""

    def __init__(self, config):
        dcfg = config.get().detection
        self.detector = MTCNN(
            min_face_size=dcfg.min_face_size,
            scale_factor=dcfg.scale_factor,
            steps_threshold=list(dcfg.steps_threshold),
        )
        self.keep_all = dcfg.keep_all
        self.margin = dcfg.margin
        self.confidence_threshold = dcfg.confidence_threshold

    def detect(self, image: np.ndarray) -> List[dict]:
        """Return list of detections: {box, keypoints, confidence}."""
        results = self.detector.detect_faces(image)
        detections = []
        for r in results:
            if r.get("confidence", 0) >= self.confidence_threshold:
                detections.append(
                    {
                        "box": r["box"],
                        "keypoints": r.get("keypoints", {}),
                        "confidence": float(r["confidence"]),
                    }
                )
        return detections

    @staticmethod
    def expand_box(box: List[int], margin: int, img_h: int, img_w: int) -> Tuple[int, int, int, int]:
        x, y, w, h = box
        x1 = max(0, x - margin // 2)
        y1 = max(0, y - margin // 2)
        x2 = min(img_w, x + w + margin // 2)
        y2 = min(img_h, y + h + margin // 2)
        return x1, y1, x2, y2
