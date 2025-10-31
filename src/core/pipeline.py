import cv2
import numpy as np
import torch
from typing import Any, Dict, List

from loguru import logger

from .detectors.mtcnn_detector import MTCNNDetector
from .recognition.facenet_embedder import FaceNetEmbedder


class Preprocessor:
    @staticmethod
    def to_rgb(image: np.ndarray) -> np.ndarray:
        if image is None:
            return image
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def crop_and_resize(img: np.ndarray, box, size: int = 160) -> np.ndarray:
        x1, y1, x2, y2 = box
        face = img[y1:y2, x1:x2]
        return cv2.resize(face, (size, size), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def to_tensor(img: np.ndarray) -> torch.Tensor:
        # HWC RGB [0,255] -> CHW float [0,1]
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return t


class FacePipeline:
    def __init__(self, config):
        self.config = config
        self.detector = MTCNNDetector(config)
        self.embedder = FaceNetEmbedder(config)
        self.margin = config.get().detection.margin

    def process_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        img = Preprocessor.to_rgb(frame_bgr)
        h, w = img.shape[:2]
        detections = self.detector.detect(img)
        faces_info: List[Dict[str, Any]] = []
        face_tensors = []
        crop_boxes = []

        for det in detections:
            box = self.detector.expand_box(det["box"], self.margin, h, w)
            crop = Preprocessor.crop_and_resize(img, box, 160)
            crop_boxes.append(box)
            face_tensors.append(Preprocessor.to_tensor(crop))

        if face_tensors:
            batch = torch.stack(face_tensors, dim=0)
            embs = self.embedder.embed(batch)
            for i, det in enumerate(detections):
                faces_info.append(
                    {
                        "box": crop_boxes[i],
                        "confidence": det["confidence"],
                        "embedding": embs[i].tolist(),
                    }
                )

        return {"faces": faces_info, "count": len(faces_info)}
