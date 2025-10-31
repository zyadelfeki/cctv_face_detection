from typing import List

import torch
from facenet_pytorch import InceptionResnetV1
import numpy as np


class FaceNetEmbedder:
    """FaceNet embedder producing 512-d or 128-d embeddings depending on model."""

    def __init__(self, config):
        rcfg = config.get().recognition
        device = rcfg.device if torch.cuda.is_available() and rcfg.device == "cuda" else "cpu"
        self.device = torch.device(device)
        # InceptionResnetV1 pretrained on VGGFace2 yields 512-d embeddings
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.embedding_size = 512

    @torch.inference_mode()
    def embed(self, face_tensors: torch.Tensor) -> np.ndarray:
        """face_tensors: N x 3 x 160 x 160 in range [0,1]. Returns N x D numpy embeddings."""
        face_tensors = face_tensors.to(self.device)
        embs = self.model(face_tensors)
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        return embs.detach().cpu().numpy()
