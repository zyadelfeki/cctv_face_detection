from typing import Any, Dict, List

import numpy as np

from loguru import logger

from ..database.embedding_index import EmbeddingIndex


class Matcher:
    def __init__(self, config):
        self.config = config
        self.threshold = config.get().recognition.similarity_threshold
        self.index = EmbeddingIndex(dim=512, metric="cosine", persist_path="./data/faiss.index")
        # Optionally load persisted index
        try:
            self.index.load()
        except Exception:
            logger.warning("No existing FAISS index found; starting fresh")

    def match(self, embeddings: np.ndarray) -> List[List[Dict[str, Any]]]:
        if embeddings.size == 0:
            return []
        results = self.index.search(embeddings, k=5)
        matched: List[List[Dict[str, Any]]] = []
        for row in results:
            row_matches: List[Dict[str, Any]] = []
            for emb_id, score in row:
                if score >= self.threshold and emb_id != -1:
                    row_matches.append({"embedding_id": emb_id, "similarity": score})
            matched.append(row_matches)
        return matched
