from pathlib import Path
from typing import Optional

import faiss
import numpy as np


class EmbeddingIndex:
    """FAISS index wrapper for cosine/L2 similarity with persistence."""

    def __init__(self, dim: int = 512, metric: str = "cosine", persist_path: Optional[str] = None):
        self.dim = dim
        self.metric = metric
        self.persist_path = persist_path or "./data/faiss.index"
        if metric == "cosine":
            # Normalize then IndexFlatIP approximates cosine
            self.index = faiss.IndexFlatIP(dim)
            self.normalize = True
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.normalize = False
        self.ids: list[int] = []

    def add(self, vectors: np.ndarray, ids: list[int]):
        assert vectors.shape[1] == self.dim
        X = vectors.astype(np.float32)
        if self.normalize:
            faiss.normalize_L2(X)
        self.index.add(X)
        self.ids.extend(ids)

    def search(self, vectors: np.ndarray, k: int = 5):
        X = vectors.astype(np.float32)
        if self.normalize:
            faiss.normalize_L2(X)
        D, I = self.index.search(X, k)
        # map local indices to global embedding row ids
        results = []
        for i in range(I.shape[0]):
            idxs = I[i]
            sims = D[i]
            results.append([(self.ids[j] if j >= 0 else -1, float(sims[t])) for t, j in enumerate(idxs)])
        return results

    def save(self):
        Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, self.persist_path)
        # ids persistence
        with open(self.persist_path + ".ids", "w", encoding="utf-8") as f:
            f.write(",".join(map(str, self.ids)))

    def load(self):
        p = Path(self.persist_path)
        if p.exists():
            self.index = faiss.read_index(self.persist_path)
            ids_file = p.with_suffix(p.suffix + ".ids")
            if ids_file.exists():
                with open(ids_file, "r", encoding="utf-8") as f:
                    txt = f.read().strip()
                    self.ids = list(map(int, txt.split(","))) if txt else []
