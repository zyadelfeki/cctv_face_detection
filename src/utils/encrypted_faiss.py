"""Encrypted FAISS Index Wrapper.

Provides:
- Transparent encryption for FAISS indexes
- Encrypted vector storage
- Encrypted ID mappings
- Drop-in replacement for plain FAISS
- Automatic encryption/decryption
"""

import pickle
import struct
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import numpy as np

from loguru import logger

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available")

from src.utils.encryption import BiometricEncryption, EncryptedData


class EncryptedFAISSIndex:
    """Encrypted wrapper for FAISS index.
    
    Provides transparent encryption:
    - Vectors encrypted at rest
    - ID mappings encrypted
    - Index operations work on plaintext (in-memory)
    - Encryption happens on save/load
    
    Compatible with regular FAISS operations.
    """
    
    def __init__(
        self,
        dimension: int,
        encryption: Optional[BiometricEncryption] = None,
        index_type: str = "IVF_PQ",
        nlist: int = 100,
        m: int = 8
    ):
        """
        Initialize encrypted FAISS index.
        
        Args:
            dimension: Vector dimension
            encryption: BiometricEncryption instance (None = no encryption)
            index_type: FAISS index type
            nlist: Number of clusters (for IVF indexes)
            m: Number of sub-quantizers (for PQ indexes)
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not installed")
        
        self.dimension = dimension
        self.encryption = encryption
        self.index_type = index_type
        self.nlist = nlist
        self.m = m
        
        # Initialize FAISS index
        self.index = None
        self.id_map: Dict[int, str] = {}  # FAISS id -> record_id
        self.is_trained = False
        
        self._init_index()
    
    def _init_index(self):
        """Initialize FAISS index structure."""
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IVF_Flat":
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
        elif self.index_type == "IVF_PQ":
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFPQ(
                quantizer, self.dimension, self.nlist, self.m, 8
            )
        elif self.index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        logger.debug(f"Initialized {self.index_type} index (dim={self.dimension})")
    
    def train(self, embeddings: np.ndarray):
        """
        Train index (required for IVF indexes).
        
        Args:
            embeddings: Training vectors (N, D)
        """
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            logger.info(f"Training {self.index_type} index with {len(embeddings)} vectors")
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            
            if len(embeddings) < self.nlist:
                logger.warning(
                    f"Not enough samples ({len(embeddings)}) for {self.nlist} clusters. "
                    "Switching to Flat index."
                )
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                self.index.train(embeddings)
            
            self.is_trained = True
            logger.info("Index training complete")
    
    def add(self, embeddings: np.ndarray, record_ids: List[str]):
        """
        Add vectors to index.
        
        Args:
            embeddings: Vectors to add (N, D)
            record_ids: Corresponding record IDs
        """
        if len(embeddings) != len(record_ids):
            raise ValueError("embeddings and record_ids must have same length")
        
        # Ensure trained
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            self.train(embeddings)
        
        # Normalize
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        # Add to index
        start_id = len(self.id_map)
        self.index.add(embeddings)
        
        # Update ID mapping
        for i, rid in enumerate(record_ids):
            self.id_map[start_id + i] = rid
        
        logger.debug(f"Added {len(embeddings)} vectors to index")
    
    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.
        
        Args:
            query: Query vector (D,) or (N, D)
            k: Number of results
        
        Returns:
            List of (record_id, similarity_score)
        """
        if self.index is None or len(self.id_map) == 0:
            return []
        
        # Ensure proper shape
        if query.ndim == 1:
            query = query.reshape(1, -1)
        query = query.astype(np.float32)
        faiss.normalize_L2(query)
        
        # Search
        scores, indices = self.index.search(query, k)
        
        # Map back to record IDs
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx in self.id_map:
                results.append((self.id_map[idx], float(score)))
        
        return results
    
    def save(self, base_path: str):
        """
        Save index to disk (with encryption if enabled).
        
        Args:
            base_path: Base path (without extension)
        """
        base = Path(base_path)
        
        if self.encryption:
            # Encrypted save
            self._save_encrypted(base)
        else:
            # Plain save
            self._save_plain(base)
    
    def _save_plain(self, base: Path):
        """Save without encryption."""
        # Save FAISS index
        faiss.write_index(self.index, str(base) + ".faiss")
        
        # Save ID map
        with open(str(base) + ".map", 'wb') as f:
            pickle.dump(self.id_map, f)
        
        logger.info(f"Saved plain FAISS index to {base}")
    
    def _save_encrypted(self, base: Path):
        """Save with encryption."""
        # Save FAISS index to bytes
        temp_index_path = str(base) + ".faiss.tmp"
        faiss.write_index(self.index, temp_index_path)
        
        with open(temp_index_path, 'rb') as f:
            index_bytes = f.read()
        
        # Encrypt index
        encrypted_index = self.encryption.encrypt_bytes(index_bytes)
        
        # Save encrypted index
        with open(str(base) + ".faiss.enc", 'wb') as f:
            f.write(encrypted_index.to_bytes())
        
        # Clean up temp file
        Path(temp_index_path).unlink()
        
        # Encrypt and save ID map
        id_map_bytes = pickle.dumps(self.id_map)
        encrypted_map = self.encryption.encrypt_bytes(id_map_bytes)
        
        with open(str(base) + ".map.enc", 'wb') as f:
            f.write(encrypted_map.to_bytes())
        
        logger.info(f"Saved encrypted FAISS index to {base}")
    
    def load(self, base_path: str):
        """
        Load index from disk (with decryption if needed).
        
        Args:
            base_path: Base path (without extension)
        """
        base = Path(base_path)
        
        # Check if encrypted or plain
        if (base.parent / (base.name + ".faiss.enc")).exists():
            if not self.encryption:
                raise RuntimeError(
                    "Index is encrypted but no encryption key provided"
                )
            self._load_encrypted(base)
        elif (base.parent / (base.name + ".faiss")).exists():
            self._load_plain(base)
        else:
            raise FileNotFoundError(f"Index not found at {base}")
    
    def _load_plain(self, base: Path):
        """Load without decryption."""
        # Load FAISS index
        self.index = faiss.read_index(str(base) + ".faiss")
        
        # Load ID map
        with open(str(base) + ".map", 'rb') as f:
            self.id_map = pickle.load(f)
        
        self.is_trained = True
        logger.info(f"Loaded plain FAISS index from {base}")
    
    def _load_encrypted(self, base: Path):
        """Load with decryption."""
        # Load and decrypt FAISS index
        with open(str(base) + ".faiss.enc", 'rb') as f:
            encrypted_bytes = f.read()
        
        encrypted_index = EncryptedData.from_bytes(encrypted_bytes)
        index_bytes = self.encryption.decrypt_bytes(encrypted_index)
        
        # Write to temp file for FAISS to read
        temp_path = str(base) + ".faiss.tmp"
        with open(temp_path, 'wb') as f:
            f.write(index_bytes)
        
        self.index = faiss.read_index(temp_path)
        Path(temp_path).unlink()
        
        # Load and decrypt ID map
        with open(str(base) + ".map.enc", 'rb') as f:
            encrypted_bytes = f.read()
        
        encrypted_map = EncryptedData.from_bytes(encrypted_bytes)
        id_map_bytes = self.encryption.decrypt_bytes(encrypted_map)
        self.id_map = pickle.loads(id_map_bytes)
        
        self.is_trained = True
        logger.info(f"Loaded encrypted FAISS index from {base}")
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        return {
            "total_vectors": len(self.id_map),
            "dimension": self.dimension,
            "index_type": self.index_type,
            "is_trained": self.is_trained,
            "encrypted": self.encryption is not None
        }


class EncryptedFaceDatabase:
    """Drop-in replacement for FaceDatabase with encryption.
    
    Automatically encrypts:
    - Face embeddings
    - FAISS vector index
    - ID mappings
    
    Usage:
        encryption = BiometricEncryption(master_key)
        db = EncryptedFaceDatabase(encryption=encryption)
        db.add_record(record)  # Automatically encrypted
    """
    
    def __init__(
        self,
        storage_path: str = "data/encrypted_face_db",
        embedding_dim: int = 512,
        encryption: Optional[BiometricEncryption] = None
    ):
        """
        Initialize encrypted face database.
        
        Args:
            storage_path: Storage directory
            embedding_dim: Embedding dimension
            encryption: BiometricEncryption instance (None = generate key)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize encryption
        if encryption is None:
            logger.warning("No encryption key provided. Generating new key.")
            logger.warning("SAVE THIS KEY SECURELY - data will be unrecoverable without it!")
            encryption = BiometricEncryption()
        
        self.encryption = encryption
        
        # Initialize encrypted FAISS index
        self.index = EncryptedFAISSIndex(
            dimension=embedding_dim,
            encryption=encryption
        )
        
        # Encrypted storage for embeddings
        self.embeddings: Dict[str, EncryptedData] = {}
        
        logger.info(f"Initialized encrypted face database at {storage_path}")
        logger.info(f"Key fingerprint: {encryption.get_key_fingerprint()}")
    
    def add_embedding(
        self,
        record_id: str,
        embedding: np.ndarray
    ):
        """
        Add encrypted embedding.
        
        Args:
            record_id: Record identifier
            embedding: Face embedding vector
        """
        # Encrypt embedding
        encrypted = self.encryption.encrypt_embedding(embedding, record_id)
        self.embeddings[record_id] = encrypted
        
        # Add to FAISS index (plaintext, in-memory)
        self.index.add(
            embedding.reshape(1, -1).astype(np.float32),
            [record_id]
        )
        
        logger.debug(f"Added encrypted embedding for {record_id}")
    
    def search_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        threshold: float = 0.6
    ) -> List[Tuple[str, float]]:
        """
        Search for similar faces.
        
        Args:
            query_embedding: Query vector
            k: Number of results
            threshold: Minimum similarity score
        
        Returns:
            List of (record_id, similarity_score)
        """
        results = self.index.search(query_embedding, k)
        
        # Filter by threshold
        return [(rid, score) for rid, score in results if score >= threshold]
    
    def get_embedding(
        self,
        record_id: str
    ) -> Optional[np.ndarray]:
        """
        Retrieve and decrypt embedding.
        
        Args:
            record_id: Record identifier
        
        Returns:
            Decrypted embedding or None
        """
        encrypted = self.embeddings.get(record_id)
        if not encrypted:
            return None
        
        return self.encryption.decrypt_embedding(encrypted)
    
    def save(self):
        """Save encrypted database to disk."""
        # Save encrypted FAISS index
        self.index.save(str(self.storage_path / "faiss_index"))
        
        # Save encrypted embeddings
        with open(self.storage_path / "embeddings.enc", 'wb') as f:
            pickle.dump(self.embeddings, f)
        
        logger.info(f"Saved encrypted face database to {self.storage_path}")
    
    def load(self):
        """Load encrypted database from disk."""
        # Load encrypted FAISS index
        self.index.load(str(self.storage_path / "faiss_index"))
        
        # Load encrypted embeddings
        embeddings_file = self.storage_path / "embeddings.enc"
        if embeddings_file.exists():
            with open(embeddings_file, 'rb') as f:
                self.embeddings = pickle.load(f)
        
        logger.info(f"Loaded encrypted face database from {self.storage_path}")
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        return {
            **self.index.get_stats(),
            "encrypted_embeddings": len(self.embeddings),
            "storage_path": str(self.storage_path),
            "key_fingerprint": self.encryption.get_key_fingerprint()
        }
