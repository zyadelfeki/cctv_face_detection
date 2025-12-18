"""Encryption Utilities for Biometric Data.

Provides:
- AES-256-GCM encryption for face embeddings
- ChaCha20-Poly1305 alternative
- Secure key derivation
- Nonce management
- Batch encryption/decryption
- Authenticated encryption with associated data (AEAD)
"""

import os
import secrets
import hashlib
import struct
import json
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import numpy as np

from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from loguru import logger


class CipherType:
    """Supported cipher types."""
    AES_256_GCM = "aes-256-gcm"
    CHACHA20_POLY1305 = "chacha20-poly1305"


@dataclass
class EncryptedData:
    """Container for encrypted data with metadata."""
    ciphertext: bytes
    nonce: bytes
    tag: bytes  # Authentication tag
    cipher_type: str
    version: int = 1
    metadata: Optional[Dict] = None
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes for storage."""
        # Format: version(1) | cipher_type_len(1) | cipher_type | nonce_len(1) | nonce | 
        #         tag_len(1) | tag | metadata_len(2) | metadata_json | ciphertext
        
        cipher_bytes = self.cipher_type.encode('utf-8')
        
        # Serialize metadata as JSON
        metadata_json = b''
        if self.metadata:
            metadata_json = json.dumps(self.metadata).encode('utf-8')
        
        parts = [
            struct.pack('B', self.version),
            struct.pack('B', len(cipher_bytes)),
            cipher_bytes,
            struct.pack('B', len(self.nonce)),
            self.nonce,
            struct.pack('B', len(self.tag)),
            self.tag,
            struct.pack('H', len(metadata_json)),  # 2 bytes for metadata length
            metadata_json,
            self.ciphertext
        ]
        
        return b''.join(parts)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'EncryptedData':
        """Deserialize from bytes."""
        offset = 0
        
        # Version
        version = struct.unpack('B', data[offset:offset+1])[0]
        offset += 1
        
        # Cipher type
        cipher_len = struct.unpack('B', data[offset:offset+1])[0]
        offset += 1
        cipher_type = data[offset:offset+cipher_len].decode('utf-8')
        offset += cipher_len
        
        # Nonce
        nonce_len = struct.unpack('B', data[offset:offset+1])[0]
        offset += 1
        nonce = data[offset:offset+nonce_len]
        offset += nonce_len
        
        # Tag
        tag_len = struct.unpack('B', data[offset:offset+1])[0]
        offset += 1
        tag = data[offset:offset+tag_len]
        offset += tag_len
        
        # Metadata
        metadata_len = struct.unpack('H', data[offset:offset+2])[0]
        offset += 2
        
        metadata = None
        if metadata_len > 0:
            metadata_json = data[offset:offset+metadata_len]
            offset += metadata_len
            metadata = json.loads(metadata_json.decode('utf-8'))
        
        # Ciphertext (remaining bytes)
        ciphertext = data[offset:]
        
        return cls(
            ciphertext=ciphertext,
            nonce=nonce,
            tag=tag,
            cipher_type=cipher_type,
            version=version,
            metadata=metadata
        )


class BiometricEncryption:
    """Encryption manager for biometric data.
    
    Features:
    - AES-256-GCM (NIST standard)
    - ChaCha20-Poly1305 (fast, modern)
    - Authenticated encryption (prevents tampering)
    - Per-record unique nonces
    - Key derivation from master key
    - Numpy array support
    """
    
    # Key sizes
    AES_KEY_SIZE = 32  # 256 bits
    CHACHA_KEY_SIZE = 32  # 256 bits
    NONCE_SIZE_GCM = 12  # 96 bits (NIST recommended)
    NONCE_SIZE_CHACHA = 12  # 96 bits
    
    def __init__(
        self,
        master_key: Optional[bytes] = None,
        cipher_type: str = CipherType.AES_256_GCM
    ):
        """
        Initialize encryption manager.
        
        Args:
            master_key: 32-byte master key (None = generate new)
            cipher_type: Cipher algorithm to use
        """
        self.cipher_type = cipher_type
        
        if master_key is None:
            master_key = secrets.token_bytes(32)
            logger.warning("Generated new master key. Save this securely!")
        
        if len(master_key) != 32:
            raise ValueError("Master key must be 32 bytes")
        
        self.master_key = master_key
        
        # Initialize cipher
        if cipher_type == CipherType.AES_256_GCM:
            self.cipher = AESGCM(master_key)
            self.nonce_size = self.NONCE_SIZE_GCM
        elif cipher_type == CipherType.CHACHA20_POLY1305:
            self.cipher = ChaCha20Poly1305(master_key)
            self.nonce_size = self.NONCE_SIZE_CHACHA
        else:
            raise ValueError(f"Unsupported cipher type: {cipher_type}")
        
        logger.info(f"Initialized biometric encryption with {cipher_type}")
    
    @staticmethod
    def generate_key() -> bytes:
        """Generate a new random 256-bit key."""
        return secrets.token_bytes(32)
    
    @staticmethod
    def derive_key(
        password: str,
        salt: Optional[bytes] = None,
        iterations: int = 100000
    ) -> Tuple[bytes, bytes]:
        """
        Derive encryption key from password.
        
        Args:
            password: User password
            salt: Salt (None = generate new)
            iterations: PBKDF2 iterations
        
        Returns:
            Tuple of (key, salt)
        """
        if salt is None:
            salt = secrets.token_bytes(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations
        )
        
        key = kdf.derive(password.encode('utf-8'))
        return key, salt
    
    def _generate_nonce(self) -> bytes:
        """Generate cryptographically secure nonce."""
        return secrets.token_bytes(self.nonce_size)
    
    def encrypt_bytes(
        self,
        plaintext: bytes,
        associated_data: Optional[bytes] = None
    ) -> EncryptedData:
        """
        Encrypt arbitrary bytes.
        
        Args:
            plaintext: Data to encrypt
            associated_data: Additional authenticated data (not encrypted)
        
        Returns:
            EncryptedData object
        """
        nonce = self._generate_nonce()
        
        # Encrypt with AEAD
        ciphertext = self.cipher.encrypt(nonce, plaintext, associated_data)
        
        # Extract tag (last 16 bytes for GCM/Poly1305)
        tag = ciphertext[-16:]
        ciphertext = ciphertext[:-16]
        
        return EncryptedData(
            ciphertext=ciphertext,
            nonce=nonce,
            tag=tag,
            cipher_type=self.cipher_type
        )
    
    def decrypt_bytes(
        self,
        encrypted: EncryptedData,
        associated_data: Optional[bytes] = None
    ) -> bytes:
        """
        Decrypt bytes.
        
        Args:
            encrypted: EncryptedData object
            associated_data: Must match encryption AAD
        
        Returns:
            Decrypted plaintext
        
        Raises:
            cryptography.exceptions.InvalidTag: If authentication fails
        """
        # Reconstruct ciphertext with tag
        ciphertext_with_tag = encrypted.ciphertext + encrypted.tag
        
        # Decrypt and verify
        plaintext = self.cipher.decrypt(
            encrypted.nonce,
            ciphertext_with_tag,
            associated_data
        )
        
        return plaintext
    
    def encrypt_embedding(
        self,
        embedding: np.ndarray,
        record_id: Optional[str] = None
    ) -> EncryptedData:
        """
        Encrypt face embedding.
        
        Args:
            embedding: Numpy array (any shape/dtype)
            record_id: Optional record ID for AAD
        
        Returns:
            EncryptedData object
        """
        # Serialize numpy array
        plaintext = self._serialize_embedding(embedding)
        
        # Use record_id as associated data if provided
        aad = record_id.encode('utf-8') if record_id else None
        
        encrypted = self.encrypt_bytes(plaintext, aad)
        
        # Store embedding metadata
        encrypted.metadata = {
            'shape': embedding.shape,
            'dtype': str(embedding.dtype),
            'record_id': record_id
        }
        
        return encrypted
    
    def decrypt_embedding(
        self,
        encrypted: EncryptedData
    ) -> np.ndarray:
        """
        Decrypt face embedding.
        
        Args:
            encrypted: EncryptedData object
        
        Returns:
            Numpy array
        """
        # Extract AAD from metadata
        record_id = encrypted.metadata.get('record_id') if encrypted.metadata else None
        aad = record_id.encode('utf-8') if record_id else None
        
        # Decrypt
        plaintext = self.decrypt_bytes(encrypted, aad)
        
        # Deserialize
        embedding = self._deserialize_embedding(
            plaintext,
            shape=encrypted.metadata['shape'] if encrypted.metadata else None,
            dtype=encrypted.metadata['dtype'] if encrypted.metadata else None
        )
        
        return embedding
    
    @staticmethod
    def _serialize_embedding(embedding: np.ndarray) -> bytes:
        """Serialize numpy array to bytes."""
        # Store dtype, shape, and data
        dtype_str = str(embedding.dtype)
        shape_bytes = struct.pack(f'{len(embedding.shape)}I', *embedding.shape)
        
        header = struct.pack(
            f'B{len(dtype_str)}sB',
            len(dtype_str),
            dtype_str.encode('utf-8'),
            len(embedding.shape)
        )
        
        return header + shape_bytes + embedding.tobytes()
    
    @staticmethod
    def _deserialize_embedding(
        data: bytes,
        shape: Optional[Tuple] = None,
        dtype: Optional[str] = None
    ) -> np.ndarray:
        """Deserialize bytes to numpy array."""
        offset = 0
        
        # Read dtype length
        dtype_len = struct.unpack('B', data[offset:offset+1])[0]
        offset += 1
        
        # Read dtype
        dtype_str = data[offset:offset+dtype_len].decode('utf-8')
        offset += dtype_len
        
        # Read shape length
        ndim = struct.unpack('B', data[offset:offset+1])[0]
        offset += 1
        
        # Read shape
        shape_tuple = struct.unpack(f'{ndim}I', data[offset:offset+ndim*4])
        offset += ndim * 4
        
        # Read array data
        array_data = data[offset:]
        
        # Reconstruct array
        arr = np.frombuffer(array_data, dtype=dtype_str)
        return arr.reshape(shape_tuple)
    
    def encrypt_batch(
        self,
        embeddings: List[np.ndarray],
        record_ids: Optional[List[str]] = None
    ) -> List[EncryptedData]:
        """
        Encrypt multiple embeddings efficiently.
        
        Args:
            embeddings: List of numpy arrays
            record_ids: Optional list of record IDs
        
        Returns:
            List of EncryptedData objects
        """
        if record_ids is None:
            record_ids = [None] * len(embeddings)
        
        return [
            self.encrypt_embedding(emb, rid)
            for emb, rid in zip(embeddings, record_ids)
        ]
    
    def decrypt_batch(self, encrypted_list: List[EncryptedData]) -> List[np.ndarray]:
        """
        Decrypt multiple embeddings.
        
        Args:
            encrypted_list: List of EncryptedData objects
        
        Returns:
            List of numpy arrays
        """
        return [self.decrypt_embedding(enc) for enc in encrypted_list]
    
    def get_key_fingerprint(self) -> str:
        """Get SHA-256 fingerprint of master key."""
        return hashlib.sha256(self.master_key).hexdigest()[:16]
    
    def rotate_key(self, new_key: bytes) -> 'BiometricEncryption':
        """
        Create new encryption manager with rotated key.
        
        Args:
            new_key: New 32-byte master key
        
        Returns:
            New BiometricEncryption instance
        """
        return BiometricEncryption(new_key, self.cipher_type)


class EncryptedEmbeddingStore:
    """Storage wrapper that encrypts embeddings transparently.
    
    Drop-in replacement for plain embedding storage.
    """
    
    def __init__(self, encryption: BiometricEncryption):
        """
        Initialize encrypted store.
        
        Args:
            encryption: BiometricEncryption instance
        """
        self.encryption = encryption
        self.storage: Dict[str, EncryptedData] = {}
    
    def store(self, record_id: str, embedding: np.ndarray):
        """Store encrypted embedding."""
        encrypted = self.encryption.encrypt_embedding(embedding, record_id)
        self.storage[record_id] = encrypted
        logger.debug(f"Stored encrypted embedding for {record_id}")
    
    def retrieve(self, record_id: str) -> Optional[np.ndarray]:
        """Retrieve and decrypt embedding."""
        encrypted = self.storage.get(record_id)
        if not encrypted:
            return None
        
        return self.encryption.decrypt_embedding(encrypted)
    
    def store_batch(self, record_ids: List[str], embeddings: List[np.ndarray]):
        """Store multiple embeddings."""
        encrypted_list = self.encryption.encrypt_batch(embeddings, record_ids)
        
        for rid, encrypted in zip(record_ids, encrypted_list):
            self.storage[rid] = encrypted
        
        logger.info(f"Stored {len(record_ids)} encrypted embeddings")
    
    def retrieve_batch(self, record_ids: List[str]) -> List[Optional[np.ndarray]]:
        """Retrieve multiple embeddings."""
        encrypted_list = [self.storage.get(rid) for rid in record_ids]
        
        results = []
        for encrypted in encrypted_list:
            if encrypted:
                results.append(self.encryption.decrypt_embedding(encrypted))
            else:
                results.append(None)
        
        return results
    
    def delete(self, record_id: str) -> bool:
        """Delete embedding."""
        if record_id in self.storage:
            del self.storage[record_id]
            return True
        return False
    
    def list_records(self) -> List[str]:
        """List all record IDs."""
        return list(self.storage.keys())
    
    def save_to_disk(self, filepath: str):
        """Save encrypted storage to disk."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.storage, f)
        logger.info(f"Saved {len(self.storage)} encrypted embeddings to {filepath}")
    
    def load_from_disk(self, filepath: str):
        """Load encrypted storage from disk."""
        import pickle
        with open(filepath, 'rb') as f:
            self.storage = pickle.load(f)
        logger.info(f"Loaded {len(self.storage)} encrypted embeddings from {filepath}")


# Global encryption instance (singleton)
_global_encryption: Optional[BiometricEncryption] = None


def get_encryption(
    master_key: Optional[bytes] = None,
    cipher_type: str = CipherType.AES_256_GCM
) -> BiometricEncryption:
    """Get global encryption instance."""
    global _global_encryption
    
    if _global_encryption is None:
        _global_encryption = BiometricEncryption(master_key, cipher_type)
    
    return _global_encryption
