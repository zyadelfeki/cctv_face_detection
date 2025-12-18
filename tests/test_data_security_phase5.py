"""Tests for Phase 5: Data Security."""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import secrets
import os

from src.utils.encryption import (
    BiometricEncryption,
    EncryptedData,
    EncryptedEmbeddingStore,
    CipherType
)
from src.utils.secure_delete import SecureDelete, DeletionMethod
from src.utils.tls_manager import TLSCertificateManager

try:
    from src.utils.encrypted_faiss import EncryptedFAISSIndex, EncryptedFaceDatabase
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class TestBiometricEncryption:
    """Test biometric data encryption."""
    
    def test_key_generation(self):
        """Test secure key generation."""
        key = BiometricEncryption.generate_key()
        
        assert len(key) == 32  # 256 bits
        assert isinstance(key, bytes)
        
        # Keys should be unique
        key2 = BiometricEncryption.generate_key()
        assert key != key2
    
    def test_key_derivation(self):
        """Test key derivation from password."""
        password = "MySecurePassword123!"
        
        key1, salt1 = BiometricEncryption.derive_key(password)
        assert len(key1) == 32
        assert len(salt1) == 16
        
        # Same password + salt = same key
        key2, _ = BiometricEncryption.derive_key(password, salt1)
        assert key1 == key2
        
        # Different salt = different key
        key3, salt3 = BiometricEncryption.derive_key(password)
        assert key1 != key3
        assert salt1 != salt3
    
    def test_aes_gcm_encryption(self):
        """Test AES-256-GCM encryption."""
        encryption = BiometricEncryption(
            cipher_type=CipherType.AES_256_GCM
        )
        
        plaintext = b"Sensitive biometric data"
        
        encrypted = encryption.encrypt_bytes(plaintext)
        
        assert isinstance(encrypted, EncryptedData)
        assert encrypted.cipher_type == CipherType.AES_256_GCM
        assert len(encrypted.nonce) == 12
        assert len(encrypted.tag) == 16
        assert encrypted.ciphertext != plaintext
    
    def test_chacha20_encryption(self):
        """Test ChaCha20-Poly1305 encryption."""
        encryption = BiometricEncryption(
            cipher_type=CipherType.CHACHA20_POLY1305
        )
        
        plaintext = b"Sensitive biometric data"
        encrypted = encryption.encrypt_bytes(plaintext)
        
        assert encrypted.cipher_type == CipherType.CHACHA20_POLY1305
        assert len(encrypted.nonce) == 12
    
    def test_encryption_decryption(self):
        """Test encryption and decryption roundtrip."""
        encryption = BiometricEncryption()
        
        plaintext = b"My secret face embedding data" * 100
        
        # Encrypt
        encrypted = encryption.encrypt_bytes(plaintext)
        
        # Decrypt
        decrypted = encryption.decrypt_bytes(encrypted)
        
        assert decrypted == plaintext
    
    def test_authenticated_encryption(self):
        """Test AEAD with associated data."""
        encryption = BiometricEncryption()
        
        plaintext = b"Sensitive data"
        aad = b"record_id_12345"  # Associated data (not encrypted)
        
        # Encrypt with AAD
        encrypted = encryption.encrypt_bytes(plaintext, aad)
        
        # Decrypt with correct AAD - should work
        decrypted = encryption.decrypt_bytes(encrypted, aad)
        assert decrypted == plaintext
        
        # Decrypt with wrong AAD - should fail
        with pytest.raises(Exception):  # cryptography.exceptions.InvalidTag
            encryption.decrypt_bytes(encrypted, b"wrong_id")
    
    def test_embedding_encryption(self):
        """Test numpy embedding encryption."""
        encryption = BiometricEncryption()
        
        # Create fake embedding
        embedding = np.random.randn(512).astype(np.float32)
        
        # Encrypt
        encrypted = encryption.encrypt_embedding(embedding, "record_001")
        
        assert encrypted.metadata is not None
        assert encrypted.metadata['shape'] == (512,)
        assert encrypted.metadata['record_id'] == "record_001"
        
        # Decrypt
        decrypted = encryption.decrypt_embedding(encrypted)
        
        assert decrypted.shape == embedding.shape
        assert decrypted.dtype == embedding.dtype
        np.testing.assert_array_almost_equal(decrypted, embedding)
    
    def test_batch_encryption(self):
        """Test batch encryption/decryption."""
        encryption = BiometricEncryption()
        
        # Create batch of embeddings
        embeddings = [np.random.randn(128).astype(np.float32) for _ in range(10)]
        record_ids = [f"record_{i:03d}" for i in range(10)]
        
        # Encrypt batch
        encrypted_list = encryption.encrypt_batch(embeddings, record_ids)
        
        assert len(encrypted_list) == 10
        
        # Decrypt batch
        decrypted_list = encryption.decrypt_batch(encrypted_list)
        
        assert len(decrypted_list) == 10
        for original, decrypted in zip(embeddings, decrypted_list):
            np.testing.assert_array_almost_equal(original, decrypted)
    
    def test_serialization(self):
        """Test EncryptedData serialization."""
        encryption = BiometricEncryption()
        
        plaintext = b"Test data" * 100
        encrypted = encryption.encrypt_bytes(plaintext)
        
        # Serialize
        serialized = encrypted.to_bytes()
        assert isinstance(serialized, bytes)
        
        # Deserialize
        deserialized = EncryptedData.from_bytes(serialized)
        
        assert deserialized.cipher_type == encrypted.cipher_type
        assert deserialized.nonce == encrypted.nonce
        assert deserialized.tag == encrypted.tag
        assert deserialized.ciphertext == encrypted.ciphertext
        
        # Decrypt deserialized
        decrypted = encryption.decrypt_bytes(deserialized)
        assert decrypted == plaintext


class TestEncryptedEmbeddingStore:
    """Test encrypted embedding storage."""
    
    def test_store_retrieve(self):
        """Test storing and retrieving embeddings."""
        encryption = BiometricEncryption()
        store = EncryptedEmbeddingStore(encryption)
        
        # Store embedding
        embedding = np.random.randn(256).astype(np.float32)
        store.store("test_001", embedding)
        
        # Retrieve
        retrieved = store.retrieve("test_001")
        
        assert retrieved is not None
        np.testing.assert_array_almost_equal(retrieved, embedding)
    
    def test_batch_operations(self):
        """Test batch store/retrieve."""
        encryption = BiometricEncryption()
        store = EncryptedEmbeddingStore(encryption)
        
        # Store batch
        embeddings = [np.random.randn(128).astype(np.float32) for _ in range(5)]
        record_ids = [f"batch_{i}" for i in range(5)]
        
        store.store_batch(record_ids, embeddings)
        
        # Retrieve batch
        retrieved = store.retrieve_batch(record_ids)
        
        assert len(retrieved) == 5
        for original, ret in zip(embeddings, retrieved):
            np.testing.assert_array_almost_equal(original, ret)
    
    def test_persistence(self):
        """Test saving/loading from disk."""
        encryption = BiometricEncryption()
        store = EncryptedEmbeddingStore(encryption)
        
        # Add data
        embeddings = [np.random.randn(64).astype(np.float32) for _ in range(3)]
        record_ids = ["persist_1", "persist_2", "persist_3"]
        store.store_batch(record_ids, embeddings)
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save
            store.save_to_disk(tmp_path)
            
            # Load into new store
            new_store = EncryptedEmbeddingStore(encryption)
            new_store.load_from_disk(tmp_path)
            
            # Verify
            for rid, orig_emb in zip(record_ids, embeddings):
                retrieved = new_store.retrieve(rid)
                np.testing.assert_array_almost_equal(retrieved, orig_emb)
        
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestSecureDeletion:
    """Test secure file deletion."""
    
    def test_simple_deletion(self):
        """Test 3-pass deletion."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"Sensitive face image data" * 1000)
            tmp_path = tmp.name
        
        try:
            assert Path(tmp_path).exists()
            
            # Secure delete
            result = SecureDelete.simple_delete(tmp_path)
            
            assert result is True
            assert not Path(tmp_path).exists()
        
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_dod_deletion(self):
        """Test DOD 7-pass deletion."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"Secret embedding" * 500)
            tmp_path = tmp.name
        
        try:
            result = SecureDelete.dod_delete(tmp_path)
            
            assert result is True
            assert not Path(tmp_path).exists()
        
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_directory_deletion(self):
        """Test deleting directory of files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create test files
            for i in range(5):
                (tmp_path / f"file_{i}.dat").write_bytes(b"data" * 100)
            
            count = SecureDelete.delete_directory(
                str(tmp_path),
                method=DeletionMethod.SIMPLE,
                recursive=False
            )
            
            assert count == 5
    
    def test_nonexistent_file(self):
        """Test deletion of nonexistent file."""
        result = SecureDelete.delete_file("/tmp/does_not_exist_12345.dat")
        assert result is False


class TestTLSCertificates:
    """Test TLS certificate management."""
    
    def test_self_signed_certificate(self):
        """Test self-signed certificate generation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = TLSCertificateManager(cert_dir=tmp_dir)
            
            # Generate certificate
            cert_pem, key_pem = manager.generate_self_signed_cert(
                domain="test.example.com",
                valid_days=30
            )
            
            assert cert_pem.startswith(b"-----BEGIN CERTIFICATE-----")
            assert key_pem.startswith(b"-----BEGIN RSA PRIVATE KEY-----")
            
            # Save certificate
            manager.save_certificate(cert_pem, key_pem, "test")
            
            # Load and verify
            info = manager.get_certificate_info("test")
            
            assert info is not None
            assert info.subject['commonName'] == "test.example.com"
            assert info.is_self_signed is True
            assert info.days_until_expiry <= 30
            assert not info.is_expired()
    
    def test_certificate_info(self):
        """Test certificate information extraction."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = TLSCertificateManager(cert_dir=tmp_dir)
            
            cert_pem, key_pem = manager.generate_self_signed_cert(
                domain="localhost",
                valid_days=365,
                organization="Test Org"
            )
            
            manager.save_certificate(cert_pem, key_pem, "localhost")
            info = manager.get_certificate_info("localhost")
            
            assert info.subject['commonName'] == "localhost"
            assert info.subject['organizationName'] == "Test Org"
            assert len(info.fingerprint) == 64  # SHA-256 hex
    
    def test_certificate_renewal_check(self):
        """Test renewal checking."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = TLSCertificateManager(
                cert_dir=tmp_dir,
                renewal_days=30
            )
            
            # Generate certificate expiring in 20 days
            cert_pem, key_pem = manager.generate_self_signed_cert(
                domain="renew-test.com",
                valid_days=20
            )
            
            manager.save_certificate(cert_pem, key_pem, "renew_test")
            info = manager.get_certificate_info("renew_test")
            
            # Should need renewal (< 30 days left)
            assert info.needs_renewal(30) is True
            assert info.needs_renewal(10) is False
    
    def test_key_permissions(self):
        """Test that private keys have secure permissions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = TLSCertificateManager(cert_dir=tmp_dir)
            
            cert_pem, key_pem = manager.generate_self_signed_cert("test.com")
            manager.save_certificate(cert_pem, key_pem, "perms_test")
            
            key_path = Path(tmp_dir) / "perms_test.key"
            
            # Check permissions (should be 0o600 = owner read/write only)
            perms = oct(key_path.stat().st_mode)[-3:]
            assert perms == "600" or os.name == 'nt'  # Skip on Windows


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not installed")
class TestEncryptedFAISS:
    """Test encrypted FAISS index."""
    
    def test_encrypted_index_creation(self):
        """Test creating encrypted FAISS index."""
        encryption = BiometricEncryption()
        index = EncryptedFAISSIndex(
            dimension=128,
            encryption=encryption
        )
        
        assert index.dimension == 128
        assert index.encryption is not None
    
    def test_add_and_search(self):
        """Test adding and searching vectors."""
        encryption = BiometricEncryption()
        index = EncryptedFAISSIndex(dimension=64, encryption=encryption)
        
        # Add vectors
        vectors = np.random.randn(10, 64).astype(np.float32)
        record_ids = [f"vec_{i}" for i in range(10)]
        
        index.add(vectors, record_ids)
        
        # Search
        query = vectors[0]  # Search for first vector
        results = index.search(query, k=3)
        
        assert len(results) > 0
        assert results[0][0] == "vec_0"  # First result should be exact match
        assert results[0][1] > 0.99  # Near-perfect similarity
    
    def test_save_load_encrypted(self):
        """Test saving and loading encrypted index."""
        encryption = BiometricEncryption()
        index = EncryptedFAISSIndex(dimension=32, encryption=encryption)
        
        # Add data
        vectors = np.random.randn(5, 32).astype(np.float32)
        record_ids = [f"save_{i}" for i in range(5)]
        index.add(vectors, record_ids)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir) / "test_index"
            
            # Save
            index.save(str(base_path))
            
            # Verify encrypted files exist
            assert (Path(str(base_path) + ".faiss.enc")).exists()
            assert (Path(str(base_path) + ".map.enc")).exists()
            
            # Load into new index
            new_index = EncryptedFAISSIndex(dimension=32, encryption=encryption)
            new_index.load(str(base_path))
            
            # Search in loaded index
            results = new_index.search(vectors[0], k=1)
            assert results[0][0] == "save_0"
    
    def test_encrypted_face_database(self):
        """Test full encrypted face database."""
        encryption = BiometricEncryption()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            db = EncryptedFaceDatabase(
                storage_path=tmp_dir,
                embedding_dim=64,
                encryption=encryption
            )
            
            # Add embeddings
            for i in range(5):
                embedding = np.random.randn(64).astype(np.float32)
                db.add_embedding(f"person_{i}", embedding)
            
            # Save
            db.save()
            
            # Load new database
            db2 = EncryptedFaceDatabase(
                storage_path=tmp_dir,
                embedding_dim=64,
                encryption=encryption
            )
            db2.load()
            
            # Verify data
            stats = db2.get_stats()
            assert stats['total_vectors'] == 5
            assert stats['encrypted'] is True


class TestIntegration:
    """Integration tests for data security."""
    
    def test_end_to_end_encryption(self):
        """Test complete encryption workflow."""
        # Generate master key
        master_key = BiometricEncryption.generate_key()
        encryption = BiometricEncryption(master_key)
        
        # Create face embedding
        face_embedding = np.random.randn(512).astype(np.float32)
        record_id = "person_12345"
        
        # Encrypt
        encrypted = encryption.encrypt_embedding(face_embedding, record_id)
        
        # Serialize (simulating storage)
        serialized = encrypted.to_bytes()
        
        # Deserialize
        deserialized = EncryptedData.from_bytes(serialized)
        
        # Decrypt
        decrypted = encryption.decrypt_embedding(deserialized)
        
        # Verify
        np.testing.assert_array_almost_equal(decrypted, face_embedding)
    
    def test_encryption_performance(self):
        """Benchmark encryption performance."""
        import time
        
        encryption = BiometricEncryption()
        
        # Create batch of embeddings
        embeddings = [np.random.randn(512).astype(np.float32) for _ in range(100)]
        record_ids = [f"perf_{i}" for i in range(100)]
        
        # Measure encryption time
        start = time.time()
        encrypted_list = encryption.encrypt_batch(embeddings, record_ids)
        encrypt_time = time.time() - start
        
        # Measure decryption time
        start = time.time()
        decrypted_list = encryption.decrypt_batch(encrypted_list)
        decrypt_time = time.time() - start
        
        # Should be fast (< 1 second for 100 embeddings)
        assert encrypt_time < 1.0
        assert decrypt_time < 1.0
        
        print(f"\nEncryption: {encrypt_time:.3f}s for 100 embeddings")
        print(f"Decryption: {decrypt_time:.3f}s for 100 embeddings")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
