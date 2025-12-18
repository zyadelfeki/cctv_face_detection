"""End-to-end integration tests.

Tests complete workflows across multiple components.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
import time

# Import all security components
from src.utils.encryption import BiometricEncryption, CipherType
from src.utils.secure_delete import SecureDelete, DeletionMethod
from src.utils.api_keys import APIKeyManager
from src.utils.rbac import RBACManager, Role
from src.utils.sessions import SessionManager
from src.utils.validators import FileValidator, URLValidator, StringValidator, ValidationError


class TestEndToEndSecurity:
    """Test complete security workflows."""
    
    def test_secure_data_lifecycle(self):
        """
        Test complete data lifecycle:
        1. Store encrypted embedding
        2. Retrieve and verify
        3. Secure deletion
        """
        # Create encryption
        encryption = BiometricEncryption()
        
        # Create fake embedding
        embedding = np.random.randn(512).astype(np.float32)
        record_id = "test_person_001"
        
        # Encrypt
        encrypted = encryption.encrypt_embedding(embedding, record_id)
        
        # Save to disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".enc") as tmp:
            tmp_path = tmp.name
            tmp.write(encrypted.to_bytes())
        
        try:
            # Verify file exists
            assert Path(tmp_path).exists()
            
            # Load and decrypt
            with open(tmp_path, 'rb') as f:
                data = f.read()
            
            from src.utils.encryption import EncryptedData
            loaded_encrypted = EncryptedData.from_bytes(data)
            decrypted = encryption.decrypt_embedding(loaded_encrypted)
            
            # Verify
            np.testing.assert_array_almost_equal(decrypted, embedding)
            
            # Secure delete
            result = SecureDelete.simple_delete(tmp_path)
            assert result is True
            assert not Path(tmp_path).exists()
        
        finally:
            # Cleanup if still exists
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_authentication_authorization_flow(self):
        """
        Test complete auth flow:
        1. Create API key
        2. Validate key
        3. Check permissions (RBAC)
        4. Create session
        5. Validate session
        """
        # Setup
        api_manager = APIKeyManager()
        rbac = RBACManager()
        session_manager = SessionManager()
        
        # 1. Create API key
        service_name = "integration_test_service"
        api_key = api_manager.create_key(
            service_name=service_name,
            description="Integration test key"
        )
        
        assert api_key.startswith("cctvfd_")
        
        # 2. Validate API key
        service = api_manager.validate_key(api_key)
        assert service == service_name
        
        # 3. Create user with role
        user_id = "test_user_001"
        rbac.assign_role(user_id, Role.ANALYST)
        
        # 4. Check permissions
        assert rbac.has_permission(user_id, "criminal:view")
        assert rbac.has_permission(user_id, "incident:create")
        assert not rbac.has_permission(user_id, "user:create")  # Admin only
        
        # 5. Create session
        session_id = session_manager.create_session(
            user_id=user_id,
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0 Test"
        )
        
        assert session_id is not None
        
        # 6. Validate session
        session = session_manager.validate_session(
            session_id,
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0 Test"
        )
        
        assert session is not None
        assert session.user_id == user_id
        
        # 7. Cleanup
        api_manager.revoke_key(api_key)
        session_manager.invalidate_session(session_id)
    
    def test_input_validation_pipeline(self):
        """
        Test input validation pipeline:
        1. Validate file upload
        2. Validate URL
        3. Sanitize strings
        """
        # Test 1: Valid image (minimal JPEG)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            # Create minimal valid JPEG
            tmp.write(b'\xFF\xD8\xFF\xE0\x00\x10JFIF')  # JPEG header
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, 'rb') as f:
                content = f.read()
            
            # This may fail with corrupted image, but should not crash
            try:
                image, metadata = FileValidator.validate_image(content)
            except ValidationError:
                pass  # Expected for invalid JPEG
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        
        # Test 2: URL validation
        # Valid HTTP URL
        try:
            result = URLValidator.validate_http_url("https://example.com/image.jpg")
            assert result['scheme'] == 'https'
        except ValidationError as e:
            pytest.fail(f"Valid URL rejected: {e}")
        
        # Invalid - private IP (SSRF prevention)
        with pytest.raises(ValidationError):
            URLValidator.validate_http_url("http://192.168.1.1/admin")
        
        with pytest.raises(ValidationError):
            URLValidator.validate_http_url("http://localhost/secret")
        
        # Valid RTSP URL (cameras allowed private IPs)
        try:
            result = URLValidator.validate_rtsp_url("rtsp://192.168.1.100:554/stream")
            assert result['scheme'] == 'rtsp'
        except ValidationError as e:
            pytest.fail(f"Valid RTSP URL rejected: {e}")
        
        # Test 3: String sanitization
        # XSS attempt
        with pytest.raises(ValidationError):
            StringValidator.sanitize_string("<script>alert('xss')</script>")
        
        # SQL injection detection
        assert StringValidator.check_sql_injection("' OR '1'='1")
        
        # Safe string
        safe = StringValidator.sanitize_string("John Doe")
        assert safe == "John Doe"
    
    def test_encryption_at_rest_integration(self):
        """
        Test that encryption works across different components.
        """
        # Test both cipher types
        for cipher in [CipherType.AES_256_GCM, CipherType.CHACHA20_POLY1305]:
            encryption = BiometricEncryption(cipher_type=cipher)
            
            # Batch encryption
            embeddings = [np.random.randn(256).astype(np.float32) for _ in range(10)]
            record_ids = [f"person_{i:03d}" for i in range(10)]
            
            encrypted_list = encryption.encrypt_batch(embeddings, record_ids)
            decrypted_list = encryption.decrypt_batch(encrypted_list)
            
            for orig, decrypted in zip(embeddings, decrypted_list):
                np.testing.assert_array_almost_equal(orig, decrypted)


class TestPerformance:
    """Performance and load tests."""
    
    def test_encryption_performance(self):
        """Test encryption performance meets requirements."""
        encryption = BiometricEncryption()
        
        # Create test data
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
        
        # Requirements: < 1 second for 100 embeddings
        assert encrypt_time < 1.0, f"Encryption too slow: {encrypt_time:.3f}s"
        assert decrypt_time < 1.0, f"Decryption too slow: {decrypt_time:.3f}s"
        
        print(f"\nEncryption: {encrypt_time:.3f}s ({100/encrypt_time:.0f} ops/s)")
        print(f"Decryption: {decrypt_time:.3f}s ({100/decrypt_time:.0f} ops/s)")
    
    def test_api_key_validation_performance(self):
        """Test API key validation is fast enough for production."""
        manager = APIKeyManager()
        
        # Create keys
        keys = []
        for i in range(10):
            key = manager.create_key(f"service_{i}")
            keys.append(key)
        
        # Measure validation time
        start = time.time()
        for _ in range(1000):
            manager.validate_key(keys[0])
        elapsed = time.time() - start
        
        # Requirement: < 1ms per validation
        per_validation = elapsed / 1000
        assert per_validation < 0.001, f"Validation too slow: {per_validation*1000:.2f}ms"
        
        print(f"\nAPI key validation: {per_validation*1000:.3f}ms per call")
    
    def test_session_validation_performance(self):
        """Test session validation performance."""
        manager = SessionManager()
        
        # Create session
        session_id = manager.create_session(
            user_id="perf_user",
            ip_address="192.168.1.1",
            user_agent="Test"
        )
        
        # Measure validation time
        start = time.time()
        for _ in range(1000):
            manager.validate_session(
                session_id,
                ip_address="192.168.1.1",
                user_agent="Test"
            )
        elapsed = time.time() - start
        
        per_validation = elapsed / 1000
        assert per_validation < 0.001, f"Session validation too slow: {per_validation*1000:.2f}ms"
        
        print(f"\nSession validation: {per_validation*1000:.3f}ms per call")


class TestSecurityBoundaries:
    """Test security boundaries and edge cases."""
    
    def test_cannot_decrypt_with_wrong_key(self):
        """Test that wrong key cannot decrypt data."""
        # Encrypt with key 1
        encryption1 = BiometricEncryption()
        embedding = np.random.randn(128).astype(np.float32)
        encrypted = encryption1.encrypt_embedding(embedding, "test")
        
        # Try to decrypt with key 2
        encryption2 = BiometricEncryption()  # Different key
        
        with pytest.raises(Exception):  # Should raise InvalidTag or similar
            encryption2.decrypt_embedding(encrypted)
    
    def test_cannot_use_expired_api_key(self):
        """Test that expired API keys are rejected."""
        manager = APIKeyManager()
        
        # Create key with 1-second expiration
        key = manager.create_key("test_service", expires_in_days=1/86400)  # 1 second
        
        # Should work immediately
        assert manager.validate_key(key) == "test_service"
        
        # Wait for expiration
        time.sleep(2)
        
        # Should be expired
        assert manager.validate_key(key) is None
    
    def test_session_hijacking_detection(self):
        """Test that session hijacking is detected."""
        manager = SessionManager()
        
        # Create session
        session_id = manager.create_session(
            user_id="user_001",
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0 Browser"
        )
        
        # Validate with same IP/UA - should work
        session = manager.validate_session(
            session_id,
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0 Browser"
        )
        assert session is not None
        
        # Validate with different IP - should be flagged
        session = manager.validate_session(
            session_id,
            ip_address="10.0.0.1",  # Different IP
            user_agent="Mozilla/5.0 Browser"
        )
        
        # Session should be flagged as suspicious
        if session:
            assert session.is_suspicious
    
    def test_rbac_permission_inheritance(self):
        """Test that permission inheritance works correctly."""
        rbac = RBACManager()
        
        # Viewer should have minimal permissions
        rbac.assign_role("viewer_user", Role.VIEWER)
        assert rbac.has_permission("viewer_user", "camera:view")
        assert not rbac.has_permission("viewer_user", "camera:control")
        
        # Admin should have all permissions
        rbac.assign_role("admin_user", Role.ADMIN)
        assert rbac.has_permission("admin_user", "camera:view")
        assert rbac.has_permission("admin_user", "camera:control")
        assert rbac.has_permission("admin_user", "user:create")
        assert rbac.has_permission("admin_user", "system:configure")


class TestResourceManagement:
    """Test resource management under load."""
    
    def test_multiple_concurrent_encryptions(self):
        """Test handling multiple concurrent encryption operations."""
        encryption = BiometricEncryption()
        
        # Simulate concurrent operations
        results = []
        for _ in range(50):
            embedding = np.random.randn(256).astype(np.float32)
            encrypted = encryption.encrypt_embedding(embedding, f"concurrent_{_}")
            decrypted = encryption.decrypt_embedding(encrypted)
            results.append(np.allclose(embedding, decrypted))
        
        # All should succeed
        assert all(results)
    
    def test_session_cleanup(self):
        """Test that expired sessions are cleaned up."""
        manager = SessionManager(
            expiration_seconds=2,  # 2 second expiration
            cleanup_interval_seconds=1
        )
        
        # Create sessions
        session_ids = []
        for i in range(5):
            sid = manager.create_session(
                user_id=f"user_{i}",
                ip_address="192.168.1.1",
                user_agent="Test"
            )
            session_ids.append(sid)
        
        # All should be valid
        active = manager.get_active_sessions_count()
        assert active == 5
        
        # Wait for expiration
        time.sleep(3)
        
        # Cleanup
        removed = manager.cleanup_expired_sessions()
        assert removed == 5
        
        # No active sessions
        active = manager.get_active_sessions_count()
        assert active == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
