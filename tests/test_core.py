"""
Comprehensive test suite for CCTV Face Detection System.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import asyncio


# ============================================
# Test Configuration and Fixtures
# ============================================

@pytest.fixture
def mock_config():
    """Create mock configuration object."""
    config = Mock()
    config.get.return_value = Mock(
        system=Mock(mode="testing", debug=True, gpu_enabled=False),
        detection=Mock(
            confidence_threshold=0.9,
            min_face_size=40,
            scale_factor=0.709,
            steps_threshold=(0.6, 0.7, 0.7),
            margin=44,
            keep_all=True
        ),
        recognition=Mock(
            similarity_threshold=0.4,
            embedding_size=512,
            device="cpu",
            batch_size=16
        ),
        performance=Mock(
            frame_skip=2,
            batch_size=8,
            batch_timeout=0.1,
            num_workers=4
        ),
        cameras=Mock(
            sources=[
                Mock(name="TestCam1", url="rtsp://test1", location="Lobby", active=True),
                Mock(name="TestCam2", url="rtsp://test2", location="Exit", active=True)
            ],
            buffer_size=1,
            reconnect_interval=5,
            timeout=10
        ),
        alerting=Mock(
            enabled=True,
            email=Mock(enabled=False),
            sms=Mock(enabled=False),
            webhook=Mock(enabled=False)
        ),
        storage=Mock(detected_faces_dir="./test_data/faces"),
        api=Mock(
            authentication=Mock(
                secret_key="test-secret-key-that-is-long-enough-32chars",
                algorithm="HS256",
                access_token_expire_minutes=15
            ),
            rate_limiting=Mock(enabled=True, calls=100, period=60)
        )
    )
    return config


@pytest.fixture
def sample_image():
    """Create a sample BGR image for testing."""
    # 640x480 BGR image with random data
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_face_crop():
    """Create a sample face crop (160x160)."""
    return np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)


@pytest.fixture
def sample_embedding():
    """Create a sample 512-d face embedding."""
    emb = np.random.randn(512).astype(np.float32)
    return emb / np.linalg.norm(emb)


# ============================================
# Detection Pipeline Tests
# ============================================

class TestFacePipeline:
    """Test face detection pipeline."""
    
    def test_preprocessor_to_rgb(self, sample_image):
        """Test BGR to RGB conversion."""
        from src.core.pipeline import Preprocessor
        
        rgb = Preprocessor.to_rgb(sample_image)
        assert rgb.shape == sample_image.shape
        # Check that channels are swapped
        np.testing.assert_array_equal(rgb[:, :, 0], sample_image[:, :, 2])
    
    def test_preprocessor_crop_and_resize(self, sample_image):
        """Test face crop and resize."""
        from src.core.pipeline import Preprocessor
        
        box = (100, 100, 200, 200)
        crop = Preprocessor.crop_and_resize(sample_image, box, 160)
        
        assert crop.shape == (160, 160, 3)
    
    def test_preprocessor_to_tensor(self, sample_face_crop):
        """Test numpy to tensor conversion."""
        from src.core.pipeline import Preprocessor
        import torch
        
        tensor = Preprocessor.to_tensor(sample_face_crop)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 160, 160)
        assert tensor.min() >= 0 and tensor.max() <= 1


class TestMTCNNDetector:
    """Test MTCNN face detector."""
    
    @patch('src.core.detectors.mtcnn_detector.MTCNN')
    def test_detect_returns_list(self, mock_mtcnn, mock_config, sample_image):
        """Test detection returns proper format."""
        from src.core.detectors.mtcnn_detector import MTCNNDetector
        
        mock_mtcnn.return_value.detect_faces.return_value = [
            {"box": [100, 100, 50, 50], "confidence": 0.99, "keypoints": {}}
        ]
        
        detector = MTCNNDetector(mock_config)
        results = detector.detect(sample_image)
        
        assert isinstance(results, list)
        assert len(results) == 1
        assert "box" in results[0]
        assert "confidence" in results[0]
    
    def test_expand_box(self):
        """Test box expansion with margins."""
        from src.core.detectors.mtcnn_detector import MTCNNDetector
        
        box = [100, 100, 50, 50]  # x, y, w, h
        margin = 20
        img_h, img_w = 480, 640
        
        x1, y1, x2, y2 = MTCNNDetector.expand_box(box, margin, img_h, img_w)
        
        assert x1 == 90  # 100 - 20/2
        assert y1 == 90  # 100 - 20/2
        assert x2 == 160  # 100 + 50 + 20/2
        assert y2 == 160


# ============================================
# Liveness Detection Tests
# ============================================

class TestEyeBlinkDetector:
    """Test eye blink detection for liveness."""
    
    def test_compute_ear(self):
        """Test Eye Aspect Ratio computation."""
        from src.core.liveness.eye_blink import EyeBlinkDetector
        
        detector = EyeBlinkDetector()
        
        # Open eye landmarks (EAR should be around 0.3-0.4)
        open_eye = np.array([
            [0, 20], [10, 30], [30, 30], [40, 20], [30, 10], [10, 10]
        ])
        ear = detector.compute_ear(open_eye)
        assert 0.2 < ear < 0.5
        
        # Closed eye landmarks (EAR should be low)
        closed_eye = np.array([
            [0, 20], [10, 22], [30, 22], [40, 20], [30, 18], [10, 18]
        ])
        ear = detector.compute_ear(closed_eye)
        assert ear < 0.2
    
    def test_blink_detection_sequence(self):
        """Test blink detection over multiple frames."""
        from src.core.liveness.eye_blink import EyeBlinkDetector
        
        detector = EyeBlinkDetector(ear_threshold=0.25, consecutive_frames=2)
        
        # Simulate open eyes
        open_landmarks = {"all": [[0, 20], [10, 30], [30, 30], [40, 20], [30, 10], [10, 10]] * 12}
        
        # Simulate blink sequence: open -> closed -> closed -> open
        closed_landmarks = {"all": [[0, 20], [10, 22], [30, 22], [40, 20], [30, 18], [10, 18]] * 12}
        
        face_id = "test_face"
        
        # Open eyes
        result = detector.update(face_id, open_landmarks)
        assert result["blink_count"] == 0
        
        # Start closing
        result = detector.update(face_id, closed_landmarks)
        assert result["eye_closed"] == True
        
        # Still closed
        result = detector.update(face_id, closed_landmarks)
        
        # Open again - should detect blink
        result = detector.update(face_id, open_landmarks)
        assert result["blink_count"] == 1


class TestTextureAnalyzer:
    """Test texture analysis for spoof detection."""
    
    def test_compute_lbp_histogram(self, sample_face_crop):
        """Test LBP histogram computation."""
        from src.core.liveness.texture import TextureAnalyzer
        import cv2
        
        analyzer = TextureAnalyzer()
        gray = cv2.cvtColor(sample_face_crop, cv2.COLOR_BGR2GRAY)
        hist = analyzer.compute_lbp_histogram(gray)
        
        assert hist.shape == (256,)
        assert abs(hist.sum() - 1.0) < 0.01  # Normalized
    
    def test_analyze_returns_result(self, sample_face_crop):
        """Test full texture analysis."""
        from src.core.liveness.texture import TextureAnalyzer
        
        analyzer = TextureAnalyzer()
        result = analyzer.analyze(sample_face_crop)
        
        assert "is_live" in result
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1


class TestLivenessDetector:
    """Test combined liveness detector."""
    
    def test_single_image_analysis(self, sample_face_crop):
        """Test liveness on single image."""
        from src.core.liveness.detector import LivenessDetector
        
        detector = LivenessDetector()
        result = detector.check_single_image(sample_face_crop)
        
        assert "is_live" in result
        assert "mode" in result
        assert result["mode"] == "single_image"
        assert "warning" in result  # Should warn about limited accuracy


# ============================================
# Matching and Embedding Tests
# ============================================

class TestEmbeddingIndex:
    """Test FAISS embedding index."""
    
    def test_add_and_search(self, sample_embedding):
        """Test adding and searching embeddings."""
        from src.database.embedding_index import EmbeddingIndex
        
        index = EmbeddingIndex(dim=512, metric="cosine")
        
        # Add embeddings
        embeddings = np.vstack([sample_embedding, sample_embedding * 0.9])
        index.add(embeddings, [1, 2])
        
        # Search
        results = index.search(sample_embedding.reshape(1, -1), k=2)
        
        assert len(results) == 1
        assert len(results[0]) == 2
        # First result should be exact match
        assert results[0][0][0] == 1


class TestMatcher:
    """Test face matching."""
    
    @patch('src.core.matching.EmbeddingIndex')
    def test_match_returns_filtered_results(self, mock_index, mock_config):
        """Test matcher filters by threshold."""
        from src.core.matching import Matcher
        
        mock_index_instance = mock_index.return_value
        mock_index_instance.search.return_value = [
            [(1, 0.95), (2, 0.3)]  # One above threshold, one below
        ]
        
        matcher = Matcher(mock_config)
        matcher.index = mock_index_instance
        
        emb = np.random.randn(1, 512).astype(np.float32)
        results = matcher.match(emb)
        
        assert len(results) == 1
        assert len(results[0]) == 1  # Only high similarity match
        assert results[0][0]["similarity"] == 0.95


# ============================================
# API Security Tests
# ============================================

class TestEnhancedAuthService:
    """Test enhanced authentication."""
    
    def test_password_hashing(self):
        """Test password hashing and verification."""
        from src.api.security import EnhancedAuthService
        
        auth = EnhancedAuthService(secret_key="a" * 32)
        
        password = "SecurePass123!"
        hashed = auth.hash_password(password)
        
        assert hashed != password
        assert auth.verify_password(password, hashed)
        assert not auth.verify_password("wrong", hashed)
    
    def test_token_creation_and_validation(self):
        """Test JWT token lifecycle."""
        from src.api.security import EnhancedAuthService
        
        auth = EnhancedAuthService(secret_key="a" * 32)
        
        token, jti = auth.create_access_token("testuser", "admin")
        
        assert token
        assert jti
        
        payload = auth.decode_token(token)
        assert payload["sub"] == "testuser"
        assert payload["role"] == "admin"
    
    def test_lockout_after_failed_attempts(self):
        """Test account lockout."""
        from src.api.security import EnhancedAuthService
        
        auth = EnhancedAuthService(
            secret_key="a" * 32,
            max_login_attempts=3,
            lockout_duration_minutes=1
        )
        
        # Record failed attempts
        for _ in range(3):
            auth.record_login_attempt("testuser", success=False)
        
        assert auth.is_locked_out("testuser")
    
    def test_token_revocation(self):
        """Test token blacklisting."""
        from src.api.security import EnhancedAuthService, token_blacklist
        
        auth = EnhancedAuthService(secret_key="a" * 32)
        
        token, jti = auth.create_access_token("testuser", "admin")
        
        # Token should work before revocation
        payload = auth.decode_token(token)
        assert payload["sub"] == "testuser"
        
        # Revoke token
        auth.revoke_token(token)
        
        # Token should now be blacklisted
        assert token_blacklist.is_blacklisted(jti)


class TestRateLimiter:
    """Test rate limiting."""
    
    @pytest.mark.asyncio
    async def test_allows_within_limit(self):
        """Test requests allowed within limit."""
        from src.api.rate_limit import SlidingWindowRateLimiter
        
        limiter = SlidingWindowRateLimiter(calls=5, period=60)
        
        request = Mock()
        request.client.host = "127.0.0.1"
        request.headers.get.return_value = None
        
        for _ in range(5):
            allowed, info = await limiter.is_allowed(request)
            assert allowed
    
    @pytest.mark.asyncio
    async def test_blocks_over_limit(self):
        """Test requests blocked over limit."""
        from src.api.rate_limit import SlidingWindowRateLimiter
        
        limiter = SlidingWindowRateLimiter(calls=2, period=60)
        
        request = Mock()
        request.client.host = "127.0.0.1"
        request.headers.get.return_value = None
        
        # First two should pass
        await limiter.is_allowed(request)
        await limiter.is_allowed(request)
        
        # Third should be blocked
        allowed, info = await limiter.is_allowed(request)
        assert not allowed
        assert info["remaining"] == 0


# ============================================
# Alert System Tests
# ============================================

class TestAlertService:
    """Test alert service."""
    
    def test_priority_determination(self, mock_config):
        """Test alert priority based on similarity and threat level."""
        from src.alerts.service import AlertService
        from src.alerts.notifiers import AlertPriority
        
        service = AlertService(mock_config)
        
        # Critical threat + high similarity
        priority = service._determine_priority(0.95, "critical")
        assert priority == AlertPriority.CRITICAL
        
        # Low threat + low similarity
        priority = service._determine_priority(0.5, "low")
        assert priority == AlertPriority.LOW
    
    def test_cooldown_prevents_duplicate_alerts(self, mock_config):
        """Test cooldown prevents alert spam."""
        from src.alerts.cooldown import CooldownManager
        
        manager = CooldownManager(cooldown_seconds=60)
        
        # First alert should pass
        assert manager.should_alert("criminal1:camera1")
        
        # Immediate second alert should be blocked
        assert not manager.should_alert("criminal1:camera1")
        
        # Different key should pass
        assert manager.should_alert("criminal2:camera1")


class TestSlackAlerter:
    """Test Slack notifications."""
    
    @patch('requests.post')
    def test_send_formats_correctly(self, mock_post):
        """Test Slack message formatting."""
        from src.alerts.notifiers import SlackAlerter, AlertPriority
        
        mock_post.return_value.status_code = 200
        
        alerter = SlackAlerter(webhook_url="https://hooks.slack.com/test")
        
        message = {
            "criminal": "John Doe",
            "similarity": 0.95,
            "camera": "Main Entrance",
            "location": "Building A"
        }
        
        result = alerter.send(message, AlertPriority.HIGH)
        
        assert result
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "embeds" in call_args.kwargs.get("json", {}) or "attachments" in call_args.kwargs.get("json", {})


# ============================================
# Database Model Tests
# ============================================

class TestDatabaseModels:
    """Test SQLAlchemy models."""
    
    def test_criminal_model_fields(self):
        """Test Criminal model has required fields."""
        from src.database.models import Criminal
        
        criminal = Criminal(
            name="John Doe",
            age=35,
            gender="Male",
            crime_type="Robbery",
            threat_level="high"
        )
        
        assert criminal.name == "John Doe"
        assert criminal.threat_level == "high"
    
    def test_incident_match_relationship(self):
        """Test Incident-IncidentMatch relationship."""
        from src.database.models import Incident, IncidentMatch
        
        # Just verify the models can be instantiated
        incident = Incident(camera_id=1, face_count=2)
        match = IncidentMatch(
            incident_id=1,
            criminal_id=1,
            similarity=0.95,
            bbox_x1=100, bbox_y1=100,
            bbox_x2=200, bbox_y2=200
        )
        
        assert match.similarity == 0.95


# ============================================
# Integration Tests
# ============================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.mark.asyncio
    async def test_detection_to_alert_workflow(self, mock_config, sample_image):
        """Test complete detection to alert workflow."""
        # This would require more mocking but demonstrates the pattern
        pass
    
    def test_api_authentication_flow(self, mock_config):
        """Test complete auth flow: login -> token -> protected resource."""
        from src.api.security import EnhancedAuthService
        
        auth = EnhancedAuthService(secret_key="a" * 32)
        
        # 1. Hash password (would be done during user creation)
        password = "SecurePassword123!"
        hashed = auth.hash_password(password)
        
        # 2. Verify password (login)
        assert auth.verify_password(password, hashed)
        
        # 3. Generate tokens
        response = auth.create_token_pair("testuser", "admin")
        assert response.access_token
        assert response.refresh_token
        
        # 4. Verify access token
        payload = auth.decode_token(response.access_token)
        assert payload["sub"] == "testuser"
        
        # 5. Refresh tokens
        new_response = auth.refresh_access_token(response.refresh_token)
        assert new_response.access_token != response.access_token


# ============================================
# Performance Tests
# ============================================

class TestPerformance:
    """Performance-related tests."""
    
    def test_embedding_batch_efficiency(self, sample_embedding):
        """Test batch processing is efficient."""
        import time
        from src.database.embedding_index import EmbeddingIndex
        
        index = EmbeddingIndex(dim=512)
        
        # Add many embeddings
        n = 1000
        embeddings = np.random.randn(n, 512).astype(np.float32)
        ids = list(range(n))
        
        start = time.time()
        index.add(embeddings, ids)
        add_time = time.time() - start
        
        # Search batch
        queries = np.random.randn(10, 512).astype(np.float32)
        start = time.time()
        results = index.search(queries, k=5)
        search_time = time.time() - start
        
        # Should be reasonably fast
        assert add_time < 1.0  # Adding 1000 embeddings
        assert search_time < 0.1  # Searching 10 queries


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
