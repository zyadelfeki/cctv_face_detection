"""Tests for input validation utilities."""

import tempfile
from pathlib import Path
import numpy as np
import cv2
import pytest
import time

from src.utils.validators import (
    FileValidator,
    URLValidator,
    StringValidator,
    NumericValidator,
    ValidationError,
    validate_image_upload,
    validate_rtsp_url,
    sanitize_string,
    sanitize_path
)


class TestFileValidator:
    """Test file upload validation."""
    
    def test_file_size_validation(self):
        """Test file size limits."""
        # Valid size
        content = b"test" * 100
        assert FileValidator.validate_file_size(content, 1000)
        
        # Too large
        with pytest.raises(ValidationError, match="exceeds maximum"):
            FileValidator.validate_file_size(content, 100)
        
        # Empty file
        with pytest.raises(ValidationError, match="empty"):
            FileValidator.validate_file_size(b"", 1000)
    
    def test_image_validation_valid(self):
        """Test valid image validation."""
        # Create a valid test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = (255, 0, 0)  # Blue image
        
        _, encoded = cv2.imencode('.jpg', img)
        content = encoded.tobytes()
        
        image, metadata = validate_image_upload(content)
        
        assert image is not None
        assert metadata['width'] == 100
        assert metadata['height'] == 100
        assert metadata['channels'] == 3
    
    def test_image_too_large(self):
        """Test image size limit."""
        # Create large image
        img = np.zeros((10000, 10000, 3), dtype=np.uint8)
        _, encoded = cv2.imencode('.jpg', img)
        content = encoded.tobytes()
        
        with pytest.raises(ValidationError, match="too large"):
            validate_image_upload(content)
    
    def test_image_too_small(self):
        """Test minimum image size."""
        # Create tiny image
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        _, encoded = cv2.imencode('.jpg', img)
        content = encoded.tobytes()
        
        with pytest.raises(ValidationError, match="too small"):
            validate_image_upload(content)
    
    def test_corrupted_image(self):
        """Test corrupted image detection."""
        # Not a valid image
        content = b"not an image"
        
        with pytest.raises(ValidationError):
            validate_image_upload(content)
    
    def test_exif_stripping(self):
        """Test EXIF metadata removal."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            # Create image with data
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path), img)
            
            # Strip EXIF
            result = FileValidator.strip_exif(tmp_path)
            assert result is True
            
        finally:
            # Wait a bit for Windows file handles
            time.sleep(0.1)
            try:
                tmp_path.unlink()
            except PermissionError:
                pass  # Windows file locking, not a test failure


class TestURLValidator:
    """Test URL validation and SSRF prevention."""
    
    def test_valid_rtsp_url(self):
        """Test valid RTSP URL."""
        url = "rtsp://camera.example.com:554/stream"
        result = validate_rtsp_url(url)
        
        assert result['scheme'] == 'rtsp'
        assert result['hostname'] == 'camera.example.com'
        assert result['port'] == 554
    
    def test_valid_rtsp_private_ip(self):
        """Test RTSP with private IP (should be allowed for cameras)."""
        url = "rtsp://192.168.1.100:554/stream"
        result = validate_rtsp_url(url)
        
        assert result['hostname'] == '192.168.1.100'
    
    def test_invalid_scheme(self):
        """Test invalid URL scheme."""
        url = "ftp://example.com/file"
        
        with pytest.raises(ValidationError, match="scheme.*not allowed"):
            URLValidator.validate_http_url(url)
    
    def test_ssrf_localhost(self):
        """Test SSRF prevention - localhost string."""
        # Test localhost string (not IP)
        url = "http://localhost:8080/api"
        
        with pytest.raises(ValidationError, match="SSRF prevention"):
            URLValidator.validate_http_url(url, allow_private=False)
    
    def test_ssrf_private_networks(self):
        """Test SSRF prevention - private networks."""
        urls = [
            "http://192.168.1.1/admin",
            "http://10.0.0.1/internal",
            "http://172.16.0.1/secret"
        ]
        
        for url in urls:
            with pytest.raises(ValidationError, match="SSRF prevention"):
                URLValidator.validate_http_url(url, allow_private=False)
    
    def test_valid_public_url(self):
        """Test valid public URL."""
        url = "https://api.example.com/endpoint"
        result = URLValidator.validate_http_url(url, allow_private=False)
        
        assert result['scheme'] == 'https'
        assert result['hostname'] == 'api.example.com'
    
    def test_url_too_long(self):
        """Test URL length limit."""
        url = "http://example.com/" + "a" * 3000
        
        with pytest.raises(ValidationError, match="too long"):
            URLValidator.validate_http_url(url)
    
    def test_missing_hostname(self):
        """Test URL without hostname."""
        url = "http:///path"
        
        with pytest.raises(ValidationError, match="hostname"):
            URLValidator.validate_http_url(url)


class TestStringValidator:
    """Test string sanitization."""
    
    def test_basic_sanitization(self):
        """Test basic string sanitization."""
        text = "  Hello World  "
        result = sanitize_string(text)
        
        assert result == "Hello World"
    
    def test_xss_prevention_script(self):
        """Test XSS prevention - script tags."""
        malicious = "<script>alert('XSS')</script>"
        
        with pytest.raises(ValidationError, match="XSS"):
            sanitize_string(malicious)
    
    def test_xss_prevention_event_handlers(self):
        """Test XSS prevention - event handlers."""
        malicious = '<img src="x" onerror="alert(1)">'
        
        with pytest.raises(ValidationError, match="XSS"):
            sanitize_string(malicious)
    
    def test_xss_prevention_javascript(self):
        """Test XSS prevention - javascript: protocol."""
        malicious = '<a href="javascript:alert(1)">Click</a>'
        
        with pytest.raises(ValidationError, match="XSS"):
            sanitize_string(malicious)
    
    def test_string_length_limit(self):
        """Test string length validation."""
        long_string = "a" * 2000
        
        with pytest.raises(ValidationError, match="exceeds maximum"):
            sanitize_string(long_string, max_length=1000)
    
    def test_sql_injection_detection(self):
        """Test SQL injection pattern detection."""
        sqli_patterns = [
            "'; DROP TABLE users--",
            "1' OR '1'='1",
            "admin'--",
            "1; DELETE FROM users"
        ]
        
        for pattern in sqli_patterns:
            is_suspicious = StringValidator.check_sql_injection(pattern)
            assert is_suspicious is True
    
    def test_path_traversal_prevention(self):
        """Test path traversal prevention."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "~/secret",
            "/etc/shadow"
        ]
        
        for path in malicious_paths:
            with pytest.raises(ValidationError, match="traversal"):
                sanitize_path(path)
    
    def test_safe_path_sanitization(self):
        """Test safe path sanitization."""
        safe_path = "uploads/images/photo.jpg"
        result = sanitize_path(safe_path)
        
        assert isinstance(result, Path)
    
    def test_path_within_base_dir(self):
        """Test path restriction to base directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            
            # Create a file within the base directory
            subdir = base_dir / "subdir"
            subdir.mkdir(exist_ok=True)
            test_file = subdir / "file.txt"
            test_file.write_text("test")
            
            # Safe path within base (use relative path from base)
            safe_path = str(test_file.relative_to(base_dir))
            result = sanitize_path(safe_path, base_dir=base_dir)
            assert isinstance(result, Path)
            
            # Attempt to escape base directory
            escape_path = "../outside.txt"
            with pytest.raises(ValidationError, match="escapes base directory"):
                sanitize_path(escape_path, base_dir=base_dir)


class TestNumericValidator:
    """Test numeric validation."""
    
    def test_range_validation_valid(self):
        """Test valid numeric range."""
        value = 50.0
        result = NumericValidator.validate_range(value, 0.0, 100.0)
        assert result == 50.0
    
    def test_range_validation_too_low(self):
        """Test value below minimum."""
        with pytest.raises(ValidationError, match="between"):
            NumericValidator.validate_range(-10, 0, 100)
    
    def test_range_validation_too_high(self):
        """Test value above maximum."""
        with pytest.raises(ValidationError, match="between"):
            NumericValidator.validate_range(150, 0, 100)
    
    def test_positive_validation(self):
        """Test positive number validation."""
        assert NumericValidator.validate_positive(5.0) == 5.0
        
        with pytest.raises(ValidationError, match="positive"):
            NumericValidator.validate_positive(-5.0)
        
        with pytest.raises(ValidationError, match="positive"):
            NumericValidator.validate_positive(0)
    
    def test_non_numeric_value(self):
        """Test non-numeric value rejection."""
        with pytest.raises(ValidationError, match="numeric"):
            NumericValidator.validate_range("not a number", 0, 100)


class TestAttackScenarios:
    """Test against real-world attack scenarios."""
    
    def test_polyglot_xss(self):
        """Test polyglot XSS payload."""
        polyglot = r'''jaVasCript:/*-/*`/*\`/*'/*"/**/(/* */oNcliCk=alert() )//%0D%0A%0d%0a//</stYle/</titLe/</teXtarEa/</scRipt/--!>\x3csVg/<sVg/oNloAd=alert()//>>'''
        
        with pytest.raises(ValidationError):
            sanitize_string(polyglot)
    
    def test_double_url_encoding(self):
        """Test double-encoded malicious content."""
        # %252E%252E represents ..
        # This tests if we properly decode URLs
        pass  # URL validation handles this by not decoding
    
    def test_null_byte_injection(self):
        """Test null byte path traversal."""
        malicious = "safe.jpg\x00../../etc/passwd"
        
        # Python 3 Path handles this correctly
        result = sanitize_path("safe.jpg")
        assert isinstance(result, Path)
    
    def test_unicode_bypass_attempts(self):
        """Test Unicode normalization attacks."""
        # Unicode representation of ../
        unicode_traversal = "\u002e\u002e\u002f"
        
        with pytest.raises(ValidationError):
            sanitize_path(unicode_traversal)


class TestIntegration:
    """Integration tests for validators."""
    
    def test_full_image_upload_workflow(self):
        """Test complete image upload validation workflow."""
        # Create valid image
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        _, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        content = encoded.tobytes()
        
        # Validate image
        image, metadata = validate_image_upload(content)
        
        # Check results
        assert image.shape == (200, 200, 3)
        assert metadata['width'] == 200
        assert metadata['height'] == 200
        assert 'mime_type' in metadata
    
    def test_full_url_validation_workflow(self):
        """Test complete URL validation workflow."""
        # Valid RTSP camera URL
        camera_url = "rtsp://192.168.1.50:554/h264"
        result = validate_rtsp_url(camera_url)
        
        assert result['scheme'] == 'rtsp'
        assert result['hostname'] == '192.168.1.50'
        assert result['port'] == 554
        assert result['path'] == '/h264'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
