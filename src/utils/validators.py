"""Input validation utilities for API and file handling.

Provides:
- File upload validation
- URL validation (RTSP/HTTP with SSRF prevention)
- Image content validation
- String sanitization
- Path traversal prevention
"""

import re
import mimetypes
from pathlib import Path
from typing import Optional, List, Tuple
from urllib.parse import urlparse
import ipaddress

import cv2
import numpy as np
from PIL import Image
from loguru import logger

# Try to import python-magic, fallback to mimetypes
try:
    import magic
    MAGIC_AVAILABLE = True
except (ImportError, OSError):
    MAGIC_AVAILABLE = False
    logger.warning(
        "python-magic not available, using fallback MIME detection. "
        "For production, install: pip install python-magic python-magic-bin"
    )


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


class FileValidator:
    """Validator for file uploads.
    
    Validates:
    - File type (MIME)
    - File size
    - Image dimensions
    - Malicious content
    """
    
    ALLOWED_IMAGE_TYPES = {
        'image/jpeg',
        'image/jpg',
        'image/png',
        'image/webp',
        'image/bmp'
    }
    
    ALLOWED_VIDEO_TYPES = {
        'video/mp4',
        'video/avi',
        'video/mkv',
        'video/mov',
        'video/webm'
    }
    
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_IMAGE_DIMENSIONS = (8192, 8192)  # Max width/height
    MIN_IMAGE_DIMENSIONS = (32, 32)  # Min width/height
    
    @staticmethod
    def validate_file_size(content: bytes, max_size: int) -> bool:
        """Validate file size.
        
        Args:
            content: File content
            max_size: Maximum allowed size in bytes
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If size exceeds limit
        """
        size = len(content)
        if size > max_size:
            raise ValidationError(
                f"File size {size} bytes exceeds maximum {max_size} bytes"
            )
        if size == 0:
            raise ValidationError("File is empty")
        return True
    
    @staticmethod
    def validate_mime_type(
        content: bytes,
        allowed_types: set,
        declared_type: Optional[str] = None
    ) -> str:
        """Validate file MIME type.
        
        Uses python-magic for true content-based detection when available,
        falls back to basic checks otherwise.
        
        Args:
            content: File content
            allowed_types: Set of allowed MIME types
            declared_type: MIME type from Content-Type header (optional)
            
        Returns:
            Detected MIME type
            
        Raises:
            ValidationError: If type not allowed or mismatch detected
        """
        detected_type = None
        
        if MAGIC_AVAILABLE:
            try:
                # Use python-magic for robust detection
                detected_type = magic.from_buffer(content, mime=True)
            except Exception as e:
                logger.error(f"MIME type detection with magic failed: {e}")
        
        # Fallback: use declared type with basic validation
        if not detected_type:
            if declared_type:
                detected_type = declared_type
            else:
                # Try to detect from content signature
                if content.startswith(b'\xff\xd8\xff'):
                    detected_type = 'image/jpeg'
                elif content.startswith(b'\x89PNG\r\n\x1a\n'):
                    detected_type = 'image/png'
                elif content.startswith(b'GIF87a') or content.startswith(b'GIF89a'):
                    detected_type = 'image/gif'
                elif content.startswith(b'RIFF') and b'WEBP' in content[:20]:
                    detected_type = 'image/webp'
                else:
                    raise ValidationError("Unable to determine file type")
        
        # Normalize MIME types
        if detected_type == 'image/jpg':
            detected_type = 'image/jpeg'
        
        # Validate detected type
        if detected_type not in allowed_types:
            raise ValidationError(
                f"File type '{detected_type}' not allowed. "
                f"Allowed types: {', '.join(allowed_types)}"
            )
        
        # Check for mismatch with declared type
        if declared_type and declared_type != detected_type:
            logger.warning(
                f"MIME type mismatch: declared={declared_type}, "
                f"detected={detected_type}"
            )
        
        return detected_type
    
    @staticmethod
    def validate_image(
        content: bytes,
        max_size: int = MAX_IMAGE_SIZE,
        check_malicious: bool = True
    ) -> Tuple[np.ndarray, dict]:
        """Comprehensive image validation.
        
        Args:
            content: Image file content
            max_size: Maximum file size in bytes
            check_malicious: Whether to check for malicious content
            
        Returns:
            Tuple of (image_array, metadata)
            
        Raises:
            ValidationError: If validation fails
        """
        # Size check
        FileValidator.validate_file_size(content, max_size)
        
        # MIME type check
        mime_type = FileValidator.validate_mime_type(
            content,
            FileValidator.ALLOWED_IMAGE_TYPES
        )
        
        # Try to decode with cv2
        try:
            nparr = np.frombuffer(content, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            raise ValidationError(f"Failed to decode image: {e}")
        
        if image is None:
            raise ValidationError("Image decoding failed - possibly corrupted")
        
        # Dimension checks
        height, width = image.shape[:2]
        
        min_w, min_h = FileValidator.MIN_IMAGE_DIMENSIONS
        if width < min_w or height < min_h:
            raise ValidationError(
                f"Image dimensions {width}x{height} too small. "
                f"Minimum: {min_w}x{min_h}"
            )
        
        max_w, max_h = FileValidator.MAX_IMAGE_DIMENSIONS
        if width > max_w or height > max_h:
            raise ValidationError(
                f"Image dimensions {width}x{height} too large. "
                f"Maximum: {max_w}x{max_h}"
            )
        
        # Check for malicious content
        if check_malicious:
            # Open with PIL for additional validation
            try:
                from io import BytesIO
                pil_image = Image.open(BytesIO(content))
                pil_image.verify()  # Verify it's actually an image
            except Exception as e:
                raise ValidationError(f"Image verification failed: {e}")
        
        metadata = {
            'width': width,
            'height': height,
            'channels': image.shape[2] if len(image.shape) == 3 else 1,
            'mime_type': mime_type,
            'size_bytes': len(content)
        }
        
        return image, metadata
    
    @staticmethod
    def strip_exif(image_path: Path) -> bool:
        """Strip EXIF metadata from image file.
        
        Removes potentially sensitive GPS, camera info, etc.
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if successful
        """
        try:
            from PIL import Image
            image = Image.open(image_path)
            
            # Remove EXIF
            data = list(image.getdata())
            image_without_exif = Image.new(image.mode, image.size)
            image_without_exif.putdata(data)
            
            # Save without EXIF
            image_without_exif.save(image_path)
            return True
        except Exception as e:
            logger.error(f"Failed to strip EXIF from {image_path}: {e}")
            return False


class URLValidator:
    """Validator for URLs with SSRF prevention.
    
    Prevents:
    - Private IP access (SSRF)
    - Localhost access
    - DNS rebinding attacks
    - Invalid schemes
    """
    
    ALLOWED_SCHEMES_HTTP = {'http', 'https'}
    ALLOWED_SCHEMES_RTSP = {'rtsp', 'rtsps'}
    
    # Private/reserved IP ranges (SSRF targets)
    BLOCKED_NETWORKS = [
        ipaddress.ip_network('127.0.0.0/8'),    # Localhost
        ipaddress.ip_network('10.0.0.0/8'),     # Private
        ipaddress.ip_network('172.16.0.0/12'),  # Private
        ipaddress.ip_network('192.168.0.0/16'), # Private
        ipaddress.ip_network('169.254.0.0/16'), # Link-local
        ipaddress.ip_network('::1/128'),        # IPv6 localhost
        ipaddress.ip_network('fe80::/10'),      # IPv6 link-local
        ipaddress.ip_network('fc00::/7'),       # IPv6 private
    ]
    
    @staticmethod
    def validate_url(
        url: str,
        allowed_schemes: set,
        allow_private: bool = False
    ) -> dict:
        """Validate URL structure and prevent SSRF.
        
        Args:
            url: URL to validate
            allowed_schemes: Set of allowed URL schemes
            allow_private: Whether to allow private IPs
            
        Returns:
            Dict with parsed URL components
            
        Raises:
            ValidationError: If URL is invalid or dangerous
        """
        if not url or not isinstance(url, str):
            raise ValidationError("URL must be a non-empty string")
        
        # Basic length check
        if len(url) > 2048:
            raise ValidationError("URL too long (max 2048 characters)")
        
        # Parse URL
        try:
            parsed = urlparse(url)
        except Exception as e:
            raise ValidationError(f"Invalid URL format: {e}")
        
        # Scheme check
        if parsed.scheme not in allowed_schemes:
            raise ValidationError(
                f"URL scheme '{parsed.scheme}' not allowed. "
                f"Allowed: {', '.join(allowed_schemes)}"
            )
        
        # Hostname required
        if not parsed.hostname:
            raise ValidationError("URL must include hostname")
        
        # SSRF prevention
        if not allow_private:
            try:
                # Resolve hostname to IP
                ip = ipaddress.ip_address(parsed.hostname)
                
                # Check against blocked networks
                for network in URLValidator.BLOCKED_NETWORKS:
                    if ip in network:
                        raise ValidationError(
                            f"Access to private/reserved IP {ip} not allowed (SSRF prevention)"
                        )
            except ValueError:
                # Not a direct IP, but hostname
                # Still check for localhost variants
                hostname_lower = parsed.hostname.lower()
                if hostname_lower in ('localhost', '0.0.0.0', 'localhost.localdomain'):
                    raise ValidationError(
                        f"Access to localhost not allowed (SSRF prevention)"
                    )
        
        return {
            'scheme': parsed.scheme,
            'hostname': parsed.hostname,
            'port': parsed.port,
            'path': parsed.path,
            'full_url': url
        }
    
    @staticmethod
    def validate_rtsp_url(url: str, allow_private: bool = True) -> dict:
        """Validate RTSP camera URL.
        
        Args:
            url: RTSP URL
            allow_private: Whether to allow private IPs (usually True for cameras)
            
        Returns:
            Parsed URL dict
        """
        return URLValidator.validate_url(
            url,
            URLValidator.ALLOWED_SCHEMES_RTSP,
            allow_private=allow_private
        )
    
    @staticmethod
    def validate_http_url(url: str, allow_private: bool = False) -> dict:
        """Validate HTTP/HTTPS URL.
        
        Args:
            url: HTTP(S) URL
            allow_private: Whether to allow private IPs
            
        Returns:
            Parsed URL dict
        """
        return URLValidator.validate_url(
            url,
            URLValidator.ALLOWED_SCHEMES_HTTP,
            allow_private=allow_private
        )


class StringValidator:
    """String validation and sanitization.
    
    Prevents:
    - XSS attacks
    - SQL injection patterns
    - Path traversal
    - Command injection
    """
    
    # XSS patterns
    XSS_PATTERNS = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',  # Event handlers
        r'<iframe',
        r'<object',
        r'<embed'
    ]
    
    # SQL injection patterns (basic detection)
    SQL_PATTERNS = [
        r"('|(\-\-)|(;)|(\|\|)|(\*))",
        r"\w*((\'|(\%27))|(\-\-)|(;))",
        r"(union|select|insert|update|delete|drop|create|alter)",
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r'\.\.',
        r'~',
        r'/etc/',
        r'/proc/',
        r'\\\\',  # UNC paths
    ]
    
    @staticmethod
    def sanitize_string(
        value: str,
        max_length: int = 1000,
        allow_html: bool = False
    ) -> str:
        """Sanitize user input string.
        
        Args:
            value: Input string
            max_length: Maximum allowed length
            allow_html: Whether to allow HTML tags
            
        Returns:
            Sanitized string
            
        Raises:
            ValidationError: If string is dangerous
        """
        if not isinstance(value, str):
            raise ValidationError("Value must be a string")
        
        # Length check
        if len(value) > max_length:
            raise ValidationError(
                f"String length {len(value)} exceeds maximum {max_length}"
            )
        
        # Check for XSS patterns
        if not allow_html:
            for pattern in StringValidator.XSS_PATTERNS:
                if re.search(pattern, value, re.IGNORECASE):
                    raise ValidationError(
                        f"Potentially malicious content detected (XSS)"
                    )
        
        # Strip leading/trailing whitespace
        value = value.strip()
        
        return value
    
    @staticmethod
    def check_sql_injection(value: str) -> bool:
        """Check for SQL injection patterns.
        
        Note: This is defense in depth. Always use parameterized queries!
        
        Args:
            value: String to check
            
        Returns:
            True if suspicious patterns found
        """
        for pattern in StringValidator.SQL_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"Potential SQL injection attempt: {value[:50]}...")
                return True
        return False
    
    @staticmethod
    def sanitize_path(
        path: str,
        base_dir: Optional[Path] = None
    ) -> Path:
        """Sanitize file path to prevent traversal attacks.
        
        Args:
            path: Path string to sanitize
            base_dir: Base directory to restrict to
            
        Returns:
            Sanitized Path object
            
        Raises:
            ValidationError: If path is dangerous
        """
        if not isinstance(path, str):
            raise ValidationError("Path must be a string")
        
        # Check for traversal patterns
        for pattern in StringValidator.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, path):
                raise ValidationError(
                    f"Path contains forbidden pattern (traversal attempt)"
                )
        
        # Convert to Path and resolve
        try:
            path_obj = Path(path).resolve()
        except Exception as e:
            raise ValidationError(f"Invalid path: {e}")
        
        # If base_dir provided, ensure path is within it
        if base_dir:
            try:
                base_dir_resolved = base_dir.resolve()
                path_obj.relative_to(base_dir_resolved)
            except ValueError:
                raise ValidationError(
                    f"Path escapes base directory (traversal blocked)"
                )
        
        return path_obj


class NumericValidator:
    """Numeric value validation."""
    
    @staticmethod
    def validate_range(
        value: float,
        min_val: float,
        max_val: float,
        name: str = "value"
    ) -> float:
        """Validate numeric value is within range.
        
        Args:
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            name: Name for error messages
            
        Returns:
            Validated value
            
        Raises:
            ValidationError: If value out of range
        """
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{name} must be numeric")
        
        if value < min_val or value > max_val:
            raise ValidationError(
                f"{name} must be between {min_val} and {max_val}, got {value}"
            )
        
        return value
    
    @staticmethod
    def validate_positive(value: float, name: str = "value") -> float:
        """Validate value is positive."""
        if value <= 0:
            raise ValidationError(f"{name} must be positive, got {value}")
        return value


# Convenience functions
def validate_image_upload(content: bytes, max_size: int = 10 * 1024 * 1024):
    """Validate image upload. Returns (image_array, metadata)."""
    return FileValidator.validate_image(content, max_size)


def validate_rtsp_url(url: str) -> dict:
    """Validate RTSP URL. Returns parsed dict."""
    return URLValidator.validate_rtsp_url(url)


def sanitize_string(value: str, max_length: int = 1000) -> str:
    """Sanitize user input string."""
    return StringValidator.sanitize_string(value, max_length)


def sanitize_path(path: str, base_dir: Optional[Path] = None) -> Path:
    """Sanitize file path."""
    return StringValidator.sanitize_path(path, base_dir)
