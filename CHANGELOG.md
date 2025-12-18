# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security Improvements (Phase 3: Input Validation & SQL Injection) - 2025-12-18

#### Added
- **Comprehensive input validation utilities** (`src/utils/validators.py`)
  - File upload validation (MIME type, size, dimensions, malicious content)
  - Image validation with cv2 and PIL verification
  - EXIF metadata stripping (removes GPS, camera info)
  - URL validation with SSRF prevention
  - RTSP URL validation for cameras (allows private IPs)
  - HTTP/HTTPS URL validation (blocks private IPs by default)
  - String sanitization (XSS pattern detection)
  - SQL injection pattern detection (defense in depth)
  - Path traversal prevention
  - Numeric range validation
  - Graceful fallback when python-magic unavailable
  
- **Enhanced API schemas** (`src/api/schemas.py`)
  - Pydantic field validators for all input fields
  - XSS prevention in string fields
  - RTSP URL validation for camera endpoints
  - Threat level and gender enumeration
  - Numeric range enforcement (confidence, similarity, age)
  - Custom validators for domain-specific fields
  - Proper error messages for validation failures
  
- **Comprehensive test suite** (`tests/test_validators.py` - 34/34 passing)
  - File upload validation tests
  - Image size and dimension tests
  - Corrupted image detection
  - URL validation and SSRF prevention tests
  - XSS prevention tests (script tags, event handlers, javascript:)
  - SQL injection pattern detection tests
  - Path traversal prevention tests
  - Attack scenario tests (polyglot XSS, Unicode bypass)
  - Integration tests for complete workflows

#### Changed
- **Dependencies**
  - Added `python-magic>=0.4.27` for MIME type detection
  - Added `python-magic-bin>=0.4.14` for Windows compatibility
  - MIME detection with content-based validation (not just extensions)

#### Fixed
- ✅ **FIXED**: No file upload validation (arbitrary file uploads)
- ✅ **FIXED**: Missing MIME type validation
- ✅ **FIXED**: No image dimension checks
- ✅ **FIXED**: EXIF metadata not stripped (privacy leak)
- ✅ **FIXED**: SSRF vulnerability in URL validation
- ✅ **FIXED**: No XSS prevention in API inputs
- ✅ **FIXED**: SQL injection detection missing
- ✅ **FIXED**: Path traversal vulnerability
- ✅ **FIXED**: No numeric range validation

#### Security
- **File Upload Security**
  - MIME type validated by content, not extension
  - File size limits enforced (10MB images, 100MB videos)
  - Image dimensions validated (32x32 to 8192x8192)
  - Malicious image detection with PIL verification
  - EXIF metadata automatically stripped
  
- **SSRF Prevention**
  - Blocks access to localhost (127.0.0.1, ::1)
  - Blocks private networks (10.0.0.0/8, 192.168.0.0/16, 172.16.0.0/12)
  - Blocks link-local addresses (169.254.0.0/16)
  - DNS rebinding protection
  - Configurable for camera URLs (allows private IPs)
  
- **XSS Prevention**
  - Script tag detection and blocking
  - Event handler detection (onclick, onerror, etc.)
  - javascript: protocol blocking
  - iframe/object/embed tag blocking
  - Polyglot XSS payload detection
  
- **Path Traversal Prevention**
  - ../ pattern detection
  - Absolute path blocking for sensitive directories
  - Base directory enforcement
  - Unicode normalization bypass prevention
  
- **SQL Injection Defense**
  - Pattern-based detection (defense in depth)
  - Warns on suspicious input
  - Note: Always use parameterized queries as primary defense

### Security Improvements (Phase 2: Exception Handling & Resource Leaks) - 2025-12-18

#### Added
- **Centralized resource manager** (`src/utils/resource_manager.py`)
  - Automatic cleanup on shutdown
  - Context managers for transactional resources
  - Memory tracking and limits (configurable via MAX_MEMORY_MB)
  - File handle management
  - Database connection pooling
  - Graceful degradation on resource exhaustion
  - Singleton pattern with thread-safe operations
  
- **Enhanced VideoStream** (`src/core/video_stream.py`)
  - Context manager support (`__enter__`, `__exit__`, `__del__`)
  - Automatic reconnection with exponential backoff
  - Thread-safe operations with locks
  - Exception handling for cv2.VideoCapture failures
  - Configurable reconnection attempts
  - Rate limiting on reconnection attempts
  
- **Comprehensive test suite** (`tests/test_resource_management.py` - 16/16 passing)
  - ResourceManager tests (singleton, cleanup, context managers)
  - FileManager tests (auto-cleanup, exception handling)
  - VideoStream thread-safety tests
  - Integration tests with multiple resource types
  - Stress tests with 100+ simultaneous resources

#### Changed
- **Dependencies**
  - Added `psutil>=5.9.0` for system resource monitoring
  - Required for memory tracking in ResourceManager

#### Fixed
- ✅ **FIXED**: VideoCapture resource leaks (streams never released)
- ✅ **FIXED**: File handles left open on exceptions
- ✅ **FIXED**: Database connections not properly pooled
- ✅ **FIXED**: No cleanup on application shutdown
- ✅ **FIXED**: Thread-unsafe video stream access
- ✅ **FIXED**: Infinite reconnection attempts on camera failures

#### Security
- Memory usage monitoring prevents DoS via resource exhaustion
- Automatic cleanup prevents file descriptor exhaustion
- Connection pooling prevents database connection exhaustion

### Security Improvements (Phase 1: Secrets Management) - 2025-12-18

#### Added
- **Comprehensive secrets management system** (`src/utils/secrets.py`)
  - Support for 5 backends: Environment, Encrypted File, AWS, Vault, Azure
  - Fernet encryption for file-based secrets (AES-128)
  - Automatic caching and validation
  - Global singleton pattern for easy access
  
- **Secrets setup tooling**
  - `scripts/setup_secrets.py` - Interactive secrets generator
  - `scripts/generate_docker_secrets.sh` - Docker secrets creation (Bash)
  - `scripts/generate_docker_secrets.ps1` - Docker secrets creation (PowerShell)
  - Cryptographically secure random key generation
  
- **Comprehensive documentation**
  - `docs/SECRETS_MANAGEMENT.md` - Full secrets management guide
  - Usage examples for all backends
  - Security best practices
  - Troubleshooting guide
  - Migration procedures
  
- **Test coverage**
  - `tests/test_secrets.py` - Comprehensive unit tests (10/10 passing)
  - Tests for all backends and edge cases
  - Validation and caching tests
  - `conftest.py` - Pytest configuration for proper imports

#### Changed
- **Docker Compose configuration** (`docker-compose.yml`)
  - Removed hardcoded passwords (CRITICAL SECURITY FIX)
  - Implemented Docker secrets for PostgreSQL and Grafana
  - Added health checks for all services
  - Added proper restart policies
  - Enabled read-only mounts for sensitive files
  
- **Environment configuration**
  - Updated `.gitignore` to prevent secrets from being committed
  - Protected `secrets/`, `config/secrets.enc`, `.env.production`
  
- **Dependencies**
  - Added `requirements-secrets.txt` for optional backends
  - Support for boto3, hvac, azure-identity, azure-keyvault-secrets
  - Updated `requirements.txt` for Python 3.11-3.13 compatibility
  - Fixed cryptography imports (PBKDF2 -> PBKDF2HMAC)

#### Security
- ✅ **FIXED**: Hardcoded database password in docker-compose.yml
- ✅ **FIXED**: Secrets visible in version control
- ✅ **IMPLEMENTED**: Encryption at rest for secrets
- ✅ **IMPLEMENTED**: Multi-backend secrets support
- ✅ **IMPLEMENTED**: Proper file permissions (400) for secrets

---

## 📊 **SECURITY AUDIT PROGRESS**

### ✅ **Completed Phases (3/10)**

| Phase | Status | Tests | Commits | Key Fixes |
|-------|--------|-------|---------|----------|
| **Phase 1: Secrets** | ✅ Complete | 10/10 | 15 | Encrypted secrets, multi-backend |
| **Phase 2: Resources** | ✅ Complete | 16/16 | 5 | Resource cleanup, memory tracking |
| **Phase 3: Input Validation** | ✅ Complete | 34/34 | 7 | XSS, SSRF, path traversal prevention |
| **TOTAL** | **60/60** ✅ | **27 commits** | **15 security fixes** |

### 🔄 **Remaining Phases (7/10)**

#### Phase 4: Authentication & Authorization
- [ ] JWT token validation improvements
- [ ] API key rotation mechanism
- [ ] Enhanced rate limiting
- [ ] Role-based access control (RBAC)
- [ ] Secure session management

#### Phase 5: Data Security
- [ ] Encryption at rest for biometric embeddings
- [ ] TLS/HTTPS certificate management
- [ ] Secure deletion of face images
- [ ] FAISS index encryption

#### Phase 6: Testing & CI/CD
- [ ] Increase test coverage to 80%+
- [ ] Add integration tests
- [ ] Restore automated testing pipeline
- [ ] Add security scanning (bandit, safety, semgrep)

#### Phase 7: Legal & Compliance
- [ ] GDPR compliance implementation
- [ ] Data Protection Impact Assessment (DPIA)
- [ ] Consent mechanism
- [ ] Audit logging

#### Phase 8: Production Hardening
- [ ] Kubernetes manifests
- [ ] Infrastructure as Code (Terraform)
- [ ] Backup automation
- [ ] Disaster recovery procedures

---

## [1.0.0] - 2024-10-31

### Added
- Initial release with face detection and recognition
- Multi-camera support
- Criminal database
- Alert system
- REST API
- Streamlit dashboard
- Docker containerization

### Known Issues (Pre-Security Audit)
- ~~Hardcoded credentials in docker-compose.yml~~ (FIXED in Phase 1)
- ~~Resource leaks in video streams~~ (FIXED in Phase 2)
- ~~Input validation missing~~ (FIXED in Phase 3)
- Missing encryption for sensitive data (Phase 5)
- Incomplete test coverage (Phase 6)
- No CI/CD pipeline (Phase 6)
- Missing GDPR compliance (Phase 7)
