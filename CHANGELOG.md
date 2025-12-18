# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security Improvements (Phase 5: Data Security) - 2025-12-18

#### Added
- **Biometric Data Encryption** (`src/utils/encryption.py`)
  - AES-256-GCM encryption (NIST standard)
  - ChaCha20-Poly1305 alternative cipher
  - Authenticated Encryption with Associated Data (AEAD)
  - Per-record unique nonces (96-bit)
  - PBKDF2 key derivation from passwords (100k iterations)
  - Numpy embedding encryption with metadata preservation
  - Batch encryption/decryption operations
  - Serialization format with version control
  - Key rotation support
  - Master key fingerprinting
  - Encrypted embedding store (drop-in replacement)
  
- **Secure File Deletion** (`src/utils/secure_delete.py`)
  - DOD 5220.22-M standard (7-pass)
  - Gutmann method (35-pass for maximum security)
  - Simple 3-pass deletion (0x00, 0xFF, random)
  - Directory deletion with recursion
  - Free space wiping
  - Metadata removal
  - Secure face image deletion helpers
  - Configurable deletion methods
  
- **TLS/HTTPS Certificate Management** (`src/utils/tls_manager.py`)
  - Self-signed certificate generation (RSA 2048/4096)
  - Let's Encrypt integration via certbot
  - Automatic certificate renewal (configurable threshold)
  - Certificate validation and expiration checking
  - Private key protection (0o600 permissions)
  - Certificate fingerprinting (SHA-256)
  - Certificate rotation support
  - Multi-certificate management
  - Domain validation with Subject Alternative Names
  - X.509 certificate parsing
  
- **Encrypted FAISS Index** (`src/utils/encrypted_faiss.py`)
  - Transparent encryption wrapper for FAISS
  - AES-256-GCM for vector data
  - Encrypted ID mappings
  - Backward compatible with plain FAISS
  - Supports IVF, PQ, HNSW indexes
  - Automatic encryption on save/load
  - In-memory plaintext operations (no performance penalty)
  - EncryptedFaceDatabase drop-in replacement
  - Cosine similarity search
  - Training and indexing support
  
- **Comprehensive test suite** (`tests/test_data_security_phase5.py` - 22/22 passing)
  - Biometric encryption tests (AES, ChaCha20)
  - Key generation and derivation tests
  - Authenticated encryption tests
  - Embedding encryption roundtrip tests
  - Batch operation tests
  - Serialization/deserialization tests
  - Secure deletion tests (simple, DOD, Gutmann)
  - TLS certificate generation tests
  - Certificate validation tests
  - Private key permission tests
  - End-to-end encryption workflow tests
  - Performance benchmarks (100 embeddings < 1s)

#### Changed
- **Dependencies**
  - Added `cryptography>=42.0.0` for encryption and TLS
  - AEAD ciphers: AES-256-GCM, ChaCha20-Poly1305
  - X.509 certificate support
  - PBKDF2 key derivation

#### Fixed
- ✅ **IMPLEMENTED**: Biometric embeddings encrypted at rest
- ✅ **IMPLEMENTED**: TLS certificate management
- ✅ **IMPLEMENTED**: Secure face image deletion
- ✅ **IMPLEMENTED**: Encrypted FAISS indexes
- ✅ **FIXED**: No encryption for sensitive biometric data
- ✅ **FIXED**: Face images stored in plaintext
- ✅ **FIXED**: FAISS indexes unencrypted on disk
- ✅ **FIXED**: No secure deletion of biometric data
- ✅ **FIXED**: Manual TLS certificate management
- ✅ **FIXED**: No certificate rotation
- ✅ **FIXED**: Missing HTTPS support

#### Security
- **Encryption Security**
  - NIST-approved AES-256-GCM cipher
  - Authenticated encryption prevents tampering
  - Per-record unique nonces prevent replay attacks
  - AEAD protects metadata integrity
  - 256-bit keys (128-bit security level)
  - Constant-time operations
  - Master key never stored in plaintext
  
- **Secure Deletion**
  - DOD 5220.22-M standard compliance
  - Multiple overwrite passes
  - Random data in final passes
  - File truncation before deletion
  - Sync to disk (fsync) after each pass
  - Prevents data recovery forensics
  
- **TLS Security**
  - Strong RSA keys (2048/4096 bit)
  - SHA-256 certificate signatures
  - Private keys protected (owner-only permissions)
  - Automatic renewal prevents expiration
  - Let's Encrypt for production certificates
  - Certificate validation and verification
  
- **FAISS Encryption**
  - Vector embeddings encrypted at rest
  - ID mappings encrypted separately
  - No performance impact on search (in-memory plaintext)
  - Transparent encryption/decryption
  - Protects against disk access attacks

### Security Improvements (Phase 4: Authentication & Authorization) - 2025-12-18

#### Added
- **API Key Management System** (`src/utils/api_keys.py`)
  - Cryptographically secure key generation (32 bytes, URL-safe)
  - Automatic key rotation with configurable intervals (90 days default)
  - Key versioning and grace periods during rotation
  - Usage tracking and analytics
  - Key expiration and revocation
  - Rate limiting per API key
  - Concurrent key limits per service
  - Audit logging for all key operations
  - Key prefix format: `cctvfd_[random]`
  
- **Role-Based Access Control (RBAC)** (`src/utils/rbac.py`)
  - Fine-grained permissions beyond simple roles
  - 4 default roles: viewer, operator, analyst, admin
  - Permission inheritance hierarchy
  - 40+ granular permissions across:
    - Camera operations (view, create, update, delete, control)
    - Criminal database (view, create, update, delete, search)
    - Incident management (view, create, update, delete, export)
    - Analytics and reporting
    - User management
    - System configuration
  - Resource-level ownership checking
  - Custom role creation
  - Dynamic permission evaluation
  - FastAPI integration with decorators
  
- **Secure Session Management** (`src/utils/sessions.py`)
  - Cryptographically secure session IDs (32 bytes)
  - Automatic session expiration (8 hours default)
  - Idle timeout (30 minutes default)
  - Session activity tracking
  - Concurrent session limits per user
  - Session hijacking detection:
    - IP address change monitoring
    - User agent validation
    - Suspicious activity flagging
  - Session refresh mechanism
  - Bulk session termination
  - Session cleanup and garbage collection
  - Comprehensive session statistics
  
- **Comprehensive test suite** (`tests/test_auth_phase4.py` - 24/24 passing)
  - API key generation and validation tests
  - Key rotation and versioning tests
  - Key revocation tests
  - RBAC role and permission tests
  - Permission inheritance tests
  - Session creation and validation tests
  - Session expiration tests
  - IP change detection tests
  - Concurrent session limit tests
  - Integration tests combining all systems
  - Security boundary tests

#### Changed
- **Dependencies**
  - Added `fastapi>=0.125.0` for RBAC decorators
  - Added `starlette>=0.40.0` for request handling

#### Fixed
- ✅ **IMPLEMENTED**: API key rotation mechanism
- ✅ **IMPLEMENTED**: Fine-grained RBAC system
- ✅ **IMPLEMENTED**: Secure session management
- ✅ **FIXED**: No session hijacking prevention
- ✅ **FIXED**: Unlimited concurrent sessions per user
- ✅ **FIXED**: No API key expiration
- ✅ **FIXED**: Simple role system without permissions
- ✅ **FIXED**: No session activity tracking

#### Security
- **API Key Security**
  - Keys stored as SHA-256 hashes, never in plaintext
  - Constant-time comparison prevents timing attacks
  - Automatic rotation prevents long-lived credentials
  - Grace periods allow zero-downtime rotation
  - Usage tracking enables anomaly detection
  
- **RBAC Security**
  - Principle of least privilege enforced
  - Permission checks at function level
  - Inheritance prevents permission drift
  - Resource ownership enforced
  - Audit logging for permission denials
  
- **Session Security**
  - Cryptographically random session IDs
  - Session fixation prevention
  - Session hijacking detection and prevention
  - Automatic cleanup prevents session table growth
  - IP and user-agent validation
  - Configurable security policies

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

### ✅ **Completed Phases (5/10)**

| Phase | Status | Tests | Commits | Key Achievements |
|-------|--------|-------|---------|------------------|
| **Phase 1: Secrets** | ✅ Complete | 10/10 | 15 | Encrypted secrets, multi-backend |
| **Phase 2: Resources** | ✅ Complete | 16/16 | 5 | Resource cleanup, memory tracking |
| **Phase 3: Input Validation** | ✅ Complete | 34/34 | 9 | XSS, SSRF, path traversal prevention |
| **Phase 4: Auth & Authorization** | ✅ Complete | 24/24 | 6 | API keys, RBAC, sessions |
| **Phase 5: Data Security** | ✅ Complete | 22/22 | 8 | Encryption at rest, secure deletion, TLS |
| **TOTAL** | **106/106** ✅ | **43 commits** | **35 security fixes** |

### 🔄 **Remaining Phases (5/10)**

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

#### Phase 9: Monitoring & Alerting
- [ ] Security event monitoring
- [ ] Anomaly detection
- [ ] Alert escalation procedures
- [ ] Security dashboards

#### Phase 10: Documentation & Training
- [ ] Security runbooks
- [ ] Incident response procedures
- [ ] Security training materials
- [ ] API security documentation

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
- ~~Weak authentication system~~ (FIXED in Phase 4)
- ~~Missing encryption for sensitive data~~ (FIXED in Phase 5)
- Incomplete test coverage (Phase 6)
- No CI/CD pipeline (Phase 6)
- Missing GDPR compliance (Phase 7)
