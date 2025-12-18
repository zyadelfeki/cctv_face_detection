# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  
- **Comprehensive test suite** (`tests/test_resource_management.py`)
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

### Remaining Work

#### Phase 3: Input Validation & SQL Injection (Next)
- [ ] API input validation with Pydantic
- [ ] SQL injection prevention (parameterized queries)
- [ ] File upload validation (type, size, malicious content)
- [ ] RTSP URL validation (SSRF prevention)
- [ ] Rate limiting on API endpoints
- [ ] XSS prevention in dashboard

#### Phase 4: Authentication & Authorization
- [ ] Implement proper JWT token validation
- [ ] Add API key rotation mechanism
- [ ] Implement rate limiting on sensitive endpoints
- [ ] Add role-based access control (RBAC)
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
- Missing encryption for sensitive data
- Incomplete test coverage
- No CI/CD pipeline
- Missing GDPR compliance
