# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security Improvements (Phase 1: Secrets Management) - 2025-12-18

#### Added
- **Comprehensive secrets management system** (`src/utils/secrets.py`)
  - Support for 5 backends: Environment, Encrypted File, AWS, Vault, Azure
  - Fernet encryption for file-based secrets (AES-128)
  - Automatic caching and validation
  - Global singleton pattern for easy access
  
- **Secrets setup tooling**
  - `scripts/setup_secrets.py` - Interactive secrets generator
  - `scripts/generate_docker_secrets.sh` - Docker secrets creation
  - Cryptographically secure random key generation
  
- **Comprehensive documentation**
  - `docs/SECRETS_MANAGEMENT.md` - Full secrets management guide
  - Usage examples for all backends
  - Security best practices
  - Troubleshooting guide
  - Migration procedures
  
- **Test coverage**
  - `tests/test_secrets.py` - Comprehensive unit tests
  - Tests for all backends and edge cases
  - Validation and caching tests

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

#### Security
- ✅ **FIXED**: Hardcoded database password in docker-compose.yml
- ✅ **FIXED**: Secrets visible in version control
- ✅ **IMPLEMENTED**: Encryption at rest for secrets
- ✅ **IMPLEMENTED**: Multi-backend secrets support
- ✅ **IMPLEMENTED**: Proper file permissions (400) for secrets

### Remaining Work

#### Phase 2: Authentication & Authorization (Planned)
- [ ] Implement proper JWT token validation
- [ ] Add API key rotation mechanism
- [ ] Implement rate limiting on sensitive endpoints
- [ ] Add role-based access control (RBAC)
- [ ] Secure session management

#### Phase 3: Data Security (Planned)
- [ ] Encryption at rest for biometric embeddings
- [ ] TLS/HTTPS certificate management
- [ ] Secure deletion of face images
- [ ] FAISS index encryption

#### Phase 4: Input Validation (Planned)
- [ ] File type validation on uploads
- [ ] Size limits enforcement
- [ ] EXIF data stripping
- [ ] RTSP URL validation (SSRF prevention)

#### Phase 5: Code Quality (Planned)
- [ ] Improve exception handling
- [ ] Fix resource leaks
- [ ] Resolve race conditions
- [ ] Add proper logging

#### Phase 6: Testing (Planned)
- [ ] Increase test coverage to 80%+
- [ ] Add integration tests
- [ ] Add load testing
- [ ] Add chaos engineering tests

#### Phase 7: CI/CD (Planned)
- [ ] Restore automated testing pipeline
- [ ] Implement deployment automation
- [ ] Add security scanning (bandit, safety, semgrep)
- [ ] Implement blue-green deployment

#### Phase 8: Legal & Compliance (Planned)
- [ ] GDPR compliance implementation
- [ ] Data Protection Impact Assessment (DPIA)
- [ ] Consent mechanism
- [ ] Data subject rights (access, erasure, portability)
- [ ] Audit logging

#### Phase 9: Monitoring & Observability (Planned)
- [ ] Prometheus dashboards
- [ ] Distributed tracing
- [ ] Structured logging
- [ ] Alerting rules
- [ ] SLO/SLI definitions

#### Phase 10: Production Hardening (Planned)
- [ ] Kubernetes manifests
- [ ] Infrastructure as Code (Terraform)
- [ ] Backup automation
- [ ] Disaster recovery procedures
- [ ] Capacity planning

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
- Hardcoded credentials in docker-compose.yml (FIXED in latest)
- Missing encryption for sensitive data
- Incomplete test coverage
- No CI/CD pipeline
- Missing GDPR compliance
