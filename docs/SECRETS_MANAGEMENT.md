# Secrets Management

Comprehensive guide for managing secrets in the CCTV Face Detection system.

## Overview

The system supports multiple secrets backends:

1. **Environment Variables** (default) - For development/testing
2. **Encrypted File** - For production deployments without cloud
3. **AWS Secrets Manager** - For AWS cloud deployments
4. **HashiCorp Vault** - For enterprise deployments
5. **Azure Key Vault** - For Azure cloud deployments

## Quick Start

### Development (Environment Variables)

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your values
vi .env

# Run the application
python main.py
```

### Production (Encrypted File)

```bash
# Generate secrets
python scripts/setup_secrets.py
# Select option 1 (Encrypted File)

# This creates:
# - config/secrets.enc (encrypted secrets)
# - .env.production (configuration template)

# Generate Docker secrets
chmod +x scripts/generate_docker_secrets.sh
./scripts/generate_docker_secrets.sh

# Deploy
docker-compose up -d
```

## Backends

### 1. Environment Variables

**Use Case:** Development, testing, simple deployments

**Setup:**
```bash
export SECRET_KEY="your-secret-key"
export JWT_SECRET_KEY="your-jwt-secret"
export DATABASE_PASSWORD="your-db-password"
```

**Pros:**
- Simple setup
- No additional dependencies
- Works everywhere

**Cons:**
- Less secure
- Visible in process list
- No encryption at rest

### 2. Encrypted File

**Use Case:** Production deployments without cloud infrastructure

**Setup:**
```bash
# Generate master key and secrets
python scripts/setup_secrets.py

# Set master key in environment
export SECRETS_MASTER_KEY="<your-master-key>"
export SECRETS_BACKEND="file"
export SECRETS_FILE_PATH="config/secrets.enc"
```

**Pros:**
- Encrypted at rest (Fernet/AES-128)
- No cloud dependencies
- Easy backup and restore

**Cons:**
- Master key must be protected
- File must be deployed with application

**Security Notes:**
- Master key should be stored separately (env var, systemd, etc.)
- Secrets file should have 400 permissions
- Use different master keys for different environments

### 3. AWS Secrets Manager

**Use Case:** AWS cloud deployments

**Setup:**
```bash
# Install AWS SDK
pip install -r requirements-secrets.txt

# Configure AWS credentials
aws configure

# Generate and upload secrets
python scripts/setup_secrets.py
# Select option 2 (AWS Secrets Manager)

# Configure application
export SECRETS_BACKEND="aws_secrets_manager"
export AWS_REGION="us-east-1"
```

**Access via IAM Role (recommended):**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ],
      "Resource": "arn:aws:secretsmanager:*:*:secret:cctv-face-detection-*"
    }
  ]
}
```

**Pros:**
- Fully managed by AWS
- Automatic rotation support
- Audit logging via CloudTrail
- Fine-grained IAM permissions

**Cons:**
- AWS-specific
- Additional cost (~$0.40/month per secret)
- Network latency

### 4. HashiCorp Vault

**Use Case:** Enterprise deployments, multi-cloud

**Setup:**
```bash
# Install Vault client
pip install -r requirements-secrets.txt

# Configure Vault
export VAULT_ADDR="https://vault.example.com:8200"
export VAULT_TOKEN="your-vault-token"
export SECRETS_BACKEND="hashicorp_vault"
export VAULT_MOUNT_POINT="secret"
export VAULT_SECRET_PATH="cctv"
```

**Pros:**
- Platform-agnostic
- Advanced security features
- Dynamic secrets support
- Built-in encryption

**Cons:**
- Requires Vault infrastructure
- More complex setup
- Learning curve

### 5. Azure Key Vault

**Use Case:** Azure cloud deployments

**Setup:**
```bash
# Install Azure SDK
pip install -r requirements-secrets.txt

# Login to Azure
az login

# Configure application
export SECRETS_BACKEND="azure_keyvault"
export AZURE_KEYVAULT_URL="https://your-vault.vault.azure.net/"
```

**Pros:**
- Native Azure integration
- Managed identity support
- Compliance certifications

**Cons:**
- Azure-specific
- Additional cost

## Required Secrets

### Application Secrets
```bash
SECRET_KEY              # Django/Flask secret key (64 chars)
JWT_SECRET_KEY          # JWT signing key (64 chars)
ENCRYPTION_KEY          # Data encryption key (32 chars)
```

### Database Credentials
```bash
DATABASE_PASSWORD       # PostgreSQL password
REDIS_PASSWORD          # Redis password
```

### External Services
```bash
TWILIO_SID             # Twilio Account SID (optional)
TWILIO_TOKEN           # Twilio Auth Token (optional)
SMTP_PASSWORD          # Email SMTP password (optional)
WEBHOOK_SECRET         # Webhook verification secret
```

### Admin Credentials
```bash
ADMIN_USERNAME         # Initial admin username
ADMIN_PASSWORD         # Initial admin password
```

## Usage in Code

### Basic Usage
```python
from src.utils.secrets import get_secret

# Get a secret
api_key = get_secret("API_KEY")

# Get with default
api_key = get_secret("API_KEY", "default_value")
```

### Advanced Usage
```python
from src.utils.secrets import get_secrets_manager, SecretsBackend

# Get manager instance
manager = get_secrets_manager()

# Get JSON secret
config = manager.get_secret_json("APP_CONFIG")

# Validate required secrets
required = ["SECRET_KEY", "DATABASE_PASSWORD", "JWT_SECRET_KEY"]
if not manager.validate_required_secrets(required):
    raise RuntimeError("Missing required secrets")
```

### Custom Backend
```python
from src.utils.secrets import SecretsManager, SecretsBackend

# Use specific backend
manager = SecretsManager(backend=SecretsBackend.AWS)
secret = manager.get_secret("MY_SECRET")
```

## Docker Integration

### Docker Secrets (Swarm Mode)
```yaml
services:
  app:
    secrets:
      - db_password
    environment:
      - DATABASE_PASSWORD_FILE=/run/secrets/db_password

secrets:
  db_password:
    external: true
```

### Docker Compose Secrets
```yaml
secrets:
  db_password:
    file: ./secrets/db_password.txt
```

## Security Best Practices

### DO
✅ Use encrypted file backend for production  
✅ Rotate secrets regularly (every 90 days)  
✅ Use different secrets per environment  
✅ Limit secret access with IAM/RBAC  
✅ Enable audit logging  
✅ Use managed identity when possible  
✅ Set proper file permissions (400/600)  
✅ Store master keys separately  

### DON'T

❌ Commit secrets to version control  
❌ Hard-code secrets in code  
❌ Share secrets via email/chat  
❌ Use same secrets across environments  
❌ Log secrets (even debug logs)  
❌ Store secrets in container images  
❌ Use weak passwords  

## Troubleshooting

### "SECRETS_MASTER_KEY not set"

**Problem:** File backend requires master key

**Solution:**
```bash
export SECRETS_MASTER_KEY="your-master-key-here"
# Or add to .env.production
```

### "Failed to decrypt secrets file"

**Problem:** Wrong master key or corrupted file

**Solution:**
1. Verify master key is correct
2. Re-generate secrets if needed:
   ```bash
   python scripts/setup_secrets.py
   ```

### "Secret not found in AWS Secrets Manager"

**Problem:** Secret doesn't exist or IAM permissions

**Solution:**
1. Verify secret exists: `aws secretsmanager describe-secret --secret-id cctv-face-detection`
2. Check IAM permissions
3. Verify AWS_REGION is correct

### "Vault authentication failed"

**Problem:** Invalid token or expired

**Solution:**
```bash
# Re-authenticate
vault login

# Verify token
vault token lookup
```

## Secret Rotation

### Manual Rotation

```bash
# 1. Generate new secret
python -c "import secrets; print(secrets.token_urlsafe(64))"

# 2. Update in secrets backend
# For file backend:
python scripts/setup_secrets.py

# For AWS:
aws secretsmanager update-secret --secret-id cctv-face-detection \
  --secret-string '{"SECRET_KEY":"new-value"}'

# 3. Restart application
docker-compose restart app
```

### Automated Rotation (AWS)

```bash
# Enable automatic rotation
aws secretsmanager rotate-secret \
  --secret-id cctv-face-detection \
  --rotation-lambda-arn arn:aws:lambda:region:account:function:rotation-function \
  --rotation-rules AutomaticallyAfterDays=90
```

## Migration Between Backends

### From Environment to Encrypted File

```bash
# 1. Export current secrets
env | grep -E '^(SECRET_|DATABASE_|JWT_)' > current_secrets.txt

# 2. Run setup
python scripts/setup_secrets.py

# 3. Manually update config/secrets.enc with values from step 1

# 4. Update .env.production
export SECRETS_BACKEND=file
```

### From File to AWS

```python
# migration_script.py
from src.utils.secrets import SecretsManager, SecretsBackend
import boto3
import json

# Read from file
file_manager = SecretsManager(backend=SecretsBackend.FILE)
secrets_dict = {
    "SECRET_KEY": file_manager.get_secret("SECRET_KEY"),
    "JWT_SECRET_KEY": file_manager.get_secret("JWT_SECRET_KEY"),
    # ... add all secrets
}

# Write to AWS
client = boto3.client('secretsmanager', region_name='us-east-1')
client.create_secret(
    Name='cctv-face-detection',
    SecretString=json.dumps(secrets_dict)
)
```

## References

- [AWS Secrets Manager Best Practices](https://docs.aws.amazon.com/secretsmanager/latest/userguide/best-practices.html)
- [HashiCorp Vault Documentation](https://developer.hashicorp.com/vault/docs)
- [Azure Key Vault Security](https://learn.microsoft.com/en-us/azure/key-vault/general/security-features)
- [OWASP Secrets Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
