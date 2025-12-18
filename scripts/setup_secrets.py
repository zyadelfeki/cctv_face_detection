#!/usr/bin/env python3
"""
Setup script for initializing secrets.
Generates secure random secrets and stores them in configured backend.
"""
import secrets
import string
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.secrets import SecretsManager, SecretsBackend
from loguru import logger


def generate_secret_key(length: int = 64) -> str:
    """Generate a cryptographically secure random key."""
    alphabet = string.ascii_letters + string.digits + string.punctuation
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def generate_jwt_secret() -> str:
    """Generate JWT secret."""
    return secrets.token_urlsafe(64)


def generate_encryption_key() -> str:
    """Generate 32-byte encryption key."""
    return secrets.token_urlsafe(32)


def setup_secrets_file():
    """Setup encrypted secrets file."""
    print("=== CCTV Face Detection - Secrets Setup ===")
    print()
    
    # Check if master key exists
    master_key = os.getenv("SECRETS_MASTER_KEY")
    if not master_key:
        print("Generating new master key...")
        master_key = generate_secret_key(64)
        print()
        print("⚠️  IMPORTANT: Save this master key securely!")
        print(f"SECRETS_MASTER_KEY={master_key}")
        print()
        print("Add this to your .env file and keep it secure!")
        print()
        
        # Set it for this session
        os.environ["SECRETS_MASTER_KEY"] = master_key
    
    # Initialize secrets manager
    manager = SecretsManager(backend=SecretsBackend.FILE)
    
    # Generate all secrets
    print("Generating secrets...")
    secrets_dict = {
        # Application secrets
        "SECRET_KEY": generate_secret_key(),
        "JWT_SECRET_KEY": generate_jwt_secret(),
        "ENCRYPTION_KEY": generate_encryption_key(),
        
        # Database credentials
        "DATABASE_PASSWORD": generate_secret_key(32),
        "REDIS_PASSWORD": generate_secret_key(32),
        
        # API keys (placeholders - replace with real values)
        "TWILIO_SID": "REPLACE_WITH_YOUR_TWILIO_SID",
        "TWILIO_TOKEN": "REPLACE_WITH_YOUR_TWILIO_TOKEN",
        "SMTP_PASSWORD": "REPLACE_WITH_YOUR_SMTP_PASSWORD",
        "WEBHOOK_SECRET": generate_secret_key(32),
        
        # Admin credentials
        "ADMIN_USERNAME": "admin",
        "ADMIN_PASSWORD": generate_secret_key(24),
    }
    
    # Save encrypted secrets
    manager.set_file_secret(secrets_dict, "config/secrets.enc")
    
    print()
    print("✅ Secrets generated and encrypted successfully!")
    print(f"📁 Secrets saved to: config/secrets.enc")
    print()
    print("🔐 Admin credentials:")
    print(f"   Username: {secrets_dict['ADMIN_USERNAME']}")
    print(f"   Password: {secrets_dict['ADMIN_PASSWORD']}")
    print()
    print("⚠️  Save these credentials securely!")
    print()
    
    # Create .env.production template
    env_template = f"""# Production Environment Variables
# DO NOT COMMIT THIS FILE TO VERSION CONTROL

# Secrets Backend Configuration
SECRETS_BACKEND=file
SECRETS_FILE_PATH=config/secrets.enc
SECRETS_MASTER_KEY={master_key}

# Application Configuration
ENVIRONMENT=production
DEBUG=false

# Database Configuration (credentials loaded from secrets)
DATABASE_URL=postgresql://cctv_user@localhost:5432/cctv_surveillance
REDIS_URL=redis://localhost:6379/0

# System Configuration
GPU_ENABLED=false
MODEL_DEVICE=cpu
MAX_WORKERS=4
MAX_CAMERAS=8

# Storage Paths
DATA_DIR=./data
LOGS_DIR=./logs
MODELS_DIR=./models
TEMP_DIR=./temp
"""
    
    env_file = Path(".env.production")
    env_file.write_text(env_template)
    print(f"📝 Created .env.production template")
    print()
    print("✅ Setup complete! Next steps:")
    print("   1. Review config/secrets.enc (encrypted)")
    print("   2. Update .env.production with your configuration")
    print("   3. Replace placeholder API keys in secrets")
    print("   4. NEVER commit .env.production or config/secrets.enc to git!")


def setup_aws_secrets():
    """Setup secrets in AWS Secrets Manager."""
    try:
        import boto3
    except ImportError:
        print("❌ boto3 not installed. Install with: pip install boto3")
        return
    
    print("=== AWS Secrets Manager Setup ===")
    print()
    region = input("AWS Region [us-east-1]: ").strip() or "us-east-1"
    secret_name = input("Secret name [cctv-face-detection]: ").strip() or "cctv-face-detection"
    
    # Generate secrets
    secrets_dict = {
        "SECRET_KEY": generate_secret_key(),
        "JWT_SECRET_KEY": generate_jwt_secret(),
        "ENCRYPTION_KEY": generate_encryption_key(),
        "DATABASE_PASSWORD": generate_secret_key(32),
        "REDIS_PASSWORD": generate_secret_key(32),
        "ADMIN_PASSWORD": generate_secret_key(24),
    }
    
    client = boto3.client('secretsmanager', region_name=region)
    
    try:
        response = client.create_secret(
            Name=secret_name,
            SecretString=str(secrets_dict)
        )
        print()
        print("✅ Secrets created in AWS Secrets Manager:")
        print(f"   ARN: {response['ARN']}")
        print(f"   Name: {secret_name}")
    except client.exceptions.ResourceExistsException:
        print(f"⚠️  Secret {secret_name} already exists. Updating...")
        client.update_secret(
            SecretId=secret_name,
            SecretString=str(secrets_dict)
        )
        print("✅ Secret updated successfully")


def main():
    print("Select secrets backend:")
    print("1. Encrypted File (recommended for development/small deployments)")
    print("2. AWS Secrets Manager (recommended for AWS deployments)")
    print("3. Environment Variables Only (least secure, for testing only)")
    
    choice = input("\nChoice [1]: ").strip() or "1"
    
    if choice == "1":
        setup_secrets_file()
    elif choice == "2":
        setup_aws_secrets()
    elif choice == "3":
        print()
        print("⚠️  Using environment variables only")
        print("Make sure to set all required secrets in .env file")
    else:
        print("Invalid choice")
        sys.exit(1)


if __name__ == "__main__":
    main()
