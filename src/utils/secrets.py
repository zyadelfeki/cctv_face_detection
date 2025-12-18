"""
Secrets management with multiple backend support.
Supports: Environment variables, AWS Secrets Manager, HashiCorp Vault, Azure Key Vault
"""
from __future__ import annotations
import os
import json
import base64
from typing import Optional, Dict, Any
from pathlib import Path
from enum import Enum
from loguru import logger
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2


class SecretsBackend(str, Enum):
    ENV = "environment"
    AWS = "aws_secrets_manager"
    VAULT = "hashicorp_vault"
    AZURE = "azure_keyvault"
    FILE = "encrypted_file"


class SecretsManager:
    """Centralized secrets management with multiple backend support."""
    
    def __init__(self, backend: SecretsBackend = SecretsBackend.ENV):
        self.backend = backend
        self._cache: Dict[str, Any] = {}
        self._cipher: Optional[Fernet] = None
        
        if backend == SecretsBackend.FILE:
            self._initialize_file_encryption()
    
    def _initialize_file_encryption(self):
        """Initialize Fernet cipher for file-based secrets."""
        master_key = os.getenv("SECRETS_MASTER_KEY")
        if not master_key:
            raise ValueError("SECRETS_MASTER_KEY environment variable required for file backend")
        
        # Derive encryption key from master key
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"cctv_face_detection_salt",  # In production, use random salt stored separately
            iterations=480000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        self._cipher = Fernet(key)
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Retrieve secret from configured backend.
        
        Args:
            key: Secret identifier
            default: Default value if secret not found
            
        Returns:
            Secret value or default
        """
        # Check cache first
        if key in self._cache:
            return self._cache[key]
        
        value = None
        
        if self.backend == SecretsBackend.ENV:
            value = os.getenv(key, default)
        
        elif self.backend == SecretsBackend.AWS:
            value = self._get_aws_secret(key, default)
        
        elif self.backend == SecretsBackend.VAULT:
            value = self._get_vault_secret(key, default)
        
        elif self.backend == SecretsBackend.AZURE:
            value = self._get_azure_secret(key, default)
        
        elif self.backend == SecretsBackend.FILE:
            value = self._get_file_secret(key, default)
        
        # Cache the value
        if value is not None:
            self._cache[key] = value
        
        return value
    
    def get_secret_json(self, key: str, default: Optional[Dict] = None) -> Optional[Dict]:
        """Get secret and parse as JSON."""
        value = self.get_secret(key)
        if value is None:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse secret {key} as JSON")
            return default
    
    def _get_aws_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve secret from AWS Secrets Manager."""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            region = os.getenv("AWS_REGION", "us-east-1")
            session = boto3.session.Session()
            client = session.client(service_name="secretsmanager", region_name=region)
            
            try:
                response = client.get_secret_value(SecretId=key)
                return response.get("SecretString", default)
            except ClientError as e:
                if e.response["Error"]["Code"] == "ResourceNotFoundException":
                    logger.warning(f"Secret {key} not found in AWS Secrets Manager")
                    return default
                raise
        except ImportError:
            logger.error("boto3 not installed. Install with: pip install boto3")
            return default
    
    def _get_vault_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve secret from HashiCorp Vault."""
        try:
            import hvac
            
            vault_url = os.getenv("VAULT_ADDR", "http://localhost:8200")
            vault_token = os.getenv("VAULT_TOKEN")
            
            if not vault_token:
                logger.error("VAULT_TOKEN not set")
                return default
            
            client = hvac.Client(url=vault_url, token=vault_token)
            
            if not client.is_authenticated():
                logger.error("Vault authentication failed")
                return default
            
            # Assuming KV v2 secrets engine at 'secret/'
            mount_point = os.getenv("VAULT_MOUNT_POINT", "secret")
            secret_path = os.getenv("VAULT_SECRET_PATH", "cctv")
            
            response = client.secrets.kv.v2.read_secret_version(
                path=f"{secret_path}/{key}",
                mount_point=mount_point
            )
            
            return response["data"]["data"].get("value", default)
        except ImportError:
            logger.error("hvac not installed. Install with: pip install hvac")
            return default
        except Exception as e:
            logger.error(f"Failed to retrieve secret from Vault: {e}")
            return default
    
    def _get_azure_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve secret from Azure Key Vault."""
        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient
            
            vault_url = os.getenv("AZURE_KEYVAULT_URL")
            if not vault_url:
                logger.error("AZURE_KEYVAULT_URL not set")
                return default
            
            credential = DefaultAzureCredential()
            client = SecretClient(vault_url=vault_url, credential=credential)
            
            try:
                secret = client.get_secret(key)
                return secret.value
            except Exception as e:
                logger.warning(f"Secret {key} not found in Azure Key Vault: {e}")
                return default
        except ImportError:
            logger.error("Azure libraries not installed. Install with: pip install azure-identity azure-keyvault-secrets")
            return default
    
    def _get_file_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve secret from encrypted file."""
        secrets_file = Path(os.getenv("SECRETS_FILE_PATH", "config/secrets.enc"))
        
        if not secrets_file.exists():
            logger.warning(f"Secrets file {secrets_file} not found")
            return default
        
        try:
            encrypted_data = secrets_file.read_bytes()
            decrypted_data = self._cipher.decrypt(encrypted_data)
            secrets = json.loads(decrypted_data.decode())
            return secrets.get(key, default)
        except Exception as e:
            logger.error(f"Failed to decrypt secrets file: {e}")
            return default
    
    def set_file_secret(self, secrets: Dict[str, Any], output_path: str = "config/secrets.enc"):
        """
        Encrypt and save secrets to file (for file backend).
        
        Args:
            secrets: Dictionary of secrets to encrypt
            output_path: Path to save encrypted secrets
        """
        if self.backend != SecretsBackend.FILE:
            raise ValueError("set_file_secret only works with FILE backend")
        
        if not self._cipher:
            raise ValueError("Encryption not initialized")
        
        # Convert to JSON and encrypt
        json_data = json.dumps(secrets).encode()
        encrypted_data = self._cipher.encrypt(json_data)
        
        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(encrypted_data)
        
        logger.info(f"Secrets encrypted and saved to {output_path}")
    
    def validate_required_secrets(self, required_keys: list[str]) -> bool:
        """
        Validate that all required secrets are available.
        
        Args:
            required_keys: List of required secret keys
            
        Returns:
            True if all secrets are available, False otherwise
        """
        missing = []
        for key in required_keys:
            if self.get_secret(key) is None:
                missing.append(key)
        
        if missing:
            logger.error(f"Missing required secrets: {', '.join(missing)}")
            return False
        
        return True


# Global instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get or create global secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        backend_name = os.getenv("SECRETS_BACKEND", "environment")
        try:
            backend = SecretsBackend(backend_name)
        except ValueError:
            logger.warning(f"Invalid secrets backend '{backend_name}', falling back to environment")
            backend = SecretsBackend.ENV
        
        _secrets_manager = SecretsManager(backend=backend)
        logger.info(f"Initialized secrets manager with backend: {backend.value}")
    
    return _secrets_manager


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Convenience function to get secret."""
    return get_secrets_manager().get_secret(key, default)
