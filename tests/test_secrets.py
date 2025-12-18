"""Tests for secrets management."""
import pytest
import os
import tempfile
from pathlib import Path
from src.utils.secrets import SecretsManager, SecretsBackend, get_secrets_manager


def test_environment_secrets():
    """Test environment variable backend."""
    os.environ["TEST_SECRET"] = "test_value"
    
    manager = SecretsManager(backend=SecretsBackend.ENV)
    assert manager.get_secret("TEST_SECRET") == "test_value"
    assert manager.get_secret("NONEXISTENT", "default") == "default"


def test_file_secrets():
    """Test encrypted file backend."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["SECRETS_MASTER_KEY"] = "test_master_key_12345678"
        secrets_file = Path(tmpdir) / "secrets.enc"
        os.environ["SECRETS_FILE_PATH"] = str(secrets_file)
        
        manager = SecretsManager(backend=SecretsBackend.FILE)
        
        # Save secrets
        test_secrets = {
            "API_KEY": "secret_api_key",
            "DATABASE_PASSWORD": "db_pass_123"
        }
        manager.set_file_secret(test_secrets, str(secrets_file))
        
        # Create new manager to test retrieval
        manager2 = SecretsManager(backend=SecretsBackend.FILE)
        assert manager2.get_secret("API_KEY") == "secret_api_key"
        assert manager2.get_secret("DATABASE_PASSWORD") == "db_pass_123"


def test_required_secrets_validation():
    """Test validation of required secrets."""
    os.environ["REQUIRED_1"] = "value1"
    os.environ["REQUIRED_2"] = "value2"
    
    manager = SecretsManager(backend=SecretsBackend.ENV)
    
    # All present
    assert manager.validate_required_secrets(["REQUIRED_1", "REQUIRED_2"]) is True
    
    # One missing
    assert manager.validate_required_secrets(["REQUIRED_1", "MISSING"]) is False


def test_json_secrets():
    """Test JSON secret parsing."""
    os.environ["JSON_SECRET"] = '{"key": "value", "number": 42}'
    
    manager = SecretsManager(backend=SecretsBackend.ENV)
    result = manager.get_secret_json("JSON_SECRET")
    
    assert result == {"key": "value", "number": 42}


def test_json_secrets_invalid():
    """Test JSON secret parsing with invalid JSON."""
    os.environ["INVALID_JSON"] = "not a json"
    
    manager = SecretsManager(backend=SecretsBackend.ENV)
    result = manager.get_secret_json("INVALID_JSON", {"default": "value"})
    
    assert result == {"default": "value"}


def test_secret_caching():
    """Test that secrets are cached after first retrieval."""
    os.environ["CACHED_SECRET"] = "original"
    
    manager = SecretsManager(backend=SecretsBackend.ENV)
    
    # First retrieval
    value1 = manager.get_secret("CACHED_SECRET")
    
    # Change environment variable
    os.environ["CACHED_SECRET"] = "changed"
    
    # Should still return cached value
    value2 = manager.get_secret("CACHED_SECRET")
    
    assert value1 == "original"
    assert value2 == "original"  # Cached


def test_file_backend_missing_master_key():
    """Test that file backend raises error without master key."""
    if "SECRETS_MASTER_KEY" in os.environ:
        del os.environ["SECRETS_MASTER_KEY"]
    
    with pytest.raises(ValueError, match="SECRETS_MASTER_KEY"):
        SecretsManager(backend=SecretsBackend.FILE)


def test_file_backend_nonexistent_file():
    """Test file backend with nonexistent secrets file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["SECRETS_MASTER_KEY"] = "test_master_key_12345678"
        os.environ["SECRETS_FILE_PATH"] = str(Path(tmpdir) / "nonexistent.enc")
        
        manager = SecretsManager(backend=SecretsBackend.FILE)
        
        # Should return default for nonexistent file
        assert manager.get_secret("ANY_KEY", "default") == "default"


def test_global_manager_singleton():
    """Test global secrets manager singleton pattern."""
    os.environ["SECRETS_BACKEND"] = "environment"
    os.environ["GLOBAL_TEST"] = "works"
    
    # Reset global instance for testing
    import src.utils.secrets
    src.utils.secrets._secrets_manager = None
    
    manager1 = get_secrets_manager()
    manager2 = get_secrets_manager()
    
    # Same instance
    assert manager1 is manager2
    assert manager1.get_secret("GLOBAL_TEST") == "works"


def test_invalid_backend_fallback():
    """Test fallback to environment backend for invalid backend name."""
    os.environ["SECRETS_BACKEND"] = "invalid_backend"
    os.environ["FALLBACK_TEST"] = "value"
    
    # Reset global instance
    import src.utils.secrets
    src.utils.secrets._secrets_manager = None
    
    manager = get_secrets_manager()
    
    # Should fall back to ENV backend
    assert manager.backend == SecretsBackend.ENV
    assert manager.get_secret("FALLBACK_TEST") == "value"
