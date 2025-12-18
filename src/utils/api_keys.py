"""API Key Management with Rotation.

Provides:
- Secure API key generation
- Automatic key rotation
- Key versioning and grace periods
- Usage tracking
- Audit logging
"""

import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum
import json

from loguru import logger


class KeyStatus(Enum):
    """API key status."""
    ACTIVE = "active"
    ROTATING = "rotating"  # Grace period during rotation
    REVOKED = "revoked"
    EXPIRED = "expired"


@dataclass
class APIKey:
    """API Key with metadata."""
    key_id: str
    key_hash: str  # Store hash, not plain key
    name: str
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime] = None
    status: KeyStatus = KeyStatus.ACTIVE
    permissions: List[str] = field(default_factory=list)
    rate_limit: Optional[Dict[str, int]] = None
    metadata: Dict = field(default_factory=dict)
    usage_count: int = 0
    version: int = 1
    
    def is_valid(self) -> bool:
        """Check if key is currently valid."""
        if self.status == KeyStatus.REVOKED:
            return False
        
        if self.status == KeyStatus.EXPIRED:
            return False
        
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        
        return True
    
    def record_usage(self):
        """Record key usage."""
        self.last_used = datetime.utcnow()
        self.usage_count += 1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "key_id": self.key_id,
            "key_hash": self.key_hash,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "status": self.status.value,
            "permissions": self.permissions,
            "rate_limit": self.rate_limit,
            "metadata": self.metadata,
            "usage_count": self.usage_count,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'APIKey':
        """Create from dictionary."""
        return cls(
            key_id=data["key_id"],
            key_hash=data["key_hash"],
            name=data["name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
            status=KeyStatus(data["status"]),
            permissions=data.get("permissions", []),
            rate_limit=data.get("rate_limit"),
            metadata=data.get("metadata", {}),
            usage_count=data.get("usage_count", 0),
            version=data.get("version", 1)
        )


class APIKeyManager:
    """Manages API keys with automatic rotation.
    
    Features:
    - Secure key generation (cryptographically random)
    - Automatic rotation with grace periods
    - Key versioning
    - Usage tracking
    - Audit logging
    """
    
    # Key format: prefix_base62(random_bytes)
    KEY_PREFIX = "cctvfd_"  # CCTV Face Detection
    KEY_LENGTH = 32  # bytes (will be base62 encoded)
    
    def __init__(
        self,
        rotation_days: int = 90,
        grace_period_days: int = 7,
        max_keys_per_service: int = 2
    ):
        """
        Initialize API key manager.
        
        Args:
            rotation_days: Days before automatic rotation
            grace_period_days: Days old key remains valid during rotation
            max_keys_per_service: Maximum active keys per service
        """
        self.rotation_days = rotation_days
        self.grace_period_days = grace_period_days
        self.max_keys_per_service = max_keys_per_service
        
        # In-memory storage (replace with DB in production)
        self._keys: Dict[str, APIKey] = {}
        self._key_by_service: Dict[str, List[str]] = {}
    
    @staticmethod
    def _generate_key_id() -> str:
        """Generate unique key ID."""
        return f"key_{secrets.token_urlsafe(16)}"
    
    @staticmethod
    def _generate_key_secret() -> str:
        """Generate secure API key."""
        random_bytes = secrets.token_bytes(APIKeyManager.KEY_LENGTH)
        # Use base62 encoding (URL-safe, no special chars)
        key_secret = secrets.token_urlsafe(APIKeyManager.KEY_LENGTH)
        return f"{APIKeyManager.KEY_PREFIX}{key_secret}"
    
    @staticmethod
    def _hash_key(key: str) -> str:
        """Hash API key for storage."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def create_key(
        self,
        name: str,
        permissions: List[str],
        expires_in_days: Optional[int] = None,
        rate_limit: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict] = None,
        skip_max_check: bool = False  # Skip max check during rotation
    ) -> tuple[APIKey, str]:
        """
        Create new API key.
        
        Args:
            name: Service/user name for the key
            permissions: List of permissions (e.g., ["read", "write"])
            expires_in_days: Optional expiration (None = use rotation_days)
            rate_limit: Optional rate limit config {"calls": 100, "period": 60}
            metadata: Optional additional metadata
            skip_max_check: Internal flag to skip max keys check (prevents recursion)
        
        Returns:
            Tuple of (APIKey object, plain key string)
            WARNING: Plain key is only returned once!
        """
        # Check max keys limit (unless during rotation)
        if not skip_max_check:
            service_keys = self._key_by_service.get(name, [])
            active_keys = [k for k in service_keys if self._keys[k].is_valid()]
            
            if len(active_keys) >= self.max_keys_per_service:
                # Auto-rotate oldest key
                oldest_key_id = min(
                    active_keys,
                    key=lambda k: self._keys[k].created_at
                )
                self.rotate_key(oldest_key_id)
        
        # Generate key
        key_id = self._generate_key_id()
        key_secret = self._generate_key_secret()
        key_hash = self._hash_key(key_secret)
        
        # Calculate expiration
        expires_at = None
        if expires_in_days is not None:
            if expires_in_days == 0:
                # Immediate expiration for testing
                expires_at = datetime.utcnow()
            else:
                expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        elif self.rotation_days:
            # Auto-expiration for rotation
            expires_at = datetime.utcnow() + timedelta(days=self.rotation_days)
        
        # Create key object
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            permissions=permissions,
            rate_limit=rate_limit,
            metadata=metadata or {}
        )
        
        # Store
        self._keys[key_id] = api_key
        
        if name not in self._key_by_service:
            self._key_by_service[name] = []
        self._key_by_service[name].append(key_id)
        
        logger.info(f"Created API key {key_id} for {name}")
        
        return api_key, key_secret
    
    def validate_key(self, key_secret: str) -> Optional[APIKey]:
        """
        Validate API key and return key object if valid.
        
        Args:
            key_secret: Plain API key
        
        Returns:
            APIKey object if valid, None otherwise
        """
        # Check format
        if not key_secret.startswith(self.KEY_PREFIX):
            return None
        
        key_hash = self._hash_key(key_secret)
        
        # Find matching key (constant-time comparison)
        for key_obj in self._keys.values():
            if secrets.compare_digest(key_obj.key_hash, key_hash):
                # Check if valid
                if not key_obj.is_valid():
                    logger.warning(f"Attempt to use invalid key {key_obj.key_id}")
                    return None
                
                # Record usage
                key_obj.record_usage()
                return key_obj
        
        logger.warning(f"Unknown API key attempted: {key_secret[:20]}...")
        return None
    
    def rotate_key(self, key_id: str) -> tuple[Optional[APIKey], Optional[str]]:
        """
        Rotate an API key.
        
        Creates a new key and marks old key for deprecation.
        
        Args:
            key_id: ID of key to rotate
        
        Returns:
            Tuple of (new APIKey, new key secret) or (None, None) if key not found
        """
        old_key = self._keys.get(key_id)
        if not old_key:
            logger.error(f"Cannot rotate: key {key_id} not found")
            return None, None
        
        # Mark old key as rotating (grace period)
        old_key.status = KeyStatus.ROTATING
        old_key.expires_at = datetime.utcnow() + timedelta(days=self.grace_period_days)
        
        # Create new key (skip max check to prevent recursion)
        new_key, new_secret = self.create_key(
            name=old_key.name,
            permissions=old_key.permissions,
            rate_limit=old_key.rate_limit,
            metadata={**old_key.metadata, "rotated_from": key_id},
            skip_max_check=True  # Important: prevents recursion
        )
        new_key.version = old_key.version + 1
        
        logger.info(
            f"Rotated key {key_id} -> {new_key.key_id}. "
            f"Old key valid until {old_key.expires_at}"
        )
        
        return new_key, new_secret
    
    def revoke_key(self, key_id: str) -> bool:
        """
        Immediately revoke an API key.
        
        Args:
            key_id: ID of key to revoke
        
        Returns:
            True if revoked, False if not found
        """
        key = self._keys.get(key_id)
        if not key:
            return False
        
        key.status = KeyStatus.REVOKED
        logger.warning(f"Revoked API key {key_id} for {key.name}")
        return True
    
    def get_key(self, key_id: str) -> Optional[APIKey]:
        """Get key by ID."""
        return self._keys.get(key_id)
    
    def list_keys(
        self,
        service_name: Optional[str] = None,
        status: Optional[KeyStatus] = None
    ) -> List[APIKey]:
        """
        List API keys.
        
        Args:
            service_name: Optional filter by service
            status: Optional filter by status
        
        Returns:
            List of APIKey objects
        """
        if service_name:
            key_ids = self._key_by_service.get(service_name, [])
            keys = [self._keys[kid] for kid in key_ids if kid in self._keys]
        else:
            keys = list(self._keys.values())
        
        if status:
            keys = [k for k in keys if k.status == status]
        
        return keys
    
    def auto_rotate_keys(self) -> List[tuple[str, str]]:
        """
        Automatically rotate keys nearing expiration.
        
        Returns:
            List of (old_key_id, new_key_id) pairs that were rotated
        """
        rotations = []
        now = datetime.utcnow()
        rotation_threshold = now + timedelta(days=7)  # Rotate 7 days before expiry
        
        for key in list(self._keys.values()):
            if key.status != KeyStatus.ACTIVE:
                continue
            
            if not key.expires_at:
                continue
            
            # Check if nearing expiration
            if key.expires_at <= rotation_threshold:
                new_key, _ = self.rotate_key(key.key_id)
                if new_key:
                    rotations.append((key.key_id, new_key.key_id))
                    logger.info(
                        f"Auto-rotated key {key.key_id} -> {new_key.key_id} "
                        f"(expires {key.expires_at})"
                    )
        
        return rotations
    
    def cleanup_expired_keys(self) -> int:
        """
        Clean up expired keys.
        
        Returns:
            Number of keys cleaned up
        """
        now = datetime.utcnow()
        count = 0
        
        for key in list(self._keys.values()):
            if key.expires_at and key.expires_at < now:
                if key.status == KeyStatus.ROTATING:
                    key.status = KeyStatus.EXPIRED
                    count += 1
                    logger.info(f"Expired rotated key {key.key_id}")
        
        return count
    
    def get_usage_stats(self, service_name: Optional[str] = None) -> Dict:
        """
        Get usage statistics.
        
        Args:
            service_name: Optional filter by service
        
        Returns:
            Usage statistics
        """
        keys = self.list_keys(service_name=service_name)
        
        total_usage = sum(k.usage_count for k in keys)
        active_keys = len([k for k in keys if k.status == KeyStatus.ACTIVE])
        
        return {
            "total_keys": len(keys),
            "active_keys": active_keys,
            "total_usage": total_usage,
            "keys_by_status": {
                status.value: len([k for k in keys if k.status == status])
                for status in KeyStatus
            }
        }


# Global instance (singleton pattern)
_api_key_manager: Optional[APIKeyManager] = None


def get_api_key_manager() -> APIKeyManager:
    """Get global API key manager instance."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager
