"""
Enhanced API security module with improved JWT handling, token refresh, and security features.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, List
import secrets
import hashlib
import re

import jwt
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, validator
from loguru import logger


# Security constants
MIN_PASSWORD_LENGTH = 12
PASSWORD_PATTERN = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]')


class TokenData(BaseModel):
    """Token payload data."""
    username: str
    role: str
    token_type: str = "access"
    jti: Optional[str] = None  # JWT ID for token revocation


class UserCreate(BaseModel):
    """User creation request with validation."""
    username: str
    password: str
    email: Optional[EmailStr] = None
    role: str = "operator"
    
    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3 or len(v) > 50:
            raise ValueError('Username must be between 3 and 50 characters')
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Username can only contain letters, numbers, underscores, and hyphens')
        return v
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < MIN_PASSWORD_LENGTH:
            raise ValueError(f'Password must be at least {MIN_PASSWORD_LENGTH} characters')
        if not PASSWORD_PATTERN.match(v):
            raise ValueError('Password must contain uppercase, lowercase, number, and special character')
        return v
    
    @validator('role')
    def validate_role(cls, v):
        allowed_roles = {'admin', 'operator', 'viewer'}
        if v not in allowed_roles:
            raise ValueError(f'Role must be one of: {", ".join(allowed_roles)}')
        return v


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_expires_in: int


class TokenBlacklist:
    """In-memory token blacklist for revocation. Replace with Redis in production."""
    
    def __init__(self):
        self._blacklist: Dict[str, datetime] = {}
        self._cleanup_interval = 3600  # 1 hour
        self._last_cleanup = datetime.utcnow()
    
    def add(self, jti: str, expires_at: datetime):
        """Add token to blacklist."""
        self._blacklist[jti] = expires_at
        self._cleanup_if_needed()
    
    def is_blacklisted(self, jti: str) -> bool:
        """Check if token is blacklisted."""
        return jti in self._blacklist
    
    def _cleanup_if_needed(self):
        """Remove expired tokens from blacklist."""
        now = datetime.utcnow()
        if (now - self._last_cleanup).seconds < self._cleanup_interval:
            return
        
        self._blacklist = {
            jti: exp for jti, exp in self._blacklist.items()
            if exp > now
        }
        self._last_cleanup = now


# Global blacklist instance
token_blacklist = TokenBlacklist()


# Password hashing configuration
pwd_context = CryptContext(
    schemes=["argon2", "bcrypt"],
    deprecated="auto",
    argon2__memory_cost=65536,
    argon2__time_cost=3,
    argon2__parallelism=4
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
http_bearer = HTTPBearer(auto_error=False)


class EnhancedAuthService:
    """Enhanced authentication service with improved security features."""
    
    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 15,
        refresh_token_expire_days: int = 7,
        max_login_attempts: int = 5,
        lockout_duration_minutes: int = 15
    ):
        if len(secret_key) < 32:
            raise ValueError("Secret key must be at least 32 characters")
        
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        self.max_login_attempts = max_login_attempts
        self.lockout_duration_minutes = lockout_duration_minutes
        
        # Track login attempts
        self._login_attempts: Dict[str, List[datetime]] = {}
        self._lockouts: Dict[str, datetime] = {}
    
    def hash_password(self, password: str) -> str:
        """Hash password using Argon2."""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception:
            return False
    
    def check_password_strength(self, password: str) -> Dict[str, any]:
        """Check password strength and return feedback."""
        issues = []
        score = 0
        
        if len(password) >= MIN_PASSWORD_LENGTH:
            score += 25
        else:
            issues.append(f"Password should be at least {MIN_PASSWORD_LENGTH} characters")
        
        if re.search(r'[A-Z]', password):
            score += 25
        else:
            issues.append("Add uppercase letters")
        
        if re.search(r'[a-z]', password):
            score += 15
        else:
            issues.append("Add lowercase letters")
        
        if re.search(r'\d', password):
            score += 20
        else:
            issues.append("Add numbers")
        
        if re.search(r'[@$!%*?&]', password):
            score += 15
        else:
            issues.append("Add special characters (@$!%*?&)")
        
        return {
            'score': score,
            'strength': 'strong' if score >= 80 else 'medium' if score >= 50 else 'weak',
            'issues': issues
        }
    
    def is_locked_out(self, username: str) -> bool:
        """Check if user is locked out."""
        lockout_until = self._lockouts.get(username)
        if lockout_until and datetime.utcnow() < lockout_until:
            return True
        elif lockout_until:
            # Lockout expired, clear it
            del self._lockouts[username]
        return False
    
    def record_login_attempt(self, username: str, success: bool):
        """Record login attempt for rate limiting."""
        now = datetime.utcnow()
        
        if success:
            # Clear attempts on successful login
            self._login_attempts.pop(username, None)
            return
        
        # Record failed attempt
        if username not in self._login_attempts:
            self._login_attempts[username] = []
        
        # Clean old attempts (outside lockout window)
        window = now - timedelta(minutes=self.lockout_duration_minutes)
        self._login_attempts[username] = [
            t for t in self._login_attempts[username] if t > window
        ]
        
        self._login_attempts[username].append(now)
        
        # Check for lockout
        if len(self._login_attempts[username]) >= self.max_login_attempts:
            self._lockouts[username] = now + timedelta(minutes=self.lockout_duration_minutes)
            logger.warning(f"User {username} locked out due to too many failed login attempts")
    
    def _generate_jti(self) -> str:
        """Generate unique JWT ID."""
        return secrets.token_urlsafe(32)
    
    def create_access_token(self, username: str, role: str) -> tuple[str, str]:
        """
        Create access token.
        
        Returns:
            Tuple of (token, jti)
        """
        jti = self._generate_jti()
        expires = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        payload = {
            "sub": username,
            "role": role,
            "type": "access",
            "jti": jti,
            "exp": expires,
            "iat": datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token, jti
    
    def create_refresh_token(self, username: str, role: str) -> tuple[str, str]:
        """
        Create refresh token.
        
        Returns:
            Tuple of (token, jti)
        """
        jti = self._generate_jti()
        expires = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        payload = {
            "sub": username,
            "role": role,
            "type": "refresh",
            "jti": jti,
            "exp": expires,
            "iat": datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token, jti
    
    def create_token_pair(self, username: str, role: str) -> TokenResponse:
        """Create access and refresh token pair."""
        access_token, _ = self.create_access_token(username, role)
        refresh_token, _ = self.create_refresh_token(username, role)
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.access_token_expire_minutes * 60,
            refresh_expires_in=self.refresh_token_expire_days * 86400
        )
    
    def decode_token(self, token: str, expected_type: str = "access") -> Dict:
        """
        Decode and validate token.
        
        Args:
            token: JWT token
            expected_type: Expected token type ("access" or "refresh")
            
        Returns:
            Token payload
            
        Raises:
            HTTPException: If token is invalid
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get("type") != expected_type:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid token type. Expected {expected_type}"
                )
            
            # Check if token is blacklisted
            jti = payload.get("jti")
            if jti and token_blacklist.is_blacklisted(jti):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}"
            )
    
    def refresh_access_token(self, refresh_token: str) -> TokenResponse:
        """
        Refresh access token using refresh token.
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            New token pair
        """
        payload = self.decode_token(refresh_token, expected_type="refresh")
        username = payload.get("sub")
        role = payload.get("role")
        
        return self.create_token_pair(username, role)
    
    def revoke_token(self, token: str):
        """Revoke a token by adding it to blacklist."""
        try:
            payload = jwt.decode(
                token, self.secret_key, 
                algorithms=[self.algorithm],
                options={"verify_exp": False}  # Allow revoking expired tokens
            )
            jti = payload.get("jti")
            exp = datetime.fromtimestamp(payload.get("exp", 0))
            
            if jti:
                token_blacklist.add(jti, exp)
                logger.info(f"Token {jti[:8]}... revoked")
        except jwt.InvalidTokenError:
            pass  # Ignore invalid tokens


def get_current_user_enhanced(
    auth_service: EnhancedAuthService,
    credentials: HTTPAuthorizationCredentials = Depends(http_bearer)
) -> Dict:
    """Dependency for getting current user from token."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    payload = auth_service.decode_token(credentials.credentials)
    return {
        "username": payload.get("sub"),
        "role": payload.get("role"),
        "jti": payload.get("jti")
    }


def require_roles(*roles: str):
    """Decorator factory for role-based access control."""
    def dependency(user: Dict = Depends(get_current_user_enhanced)) -> Dict:
        if user.get("role") not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This action requires one of these roles: {', '.join(roles)}"
            )
        return user
    return dependency


class APIKeyAuth:
    """API Key authentication for service-to-service communication."""
    
    def __init__(self, valid_keys: Dict[str, Dict]):
        """
        Initialize with valid API keys.
        
        Args:
            valid_keys: Dict mapping API keys to their metadata
                        e.g., {"key123": {"name": "service1", "permissions": ["read"]}}
        """
        self._keys = valid_keys
    
    def validate(self, api_key: str) -> Optional[Dict]:
        """Validate API key and return metadata if valid."""
        # Use constant-time comparison to prevent timing attacks
        for key, metadata in self._keys.items():
            if secrets.compare_digest(key, api_key):
                return metadata
        return None
    
    def __call__(self, request: Request) -> Dict:
        """FastAPI dependency for API key validation."""
        api_key = request.headers.get("X-API-Key")
        
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required"
            )
        
        metadata = self.validate(api_key)
        if not metadata:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        return metadata
