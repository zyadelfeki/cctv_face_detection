"""Secure Session Management.

Provides:
- Session creation and validation
- Automatic expiration
- Activity tracking
- Concurrent session limits
- Session hijacking prevention
"""

import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger


class SessionStatus(Enum):
    """Session status."""
    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    SUSPICIOUS = "suspicious"


@dataclass
class Session:
    """User session with security tracking."""
    session_id: str
    username: str
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    status: SessionStatus = SessionStatus.ACTIVE
    metadata: Dict = field(default_factory=dict)
    
    # Security tracking
    request_count: int = 0
    last_ip: Optional[str] = None
    last_user_agent: Optional[str] = None
    ip_changes: int = 0
    suspicious_activity: List[str] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Check if session is currently valid."""
        if self.status != SessionStatus.ACTIVE:
            return False
        
        if datetime.utcnow() > self.expires_at:
            return False
        
        return True
    
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.utcnow() > self.expires_at
    
    def update_activity(self, ip_address: str, user_agent: str):
        """Update session activity."""
        self.last_activity = datetime.utcnow()
        self.request_count += 1
        
        # Track IP changes (potential session hijacking)
        if self.last_ip and self.last_ip != ip_address:
            self.ip_changes += 1
            self.suspicious_activity.append(
                f"IP change from {self.last_ip} to {ip_address} at {datetime.utcnow()}"
            )
            logger.warning(
                f"Session {self.session_id[:8]}... IP change detected: "
                f"{self.last_ip} -> {ip_address}"
            )
        
        self.last_ip = ip_address
        self.last_user_agent = user_agent
    
    def mark_suspicious(self, reason: str):
        """Mark session as suspicious."""
        self.status = SessionStatus.SUSPICIOUS
        self.suspicious_activity.append(f"{reason} at {datetime.utcnow()}")
        logger.warning(
            f"Session {self.session_id[:8]}... marked suspicious: {reason}"
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "username": self.username,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "status": self.status.value,
            "metadata": self.metadata,
            "request_count": self.request_count,
            "ip_changes": self.ip_changes
        }


class SessionManager:
    """Manages user sessions with security features.
    
    Features:
    - Secure session ID generation
    - Automatic expiration
    - Activity tracking
    - Concurrent session limits
    - Session hijacking detection
    """
    
    SESSION_ID_LENGTH = 32  # bytes
    
    def __init__(
        self,
        session_lifetime_minutes: int = 480,  # 8 hours
        idle_timeout_minutes: int = 30,
        max_sessions_per_user: int = 5,
        max_ip_changes: int = 3
    ):
        """
        Initialize session manager.
        
        Args:
            session_lifetime_minutes: Maximum session lifetime
            idle_timeout_minutes: Idle timeout before expiration
            max_sessions_per_user: Maximum concurrent sessions per user
            max_ip_changes: Maximum allowed IP changes before flagging
        """
        self.session_lifetime = timedelta(minutes=session_lifetime_minutes)
        self.idle_timeout = timedelta(minutes=idle_timeout_minutes)
        self.max_sessions_per_user = max_sessions_per_user
        self.max_ip_changes = max_ip_changes
        
        # In-memory storage (replace with Redis in production)
        self._sessions: Dict[str, Session] = {}
        self._user_sessions: Dict[str, List[str]] = {}  # username -> [session_ids]
    
    @staticmethod
    def _generate_session_id() -> str:
        """Generate cryptographically secure session ID."""
        return secrets.token_urlsafe(SessionManager.SESSION_ID_LENGTH)
    
    def create_session(
        self,
        username: str,
        ip_address: str,
        user_agent: str,
        metadata: Optional[Dict] = None
    ) -> Session:
        """
        Create new session.
        
        Args:
            username: User identifier
            ip_address: Client IP address
            user_agent: Client user agent
            metadata: Optional session metadata
        
        Returns:
            Created Session object
        """
        # Check concurrent session limit
        user_session_ids = self._user_sessions.get(username, [])
        active_sessions = [
            sid for sid in user_session_ids
            if sid in self._sessions and self._sessions[sid].is_valid()
        ]
        
        if len(active_sessions) >= self.max_sessions_per_user:
            # Terminate oldest session
            oldest_sid = min(
                active_sessions,
                key=lambda sid: self._sessions[sid].created_at
            )
            self.terminate_session(oldest_sid)
            logger.info(
                f"Terminated oldest session {oldest_sid[:8]}... for {username} "
                f"(concurrent limit reached)"
            )
        
        # Generate session
        session_id = self._generate_session_id()
        now = datetime.utcnow()
        
        session = Session(
            session_id=session_id,
            username=username,
            created_at=now,
            last_activity=now,
            expires_at=now + self.session_lifetime,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata or {},
            last_ip=ip_address,
            last_user_agent=user_agent
        )
        
        # Store session
        self._sessions[session_id] = session
        
        if username not in self._user_sessions:
            self._user_sessions[username] = []
        self._user_sessions[username].append(session_id)
        
        logger.info(f"Created session {session_id[:8]}... for {username}")
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        return self._sessions.get(session_id)
    
    def validate_session(
        self,
        session_id: str,
        ip_address: str,
        user_agent: str
    ) -> Optional[Session]:
        """
        Validate session and update activity.
        
        Args:
            session_id: Session identifier
            ip_address: Client IP address
            user_agent: Client user agent
        
        Returns:
            Session if valid, None otherwise
        """
        session = self._sessions.get(session_id)
        
        if not session:
            logger.warning(f"Session not found: {session_id[:8]}...")
            return None
        
        # Check if expired
        if session.is_expired():
            session.status = SessionStatus.EXPIRED
            logger.info(f"Session {session_id[:8]}... expired")
            return None
        
        # Check idle timeout
        if datetime.utcnow() - session.last_activity > self.idle_timeout:
            session.status = SessionStatus.EXPIRED
            logger.info(f"Session {session_id[:8]}... idle timeout")
            return None
        
        # Check if valid
        if not session.is_valid():
            return None
        
        # Update activity
        session.update_activity(ip_address, user_agent)
        
        # Check for suspicious activity
        if session.ip_changes > self.max_ip_changes:
            session.mark_suspicious(f"Too many IP changes ({session.ip_changes})")
            return None
        
        return session
    
    def refresh_session(self, session_id: str) -> bool:
        """
        Refresh session expiration.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if refreshed, False if session not found
        """
        session = self._sessions.get(session_id)
        if not session or not session.is_valid():
            return False
        
        session.expires_at = datetime.utcnow() + self.session_lifetime
        logger.debug(f"Refreshed session {session_id[:8]}...")
        return True
    
    def terminate_session(self, session_id: str) -> bool:
        """
        Terminate a session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if terminated, False if not found
        """
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        session.status = SessionStatus.TERMINATED
        logger.info(f"Terminated session {session_id[:8]}... for {session.username}")
        return True
    
    def terminate_user_sessions(self, username: str) -> int:
        """
        Terminate all sessions for a user.
        
        Args:
            username: User identifier
        
        Returns:
            Number of sessions terminated
        """
        session_ids = self._user_sessions.get(username, [])
        count = 0
        
        for sid in session_ids:
            if self.terminate_session(sid):
                count += 1
        
        logger.info(f"Terminated {count} sessions for {username}")
        return count
    
    def list_user_sessions(self, username: str) -> List[Session]:
        """
        List all active sessions for a user.
        
        Args:
            username: User identifier
        
        Returns:
            List of active Session objects
        """
        session_ids = self._user_sessions.get(username, [])
        return [
            self._sessions[sid]
            for sid in session_ids
            if sid in self._sessions and self._sessions[sid].is_valid()
        ]
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired and terminated sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        count = 0
        expired_ids = []
        
        for sid, session in self._sessions.items():
            if session.is_expired() or session.status == SessionStatus.TERMINATED:
                expired_ids.append(sid)
        
        for sid in expired_ids:
            session = self._sessions.pop(sid)
            count += 1
            
            # Remove from user sessions
            username = session.username
            if username in self._user_sessions:
                self._user_sessions[username] = [
                    s for s in self._user_sessions[username] if s != sid
                ]
        
        if count > 0:
            logger.info(f"Cleaned up {count} expired/terminated sessions")
        
        return count
    
    def get_session_stats(self) -> Dict:
        """
        Get session statistics.
        
        Returns:
            Statistics dictionary
        """
        active = sum(1 for s in self._sessions.values() if s.is_valid())
        expired = sum(1 for s in self._sessions.values() if s.is_expired())
        suspicious = sum(1 for s in self._sessions.values() if s.status == SessionStatus.SUSPICIOUS)
        
        return {
            "total_sessions": len(self._sessions),
            "active_sessions": active,
            "expired_sessions": expired,
            "suspicious_sessions": suspicious,
            "unique_users": len(self._user_sessions)
        }


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
