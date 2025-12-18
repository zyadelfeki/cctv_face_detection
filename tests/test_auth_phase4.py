"""Tests for Phase 4: Authentication & Authorization systems."""

import pytest
import time
from datetime import datetime, timedelta

from src.utils.api_keys import APIKeyManager, KeyStatus
from src.utils.rbac import RBACManager, Permission, Role
from src.utils.sessions import SessionManager, SessionStatus


class TestAPIKeyManagement:
    """Test API key rotation and management."""
    
    def test_key_generation(self):
        """Test secure key generation."""
        manager = APIKeyManager()
        
        key, secret = manager.create_key(
            name="test_service",
            permissions=["read", "write"]
        )
        
        assert key is not None
        assert secret.startswith("cctvfd_")
        assert len(secret) > 40  # Should be long and secure
        assert key.status == KeyStatus.ACTIVE
    
    def test_key_validation(self):
        """Test key validation."""
        manager = APIKeyManager()
        
        key, secret = manager.create_key(
            name="test_service",
            permissions=["read"]
        )
        
        # Valid key
        validated = manager.validate_key(secret)
        assert validated is not None
        assert validated.key_id == key.key_id
        assert validated.usage_count == 1
        
        # Invalid key
        invalid = manager.validate_key("cctvfd_invalid")
        assert invalid is None
    
    def test_key_rotation(self):
        """Test automatic key rotation."""
        manager = APIKeyManager(grace_period_days=1)
        
        old_key, old_secret = manager.create_key(
            name="service1",
            permissions=["read", "write"]
        )
        
        # Rotate key
        new_key, new_secret = manager.rotate_key(old_key.key_id)
        
        assert new_key is not None
        assert new_secret is not None
        assert new_key.key_id != old_key.key_id
        assert new_key.version == old_key.version + 1
        
        # Old key should be in ROTATING status
        old_key_updated = manager.get_key(old_key.key_id)
        assert old_key_updated.status == KeyStatus.ROTATING
        
        # Both keys should work during grace period
        assert manager.validate_key(old_secret) is not None
        assert manager.validate_key(new_secret) is not None
    
    def test_key_revocation(self):
        """Test key revocation."""
        manager = APIKeyManager()
        
        key, secret = manager.create_key(
            name="service1",
            permissions=["read"]
        )
        
        # Revoke key
        assert manager.revoke_key(key.key_id) is True
        
        # Key should no longer validate
        assert manager.validate_key(secret) is None
        
        # Key status should be REVOKED
        revoked_key = manager.get_key(key.key_id)
        assert revoked_key.status == KeyStatus.REVOKED
    
    def test_max_keys_per_service(self):
        """Test maximum keys per service limit."""
        manager = APIKeyManager(max_keys_per_service=2)
        
        # Create 2 keys
        key1, _ = manager.create_key(name="service1", permissions=["read"])
        key2, _ = manager.create_key(name="service1", permissions=["read"])
        
        # Creating 3rd key should auto-rotate oldest
        key3, _ = manager.create_key(name="service1", permissions=["read"])
        
        # First key should be in ROTATING status
        key1_updated = manager.get_key(key1.key_id)
        assert key1_updated.status == KeyStatus.ROTATING
    
    def test_key_expiration(self):
        """Test key expiration."""
        manager = APIKeyManager()
        
        key, secret = manager.create_key(
            name="service1",
            permissions=["read"],
            expires_in_days=0  # Immediate expiration for testing
        )
        
        # Should not validate (expired)
        time.sleep(0.1)
        assert manager.validate_key(secret) is None
    
    def test_usage_tracking(self):
        """Test key usage tracking."""
        manager = APIKeyManager()
        
        key, secret = manager.create_key(
            name="service1",
            permissions=["read"]
        )
        
        # Use key multiple times
        for _ in range(5):
            manager.validate_key(secret)
        
        # Check usage count
        key_updated = manager.get_key(key.key_id)
        assert key_updated.usage_count == 5
        assert key_updated.last_used is not None


class TestRBAC:
    """Test Role-Based Access Control."""
    
    def test_default_roles(self):
        """Test default role creation."""
        rbac = RBACManager()
        
        roles = rbac.list_roles()
        role_names = [r["name"] for r in roles]
        
        assert "viewer" in role_names
        assert "operator" in role_names
        assert "analyst" in role_names
        assert "admin" in role_names
    
    def test_role_permissions(self):
        """Test role permissions."""
        rbac = RBACManager()
        
        viewer_perms = rbac.get_role_permissions("viewer")
        admin_perms = rbac.get_role_permissions("admin")
        
        # Viewer should have basic permissions
        assert Permission.CAMERA_VIEW in viewer_perms
        assert Permission.CAMERA_DELETE not in viewer_perms
        
        # Admin should have all permissions (inheritance)
        assert Permission.CAMERA_VIEW in admin_perms
        assert Permission.CAMERA_DELETE in admin_perms
        assert Permission.SYSTEM_CONFIG in admin_perms
    
    def test_role_inheritance(self):
        """Test role inheritance."""
        rbac = RBACManager()
        
        # Operator inherits from viewer
        operator_perms = rbac.get_role_permissions("operator")
        
        # Should have viewer permissions
        assert Permission.CAMERA_VIEW in operator_perms
        # Plus operator-specific permissions
        assert Permission.CAMERA_CREATE in operator_perms
    
    def test_permission_checking(self):
        """Test permission checking."""
        rbac = RBACManager()
        
        # Assign roles
        rbac.assign_role("alice", "viewer")
        rbac.assign_role("bob", "admin")
        
        # Check permissions
        assert rbac.check_permission("alice", Permission.CAMERA_VIEW) is True
        assert rbac.check_permission("alice", Permission.CAMERA_DELETE) is False
        
        assert rbac.check_permission("bob", Permission.CAMERA_VIEW) is True
        assert rbac.check_permission("bob", Permission.CAMERA_DELETE) is True
    
    def test_custom_role_creation(self):
        """Test creating custom roles."""
        rbac = RBACManager()
        
        # Create custom role
        custom_role = rbac.create_role(
            name="monitor",
            permissions={Permission.CAMERA_VIEW, Permission.INCIDENT_VIEW},
            description="Monitoring only"
        )
        
        assert custom_role.name == "monitor"
        assert len(custom_role.permissions) == 2
        
        # Assign and test
        rbac.assign_role("charlie", "monitor")
        assert rbac.check_permission("charlie", Permission.CAMERA_VIEW) is True
        assert rbac.check_permission("charlie", Permission.CAMERA_CREATE) is False
    
    def test_no_role_assigned(self):
        """Test behavior when user has no role."""
        rbac = RBACManager()
        
        # User with no role should have no permissions
        assert rbac.check_permission("nobody", Permission.CAMERA_VIEW) is False


class TestSessionManagement:
    """Test session management."""
    
    def test_session_creation(self):
        """Test session creation."""
        manager = SessionManager()
        
        session = manager.create_session(
            username="alice",
            ip_address="192.168.1.1",
            user_agent="TestBrowser/1.0"
        )
        
        assert session is not None
        assert session.username == "alice"
        assert session.status == SessionStatus.ACTIVE
        assert session.is_valid() is True
    
    def test_session_validation(self):
        """Test session validation."""
        manager = SessionManager()
        
        session = manager.create_session(
            username="alice",
            ip_address="192.168.1.1",
            user_agent="TestBrowser/1.0"
        )
        
        # Valid session
        validated = manager.validate_session(
            session.session_id,
            "192.168.1.1",
            "TestBrowser/1.0"
        )
        assert validated is not None
        assert validated.request_count == 1
    
    def test_session_expiration(self):
        """Test session expiration."""
        manager = SessionManager(
            session_lifetime_minutes=0,  # Immediate expiration
            idle_timeout_minutes=0
        )
        
        session = manager.create_session(
            username="alice",
            ip_address="192.168.1.1",
            user_agent="TestBrowser/1.0"
        )
        
        time.sleep(0.1)
        
        # Should be expired
        validated = manager.validate_session(
            session.session_id,
            "192.168.1.1",
            "TestBrowser/1.0"
        )
        assert validated is None
        assert session.is_expired() is True
    
    def test_ip_change_detection(self):
        """Test IP change detection (session hijacking prevention)."""
        manager = SessionManager(max_ip_changes=1)
        
        session = manager.create_session(
            username="alice",
            ip_address="192.168.1.1",
            user_agent="TestBrowser/1.0"
        )
        
        # First IP change - should still work
        validated = manager.validate_session(
            session.session_id,
            "192.168.1.2",
            "TestBrowser/1.0"
        )
        assert validated is not None
        assert validated.ip_changes == 1
        
        # Second IP change - should be flagged
        validated = manager.validate_session(
            session.session_id,
            "192.168.1.3",
            "TestBrowser/1.0"
        )
        assert validated is None
        assert session.status == SessionStatus.SUSPICIOUS
    
    def test_concurrent_session_limit(self):
        """Test concurrent session limit."""
        manager = SessionManager(max_sessions_per_user=2)
        
        # Create 2 sessions
        session1 = manager.create_session("alice", "192.168.1.1", "Browser1")
        session2 = manager.create_session("alice", "192.168.1.2", "Browser2")
        
        # Both should be active
        assert session1.is_valid() is True
        assert session2.is_valid() is True
        
        # Create 3rd session - should terminate oldest
        session3 = manager.create_session("alice", "192.168.1.3", "Browser3")
        
        # First session should be terminated
        session1_updated = manager.get_session(session1.session_id)
        assert session1_updated.status == SessionStatus.TERMINATED
    
    def test_session_refresh(self):
        """Test session refresh."""
        manager = SessionManager(session_lifetime_minutes=10)
        
        session = manager.create_session(
            username="alice",
            ip_address="192.168.1.1",
            user_agent="TestBrowser/1.0"
        )
        
        original_expires = session.expires_at
        time.sleep(0.1)
        
        # Refresh session
        assert manager.refresh_session(session.session_id) is True
        
        # Expiration should be extended
        session_updated = manager.get_session(session.session_id)
        assert session_updated.expires_at > original_expires
    
    def test_terminate_user_sessions(self):
        """Test terminating all user sessions."""
        manager = SessionManager()
        
        # Create multiple sessions for user
        session1 = manager.create_session("alice", "192.168.1.1", "Browser1")
        session2 = manager.create_session("alice", "192.168.1.2", "Browser2")
        
        # Terminate all sessions
        count = manager.terminate_user_sessions("alice")
        assert count == 2
        
        # Both should be terminated
        assert manager.get_session(session1.session_id).status == SessionStatus.TERMINATED
        assert manager.get_session(session2.session_id).status == SessionStatus.TERMINATED
    
    def test_session_cleanup(self):
        """Test expired session cleanup."""
        manager = SessionManager(session_lifetime_minutes=0)
        
        # Create expired session
        session = manager.create_session("alice", "192.168.1.1", "Browser1")
        time.sleep(0.1)
        
        # Cleanup
        count = manager.cleanup_expired_sessions()
        assert count >= 1
        
        # Session should be removed
        assert manager.get_session(session.session_id) is None


class TestIntegration:
    """Integration tests combining multiple systems."""
    
    def test_rbac_with_sessions(self):
        """Test RBAC with session management."""
        rbac = RBACManager()
        session_mgr = SessionManager()
        
        # Create user with role
        rbac.assign_role("alice", "operator")
        
        # Create session
        session = session_mgr.create_session(
            username="alice",
            ip_address="192.168.1.1",
            user_agent="TestBrowser/1.0"
        )
        
        # Check permissions
        assert rbac.check_permission("alice", Permission.CAMERA_VIEW) is True
        assert session.is_valid() is True
    
    def test_api_key_with_rbac(self):
        """Test API keys with permission checking."""
        api_key_mgr = APIKeyManager()
        
        # Create key with specific permissions
        key, secret = api_key_mgr.create_key(
            name="service1",
            permissions=["camera:view", "incident:view"]
        )
        
        # Validate key
        validated_key = api_key_mgr.validate_key(secret)
        assert validated_key is not None
        assert "camera:view" in validated_key.permissions
    
    def test_security_audit_trail(self):
        """Test that security events are logged."""
        session_mgr = SessionManager(max_ip_changes=1)
        
        session = session_mgr.create_session("alice", "192.168.1.1", "Browser1")
        
        # Trigger suspicious activity
        session_mgr.validate_session(session.session_id, "192.168.1.2", "Browser1")
        session_mgr.validate_session(session.session_id, "192.168.1.3", "Browser1")
        
        # Should have audit trail
        assert len(session.suspicious_activity) > 0
        assert session.status == SessionStatus.SUSPICIOUS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
