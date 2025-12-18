"""Role-Based Access Control (RBAC) System.

Provides:
- Fine-grained permissions
- Resource-level access control
- Permission inheritance
- Dynamic permission evaluation
- Audit logging
"""

from enum import Enum
from typing import Set, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from functools import wraps

from fastapi import HTTPException, status, Request
from loguru import logger


class Permission(Enum):
    """System permissions."""
    
    # Camera permissions
    CAMERA_VIEW = "camera:view"
    CAMERA_CREATE = "camera:create"
    CAMERA_UPDATE = "camera:update"
    CAMERA_DELETE = "camera:delete"
    CAMERA_CONTROL = "camera:control"  # Start/stop streams
    
    # Criminal database permissions
    CRIMINAL_VIEW = "criminal:view"
    CRIMINAL_CREATE = "criminal:create"
    CRIMINAL_UPDATE = "criminal:update"
    CRIMINAL_DELETE = "criminal:delete"
    CRIMINAL_SEARCH = "criminal:search"
    
    # Incident permissions
    INCIDENT_VIEW = "incident:view"
    INCIDENT_CREATE = "incident:create"
    INCIDENT_UPDATE = "incident:update"
    INCIDENT_DELETE = "incident:delete"
    INCIDENT_EXPORT = "incident:export"
    
    # Analytics permissions
    ANALYTICS_VIEW = "analytics:view"
    ANALYTICS_EXPORT = "analytics:export"
    
    # User management permissions
    USER_VIEW = "user:view"
    USER_CREATE = "user:create"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    
    # System permissions
    SYSTEM_CONFIG = "system:config"
    SYSTEM_LOGS = "system:logs"
    SYSTEM_BACKUP = "system:backup"
    SYSTEM_HEALTH = "system:health"
    
    # API permissions
    API_READ = "api:read"
    API_WRITE = "api:write"
    API_ADMIN = "api:admin"


class Action(Enum):
    """CRUD actions for resources."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"


@dataclass
class Role:
    """User role with permissions."""
    name: str
    permissions: Set[Permission]
    inherits_from: Optional['Role'] = None
    description: str = ""
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if role has permission (including inherited)."""
        if permission in self.permissions:
            return True
        
        if self.inherits_from:
            return self.inherits_from.has_permission(permission)
        
        return False
    
    def get_all_permissions(self) -> Set[Permission]:
        """Get all permissions including inherited."""
        perms = self.permissions.copy()
        
        if self.inherits_from:
            perms.update(self.inherits_from.get_all_permissions())
        
        return perms


class RBACManager:
    """Manages roles and permissions.
    
    Features:
    - Role hierarchy with inheritance
    - Resource-specific permissions
    - Dynamic permission checking
    - Audit logging
    """
    
    def __init__(self):
        self._roles: Dict[str, Role] = {}
        self._user_roles: Dict[str, str] = {}  # username -> role_name
        self._resource_owners: Dict[str, str] = {}  # resource_id -> owner_username
        
        # Initialize default roles
        self._init_default_roles()
    
    def _init_default_roles(self):
        """Initialize system default roles."""
        
        # Viewer role - read-only access
        viewer = Role(
            name="viewer",
            permissions={
                Permission.CAMERA_VIEW,
                Permission.INCIDENT_VIEW,
                Permission.ANALYTICS_VIEW,
                Permission.SYSTEM_HEALTH,
                Permission.API_READ
            },
            description="Read-only access to system resources"
        )
        
        # Operator role - can manage cameras and incidents
        operator = Role(
            name="operator",
            permissions={
                Permission.CAMERA_CREATE,
                Permission.CAMERA_UPDATE,
                Permission.CAMERA_CONTROL,
                Permission.CRIMINAL_VIEW,
                Permission.CRIMINAL_SEARCH,
                Permission.INCIDENT_CREATE,
                Permission.INCIDENT_UPDATE,
                Permission.API_WRITE
            },
            inherits_from=viewer,
            description="Operational access - manage cameras and incidents"
        )
        
        # Analyst role - can manage criminal database
        analyst = Role(
            name="analyst",
            permissions={
                Permission.CRIMINAL_CREATE,
                Permission.CRIMINAL_UPDATE,
                Permission.CRIMINAL_DELETE,
                Permission.INCIDENT_EXPORT,
                Permission.ANALYTICS_EXPORT
            },
            inherits_from=operator,
            description="Analyst access - manage criminal database and analytics"
        )
        
        # Admin role - full access
        admin = Role(
            name="admin",
            permissions={
                Permission.CAMERA_DELETE,
                Permission.INCIDENT_DELETE,
                Permission.USER_VIEW,
                Permission.USER_CREATE,
                Permission.USER_UPDATE,
                Permission.USER_DELETE,
                Permission.SYSTEM_CONFIG,
                Permission.SYSTEM_LOGS,
                Permission.SYSTEM_BACKUP,
                Permission.API_ADMIN
            },
            inherits_from=analyst,
            description="Full administrative access"
        )
        
        # Register roles
        self._roles = {
            "viewer": viewer,
            "operator": operator,
            "analyst": analyst,
            "admin": admin
        }
    
    def create_role(
        self,
        name: str,
        permissions: Set[Permission],
        inherits_from: Optional[str] = None,
        description: str = ""
    ) -> Role:
        """
        Create a custom role.
        
        Args:
            name: Role name
            permissions: Set of permissions
            inherits_from: Optional parent role name
            description: Role description
        
        Returns:
            Created Role
        """
        parent_role = None
        if inherits_from:
            parent_role = self._roles.get(inherits_from)
            if not parent_role:
                raise ValueError(f"Parent role '{inherits_from}' not found")
        
        role = Role(
            name=name,
            permissions=permissions,
            inherits_from=parent_role,
            description=description
        )
        
        self._roles[name] = role
        logger.info(f"Created role '{name}' with {len(permissions)} permissions")
        return role
    
    def assign_role(self, username: str, role_name: str):
        """
        Assign role to user.
        
        Args:
            username: User identifier
            role_name: Role to assign
        """
        if role_name not in self._roles:
            raise ValueError(f"Role '{role_name}' not found")
        
        self._user_roles[username] = role_name
        logger.info(f"Assigned role '{role_name}' to user '{username}'")
    
    def get_user_role(self, username: str) -> Optional[Role]:
        """Get user's role."""
        role_name = self._user_roles.get(username)
        if not role_name:
            return None
        return self._roles.get(role_name)
    
    def check_permission(
        self,
        username: str,
        permission: Permission,
        resource_id: Optional[str] = None
    ) -> bool:
        """
        Check if user has permission.
        
        Args:
            username: User identifier
            permission: Required permission
            resource_id: Optional resource ID for ownership check
        
        Returns:
            True if user has permission
        """
        # Get user role
        role = self.get_user_role(username)
        if not role:
            logger.warning(f"User '{username}' has no role assigned")
            return False
        
        # Check role permission
        has_perm = role.has_permission(permission)
        
        # If checking specific resource, verify ownership or higher permission
        if resource_id and not has_perm:
            owner = self._resource_owners.get(resource_id)
            if owner == username:
                # User owns the resource
                has_perm = True
        
        if not has_perm:
            logger.warning(
                f"User '{username}' (role: {role.name}) denied permission "
                f"'{permission.value}' for resource '{resource_id}'"
            )
        
        return has_perm
    
    def require_permission(self, permission: Permission):
        """
        Decorator for requiring permission.
        
        Usage:
            @rbac.require_permission(Permission.CAMERA_CREATE)
            async def create_camera(...):
                pass
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract request and user from kwargs
                request = kwargs.get('request')
                if not request:
                    for arg in args:
                        if isinstance(arg, Request):
                            request = arg
                            break
                
                if not request:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Cannot determine request context"
                    )
                
                # Get user from request state (set by auth middleware)
                user = getattr(request.state, 'user', None)
                if not user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required"
                    )
                
                username = user.get('username')
                
                # Check permission
                if not self.check_permission(username, permission):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Permission '{permission.value}' required"
                    )
                
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def require_any_permission(self, *permissions: Permission):
        """
        Decorator for requiring any of the given permissions.
        
        Usage:
            @rbac.require_any_permission(
                Permission.CAMERA_VIEW,
                Permission.CAMERA_UPDATE
            )
            async def get_camera(...):
                pass
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                request = kwargs.get('request')
                if not request:
                    for arg in args:
                        if isinstance(arg, Request):
                            request = arg
                            break
                
                user = getattr(request.state, 'user', None)
                if not user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required"
                    )
                
                username = user.get('username')
                
                # Check if user has any of the permissions
                has_any = any(
                    self.check_permission(username, perm)
                    for perm in permissions
                )
                
                if not has_any:
                    perm_names = [p.value for p in permissions]
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"One of these permissions required: {', '.join(perm_names)}"
                    )
                
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def set_resource_owner(self, resource_id: str, username: str):
        """Set resource owner for ownership-based access control."""
        self._resource_owners[resource_id] = username
    
    def get_role_permissions(self, role_name: str) -> Set[Permission]:
        """Get all permissions for a role (including inherited)."""
        role = self._roles.get(role_name)
        if not role:
            return set()
        return role.get_all_permissions()
    
    def list_roles(self) -> List[Dict]:
        """List all roles with their permissions."""
        return [
            {
                "name": role.name,
                "description": role.description,
                "permissions": [p.value for p in role.get_all_permissions()],
                "inherits_from": role.inherits_from.name if role.inherits_from else None
            }
            for role in self._roles.values()
        ]


# Global RBAC manager instance
_rbac_manager: Optional[RBACManager] = None


def get_rbac_manager() -> RBACManager:
    """Get global RBAC manager instance."""
    global _rbac_manager
    if _rbac_manager is None:
        _rbac_manager = RBACManager()
    return _rbac_manager


# Convenience dependency for FastAPI
def check_permission(permission: Permission):
    """
    FastAPI dependency for permission checking.
    
    Usage:
        @app.get("/cameras")
        async def list_cameras(
            user: dict = Depends(check_permission(Permission.CAMERA_VIEW))
        ):
            pass
    """
    def dependency(request: Request) -> Dict:
        user = getattr(request.state, 'user', None)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        rbac = get_rbac_manager()
        username = user.get('username')
        
        if not rbac.check_permission(username, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission.value}' required"
            )
        
        return user
    
    return dependency
