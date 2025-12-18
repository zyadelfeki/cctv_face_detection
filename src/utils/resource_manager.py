"""Centralized resource management for the CCTV system.

Manages:
- File handles
- Database connections
- Video streams
- Memory usage
- Cleanup on shutdown
"""

import atexit
import weakref
import threading
import psutil
import os
from typing import Any, Callable, Dict, List, Optional, Set
from contextlib import contextmanager
from pathlib import Path
from loguru import logger


class ResourceManager:
    """Global resource manager with automatic cleanup.
    
    Tracks all open resources and ensures they're properly closed
    on application shutdown or errors.
    
    Usage:
        rm = ResourceManager()
        
        # Register a resource
        file_handle = open('data.txt')
        rm.register(file_handle, file_handle.close)
        
        # Or use context manager
        with rm.managed_resource(file_handle, file_handle.close):
            data = file_handle.read()
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize resource manager."""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._resources: Dict[int, tuple] = {}  # id -> (resource, cleanup_fn, name)
        self._resource_lock = threading.RLock()
        self._shutdown_hooks: List[Callable] = []
        self._max_memory_mb = int(os.getenv('MAX_MEMORY_MB', '2048'))
        
        # Register cleanup on exit
        atexit.register(self.cleanup_all)
        
        logger.info("Resource manager initialized")
    
    def register(
        self,
        resource: Any,
        cleanup_fn: Callable,
        name: Optional[str] = None
    ) -> int:
        """Register a resource for cleanup.
        
        Args:
            resource: The resource object
            cleanup_fn: Function to call to cleanup (e.g., close())
            name: Optional name for debugging
            
        Returns:
            Resource ID for later unregistration
        """
        resource_id = id(resource)
        resource_name = name or f"{type(resource).__name__}_{resource_id}"
        
        with self._resource_lock:
            if resource_id in self._resources:
                logger.warning(f"Resource {resource_name} already registered")
                return resource_id
            
            self._resources[resource_id] = (resource, cleanup_fn, resource_name)
            logger.debug(f"Registered resource: {resource_name}")
        
        return resource_id
    
    def unregister(self, resource_id: int, cleanup: bool = True):
        """Unregister and optionally cleanup a resource.
        
        Args:
            resource_id: ID returned from register()
            cleanup: Whether to call cleanup function
        """
        with self._resource_lock:
            if resource_id not in self._resources:
                return
            
            resource, cleanup_fn, name = self._resources.pop(resource_id)
            
            if cleanup:
                try:
                    cleanup_fn()
                    logger.debug(f"Cleaned up resource: {name}")
                except Exception as e:
                    logger.error(f"Error cleaning up {name}: {e}")
    
    @contextmanager
    def managed_resource(self, resource: Any, cleanup_fn: Callable, name: Optional[str] = None):
        """Context manager for automatic resource cleanup.
        
        Example:
            with rm.managed_resource(file, file.close, "data_file"):
                data = file.read()
            # file automatically closed
        """
        resource_id = self.register(resource, cleanup_fn, name)
        try:
            yield resource
        finally:
            self.unregister(resource_id, cleanup=True)
    
    def cleanup_all(self):
        """Cleanup all registered resources."""
        logger.info("Cleaning up all resources...")
        
        with self._resource_lock:
            # Get all resources (copy to avoid modification during iteration)
            resources = list(self._resources.items())
            
            for resource_id, (resource, cleanup_fn, name) in resources:
                try:
                    cleanup_fn()
                    logger.debug(f"Cleaned up: {name}")
                except Exception as e:
                    logger.error(f"Error cleaning up {name}: {e}")
            
            self._resources.clear()
        
        # Run shutdown hooks
        for hook in self._shutdown_hooks:
            try:
                hook()
            except Exception as e:
                logger.error(f"Error in shutdown hook: {e}")
        
        logger.info("Resource cleanup complete")
    
    def add_shutdown_hook(self, hook: Callable):
        """Add a function to call on shutdown.
        
        Args:
            hook: Function to call (no arguments)
        """
        self._shutdown_hooks.append(hook)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics.
        
        Returns:
            Dict with memory stats in MB
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Physical memory
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual memory
            "percent": process.memory_percent(),
            "available_system_mb": psutil.virtual_memory().available / 1024 / 1024
        }
    
    def check_memory_limit(self) -> bool:
        """Check if process is within memory limits.
        
        Returns:
            True if within limits, False if exceeded
        """
        usage = self.get_memory_usage()
        if usage['rss_mb'] > self._max_memory_mb:
            logger.warning(
                f"Memory limit exceeded: {usage['rss_mb']:.1f}MB > "
                f"{self._max_memory_mb}MB"
            )
            return False
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resource manager statistics.
        
        Returns:
            Dict with stats
        """
        with self._resource_lock:
            resource_types = {}
            for _, (resource, _, name) in self._resources.items():
                res_type = type(resource).__name__
                resource_types[res_type] = resource_types.get(res_type, 0) + 1
            
            return {
                "total_resources": len(self._resources),
                "resource_types": resource_types,
                "memory_usage": self.get_memory_usage(),
                "shutdown_hooks": len(self._shutdown_hooks)
            }


class FileManager:
    """Manage file handles with automatic cleanup."""
    
    def __init__(self, resource_manager: Optional[ResourceManager] = None):
        self.rm = resource_manager or ResourceManager()
        self._open_files: Set[int] = set()
    
    @contextmanager
    def open(self, path: Path, mode: str = 'r', **kwargs):
        """Open file with automatic cleanup.
        
        Example:
            fm = FileManager()
            with fm.open(Path('data.txt'), 'r') as f:
                data = f.read()
        """
        file_handle = None
        try:
            file_handle = open(path, mode, **kwargs)
            file_id = id(file_handle)
            self._open_files.add(file_id)
            
            with self.rm.managed_resource(
                file_handle,
                file_handle.close,
                f"file:{path}"
            ):
                yield file_handle
                
        except Exception as e:
            logger.error(f"Error opening file {path}: {e}")
            if file_handle:
                try:
                    file_handle.close()
                except:
                    pass
            raise
        finally:
            if file_handle:
                file_id = id(file_handle)
                self._open_files.discard(file_id)
    
    def close_all(self):
        """Close all managed files."""
        # The resource manager will handle cleanup
        self._open_files.clear()


class DatabaseConnectionPool:
    """Simple database connection pool."""
    
    def __init__(
        self,
        create_connection: Callable,
        max_connections: int = 10,
        resource_manager: Optional[ResourceManager] = None
    ):
        self.create_connection = create_connection
        self.max_connections = max_connections
        self.rm = resource_manager or ResourceManager()
        self._pool: List[Any] = []
        self._in_use: Set[int] = set()
        self._lock = threading.Lock()
        
        # Register cleanup
        self.rm.add_shutdown_hook(self.close_all)
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool.
        
        Example:
            pool = DatabaseConnectionPool(create_pg_connection)
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(...)
        """
        conn = None
        try:
            with self._lock:
                # Try to reuse existing connection
                if self._pool:
                    conn = self._pool.pop()
                # Create new connection if pool empty and under limit
                elif len(self._in_use) < self.max_connections:
                    conn = self.create_connection()
                else:
                    raise RuntimeError("Connection pool exhausted")
                
                self._in_use.add(id(conn))
            
            yield conn
            
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                try:
                    conn.close()
                except:
                    pass
                conn = None
            raise
            
        finally:
            if conn:
                conn_id = id(conn)
                with self._lock:
                    self._in_use.discard(conn_id)
                    # Return to pool if connection is still good
                    try:
                        # Check if connection is alive (implementation depends on DB)
                        self._pool.append(conn)
                    except:
                        try:
                            conn.close()
                        except:
                            pass
    
    def close_all(self):
        """Close all connections in pool."""
        with self._lock:
            for conn in self._pool:
                try:
                    conn.close()
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
            self._pool.clear()
            self._in_use.clear()


# Global instance
_resource_manager = ResourceManager()


def get_resource_manager() -> ResourceManager:
    """Get global resource manager instance."""
    return _resource_manager
