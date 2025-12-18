"""Tests for resource management."""

import os
import tempfile
import time
from pathlib import Path
import pytest
import numpy as np

from src.utils.resource_manager import (
    ResourceManager,
    FileManager,
    get_resource_manager
)


class TestResourceManager:
    """Test ResourceManager functionality."""
    
    def test_singleton_pattern(self):
        """Test that ResourceManager is a singleton."""
        rm1 = ResourceManager()
        rm2 = ResourceManager()
        assert rm1 is rm2
        
        rm3 = get_resource_manager()
        assert rm1 is rm3
    
    def test_register_and_cleanup(self):
        """Test resource registration and cleanup."""
        rm = ResourceManager()
        
        # Create a mock resource
        cleanup_called = [False]
        
        def cleanup_fn():
            cleanup_called[0] = True
        
        resource = object()
        resource_id = rm.register(resource, cleanup_fn, "test_resource")
        
        assert resource_id == id(resource)
        assert cleanup_called[0] is False
        
        # Cleanup
        rm.unregister(resource_id, cleanup=True)
        assert cleanup_called[0] is True
    
    def test_managed_resource_context(self):
        """Test context manager for resources."""
        rm = ResourceManager()
        cleanup_called = [False]
        
        def cleanup_fn():
            cleanup_called[0] = True
        
        resource = object()
        
        with rm.managed_resource(resource, cleanup_fn, "ctx_resource"):
            assert cleanup_called[0] is False
        
        # Should be cleaned up after context
        assert cleanup_called[0] is True
    
    def test_cleanup_on_exception(self):
        """Test that resources are cleaned up even on exceptions."""
        rm = ResourceManager()
        cleanup_called = [False]
        
        def cleanup_fn():
            cleanup_called[0] = True
        
        resource = object()
        
        try:
            with rm.managed_resource(resource, cleanup_fn, "exc_resource"):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should still be cleaned up
        assert cleanup_called[0] is True
    
    def test_memory_tracking(self):
        """Test memory usage tracking."""
        rm = ResourceManager()
        
        memory_stats = rm.get_memory_usage()
        
        assert 'rss_mb' in memory_stats
        assert 'vms_mb' in memory_stats
        assert 'percent' in memory_stats
        assert 'available_system_mb' in memory_stats
        
        # Memory usage should be positive
        assert memory_stats['rss_mb'] > 0
        assert memory_stats['available_system_mb'] > 0
    
    def test_memory_limit_check(self):
        """Test memory limit checking."""
        rm = ResourceManager()
        
        # Should be within limits for tests
        assert rm.check_memory_limit() is True
    
    def test_stats(self):
        """Test getting resource manager stats."""
        rm = ResourceManager()
        
        # Register some test resources
        rm.register(object(), lambda: None, "test1")
        rm.register("string", lambda: None, "test2")
        rm.register(123, lambda: None, "test3")
        
        stats = rm.get_stats()
        
        assert 'total_resources' in stats
        assert stats['total_resources'] >= 3
        assert 'resource_types' in stats
        assert 'memory_usage' in stats
    
    def test_shutdown_hooks(self):
        """Test shutdown hook registration."""
        rm = ResourceManager()
        hook_called = [False]
        
        def shutdown_hook():
            hook_called[0] = True
        
        rm.add_shutdown_hook(shutdown_hook)
        
        # Call cleanup_all to trigger hooks
        rm.cleanup_all()
        
        assert hook_called[0] is True


class TestFileManager:
    """Test FileManager functionality."""
    
    def test_file_open_and_close(self):
        """Test automatic file closure."""
        fm = FileManager()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write("test content")
        
        try:
            # Open and read file
            with fm.open(tmp_path, 'r') as f:
                content = f.read()
                assert content == "test content"
            
            # File should be closed after context
            # Try to open again (should work if properly closed)
            with fm.open(tmp_path, 'r') as f:
                content = f.read()
        
        finally:
            # Cleanup
            tmp_path.unlink()
    
    def test_file_exception_handling(self):
        """Test file closure on exceptions."""
        fm = FileManager()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write("test")
        
        try:
            with pytest.raises(ValueError):
                with fm.open(tmp_path, 'r') as f:
                    raise ValueError("Test error")
            
            # File should still be accessible after exception
            with fm.open(tmp_path, 'r') as f:
                content = f.read()
                assert content == "test"
        
        finally:
            tmp_path.unlink()
    
    def test_file_write(self):
        """Test file writing with auto-cleanup."""
        fm = FileManager()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.txt"
            
            # Write file
            with fm.open(file_path, 'w') as f:
                f.write("Hello, World!")
            
            # Read back
            with fm.open(file_path, 'r') as f:
                content = f.read()
                assert content == "Hello, World!"


class TestVideoStreamResourceManagement:
    """Test VideoStream resource management."""
    
    def test_context_manager(self):
        """Test VideoStream context manager."""
        # We can't test with real camera, but we can test the interface
        from src.core.video_stream import VideoStream
        
        # Create stream (won't connect)
        stream = VideoStream("fake_url.mp4")
        
        # Test that context manager methods exist
        assert hasattr(stream, '__enter__')
        assert hasattr(stream, '__exit__')
        assert hasattr(stream, '__del__')
    
    def test_close_idempotent(self):
        """Test that close() can be called multiple times safely."""
        from src.core.video_stream import VideoStream
        
        stream = VideoStream("fake_url.mp4")
        
        # Should not raise error
        stream.close()
        stream.close()  # Second call should be safe
        stream.close()  # Third call should be safe
    
    def test_thread_safety(self):
        """Test that VideoStream is thread-safe."""
        from src.core.video_stream import VideoStream
        import threading
        
        stream = VideoStream("fake_url.mp4")
        errors = []
        
        def access_stream():
            try:
                stream.is_opened()
                stream.close()
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = [threading.Thread(target=access_stream) for _ in range(10)]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Should not have any threading errors
        assert len(errors) == 0


class TestIntegration:
    """Integration tests for resource management."""
    
    def test_multiple_resources(self):
        """Test managing multiple different resource types."""
        rm = ResourceManager()
        fm = FileManager(rm)
        
        cleanup_counts = {'file': 0, 'obj': 0}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write("test")
        
        try:
            # Mix file and object resources
            with fm.open(tmp_path, 'r') as f:
                obj = object()
                
                def obj_cleanup():
                    cleanup_counts['obj'] += 1
                
                with rm.managed_resource(obj, obj_cleanup, "test_obj"):
                    # Both resources active
                    content = f.read()
                    assert content == "test"
                
                # Object cleaned up
                assert cleanup_counts['obj'] == 1
            
            # File also cleaned up (implicitly)
        
        finally:
            tmp_path.unlink()
    
    def test_stress_test(self):
        """Stress test with many resources."""
        rm = ResourceManager()
        cleanup_count = [0]
        
        def cleanup_fn():
            cleanup_count[0] += 1
        
        # Register many resources
        n_resources = 100
        resource_ids = []
        
        for i in range(n_resources):
            obj = f"resource_{i}"
            rid = rm.register(obj, cleanup_fn, f"res_{i}")
            resource_ids.append(rid)
        
        # Cleanup all
        for rid in resource_ids:
            rm.unregister(rid, cleanup=True)
        
        assert cleanup_count[0] == n_resources


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
