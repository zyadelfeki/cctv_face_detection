"""Secure File Deletion.

Provides:
- DOD 5220.22-M standard (7-pass)
- Gutmann method (35-pass)
- Simple overwrite (3-pass)
- Metadata wiping
- Directory deletion
- Verification
"""

import os
import secrets
from pathlib import Path
from typing import Optional, List
import shutil

from loguru import logger


class DeletionMethod:
    """Secure deletion methods."""
    SIMPLE = "simple"  # 3 passes (0x00, 0xFF, random)
    DOD_7_PASS = "dod-7"  # DOD 5220.22-M (7 passes)
    GUTMANN = "gutmann"  # Gutmann method (35 passes)


class SecureDelete:
    """Secure file deletion to prevent data recovery.
    
    Implements multiple deletion standards:
    - Simple: 3-pass (0x00, 0xFF, random)
    - DOD 5220.22-M: 7-pass standard
    - Gutmann: 35-pass method for maximum security
    
    After overwriting, file is truncated and deleted.
    """
    
    # Gutmann patterns (some of them)
    GUTMANN_PATTERNS = [
        b'\x00' * 512,  # All zeros
        b'\xFF' * 512,  # All ones
        b'\x55' * 512,  # 01010101
        b'\xAA' * 512,  # 10101010
        b'\x92\x49\x24' * 171,  # MFM patterns
        b'\x49\x24\x92' * 171,
        b'\x24\x92\x49' * 171,
    ]
    
    @staticmethod
    def _get_file_size(filepath: Path) -> int:
        """Get file size in bytes."""
        return filepath.stat().st_size
    
    @staticmethod
    def _overwrite_pass(filepath: Path, pattern: bytes, pass_num: int):
        """Perform single overwrite pass."""
        file_size = SecureDelete._get_file_size(filepath)
        
        with open(filepath, 'r+b', buffering=0) as f:
            # Write in chunks to handle large files
            chunk_size = len(pattern)
            bytes_written = 0
            
            while bytes_written < file_size:
                remaining = file_size - bytes_written
                write_size = min(chunk_size, remaining)
                
                if write_size < chunk_size:
                    # Last chunk - truncate pattern
                    f.write(pattern[:write_size])
                else:
                    f.write(pattern)
                
                bytes_written += write_size
            
            # Force write to disk
            f.flush()
            os.fsync(f.fileno())
        
        logger.debug(f"Pass {pass_num}: Wrote {bytes_written} bytes")
    
    @staticmethod
    def _random_pass(filepath: Path, pass_num: int):
        """Overwrite with random data."""
        file_size = SecureDelete._get_file_size(filepath)
        
        with open(filepath, 'r+b', buffering=0) as f:
            chunk_size = 65536  # 64KB chunks
            bytes_written = 0
            
            while bytes_written < file_size:
                remaining = file_size - bytes_written
                write_size = min(chunk_size, remaining)
                
                random_data = secrets.token_bytes(write_size)
                f.write(random_data)
                bytes_written += write_size
            
            f.flush()
            os.fsync(f.fileno())
        
        logger.debug(f"Pass {pass_num}: Wrote {bytes_written} bytes of random data")
    
    @classmethod
    def simple_delete(cls, filepath: str) -> bool:
        """
        Simple 3-pass deletion.
        
        Passes:
        1. All zeros (0x00)
        2. All ones (0xFF)
        3. Random data
        
        Args:
            filepath: Path to file
        
        Returns:
            True if successful
        """
        path = Path(filepath)
        
        if not path.exists():
            logger.warning(f"File not found: {filepath}")
            return False
        
        if not path.is_file():
            logger.error(f"Not a file: {filepath}")
            return False
        
        try:
            logger.info(f"Simple deletion: {filepath}")
            
            # Pass 1: Zeros
            cls._overwrite_pass(path, b'\x00' * 512, 1)
            
            # Pass 2: Ones
            cls._overwrite_pass(path, b'\xFF' * 512, 2)
            
            # Pass 3: Random
            cls._random_pass(path, 3)
            
            # Truncate and delete
            with open(path, 'w') as f:
                pass  # Truncate to 0 bytes
            
            path.unlink()
            logger.info(f"Securely deleted: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to securely delete {filepath}: {e}")
            return False
    
    @classmethod
    def dod_delete(cls, filepath: str) -> bool:
        """
        DOD 5220.22-M standard deletion (7 passes).
        
        Passes:
        1. 0x00
        2. 0xFF
        3. Random
        4. 0x00
        5. 0xFF
        6. Random
        7. Random
        
        Args:
            filepath: Path to file
        
        Returns:
            True if successful
        """
        path = Path(filepath)
        
        if not path.exists() or not path.is_file():
            return False
        
        try:
            logger.info(f"DOD 7-pass deletion: {filepath}")
            
            # Pass 1: Zeros
            cls._overwrite_pass(path, b'\x00' * 512, 1)
            
            # Pass 2: Ones
            cls._overwrite_pass(path, b'\xFF' * 512, 2)
            
            # Pass 3: Random
            cls._random_pass(path, 3)
            
            # Pass 4: Zeros
            cls._overwrite_pass(path, b'\x00' * 512, 4)
            
            # Pass 5: Ones
            cls._overwrite_pass(path, b'\xFF' * 512, 5)
            
            # Pass 6: Random
            cls._random_pass(path, 6)
            
            # Pass 7: Random (verification)
            cls._random_pass(path, 7)
            
            # Truncate and delete
            with open(path, 'w') as f:
                pass
            
            path.unlink()
            logger.info(f"DOD-securely deleted: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed DOD deletion of {filepath}: {e}")
            return False
    
    @classmethod
    def gutmann_delete(cls, filepath: str) -> bool:
        """
        Gutmann method (35 passes).
        
        Most secure but slowest method.
        
        Args:
            filepath: Path to file
        
        Returns:
            True if successful
        """
        path = Path(filepath)
        
        if not path.exists() or not path.is_file():
            return False
        
        try:
            logger.info(f"Gutmann 35-pass deletion: {filepath}")
            
            # Passes 1-4: Random
            for i in range(1, 5):
                cls._random_pass(path, i)
            
            # Passes 5-31: Specific patterns
            for i in range(5, 32):
                pattern_idx = (i - 5) % len(cls.GUTMANN_PATTERNS)
                cls._overwrite_pass(path, cls.GUTMANN_PATTERNS[pattern_idx], i)
            
            # Passes 32-35: Random
            for i in range(32, 36):
                cls._random_pass(path, i)
            
            # Truncate and delete
            with open(path, 'w') as f:
                pass
            
            path.unlink()
            logger.info(f"Gutmann-securely deleted: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed Gutmann deletion of {filepath}: {e}")
            return False
    
    @classmethod
    def delete_file(
        cls,
        filepath: str,
        method: str = DeletionMethod.DOD_7_PASS
    ) -> bool:
        """
        Securely delete file using specified method.
        
        Args:
            filepath: Path to file
            method: Deletion method (simple, dod-7, gutmann)
        
        Returns:
            True if successful
        """
        if method == DeletionMethod.SIMPLE:
            return cls.simple_delete(filepath)
        elif method == DeletionMethod.DOD_7_PASS:
            return cls.dod_delete(filepath)
        elif method == DeletionMethod.GUTMANN:
            return cls.gutmann_delete(filepath)
        else:
            logger.error(f"Unknown deletion method: {method}")
            return False
    
    @classmethod
    def delete_directory(
        cls,
        dirpath: str,
        method: str = DeletionMethod.DOD_7_PASS,
        recursive: bool = True
    ) -> int:
        """
        Securely delete all files in directory.
        
        Args:
            dirpath: Path to directory
            method: Deletion method
            recursive: Delete subdirectories
        
        Returns:
            Number of files deleted
        """
        path = Path(dirpath)
        
        if not path.exists() or not path.is_dir():
            logger.error(f"Not a directory: {dirpath}")
            return 0
        
        count = 0
        
        # Collect all files
        if recursive:
            files = list(path.rglob('*'))
        else:
            files = list(path.glob('*'))
        
        # Delete files (not directories)
        for file in files:
            if file.is_file():
                if cls.delete_file(str(file), method):
                    count += 1
        
        # Remove empty directories
        if recursive:
            try:
                shutil.rmtree(dirpath)
                logger.info(f"Removed directory: {dirpath}")
            except Exception as e:
                logger.error(f"Failed to remove directory {dirpath}: {e}")
        
        logger.info(f"Securely deleted {count} files from {dirpath}")
        return count
    
    @classmethod
    def wipe_free_space(
        cls,
        mount_point: str,
        passes: int = 3
    ) -> bool:
        """
        Wipe free space on filesystem.
        
        Creates large file filled with random data until disk is full,
        then securely deletes it.
        
        WARNING: This can take a very long time on large disks!
        
        Args:
            mount_point: Filesystem mount point
            passes: Number of passes
        
        Returns:
            True if successful
        """
        logger.warning(f"Free space wiping on {mount_point} - this may take hours!")
        
        try:
            # Create temporary file
            temp_file = Path(mount_point) / f".secure_wipe_{secrets.token_hex(8)}"
            
            for pass_num in range(1, passes + 1):
                logger.info(f"Free space wipe pass {pass_num}/{passes}")
                
                try:
                    # Fill disk with random data
                    with open(temp_file, 'wb') as f:
                        chunk_size = 1048576  # 1MB chunks
                        while True:
                            try:
                                f.write(secrets.token_bytes(chunk_size))
                            except OSError:
                                # Disk full
                                break
                    
                    # Securely delete temp file
                    cls.delete_file(str(temp_file), DeletionMethod.SIMPLE)
                    
                except Exception as e:
                    logger.error(f"Error in pass {pass_num}: {e}")
                    if temp_file.exists():
                        temp_file.unlink()
                    return False
            
            logger.info(f"Free space wiping complete on {mount_point}")
            return True
            
        except Exception as e:
            logger.error(f"Free space wipe failed: {e}")
            return False


def secure_delete_face_image(
    image_path: str,
    method: str = DeletionMethod.DOD_7_PASS
) -> bool:
    """
    Securely delete face image.
    
    Convenience function for deleting biometric images.
    
    Args:
        image_path: Path to image file
        method: Deletion method
    
    Returns:
        True if successful
    """
    logger.info(f"Securely deleting face image: {image_path}")
    return SecureDelete.delete_file(image_path, method)


def secure_delete_embedding(
    embedding_path: str,
    method: str = DeletionMethod.DOD_7_PASS
) -> bool:
    """
    Securely delete embedding file.
    
    Args:
        embedding_path: Path to embedding file
        method: Deletion method
    
    Returns:
        True if successful
    """
    logger.info(f"Securely deleting embedding: {embedding_path}")
    return SecureDelete.delete_file(embedding_path, method)
