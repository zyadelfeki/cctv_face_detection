from typing import Optional
import time
import threading

import numpy as np
import cv2
from loguru import logger


class VideoStream:
    """Thread-safe video stream with automatic resource management.
    
    Features:
    - Context manager support for guaranteed cleanup
    - Automatic reconnection on failures
    - Thread-safe operations
    - Proper resource release
    
    Example:
        with VideoStream(url) as stream:
            frame = stream.read()
    """
    
    def __init__(
        self,
        url: str,
        *,
        buffer_size: int = 1,
        reconnect_interval: int = 5,
        timeout: int = 10,
        max_reconnect_attempts: int = 3
    ):
        self.url = url
        self.buffer_size = buffer_size
        self.reconnect_interval = reconnect_interval
        self.timeout = timeout
        self.max_reconnect_attempts = max_reconnect_attempts
        self.cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._reconnect_count = 0
        self._last_error_time = 0

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False

    def __del__(self):
        """Destructor - final cleanup."""
        try:
            self.close()
        except Exception as e:
            logger.error(f"Error in VideoStream destructor: {e}")

    def open(self):
        """Open video stream with error handling."""
        with self._lock:
            if self.cap is not None and self.cap.isOpened():
                return  # Already open
            
            try:
                self.cap = cv2.VideoCapture(self.url)
                
                if not self.cap.isOpened():
                    raise RuntimeError(f"Failed to open video stream: {self.url}")
                
                # Reduce internal buffering for low-latency
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
                
                logger.info(f"Opened video stream: {self.url}")
                self._reconnect_count = 0
                
            except Exception as e:
                logger.error(f"Error opening video stream {self.url}: {e}")
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                raise

    def read(self) -> Optional[np.ndarray]:
        """Read frame with automatic reconnection on failure.
        
        Returns:
            Frame as numpy array, or None if read failed
        """
        with self._lock:
            # Ensure stream is open
            if self.cap is None or not self.cap.isOpened():
                try:
                    self._reconnect()
                except Exception as e:
                    logger.error(f"Failed to reconnect: {e}")
                    return None
            
            try:
                ok, frame = self.cap.read()
                
                if not ok:
                    logger.warning(f"Failed to read frame from {self.url}")
                    # Try to reconnect
                    self._reconnect()
                    return None
                
                return frame
                
            except Exception as e:
                logger.error(f"Exception while reading frame: {e}")
                self._reconnect()
                return None

    def _reconnect(self):
        """Internal reconnection with exponential backoff."""
        current_time = time.time()
        
        # Rate limit reconnection attempts
        if current_time - self._last_error_time < self.reconnect_interval:
            time.sleep(0.1)
            return
        
        self._last_error_time = current_time
        
        if self._reconnect_count >= self.max_reconnect_attempts:
            raise RuntimeError(
                f"Max reconnection attempts ({self.max_reconnect_attempts}) "
                f"exceeded for {self.url}"
            )
        
        logger.info(
            f"Attempting to reconnect ({self._reconnect_count + 1}/"
            f"{self.max_reconnect_attempts})..."
        )
        
        # Close existing connection
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.warning(f"Error releasing capture during reconnect: {e}")
            finally:
                self.cap = None
        
        # Exponential backoff
        wait_time = self.reconnect_interval * (2 ** self._reconnect_count)
        time.sleep(min(wait_time, 60))  # Cap at 60 seconds
        
        self._reconnect_count += 1
        
        try:
            self.open()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            raise

    def close(self):
        """Close video stream and release resources."""
        with self._lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                    logger.info(f"Closed video stream: {self.url}")
                except Exception as e:
                    logger.error(f"Error releasing video capture: {e}")
                finally:
                    self.cap = None

    def is_opened(self) -> bool:
        """Check if stream is open."""
        with self._lock:
            return self.cap is not None and self.cap.isOpened()

    def get_property(self, prop_id: int) -> float:
        """Get stream property.
        
        Args:
            prop_id: OpenCV property ID (e.g., cv2.CAP_PROP_FPS)
            
        Returns:
            Property value, or 0.0 if not available
        """
        with self._lock:
            if self.cap is None or not self.cap.isOpened():
                return 0.0
            try:
                return self.cap.get(prop_id)
            except Exception as e:
                logger.error(f"Error getting property {prop_id}: {e}")
                return 0.0
