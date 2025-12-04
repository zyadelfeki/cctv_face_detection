"""
Enhanced rate limiting with sliding window, burst allowance, and per-user limits.
"""

import time
from typing import Callable, Dict, Optional, Tuple
from collections import defaultdict
import asyncio

from fastapi import Request, HTTPException, status
from loguru import logger


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter with burst allowance.
    
    Uses a sliding window algorithm for more accurate rate limiting
    compared to fixed window approaches.
    """
    
    def __init__(
        self,
        calls: int,
        period: int,
        burst_multiplier: float = 1.5,
        cleanup_interval: int = 300
    ):
        """
        Initialize rate limiter.
        
        Args:
            calls: Maximum calls allowed in period
            period: Time period in seconds
            burst_multiplier: Allow short bursts up to calls * burst_multiplier
            cleanup_interval: How often to clean up old entries (seconds)
        """
        self.calls = calls
        self.period = period
        self.burst_limit = int(calls * burst_multiplier)
        self.cleanup_interval = cleanup_interval
        
        # Store: key -> list of timestamps
        self._requests: Dict[str, list] = defaultdict(list)
        self._last_cleanup = time.time()
        self._lock = asyncio.Lock()
    
    def _get_key(self, request: Request, key_func: Optional[Callable] = None) -> str:
        """Generate rate limit key for request."""
        if key_func:
            return key_func(request)
        
        # Default: use IP address
        client_ip = request.client.host if request.client else "unknown"
        
        # Check for forwarded IP (behind proxy)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        
        return f"ip:{client_ip}"
    
    def _cleanup(self):
        """Remove expired entries."""
        now = time.time()
        if now - self._last_cleanup < self.cleanup_interval:
            return
        
        cutoff = now - self.period
        keys_to_delete = []
        
        for key, timestamps in self._requests.items():
            # Filter old timestamps
            self._requests[key] = [t for t in timestamps if t > cutoff]
            if not self._requests[key]:
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del self._requests[key]
        
        self._last_cleanup = now
    
    async def is_allowed(
        self,
        request: Request,
        key_func: Optional[Callable] = None
    ) -> Tuple[bool, Dict[str, int]]:
        """
        Check if request is allowed.
        
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        async with self._lock:
            self._cleanup()
            
            key = self._get_key(request, key_func)
            now = time.time()
            cutoff = now - self.period
            
            # Get recent requests
            timestamps = self._requests[key]
            recent = [t for t in timestamps if t > cutoff]
            
            # Calculate remaining
            remaining = self.calls - len(recent)
            reset_at = int(now + self.period) if recent else int(now + self.period)
            
            info = {
                "limit": self.calls,
                "remaining": max(0, remaining),
                "reset": reset_at,
                "retry_after": int(self.period - (now - recent[0])) if recent and remaining <= 0 else 0
            }
            
            # Check burst limit (hard limit)
            if len(recent) >= self.burst_limit:
                return False, info
            
            # Check normal limit
            if len(recent) >= self.calls:
                return False, info
            
            # Record this request
            self._requests[key].append(now)
            self._requests[key] = [t for t in self._requests[key] if t > cutoff]
            info["remaining"] = max(0, self.calls - len(self._requests[key]))
            
            return True, info
    
    async def __call__(self, request: Request):
        """FastAPI dependency."""
        allowed, info = await self.is_allowed(request)
        
        # Add rate limit headers
        request.state.rate_limit_info = info
        
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(info["limit"]),
                    "X-RateLimit-Remaining": str(info["remaining"]),
                    "X-RateLimit-Reset": str(info["reset"]),
                    "Retry-After": str(info["retry_after"])
                }
            )


class TieredRateLimiter:
    """
    Tiered rate limiter with different limits for different user roles.
    """
    
    def __init__(self, tiers: Dict[str, Dict[str, int]]):
        """
        Initialize with tier configuration.
        
        Args:
            tiers: Dict mapping tier names to rate limit config
                   e.g., {"admin": {"calls": 1000, "period": 60},
                          "user": {"calls": 100, "period": 60}}
        """
        self._limiters = {
            tier: SlidingWindowRateLimiter(
                calls=config["calls"],
                period=config["period"],
                burst_multiplier=config.get("burst", 1.5)
            )
            for tier, config in tiers.items()
        }
        self._default_tier = "anonymous"
        
        # Add default tier if not present
        if self._default_tier not in self._limiters:
            self._limiters[self._default_tier] = SlidingWindowRateLimiter(
                calls=30, period=60
            )
    
    def get_tier(self, request: Request) -> str:
        """Get user tier from request."""
        # Check for tier in request state (set by auth middleware)
        if hasattr(request.state, "user"):
            return request.state.user.get("role", self._default_tier)
        return self._default_tier
    
    async def __call__(self, request: Request):
        """FastAPI dependency."""
        tier = self.get_tier(request)
        limiter = self._limiters.get(tier, self._limiters[self._default_tier])
        await limiter(request)


class EndpointRateLimiter:
    """
    Per-endpoint rate limiter with different limits for different endpoints.
    """
    
    def __init__(self, default_calls: int = 100, default_period: int = 60):
        self._limiters: Dict[str, SlidingWindowRateLimiter] = {}
        self._default_calls = default_calls
        self._default_period = default_period
    
    def limit(
        self,
        calls: Optional[int] = None,
        period: Optional[int] = None,
        key_func: Optional[Callable] = None
    ):
        """
        Decorator for applying rate limits to endpoints.
        
        Usage:
            limiter = EndpointRateLimiter()
            
            @app.get("/heavy")
            @limiter.limit(calls=10, period=60)
            async def heavy_endpoint():
                pass
        """
        calls = calls or self._default_calls
        period = period or self._default_period
        
        def decorator(func):
            endpoint_key = f"{func.__module__}.{func.__name__}"
            self._limiters[endpoint_key] = SlidingWindowRateLimiter(
                calls=calls, period=period
            )
            
            async def wrapper(request: Request, *args, **kwargs):
                await self._limiters[endpoint_key](request)
                return await func(request, *args, **kwargs)
            
            wrapper.__name__ = func.__name__
            wrapper.__doc__ = func.__doc__
            return wrapper
        
        return decorator


def use_rate_limit(calls: int, period: int):
    """Simple rate limit dependency factory."""
    limiter = SlidingWindowRateLimiter(calls, period)
    
    async def dependency(request: Request):
        await limiter(request)
    
    return dependency


# Pre-configured limiters for common use cases
STANDARD_RATE_LIMIT = SlidingWindowRateLimiter(calls=100, period=60)
STRICT_RATE_LIMIT = SlidingWindowRateLimiter(calls=20, period=60)
AUTH_RATE_LIMIT = SlidingWindowRateLimiter(calls=5, period=60)  # For login endpoints
UPLOAD_RATE_LIMIT = SlidingWindowRateLimiter(calls=10, period=300)  # For file uploads

