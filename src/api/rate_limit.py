import time
from typing import Callable

from fastapi import Request, HTTPException


class RateLimiter:
    def __init__(self, calls: int, period: int):
        self.calls = calls
        self.period = period
        self.store = {}

    def __call__(self, request: Request):
        now = time.time()
        key = request.client.host
        window = int(now // self.period)
        counter_key = f"{key}:{window}"
        count = self.store.get(counter_key, 0)
        if count >= self.calls:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        self.store[counter_key] = count + 1


def use_rate_limit(calls: int, period: int):
    limiter = RateLimiter(calls, period)
    async def dependency(request: Request):
        limiter(request)
    return dependency
