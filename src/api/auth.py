from datetime import datetime, timedelta
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext

# Simple in-memory user store for demo; replace with DB-backed users
USERS = {
    "admin": {
        "username": "admin",
        "role": "admin",
        "password_hash": "$2b$12$Hk9YI0t8vD5TQ0dV4nKc6O8mfr7QkJk1fLQ8v8U9XxqgW7Oa3k0We"  # bcrypt for 'admin123'
    },
    "operator": {
        "username": "operator",
        "role": "operator",
        "password_hash": "$2b$12$4n2m0v7f6W2nZ5m6QGfZGe8PpQO4X1W0JmO3B2H8Lqf2Ue6SNo0s6"  # bcrypt for 'operator123'
    }
}

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


class AuthService:
    def __init__(self, secret_key: str, algorithm: str = "HS256", access_token_expire_minutes: int = 30):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes

    def verify_password(self, plain_password: str, password_hash: str) -> bool:
        return pwd_context.verify(plain_password, password_hash)

    def authenticate_user(self, username: str, password: str) -> Optional[dict]:
        user = USERS.get(username)
        if not user:
            return None
        if not self.verify_password(password, user["password_hash"]):
            return None
        return user

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=self.access_token_expire_minutes))
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def decode_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.PyJWTError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")


def get_current_user(auth: 'AuthService' , token: str = Depends(oauth2_scheme)) -> dict:
    payload = auth.decode_token(token)
    username = payload.get("sub")
    if not username or username not in USERS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
    return USERS[username]


def require_role(*roles: str):
    def dependency(user: dict = Depends(get_current_user)):
        if user.get("role") not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return user
    return dependency
