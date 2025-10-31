from datetime import timedelta

from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.security import OAuth2PasswordRequestForm

from .auth import AuthService


def create_auth_router(auth: AuthService) -> APIRouter:
    router = APIRouter(prefix="/auth", tags=["auth"])

    @router.post("/token")
    async def login(form_data: OAuth2PasswordRequestForm = Depends()):
        user = auth.authenticate_user(form_data.username, form_data.password)
        if not user:
            raise HTTPException(status_code=400, detail="Incorrect username or password")
        access_token_expires = timedelta(minutes=auth.access_token_expire_minutes)
        access_token = auth.create_access_token(data={"sub": user["username"], "role": user["role"]}, expires_delta=access_token_expires)
        return {"access_token": access_token, "token_type": "bearer"}

    return router
