# routers/auth_router.py
from typing import Optional
from fastapi import APIRouter, Depends, status, Request, Response
from sqlalchemy.orm import Session
from database import get_db
from schemas.auth_schemas import (
    UserCreate, UserLogin, UserResponse, 
    TokenResponse, TokenRefreshRequest, MessageResponse
)
from services.auth_service import AuthService, get_current_user
from repositories.user_repository import UserRepository
from models.auth_models import User

auth_router = APIRouter(prefix="/auth", tags=["Authentication"])


@auth_router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    """Register a new user (Clinician or Parent)."""
    return AuthService.register_user(db, user_in)


@auth_router.post("/login", response_model=TokenResponse)
def login(login_in: UserLogin, response: Response, db: Session = Depends(get_db)):
    """Authenticate credentials and return JWT access/refresh tokens, setting HttpOnly cookies."""
    user = AuthService.authenticate_user(db, login_in.username_or_email, login_in.password)
    tokens = AuthService.create_tokens(user)
    # Persist refresh token in DB
    UserRepository.update_refresh_token(db, user.id, tokens.refresh_token)

    # Set secure HttpOnly cookies for browser navigation
    response.set_cookie(
        key="access_token",
        value=tokens.access_token,
        httponly=True,
        max_age=tokens.expires_in,
        samesite="lax",
        secure=False,  # Set to True in production with HTTPS
        path="/"
    )
    response.set_cookie(
        key="refresh_token",
        value=tokens.refresh_token,
        httponly=True,
        max_age=7 * 24 * 3600,
        samesite="lax",
        secure=False,
        path="/"
    )
    return tokens


@auth_router.post("/refresh", response_model=TokenResponse)
def refresh(refresh_in: TokenRefreshRequest, response: Response, db: Session = Depends(get_db)):
    """Refresh expired access token using a valid refresh token."""
    tokens = AuthService.refresh_tokens(db, refresh_in.refresh_token)
    response.set_cookie(
        key="access_token",
        value=tokens.access_token,
        httponly=True,
        max_age=tokens.expires_in,
        samesite="lax",
        secure=False,
        path="/"
    )
    return tokens


@auth_router.post("/logout", response_model=MessageResponse, status_code=status.HTTP_200_OK)
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    """Invalidate current user refresh token and clear cookies."""
    token = None
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]
    elif "access_token" in request.cookies:
        token = request.cookies.get("access_token")

    if token:
        try:
            token_data = AuthService.verify_access_token(token)
            user = UserRepository.get_by_username(db, token_data.username)
            if user:
                UserRepository.update_refresh_token(db, user.id, None)
        except Exception:
            pass

    response.delete_cookie("access_token", path="/")
    response.delete_cookie("refresh_token", path="/")
    return MessageResponse(detail="Logged out successfully.")


@auth_router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Retrieve details of the currently authenticated user."""
    return current_user
