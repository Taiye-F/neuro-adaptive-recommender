# services/auth_service.py
import os
from datetime import datetime, timezone, timedelta
from typing import Optional, List
import bcrypt
import jwt
from sqlalchemy.orm import Session
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from database import get_db, SessionLocal
from models.auth_models import User
from repositories.user_repository import UserRepository
from schemas.auth_schemas import UserCreate, TokenResponse, TokenData

SECRET_KEY = os.getenv("JWT_SECRET")
if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET environment variable is missing. The authentication service cannot start without a secure secret key.")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Security scheme for token extraction
security = HTTPBearer(auto_error=False)

class AuthService:
    @staticmethod
    def hash_password(password: str) -> str:
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        try:
            return bcrypt.checkpw(
                plain_password.encode("utf-8"), 
                hashed_password.encode("utf-8")
            )
        except Exception:
            return False

    @staticmethod
    def create_tokens(user: User) -> TokenResponse:
        now = datetime.now(timezone.utc)
        
        # Access Token (contains username, email, role)
        access_expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_payload = {
            "sub": user.username,
            "email": user.email,
            "role": user.role,
            "exp": access_expire
        }
        access_token = jwt.encode(access_payload, SECRET_KEY, algorithm=ALGORITHM)
        
        # Refresh Token (contains username only)
        refresh_expire = now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        refresh_payload = {
            "sub": user.username,
            "exp": refresh_expire
        }
        refresh_token = jwt.encode(refresh_payload, SECRET_KEY, algorithm=ALGORITHM)
        
        expires_in = ACCESS_TOKEN_EXPIRE_MINUTES * 60
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=expires_in,
            role=user.role
        )

    @staticmethod
    def register_user(db: Session, user_in: UserCreate) -> User:
        # Check if username exists
        if UserRepository.get_by_username(db, user_in.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username is already registered."
            )
        # Check if email exists
        if UserRepository.get_by_email(db, user_in.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email is already registered."
            )
        
        hashed = AuthService.hash_password(user_in.password)
        db_user = UserRepository.create(db, user_in, hashed)
        return db_user

    @staticmethod
    def authenticate_user(db: Session, username_or_email: str, password: str) -> User:
        user = UserRepository.get_by_username_or_email(db, username_or_email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username/email or password.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not AuthService.verify_password(password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username/email or password.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User account is inactive."
            )
            
        return user

    @staticmethod
    def verify_access_token(token: str) -> TokenData:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            email: str = payload.get("email")
            role: str = payload.get("role")
            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token claims.",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return TokenData(username=username, email=email, role=role)
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @staticmethod
    def refresh_tokens(db: Session, refresh_token: str) -> TokenResponse:
        try:
            payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token claims."
                )
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has expired."
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token."
            )
            
        user = UserRepository.get_by_username(db, username)
        
        # Token rotation check: compares the incoming refresh token against the last active
        # refresh token saved in the database. If they don't match, it signifies either the
        # token has been reused (potential replay attack) or it has been revoked.
        if not user or user.refresh_token != refresh_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token is revoked or invalid."
            )
            
        tokens = AuthService.create_tokens(user)
        # Update user's active refresh token in database
        UserRepository.update_refresh_token(db, user.id, tokens.refresh_token)
        return tokens

# Dependency injection helpers
async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    token = None
    if credentials and credentials.credentials:
        token = credentials.credentials
    elif "access_token" in request.cookies:
        token = request.cookies.get("access_token")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token_data = AuthService.verify_access_token(token)
    user = UserRepository.get_by_username(db, token_data.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user."
        )
    return user

class RoleChecker:
    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Resource restricted to roles: {', '.join(self.allowed_roles)}."
            )
        return current_user
