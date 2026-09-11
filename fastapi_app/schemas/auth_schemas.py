# schemas/auth_schemas.py
from datetime import datetime
from typing import Optional
import re
from pydantic import BaseModel, Field, field_validator, ConfigDict
from models.auth_models import UserRole

EMAIL_REGEX = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Alphanumeric username")
    email: str = Field(..., min_length=5, max_length=100, description="User email address")
    password: str = Field(..., min_length=6, max_length=100, description="Plaintext password")
    role: UserRole = Field(default=UserRole.PARENT, description="User role (parent or clinician)")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if not re.match(EMAIL_REGEX, v):
            raise ValueError("Invalid email address format.")
        return v.lower()

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Username must contain only alphanumeric characters, underscores, or hyphens.")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter.")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter.")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one number.")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError("Password must contain at least one special character.")
        return v

class UserLogin(BaseModel):
    username_or_email: str = Field(..., description="Username or email address")
    password: str = Field(..., description="User password")

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: UserRole
    is_active: bool
    subscription_tier: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # in seconds
    role: Optional[str] = "parent"

class TokenRefreshRequest(BaseModel):
    refresh_token: str

class TokenData(BaseModel):
    username: str = Field(..., description="Required username claim")
    email: Optional[str] = None
    role: Optional[str] = None

class MessageResponse(BaseModel):
    detail: str = Field(..., description="Feedback message")
