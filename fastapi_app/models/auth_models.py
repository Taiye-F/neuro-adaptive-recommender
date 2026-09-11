# models/auth_models.py
from datetime import datetime, timezone
import enum
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import relationship
from database import Base

class UserRole(str, enum.Enum):
    PARENT = "parent"
    CLINICIAN = "clinician"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, nullable=False, default=UserRole.PARENT.value)  # Stores "parent" or "clinician"
    is_active = Column(Boolean, default=True, nullable=False)
    subscription_tier = Column(String, default="free", nullable=False)  # Future subscription support
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    refresh_token = Column(String, nullable=True)  # Store current active refresh token

    # Relationships
    children = relationship("ChildProfile", back_populates="user", cascade="all, delete-orphan")
    reminders = relationship("ReminderNotification", back_populates="user", cascade="all, delete-orphan")
    assigned_patients = relationship("ClinicianPatientAssignment", back_populates="clinician", cascade="all, delete-orphan")
    clinical_notes = relationship("ClinicalNote", back_populates="clinician", cascade="all, delete-orphan")
