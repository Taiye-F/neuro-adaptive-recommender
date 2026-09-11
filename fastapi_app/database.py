# database.py
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Ensure .env is loaded regardless of working directory
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if (ROOT_DIR / ".env").exists():
    load_dotenv(ROOT_DIR / ".env", override=False)
elif (BASE_DIR / ".env").exists():
    load_dotenv(BASE_DIR / ".env", override=False)

log = logging.getLogger(__name__)

# Get environment config
ENVIRONMENT = os.getenv("ENVIRONMENT", "DEVELOPMENT").upper()
DATABASE_URL = os.getenv("DATABASE_URL")

sqlite_url = f"sqlite:///{BASE_DIR}/recommender.db"
connect_args_sqlite = {"check_same_thread": False}

if ENVIRONMENT == "PRODUCTION":
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is required in PRODUCTION mode.")
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    if not DATABASE_URL.startswith("postgresql"):
        raise ValueError("DATABASE_URL must be a valid PostgreSQL connection string in PRODUCTION mode.")
    engine = create_engine(DATABASE_URL)
else:
    # In DEVELOPMENT mode: try PostgreSQL if configured, but fall back gracefully to SQLite
    if DATABASE_URL and (DATABASE_URL.startswith("postgresql") or DATABASE_URL.startswith("postgres://")):
        db_url = DATABASE_URL.replace("postgres://", "postgresql://", 1)
        try:
            test_engine = create_engine(db_url)
            with test_engine.connect():
                pass
            engine = test_engine
            DATABASE_URL = db_url
            log.info("✓ Connected to PostgreSQL in DEVELOPMENT mode.")
        except Exception as e:
            log.warning(
                f"⚠️ PostgreSQL connection failed in DEVELOPMENT ({e}). "
                f"Falling back safely to local SQLite ({sqlite_url})."
            )
            DATABASE_URL = sqlite_url
            engine = create_engine(DATABASE_URL, connect_args=connect_args_sqlite)
    else:
        DATABASE_URL = sqlite_url
        engine = create_engine(DATABASE_URL, connect_args=connect_args_sqlite)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency to get db session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
