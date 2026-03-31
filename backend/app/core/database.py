"""
AuroraML Database Configuration
SQLAlchemy engine, session management, and base model.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.core.config import settings


# Connection arguments based on DB type
is_sqlite = settings.DATABASE_URL.startswith("sqlite")
connect_args = {"check_same_thread": False} if is_sqlite else {}

# Engine configuration
engine_args = {
    "connect_args": connect_args,
    "echo": settings.DEBUG,
}

# Only add connection pooling for non-SQLite databases
if not is_sqlite:
    engine_args.update({
        "pool_size": 10,
        "max_overflow": 20,
        "pool_pre_ping": True,
    })

engine = create_engine(settings.DATABASE_URL, **engine_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


def get_db():
    """FastAPI dependency that provides a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all database tables. Call on startup."""
    # Ensure models are imported so metadata is fully populated.
    from app.models import notification  # noqa: F401
    from app.models import prediction_event  # noqa: F401
    from app.models import monitoring_snapshot  # noqa: F401
    Base.metadata.create_all(bind=engine)
