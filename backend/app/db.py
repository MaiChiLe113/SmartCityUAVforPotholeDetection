"""Database engine and session. SQLite by default; set DATABASE_URL for Postgres."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import API


class Base(DeclarativeBase):
    pass


_connect_args = {"check_same_thread": False} if API.database_url.startswith("sqlite") else {}
if API.database_url.startswith("sqlite:///"):
    Path(API.database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(API.database_url, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """Create all tables. Replace with Alembic migrations once the schema settles."""
    from app import models  # noqa: F401  (registers the tables)

    Base.metadata.create_all(engine)


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
