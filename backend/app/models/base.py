"""Base model and db session."""

from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


def get_engine(database_url: str):
    return create_engine(database_url, pool_pre_ping=True)


def get_session_factory(engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db(engine):
    from app.models import video  # noqa: F401
    Base.metadata.create_all(bind=engine)


def init_db_with_retry(engine, *, retries: int = 30, delay_sec: float = 2.0) -> None:
    """
    Create tables once DB is reachable.

    This is mainly for local dev when Postgres may still be starting.
    """
    from time import sleep

    from app.models import video  # noqa: F401

    last_exc: Optional[Exception] = None
    for _ in range(retries):
        try:
            Base.metadata.create_all(bind=engine)
            return
        except OperationalError as e:
            last_exc = e
            sleep(delay_sec)
    if last_exc:
        raise last_exc
