"""Base model and db session."""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


def get_engine(database_url: str):
    return create_engine(database_url, pool_pre_ping=True)


def get_session_factory(engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db(engine):
    from app.models import video  # noqa: F401
    Base.metadata.create_all(bind=engine)
