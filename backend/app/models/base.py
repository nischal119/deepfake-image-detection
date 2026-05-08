 

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
    from app.models import video    
    Base.metadata.create_all(bind=engine)


def init_db_with_retry(engine, *, retries: int = 30, delay_sec: float = 2.0) -> None:
    from time import sleep

    from app.models import video    

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
