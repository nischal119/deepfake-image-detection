"""Flask extensions (initialized without app, init_app in factory)."""

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
