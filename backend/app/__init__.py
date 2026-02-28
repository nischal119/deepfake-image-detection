"""Flask application factory."""

from flask import Flask

from config import Config, init_upload_dir
from app.extensions import limiter


def create_app(config_class=Config) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_class)
    init_upload_dir(app.config)

    app.config.setdefault("RATELIMIT_DEFAULT", config_class.RATE_LIMIT)
    limiter.init_app(app)

    from app.blueprints.video_api import video_api
    app.register_blueprint(video_api, url_prefix="/api/video")

    from app.models import get_engine, get_session_factory, init_db
    engine = get_engine(app.config["DATABASE_URL"])
    session_factory = get_session_factory(engine)
    init_db(engine)

    @app.before_request
    def create_session():
        from flask import g
        g.db_session = session_factory()

    @app.teardown_appcontext
    def shutdown_session(exception=None):
        from flask import g
        session = getattr(g, "db_session", None)
        if session:
            session.close()

    return app
