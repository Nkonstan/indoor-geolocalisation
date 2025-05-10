from flask import Flask
from app.config import Config
import os
import logging
from app.utils.logging import setup_logging


def create_app(config_class=Config):
    app = Flask(__name__,
                static_folder='/app/static',
                static_url_path='/static')
    app.config.from_object(config_class)

    # Enable debug mode
    app.config['DEBUG'] = True

    # Initialize logging once
    setup_logging()  # This will set up all logging configuration

    # Set Flask app logger level - don't use basicConfig
    app.logger.setLevel(logging.DEBUG)

    # Ensure static directory exists
    os.makedirs(app.static_folder, exist_ok=True)

    # Register blueprints
    from app.routes.main import main_bp
    app.register_blueprint(main_bp)

    return app



