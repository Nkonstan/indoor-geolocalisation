# Makes the routes package importable and exposes blueprints
from app.routes.main import main_bp

__all__ = ['main_bp']