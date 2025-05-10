# Makes the utils package importable and exposes utility functions
from app.utils.logging import setup_logging
from app.utils.exceptions import AppError, ModelError, ProcessingError
from app.utils.gpu_utils import gpu_memory_manager  # Add this line
__all__ = ['setup_logging', 'AppError', 'ModelError', 'ProcessingError', 'gpu_memory_manager']
