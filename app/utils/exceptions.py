class AppError(Exception):
    """Base application error"""
    pass

class ModelError(AppError):
    """Model-related errors"""
    pass

class ProcessingError(AppError):
    """Processing-related errors"""
    pass