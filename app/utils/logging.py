import logging
import sys

_is_logging_configured = False


def setup_logging():
    global _is_logging_configured
    if _is_logging_configured:
        return

    # Clear any existing handlers
    root = logging.getLogger()
    for handler in root.handlers:
        root.removeHandler(handler)

    # Create single handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # Set levels for noisy modules
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('huggingface_hub').setLevel(logging.WARNING)

    _is_logging_configured = True