from contextlib import contextmanager
import torch
import gc
from app.utils.logging import setup_logging
import logging
from typing import Optional
import time
import sys

setup_logging()
logger = logging.getLogger(__name__)

@contextmanager
def gpu_memory_manager():
    """Enhanced context manager for aggressive GPU memory cleanup"""
    operation_start_time = time.time()  # Add timing
    try:
        if torch.cuda.is_available():
            # Initial cleanup
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"GPU Memory Status - Before operation: "
                             f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB, "
                             f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB")
        yield
    finally:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                # Force delete CUDA tensors
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) and obj.is_cuda:
                            del obj
                    except Exception:
                        continue
                # Multiple cleanup passes
                for _ in range(3):
                    torch.cuda.empty_cache()
                    gc.collect()
                # Final synchronization
                torch.cuda.synchronize()
                # Log final memory state with operation duration
                operation_time = time.time() - operation_start_time
                logger.debug(f"GPU Memory Status - After cleanup "
                           f"(Operation took {operation_time:.2f}s): "
                           f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB, "
                           f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB")
            except Exception as e:
                logger.error(f"GPU cleanup failed: {e}")

def cleanup_gpu_memory():
    """Cleanup GPU memory - call at end of requests only."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"GPU: {torch.cuda.memory_allocated() / 1e9:.2f}GB allocated, "
                f"{torch.cuda.memory_reserved() / 1e9:.2f}GB reserved"
            )