import time
from contextlib import contextmanager
from collections import defaultdict
import logging
from app.utils.logging import setup_logging
setup_logging()  # Will only configure once now
logger = logging.getLogger(__name__)


class ProcessTimer:
    def __init__(self):
        self.timings = defaultdict(float)
        self.start_times = {}
        self.total_start = None

    def start_process(self, name):
        self.start_times[name] = time.time()
        if self.total_start is None:
            self.total_start = time.time()

    def end_process(self, name):
        if name in self.start_times:
            self.timings[name] = time.time() - self.start_times[name]
            del self.start_times[name]

    @contextmanager
    def time_process(self, name):
        self.start_process(name)
        try:
            yield
        finally:
            self.end_process(name)

    def get_total_time(self):
        if self.total_start is not None:
            return time.time() - self.total_start
        return 0

    def print_summary(self):
        total_time = self.get_total_time()

        # Create a formatted summary
        summary = "\n" + "=" * 50 + "\n"
        summary += "PERFORMANCE SUMMARY\n"
        summary += "=" * 50 + "\n\n"

        # Add individual process times
        summary += "Process Timings:\n"
        summary += "-" * 50 + "\n"
        for process, duration in sorted(self.timings.items()):
            percentage = (duration / total_time) * 100 if total_time > 0 else 0
            summary += f"{process:<30} {duration:6.2f}s ({percentage:5.1f}%)\n"

        # Add total time
        summary += "-" * 50 + "\n"
        summary += f"{'Total Time:':<30} {total_time:6.2f}s (100.0%)\n"
        summary += "=" * 50 + "\n"

        logger.info(summary)
        return summary