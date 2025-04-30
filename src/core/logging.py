import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
        log_file: Optional[Path] = None,
        level: int = logging.INFO,
        formatter: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
):
    """
    Setup logging configuration.

    Args:
        log_file: Path to the log file. If None, logs will be printed to stdout.
        level: Logging level.
        formatter: Log formatter.
    """
    handlers = [
        logging.StreamHandler(sys.stdout),
    ]

    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=formatter,
        handlers=handlers,
    )
