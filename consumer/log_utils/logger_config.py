import logging
import sys
from pathlib import Path


def setup_logger(name: str, log_file: str, level=logging.INFO):
    """Tạo logger riêng cho mỗi module"""

    # Tạo logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Tránh duplicate handlers
    if logger.handlers:
        return logger

    # Format
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    log_path = Path("logs") / log_file
    log_path.parent.mkdir(exist_ok=True)

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger