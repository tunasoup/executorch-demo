import logging
import logging.handlers
from pathlib import Path


def create_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Create a custom logger.

    Contains a stream handler for console output and a filehandler with a hardcoded
    path. Filehandler is always at Debug level.

    Args:
        name (str): Name of the logger, recommended to be __name__.
        level (int, optional): Logging level for console. Defaults to logging.DEBUG.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Handlers never show lower levels than the logger itself

    log_file = get_log_dir() / "app.log"
    msg_format = "%(asctime)s - %(name)s - %(levelname)8s: %(message)s"

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(msg_format, datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level=level)
    logger.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, mode="a", encoding="utf-8", maxBytes=5 * 1024 * 1024, backupCount=4
    )
    file_formatter = logging.Formatter(msg_format, datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    return logger


def get_root_dir() -> Path:
    """Get the root directory of the project.

    Assumes this file is located at src/package/utils.py

    Returns:
        Path: Path to the root directory of the project.
    """
    return Path(__file__).parent.parent.parent


def get_log_dir() -> Path:
    """Get the log directory of the project.

    Returns:
        Path: Path to the log directory of the project.
    """
    log_dir = get_root_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_data_dir() -> Path:
    """Get the data directory.

    Returns:
        Path: Path to the data directory.
    """
    return get_root_dir() / "data"


def get_data_raw_dir() -> Path:
    """Get the directory with unprocessed data.

    Returns:
        Path: Path to the raw data directory.
    """
    return get_data_dir() / "raw"


def get_model_dir() -> Path:
    """Get the model directory.

    Returns:
        Path: Path to the model directory
    """
    return get_root_dir() / "models"
