"""
Structured logging for VoxLens modules.
Uses loguru with JSON or text output.
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
from typing import Optional
from loguru import logger
from .config import get_config


def _json_serializer(record: dict) -> str:
    """Serialize log record as JSON."""
    log_entry = {
        "timestamp": record["time"].astimezone().isoformat(),
        "level": record["level"].name,
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
    }
    # Add extra fields
    for key, value in record["extra"].items():
        log_entry[key] = value
    return json.dumps(log_entry, ensure_ascii=False)


def _json_formatter(record: dict) -> str:
    """Format string for JSON logging - returns format template for loguru."""
    return _json_serializer(record)


def _text_formatter(record: dict) -> str:
    """Format log record as human-readable text."""
    time_str = record["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    level_str = record["level"].name.ljust(8)
    name = record["name"]
    func = record["function"]
    line = record["line"]
    msg = record["message"]
    extra = " ".join(f"{k}={v}" for k, v in record["extra"].items() if v is not None)
    extra_str = f" [{extra}]" if extra else ""
    return f"{time_str} | {level_str} | {name}:{func}:{line} | {msg}{extra_str}"


def setup_logging(
    level: Optional[str] = None,
    format_type: Optional[str] = None,
    log_dir: Optional[str | Path] = None,
    rotation: Optional[str] = None,
    retention: Optional[int] = None,
) -> None:
    """
    Configure global logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format_type: "json" or "text"
        log_dir: Directory for log files
        rotation: Rotation policy (e.g., "daily", "10 MB")
        retention: Number of log files to keep
    """
    config = get_config()
    
    level = level or config.logging.level
    format_type = format_type or config.logging.format
    log_dir = Path(log_dir or config.log_dir)
    rotation = rotation or config.logging.file_rotation
    retention = retention or config.logging.max_files
    
    # Remove default handler
    logger.remove()
    
    # Console handler
    if format_type == "json":
        logger.add(sys.stdout, level=level, serialize=True)
    else:
        logger.add(sys.stdout, level=level, format=_text_formatter)
    
    # File handler
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "voxlens_{time:YYYY-MM-DD}.log"
    
    if format_type == "json":
        logger.add(
            log_file,
            level=level,
            rotation=rotation,
            retention=retention,
            serialize=True,
        )
    else:
        logger.add(
            log_file,
            level=level,
            rotation=rotation,
            retention=retention,
            format=_text_formatter,
        )


def get_logger(name: str):
    """Get a logger instance for a module."""
    return logger.bind(module=name)


class LogContext:
    """Context manager for adding structured context to logs."""
    
    def __init__(self, logger_instance, **context):
        self.logger = logger_instance
        self.context = context
        self.token = None
    
    def __enter__(self):
        self.token = self.logger.contextualize(**self.context)
        self.token.__enter__()
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.token.__exit__(exc_type, exc_val, exc_tb)


def log_input(logger_instance, module: str, data: dict, **extra):
    """Log an input to a module."""
    logger_instance.bind(module=module, direction="input", **extra).info(
        f"Input received: {json.dumps(data, default=str)[:200]}"
    )


def log_output(logger_instance, module: str, data: dict, **extra):
    """Log an output from a module."""
    logger_instance.bind(module=module, direction="output", **extra).info(
        f"Output emitted: {json.dumps(data, default=str)[:200]}"
    )


def log_error(logger_instance, module: str, error: Exception, **extra):
    """Log an error with context."""
    logger_instance.bind(module=module, error_type=type(error).__name__, **extra).error(
        f"Error: {error}"
    )