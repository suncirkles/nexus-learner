"""
core/logging_config.py
-----------------------
Central logging configuration for Nexus Learner.

Call ``setup_logging()`` once at application startup (top of app.py).
All modules that do ``logging.getLogger(__name__)`` automatically inherit
this configuration without needing any local basicConfig() calls.

Features
--------
- RotatingFileHandler  : logs/{LOG_FILE}, rotates at LOG_MAX_BYTES, keeps LOG_BACKUP_COUNT files
- StreamHandler        : console output (useful during development / Streamlit terminal)
- Level / format       : driven by Settings in core/config.py (overrideable via .env)
- Idempotent           : safe to call multiple times (e.g. during hot-reload)
"""

import logging
import logging.handlers
import os
from pathlib import Path
from core.context import get_request_id, get_session_id


class ContextFilter(logging.Filter):
    """Filter that injects context IDs (request_id, session_id) into log records."""
    def filter(self, record):
        record.request_id = get_request_id()
        record.session_id = get_session_id()
        return True


def setup_logging() -> None:
    """Configure the root logger with file rotation + console output.

    Settings are read from ``core.config.settings`` so they can be
    overridden via environment variables or the ``.env`` file:

        LOG_LEVEL=DEBUG
        LOG_DIR=logs
        LOG_FILE=nexus_learner.log
        LOG_MAX_BYTES=10485760
        LOG_BACKUP_COUNT=5
    """
    from core.config import settings  # late import avoids circular deps

    root_logger = logging.getLogger()

    # Idempotency guard — don't add duplicate handlers on Streamlit hot-reload
    if root_logger.handlers:
        # Handlers already configured; just ensure level is current
        root_logger.setLevel(settings.LOG_LEVEL.upper())
        return

    numeric_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    # Updated format string to include req and sess IDs
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] [req:%(request_id)s] [sess:%(session_id)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Rotating file handler ---
    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / settings.LOG_FILE

    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_path),
        maxBytes=settings.LOG_MAX_BYTES,
        backupCount=settings.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(fmt)
    file_handler.addFilter(ContextFilter())

    # --- Console (stream) handler ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(fmt)
    console_handler.addFilter(ContextFilter())

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Silence overly chatty third-party loggers at WARNING+
    for noisy in ("httpx", "httpcore", "openai", "anthropic", "langchain", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root_logger.info(
        "Logging initialised — level=%s file=%s",
        settings.LOG_LEVEL.upper(),
        log_path.resolve(),
    )
