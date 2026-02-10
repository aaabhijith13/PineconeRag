import logging
import sys
from typing import Optional


class LoggerSetup:
    """
    Sets up structured logging for a Dockerized application.
    Logs go to stdout only — no file logging.
    Prevents duplicate handler setups across multiple imports.
    """

    _is_configured = False

    @classmethod
    def setup(cls, name: Optional[str] = None) -> logging.Logger:
        """
        Set up and return a named logger.

        :param name: Optional logger name (usually __name__)
        :return: Configured logger instance
        """
        if cls._is_configured:
            return logging.getLogger(name)

        log_level = getattr(logging, "INFO", logging.INFO)

        # Clear existing handlers to prevent duplication
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.handlers.clear()

        # StreamHandler for stdout (Docker logging best practice)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

        root_logger.addHandler(stream_handler)

        cls._is_configured = True
        return logging.getLogger(name)
