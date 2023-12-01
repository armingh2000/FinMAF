import logging
import os
import sys
from src.utils import mkpath

global org_stderr


def setup_logger(name, log_file, level=logging.INFO):
    """
    Sets up a logger with a specified name, log file, and level.
    """
    mkpath(log_file)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    redirect_stderr(logger)

    return logger


def log_to_console(level=logging.INFO):
    """
    Configures logging to output to the console.
    """
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.ERROR):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())


def redirect_stderr(logger):
    """
    Redirects stderr to the given logger.
    """
    global org_stderr
    org_stderr = sys.stderr
    sys.stderr = StreamToLogger(logger, logging.ERROR)


def revert_streams():
    global org_stderr
    sys.stderr = org_stderr


# Setup console logger
log_to_console()
