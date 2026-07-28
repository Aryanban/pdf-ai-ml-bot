"""Application logging setup."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a consistently configured module logger."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger(name)
