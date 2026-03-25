import os
import logging
from logging.handlers import RotatingFileHandler
from logging_loki import LokiHandler


logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def get_loki_logger(service: str, console: bool = True):
    
    logger = logging.getLogger(service)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent propagation to root logger

    if not logger.handlers:
        loki_handler = LokiHandler(
            url="http://172.31.21.66:3100/loki/api/v1/push",
            tags={"application": "events"},
            version="1",
        )
        logger.addHandler(loki_handler)

        if console:
            logger.addHandler(logging.StreamHandler())

    return logger