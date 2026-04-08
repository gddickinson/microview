import logging
from logging.handlers import RotatingFileHandler

class LoggingOperations:
    @staticmethod
    def setup_logging():
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = RotatingFileHandler('microview.log', maxBytes=1e6, backupCount=5)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
