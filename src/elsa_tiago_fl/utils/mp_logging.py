import multiprocessing as mp
import logging

def set_logs_level():
    # Create a logger instance
    logger_mp = mp.get_logger()
    logger_mp.setLevel(logging.DEBUG)

    # Create a file handler for DEBUG and INFO level logs
    file_handler = logging.FileHandler('logs/multiprocessing_logs.log')
    file_handler.setLevel(logging.DEBUG)

    # Create a stream handler for WARNING and above level logs
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the handlers to the logger_mp
    logger_mp.addHandler(file_handler)
    logger_mp.addHandler(stream_handler)



def configure_logging():
    # Create a file handler for all log levels
    file_handler = logging.FileHandler('logs/multiprocessing_logs.log')
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Get the root logger and add the file handler
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)