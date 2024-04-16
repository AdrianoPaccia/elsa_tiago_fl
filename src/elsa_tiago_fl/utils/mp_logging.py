import multiprocessing as mp
import logging

def set_logs_level():
    # Create a logger instance
    logger_mp = mp.get_logger()
    logger_mp.setLevel(logging.DEBUG)

    # Create a file handler for DEBUG and INFO level logs
    file_handler = logging.FileHandler('/save/mp_logs/multiprocessing_logs.log')
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
