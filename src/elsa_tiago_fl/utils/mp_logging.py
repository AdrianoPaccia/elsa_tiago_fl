import multiprocessing as mp
import logging
import os

def set_logs_level():
    logger_mp = mp.get_logger()
    logger_mp.setLevel(logging.DEBUG)

    # file handler for DEBUG and INFO level logs
    log_file = str(os.getcwd()) + '/save/mp_logs/multiprocessing_logs.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # stream handler for WARNING and above level logs
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)

    # formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # add the handlers to the logger_mp
    logger_mp.addHandler(file_handler)
    logger_mp.addHandler(stream_handler)
