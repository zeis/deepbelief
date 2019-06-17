import sys
import logging


def init_logging(filename=None,
                 file_level=logging.DEBUG,
                 console_level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '[%(asctime)s] - %(levelname)s - %(name)s - %(message)s')

    if filename:
        fh = logging.FileHandler(filename, mode='w')
        fh.setLevel(file_level)
        fh.setFormatter(formatter)
        logging.getLogger().addHandler(fh)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(formatter)
    logging.getLogger().addHandler(ch)
