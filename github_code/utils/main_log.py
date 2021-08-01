# log_with_config.py
import logging

def init(logname, ProgName):
    """
    Based on http://docs.python.org/howto/logging.html#configuring-logging
    DEBUG,     INFO,     WARNING,     ERROR,     CRITICAL
    """
    global log
    logging.basicConfig(
        handlers=[logging.FileHandler(logname, 'a', 'utf-8')],
        format='%(asctime)s %(message)s [%(name)s %(levelname)s]',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO)
    log = logging.getLogger(ProgName)


def printL(log_str):
    log.warning(log_str)
    print(log_str)
