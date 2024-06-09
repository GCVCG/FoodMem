import logging

from mmcv.utils import get_logger


def get_root_logger(log_file=None, log_level=logging.INFO):
    """
    Retrieves the root logger.

    The logger will be initialized if it has not been previously initialized. By
    default, a StreamHandler will be added. If `log_file` is specified, a FileHandler
    will also be added. The name of the root logger is the name of the top-level
    package, for example, "mmseg".

    Args:
        log_file (str | None): The name of the log file. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The level of the root logger. Note that only the process
            with rank 0 will be affected, while other processes will set the level
            to "Error" and remain silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """

    # Retrieves or initializes the logger with the name 'mmseg'
    logger = get_logger(name='mmseg', log_file=log_file, log_level=log_level)

    return logger
