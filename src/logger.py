import logging
import sys

def get_logger(logger_name: str) -> logging.Logger:
    """
    Creates and configures a centralized logger for the project.

    Args:
        logger_name (str): The name of the logger to create.

    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate logs
    if logger.hasHandlers():
        return logger
        
    # Define a professional format: [Timestamp] | [Level] | [Script Name] | Message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Send logs to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger