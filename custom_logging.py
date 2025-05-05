import logging
import coloredlogs

def setup_custom_logger(logger_name: str = 'Rag_app_BM25_log'):
    """
    Sets up a custom logger for the specified logger name.
    Only logs from this logger will be displayed. 
    All other library logs are suppressed.
    """
    # Create the logger with the specified name
    logger = logging.getLogger(logger_name)
    
    # Set the logger's logging level (DEBUG or INFO)
    logger.setLevel(logging.DEBUG)

    # Prevent adding multiple handlers
    if not logger.hasHandlers():
        # Add a colored handler for the logger
        coloredlogs.install(level='DEBUG', fmt='** My Logs - %(levelname)s %(name)s => %(message)s')

    # Suppress logs from all other libraries (set them to CRITICAL to suppress their logs)
    logging.getLogger().setLevel(logging.CRITICAL)  # This suppresses all logs at or below CRITICAL

    # Optionally suppress logs from specific libraries
    logging.getLogger('urllib3').setLevel(logging.CRITICAL)
    logging.getLogger('requests').setLevel(logging.CRITICAL)
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)
    logging.getLogger('pyrogram').setLevel(logging.CRITICAL)
    logging.getLogger('httpx').setLevel(logging.CRITICAL)

    return logger
