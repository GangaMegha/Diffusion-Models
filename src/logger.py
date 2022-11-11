import logging

def setup_logger(logger, filename, file_format, stream_format, level=logging.INFO):
    
    # set log level
    logger.setLevel(level)
    
    if filename:
        # create formatter
        file_formatter = logging.Formatter(file_format)
        
        # define file handler and set formatter
        file_handler = logging.FileHandler(filename, mode='w')
        file_handler.setFormatter(file_formatter)

        # add file handler to logger
        logger.addHandler(file_handler)
    
    # create formatter
    stream_formatter = logging.Formatter(stream_format)

    # create console handler and set level to debug
    stream_handler = logging.StreamHandler()


    # add formatter to ch
    stream_handler.setFormatter(stream_formatter)

    # add ch to logger
    logger.addHandler(stream_handler)

    return logger

if __name__ == "__main__":
    logger = logging.getLogger('logger')
    logger = setup_logger(logger, '', '[%(filename)s:%(lineno)s]: %(message)s', '[%(name)s:%(lineno)d]: %(message)s', logging.INFO)
    # 'application' code
    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warn message')
    logger.error('error message')
    logger.critical('critical message')