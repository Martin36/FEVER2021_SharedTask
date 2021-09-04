import logging

loggers = {}


def get_logger(name="logger"):
    global loggers

    if loggers.get(name):
        return loggers.get(name)
    else:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s: [ %(message)s ]", "%m/%d/%Y %I:%M:%S %p")
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)
        loggers[name] = logger
        return logger
