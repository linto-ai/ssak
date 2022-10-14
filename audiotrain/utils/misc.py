def flatten(l):
    return [item for sublist in l for item in sublist]

def log_time(logger, text, t):
    logger.info(f"{text} {t}")