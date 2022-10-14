import time
import logging
logging.basicConfig()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TIC = {}

def tic(name = ""):
    global TIC
    TIC[name] = time.time()

def toc(name = ""):
    global TIC
    t = time.time() - TIC.get(name, TIC[""])
    logger.info(f"{name}: {t} sec")
