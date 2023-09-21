import os

# So that index of GPU is the same everywhere
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"

import time
import logging

import py3nvml.py3nvml as pynvml # GPU management

logging.basicConfig(
    # format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    # datefmt="%m/%d/%Y %H:%M:%S",
    # handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__.split(".")[0])
logger.setLevel(logging.INFO)

TIC = {}
GPUMEMPEAK = {}
TIMES = {}

def tic(name = ""):
    """ start clock
    Args:
        name: name of the clock
    """
    global TIC, GPUMEMPEAK
    TIC[name] = time.time()
    GPUMEMPEAK[name] = gpu_usage(name, verbose = False)

def toc(name = "", stream = None, log_mem_usage = False, total=False):
    """ end clock and print time elapsed since the last tic 
    Args:
        name: name of the clock
        log_mem_usage: if True, log GPU memory usage
    """
    global TIC
    t = time.time() - TIC.get(name, TIC[""])
    TIMES[name] = TIMES.get(name, 0) + t
    if total:
        t = TIMES[name]
    s = f"TIMING {name}: took {t} sec"
    logger.info(s)
    if stream:        
        print(s, file = stream)
    if log_mem_usage:
        gpu_usage(name)
        log_gpu_gpu_mempeak(name)
    return t

def get_num_gpus():
    try:
        pynvml.nvmlInit() # Can throw pynvml.NVMLError_DriverNotLoaded if driver problem
    except pynvml.NVMLError_DriverNotLoaded:
        import torch
        if torch.cuda.is_available():
            raise RuntimeError("CUDA is available but pynvml.NVMLError_DriverNotLoaded. This is probably because you are using a conda environment. Try to install nvidia-smi in the conda environment.")
        return 0
    return pynvml.nvmlDeviceGetCount()
    
def has_gpu():
    return get_num_gpus() > 0

def gpu_mempeak(name = ""):
    """ measure / return peak GPU memory usage peak (since last tic) """
    global GPUMEMPEAK
    GPUMEMPEAK[name] = max(GPUMEMPEAK.get(name, GPUMEMPEAK[""]), gpu_usage(verbose = False))
    return GPUMEMPEAK[name]

def log_gpu_gpu_mempeak(name = ""):
    """ log peak GPU memory usage """
    if has_gpu():
        logger.info(f"GPU MEMORY PEAK {name}: {gpu_mempeak(name)} MB")

def gpu_usage(name = "", index = None, verbose = True, stream = None, minimum = 10):
    """
    Args:
        name: name of the clock
        index: GPU index
        stream: stream to log to
        minimum: Minimum memory usage to report the mem usage (per GPU)
    """
    if verbose is None:
        verbose = (stream == None)
    summemused = 0
    indices = range(get_num_gpus())
    if index is None:
        pass
    elif isinstance(index, int):
        assert index in indices, "Got index %d but only %d GPUs available" % (index, indices)
        indices = [index]
    else:
        for i in index:
            assert i in indices, "Got index %d but only %d GPUs available" % (i, indices)
        indices = index
    for igpu in indices:
        handle = _get_gpu_handle(igpu)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpuname = pynvml.nvmlDeviceGetName(handle)
        # use = pynvml.nvmlDeviceGetUtilizationRates(handle) # This info does not seem to be reliable
        memused = info.used // 1024**2
        memtotal = info.total // 1024**2
        if memused >= minimum: # There is always a residual GPU memory used (1 or a few MB). Less than 10 MB usually means nothing.
            summemused+= memused
            s = f"GPU MEMORY {name} : {igpu+1}/{len(indices)} {gpuname}: mem {memused} / {memtotal} MB"
            if verbose:
                logger.info(s)
            if stream is not None:
                stream.write(f"{time.time()} {s}")
                stream.flush()
                
    return summemused

def gpu_total_memory(index = 0):
    handle = _get_gpu_handle(index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.total // 1024**2

def gpu_free_memory(index = 0):
    handle = _get_gpu_handle(index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.free // 1024**2

def _get_gpu_handle(index = 0):
    if not has_gpu():
        raise RuntimeError(f"No GPU available")
    pynvml.nvmlInit()
    try:
        return pynvml.nvmlDeviceGetHandleByIndex(index) 
    except:
        raise RuntimeError(f"Could not access GPU at index {index}")