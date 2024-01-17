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

# Global variables
TIC = {}
GPUMEMPEAK = {}
TIMES = {}
NUM_GPU_CACHED = None


def tic(name = ""):
    """ start clock
    Args:
        name: name of the clock
    """
    global TIC
    TIC[name] = time.time()


def toc(name = "", stream = None, verbose = True, log_mem_usage = False, total=False):
    """ end clock and returns time elapsed since the last tic 
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
    if stream:        
        print(s, file = stream)
    if verbose:
        logger.info(s)
        if log_mem_usage:
            vram_usage(name, ignore_errors=True)
            vram_peak(name)
    return t


def get_num_gpus(ignore_errors = False):
    """
        Returns the number of GPUs available
    """
    global NUM_GPU_CACHED
    if NUM_GPU_CACHED is not None:
        return NUM_GPU_CACHED
    try:
        pynvml.nvmlInit() # Can throw pynvml.NVMLError_DriverNotLoaded if driver problem
    except pynvml.NVMLError_DriverNotLoaded:
        import torch
        if torch.cuda.is_available():
            raise RuntimeError("CUDA is available but pynvml.NVMLError_DriverNotLoaded. This is probably because you are using a conda environment. Try to install nvidia-smi in the conda environment.")
        return 0
    except Exception as unexpected_error:
        if ignore_errors:
            return 0
        raise unexpected_error
    NUM_GPU_CACHED = pynvml.nvmlDeviceGetCount()
    return NUM_GPU_CACHED
    
def has_gpu():
    """
        Returns True if GPU is available
    """
    return get_num_gpus() > 0

def vram_peak(name="", index = None, ignore_errors=False, verbose = True, **kwargs):
    """
        Measures and returns peak VRAM usage (maximum GPU memory) and logs it (with logger.info).

        See vram_usage() for arguments
    """
    global GPUMEMPEAK
    if ignore_errors and not has_gpu():
        return 0
    key = f"{name}::{index}"
    GPUMEMPEAK[key] = max(GPUMEMPEAK.get(key, 0), vram_usage(name=name, index=index, verbose=verbose, ignore_errors=False, **kwargs))
    if verbose:
        logger.info(f"GPU MEMORY PEAK {key}: {GPUMEMPEAK[key]} MB")
    return GPUMEMPEAK[key]

def vram_usage(name = "", index = None, ignore_errors=False, verbose = True, stream = None, minimum = 10):
    """
        Returns the VRAM usage (GPU memory) and logs it (with logger.info).

    Args:
        name: str
            an arbitrary name for this measure (that will be used in the log). Can be left empty for simple usage.
        index: list or int or None
            GPU index or list of indices (if None, all available GPUs are considered)
        ignore_errors: bool
            Do not raise errors if GPU is not available
        verbose: bool
            Use false to disable logging
        stream: stream (with .write() and .flush() methods)
            a stream to write the log to
        minimum: int or float
            Minimum memory usage to report the mem usage (in MiB per GPU)

    Returns:
        Total memory usage in MiB
    """
    if verbose is None:
        verbose = (stream == None)
    summemused = 0
    indices = range(get_num_gpus(ignore_errors=ignore_errors))
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
                stream.write(f"{time.time()} {s}\n")
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