import os

# So that index of GPU is the same everywhere
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"

import time
import py3nvml.py3nvml as pynvml # VRAM (GPU) monitoring
import psutil # RAM monitoring

import logging
from datetime import datetime

logging.basicConfig(
    # format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    # datefmt="%m/%d/%Y %H:%M:%S",
    # handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__.split(".")[0])
logger.setLevel(logging.INFO)

"""
This module provides helpers to monitor timing and memory usages.

Example usage (minimal):

```python
from linastt.utils.logs import tic, toc, vram_peak, ram_peak
import torch

tic()

print(f"> {ram_peak()=}")
print(f"> {vram_peak()=}")
print(f"> {vram_peak(index=0)=}")

print("Allocating memory...")
x = torch.rand(1000, 1000, device="cpu")
y = torch.rand(1000, 1000, device="cuda:0")

print(f"> {ram_peak()=}")
print(f"> {vram_peak()=}")
print(f"> {vram_peak(index=0)=}")

print(f"> {toc()=}")
```	

The above should print something like:

```
INFO:linastt:RAM_CURRENT : 356.95703125 MB
INFO:linastt:RAM_PEAK : 356.95703125 MB
> ram_peak()=356.95703125
INFO:linastt:VRAM_CURRENT : 1/2 NVIDIA GeForce GTX TITAN X: 81 / 12288 MB
INFO:linastt:VRAM_CURRENT : 2/2 NVIDIA GeForce GTX 1080 Ti: 91 / 11264 MB
INFO:linastt:VRAM_PEAK : 172 MB
> vram_peak()=172
INFO:linastt:VRAM_CURRENT : 1/1 NVIDIA GeForce GTX TITAN X: 81 / 12288 MB
INFO:linastt:VRAM_PEAK : 81 MB
> vram_peak(index=0)=81
Allocating memory...
INFO:linastt:RAM_CURRENT : 445.61328125 MB
INFO:linastt:RAM_PEAK : 445.61328125 MB
> ram_peak()=445.61328125
INFO:linastt:VRAM_CURRENT : 1/2 NVIDIA GeForce GTX TITAN X: 204 / 12288 MB
INFO:linastt:VRAM_CURRENT : 2/2 NVIDIA GeForce GTX 1080 Ti: 93 / 11264 MB
INFO:linastt:VRAM_PEAK : 297 MB
> vram_peak()=297
INFO:linastt:VRAM_CURRENT : 1/1 NVIDIA GeForce GTX TITAN X: 204 / 12288 MB
INFO:linastt:VRAM_PEAK : 204 MB
> vram_peak(index=0)=204
INFO:linastt:TIMING : took 0.7578799724578857 sec
> toc()=0.7578799724578857
```
"""

############################################
# Helpers for timing

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
    s = f"TIMING{_name_to_suffix(name)} : {t:.3f} sec"
    if stream:        
        print(s, file = stream)
    if verbose:
        logger.info(s)
        if log_mem_usage:
            vram_usage(name, ignore_errors=True)
            vram_peak(name)
    return t


############################################
# Helpers to monitor GPU and VRAM memory

def get_num_gpus(ignore_errors = False):
    """
        Returns the number of GPUs available
    """
    global NUM_GPU
    if NUM_GPU is not None:
        return NUM_GPU
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
    NUM_GPU = pynvml.nvmlDeviceGetCount()
    return NUM_GPU
    
def has_gpu():
    """
        Returns True if GPU is available
    """
    return get_num_gpus() > 0

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
        memused = info.used / 1024**2
        memtotal = info.total / 1024**2
        if memused >= minimum: # There is always a residual GPU memory used (1 or a few MB). Less than 10 MB usually means nothing.
            summemused+= memused
            s = f"VRAM_CURRENT{_name_to_suffix(name)} : {igpu+1}/{len(indices)} {gpuname} (max {memtotal:.0f} MB): {memused:.0f} MB"
            if verbose:
                logger.info(s)
            if stream is not None:
                stream.write(f"{s}\n")
                stream.flush()
                
    return summemused


def vram_peak(name="", index = None, ignore_errors=False, verbose = True, **kwargs):
    """
        Measures and returns peak VRAM usage (maximum GPU memory) and logs it (with logger.info).

        See vram_usage() for arguments
    """
    global VRAM_PEAKS
    if ignore_errors and not has_gpu():
        return 0
    key = f"{name}::{index}"
    VRAM_PEAKS[key] = max(VRAM_PEAKS.get(key, 0), vram_usage(name=name, index=index, verbose=verbose, ignore_errors=False, **kwargs))
    if verbose:
        logger.info(f"VRAM_PEAK{_name_to_suffix(name)} : {VRAM_PEAKS[key]} MB")
    return VRAM_PEAKS[key]


def vram_free(index = 0):
    """
        Returns the free GPU memory (in MiB) of the GPU at index `index`

        Args:
            index: int
                GPU index
    """
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


############################################
# Helpers to monitor RAM memory

def ram_usage(name = "", pids = None, verbose = True, stream = None):
    """
        Returns the RAM usage (in MiB) of the current process and all its subprocesses (if pids is None)

    Args:
        name: str
            an arbitrary name for this measure (that will be used in the log). Can be left empty for simple usage.
        pids: list
            list of process IDs (if None, current process and all its subprocesses)
        verbose: bool
            Use false to disable logging
        stream: stream (with .write() and .flush() methods)
            a stream to write the log to
    Returns:
        Total memory usage in MiB

    """
    if pids is None:
        # Get PIDs of the current process and all its subprocesses
        current_pid = get_current_pid()
        pids = [current_pid] + get_subprocess_pids(current_pid)
    ram = sum([psutil.Process(pid=p).memory_info().rss / (1024 * 1024) for p in pids])
    if verbose or stream:
        s = f"RAM_CURRENT{_name_to_suffix(name)} : {ram:.3f} MB"
        if verbose:
            logger.info(s)
        if stream is not None:
            stream.write(f"{s}\n")
            stream.flush()
    return ram

def ram_peak(name="", pids = None, verbose = True, **kwargs):
    """
        Measures and returns peak RAM usage and logs it (with logger.info).

        See ram_usage() for arguments
    """
    global RAM_PEAKS
    key = f"{name}::{pids}"
    RAM_PEAKS[key] = max(RAM_PEAKS.get(key, 0), ram_usage(name=name, pids=pids, verbose=verbose,  **kwargs))
    if verbose:
        logger.info(f"RAM_PEAK{_name_to_suffix(name)} : {RAM_PEAKS[key]:.3f} MB")
    return RAM_PEAKS[key]

def get_current_pid():
    return os.getpid()

def get_subprocess_pids(parent_pid):
    subprocess_pids = []
    parent_process = psutil.Process(parent_pid)
    children = parent_process.children(recursive=True)
    subprocess_pids = [child.pid for child in children]
    return subprocess_pids

def get_pids(python_script_name, **kwargs):
    """
        Returns the list of PIDs of the processes that run the python script `python_script_name`

        See get_processes() for arguments
    """
    return [p.info["pid"] for p in get_processes(python_script_name, **kwargs)]

def get_processes(python_script_name, pids_to_ignore=[]):
    """
        Returns the list of processes that run the python script `python_script_name`
    """
    for p in psutil.process_iter(attrs=[]):
        if p.info["pid"] in pids_to_ignore:
            continue
        cmdline = p.info["cmdline"]
        if len(cmdline) >= 2 and cmdline[0].startswith("python") and cmdline[1] == python_script_name:
            yield p
            for child in p.children(recursive=True):
                yield child


############################################
# Internal 

# Global variables
TIC = {}
TIMES = {}
NUM_GPU = None
VRAM_PEAKS = {}
RAM_PEAKS = {}

def _name_to_suffix(
        name,
        datefmt=None # "%m/%d/%Y|%H:%M:%S"
    ):
    suffix = ""
    if datefmt:
        formatted_time = datetime.now().strftime(datefmt)
        suffix += f"({formatted_time})"
    if name:
        suffix += f" {name}"
    return suffix
