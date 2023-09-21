import os
import sys
from .misc import get_cache_dir

DISABLE_GPU = False

# So that index of GPU is the same everywhere
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"

# For RuntimeError: CUDA out of memory. Tried to allocate 1.03 GiB (GPU 0; 11.93 GiB total capacity; 7.81 GiB already allocated; 755.69 MiB free; 10.59 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF  
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128" 

def _set_visible_gpus(s):
    global DISABLE_GPU
    if isinstance(s, str):
        if s == "auto":
            # Choose the GPU with the most free memory
            from linastt.utils.logs import get_num_gpus, gpu_free_memory
            # GPUs sorted by decreasing free memory
            gpus = list(reversed(sorted(range(get_num_gpus()), key = gpu_free_memory)))
            s = str(gpus[0]) if len(gpus) else ""
        return _set_visible_gpus(s.split(",") if s else [])
    if isinstance(s, list):
        s = ','.join([str(int(si)) for si in s])
    if not s:
        DISABLE_GPU = True
    os.environ["CUDA_VISIBLE_DEVICES"] = s

for i, arg in enumerate(sys.argv[1:]):
    if arg in ["--gpus", "--gpu"]:
        _set_visible_gpus(sys.argv[i+2])
    elif arg.startswith("--gpus=") or arg.startswith("--gpu="):
        _set_visible_gpus(arg.split("=")[-1])

# To address the following error when importing librosa
#   RuntimeError: cannot cache function '__shear_dense': no locator available for file '/usr/local/lib/python3.9/site-packages/librosa/util/utils.py'
# See https://stackoverflow.com/questions/59290386/runtimeerror-at-cannot-cache-function-shear-dense-no-locator-available-fo
os.environ["NUMBA_CACHE_DIR"] = "/tmp"

# Disable warnings of type "This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if not os.environ.get("HOME"):
    path = os.path.dirname(os.path.abspath(__file__))
    if path.startswith("/home/"):
        os.environ["HOME"] = "/".join(os.environ["HOME"].split("/")[:3])

# Set cache directory
os.environ["HUGGINGFACE_HUB_CACHE"] = get_cache_dir("huggingface/hub")
os.environ["TRANSFORMERS_CACHE"] = get_cache_dir("huggingface/hub")
import datasets
datasets.config.HF_MODULES_CACHE = get_cache_dir("huggingface/modules")
datasets.config.HF_DATASETS_CACHE = get_cache_dir("huggingface/datasets")
datasets.config.HF_METRICS_CACHE = get_cache_dir("huggingface/metrics")
datasets.config.DOWNLOADED_DATASETS_PATH = get_cache_dir("huggingface/datasets/downloads")

# Importing torch must be done after having set the CUDA-related environment variables
import torch
import multiprocessing

def auto_device():
    return torch.device('cuda:0') if (torch.cuda.is_available() and not DISABLE_GPU) else torch.device("cpu")

def use_gpu():
    return torch.cuda.is_available() and not DISABLE_GPU

if not use_gpu():
    # Use maximum number of threads
    torch.set_num_threads(multiprocessing.cpu_count())
