import os
import sys

def _set_visible_gpus(s):
    if isinstance(s, str):
        return _set_visible_gpus(s.split(","))
    if isinstance(s, list):
        s = ','.join([str(int(si)) for si in s])
    os.environ["CUDA_VISIBLE_DEVICES"] = s

for i, arg in enumerate(sys.argv[1:]):
    if arg == "--gpu":
        _set_visible_gpus(sys.argv[i+2])
    elif arg.startswith("--gpu="):
        _set_visible_gpus(arg.split("=")[-1])

# So that index of GPU is the same everywhere
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"

# For RuntimeError: CUDA out of memory. Tried to allocate 1.03 GiB (GPU 0; 11.93 GiB total capacity; 7.81 GiB already allocated; 755.69 MiB free; 10.59 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF  
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128" 

# To address the following error when importing librosa
#   RuntimeError: cannot cache function '__shear_dense': no locator available for file '/usr/local/lib/python3.9/site-packages/librosa/util/utils.py'
# See https://stackoverflow.com/questions/59290386/runtimeerror-at-cannot-cache-function-shear-dense-no-locator-available-fo
os.environ["NUMBA_CACHE_DIR"] = "/tmp"

if not os.environ.get("HOME"):
    path = os.path.dirname(os.path.abspath(__file__))
    if path.startswith("/home/"):
        os.environ["HOME"] = "/".join(os.environ["HOME"].split("/")[:3])

# Importing torch must be done after having set the CUDA-related environment variables
import torch
import multiprocessing

def auto_device():
    return torch.device('cuda:0') if torch.cuda.is_available() else "cpu"

if not torch.cuda.is_available():
    # Use maximum number of threads
    torch.set_num_threads(multiprocessing.cpu_count())
