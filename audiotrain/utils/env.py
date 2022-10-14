import os

# So that index of GPU is the same everywhere
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"

# For RuntimeError: CUDA out of memory. Tried to allocate 1.03 GiB (GPU 0; 11.93 GiB total capacity; 7.81 GiB already allocated; 755.69 MiB free; 10.59 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF  
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128" 

import torch

def auto_device():
    return torch.device('cuda:0') if torch.cuda.is_available() else "cpu"