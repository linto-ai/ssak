import os
import tempfile

def flatten(l):
    return [item for sublist in l for item in sublist]

def get_cache_dir(name = None):
    cache_dir = tempfile.gettempdir()
    if os.environ.get("HOME"):
        cache_dir = os.path.join(os.environ["HOME"], ".cache") # os.path.expanduser("~/.cache")
    else:
        for folder in [
            "/usr/share",
            "/workspace",
            "/opt",
        ]:
            # Check write access
            if os.access(folder, os.W_OK):
                cache_dir = os.path.join(folder, ".cache")
                break
    if name:
        cache_dir = os.path.join(cache_dir, name)
    return cache_dir