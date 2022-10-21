import os
import tempfile
import hashlib
import pickle
import shutil

def flatten(l):
    """
    flatten a list of lists
    """
    return [item for sublist in l for item in sublist]

def get_cache_dir(name = None):
    """
    return the cache directory for a given (library) name
    """
    cache_dir = tempfile.gettempdir()
    # TODO (if nessary): consider environment variables related to name
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

def hashmd5(obj):
    """
    Hash an object into a deterministic string
    """
    return hashlib.md5(pickle.dumps(obj)).hexdigest()

def save_source_dir(parentdir):
    src_dir = parentdir+"/src"
    i = 0
    while os.path.isdir(src_dir):
        i += 1
        src_dir = parentdir+"/src-"+str(i)
    os.makedirs(src_dir) #, exist_ok=True)
    whattocopy = [
        os.path.dirname(os.path.dirname(__file__))
    ]
    for what in whattocopy:
        dest = src_dir+"/"+os.path.basename(what)
        if os.path.isfile(what):
            shutil.copy(what, dest)
        else:
            shutil.copytree(what, dest, dirs_exist_ok=True)
