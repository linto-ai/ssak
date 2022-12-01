import os
import sys
import tempfile
import hashlib
import pickle
import shutil
import types

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
    if os.environ.get("HOME") and os.access(os.environ["HOME"], os.W_OK):
        cache_dir = os.path.join(os.environ["HOME"], ".cache") # os.path.expanduser("~/.cache")
    elif __file__.startswith("/home/") and os.access("/".join(__file__.split("/")[:3]), os.W_OK):
        cache_dir = os.path.join("/".join(__file__.split("/")[:3]), ".cache")
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

def save_source_dir(parentdir, add_packages = None):
    src_dir = parentdir+"/src"
    i = 0
    while os.path.isdir(src_dir):
        i += 1
        src_dir = parentdir+"/src-"+str(i)
    os.makedirs(src_dir) #, exist_ok=True)
    whattocopy = [
        os.path.dirname(os.path.dirname(__file__))
    ] + [
        arg for arg in sys.argv if os.path.isfile(arg)
    ]
    if add_packages:
        if not isinstance(add_packages, list):
            add_packages = [add_packages]
        for package in add_packages:
            if isinstance(package, types.ModuleType):
                package = os.path.dirname(package.__file__)
            assert isinstance(package, str)
            whattocopy.append(package)
    for what in whattocopy:
        dest = src_dir+"/"+os.path.basename(what)
        if os.path.isfile(what):
            shutil.copy(what, dest)
        else:
            shutil.copytree(what, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.pyc", "__pycache__"))

# Return the longest prefix of all list elements.
def commonprefix(m, end = None):
    "Given a list of pathnames, returns the longest common leading component"
    if not m: return ''
    s1 = min(m)
    s2 = max(m)
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
    if end:
        while len(s1) and not s1.endswith(end):
            s1 = s1[:-1]
    return s1

def remove_commonprefix(l, end = None):
    cp = commonprefix(l, end)
    return [s[len(cp):] for s in l]

class suppress_stderr(object):
    def __enter__(self):
        self.errnull_file = open(os.devnull, 'w')
        self.old_stderr_fileno_undup    = sys.stderr.fileno()
        self.old_stderr_fileno = os.dup ( sys.stderr.fileno() )
        self.old_stderr = sys.stderr
        os.dup2 ( self.errnull_file.fileno(), self.old_stderr_fileno_undup )
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stderr = self.old_stderr
        os.dup2 ( self.old_stderr_fileno, self.old_stderr_fileno_undup )
        os.close ( self.old_stderr_fileno )
        self.errnull_file.close()
