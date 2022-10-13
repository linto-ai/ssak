import os
import sys
import unittest
import shutil
import pickle
import hashlib
import numpy as np
import torch

class Test(unittest.TestCase):

    def setUp(self):
        #print("Running", self.__class__.__name__)
        os.environ["DATAPATH"] = self.get_data_path()

    def tearDown(self):
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
    
    def get_data_path(self, fn = None):
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data"
        )
        if fn:
            data_path = os.path.join(data_path, fn)
        self.assertTrue(os.path.exists(data_path), f"Cannot find {data_path}")
        return data_path

    def loosehash(self, obj):
        # Hash the approximation of an object into a deterministic string
        return self.hash(self.loose(obj))

    def hash(self, obj):
        # Hash an object into a deterministic string
        return hashlib.md5(pickle.dumps(obj)).hexdigest()

    def loose(self, obj, t = None):
        # Return an approximative value of an object
        if isinstance(obj, list):
            try:
                stats = {"mean": np.mean(obj), "std": np.std(obj)}
            except TypeError:
                return [self.loose(a) for a in obj]
            return self.loose({"type": t if t else "list", "len": len(obj)} | stats)
        if isinstance(obj, float):
            f = round(obj, 3)
            return 0.0 if f == -0.0 else f
        if isinstance(obj, dict):
            return {k: self.loose(v) for k, v in obj.items()}
        if isinstance(obj, tuple):
            return tuple(self.loose(list(obj)))
        if isinstance(obj, set):
            return self.loose(list(obj), "set")
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return self.loose(obj.tolist())
        return obj

    # def test_paths(self):
    #     print("test_paths")
    #     self.assertTrue(os.path.isdir(self.get_data_path("audio")))
    #     self.assertTrue(os.path.isdir(self.get_data_path("kaldi")))


# Import all classes in the current directory
# https://stackoverflow.com/questions/6246458/import-all-classes-in-directory
path = os.path.dirname(os.path.abspath(__file__))

for py in [f[:-3] for f in os.listdir(path) if f.endswith('.py') and f != '__init__.py']:
    mod = __import__('.'.join([__name__, py]), fromlist=[py])
    classes = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]
    for cls in classes:
        setattr(sys.modules[__name__], cls.__name__, cls)

