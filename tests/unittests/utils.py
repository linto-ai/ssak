import os
import sys
import unittest
import shutil
import pickle
import hashlib
import numpy as np
import torch
import subprocess

class Test(unittest.TestCase):

    def setUp(self):
        #print("Running", self.__class__.__name__)
        os.environ["DATAPATH"] = self.get_data_path()

    def tearDown(self):
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
    
    def get_data_path(self, fn = None, check = True):
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data"
        )
        if fn:
            data_path = os.path.join(data_path, fn)
        if check:
            self.assertTrue(os.path.exists(data_path), f"Cannot find {data_path}")
        return data_path

    def get_lib_path(self, fn = None):
        lib_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "audiotrain"
        )
        if fn:
            lib_path = os.path.join(lib_path, fn)
        self.assertTrue(os.path.exists(lib_path), f"Cannot find {lib_path}")
        return lib_path

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

    def assertClose(self, first, second, msg = None):
        return self.assertEqual(self.loose(first), self.loose(second), msg = f"{first} != {second}" if msg is None else msg)

    def assertRun(self, cmd):
        if isinstance(cmd, str):
            return self.assertRunCommand(cmd.split())
        print("Running:", " ".join(cmd))
        p = subprocess.Popen(cmd, 
            env = dict(os.environ, PYTHONPATH = os.pathsep.join(sys.path)), # Otherwise ".local" path might be missing
        )
        p.communicate()
        self.assertEqual(p.returncode, 0)


    # def test_paths(self):
    #     print("test_paths")
    #     self.assertTrue(os.path.isdir(self.get_data_path("audio")))
    #     self.assertTrue(os.path.isdir(self.get_data_path("kaldi")))