import os
import sys
import unittest
import shutil
import pickle
import hashlib
import numpy as np
import torch
import subprocess
import tempfile

class Test(unittest.TestCase):

    def setUp(self):
        #print("Running", self.__class__.__name__)
        os.environ["DATAPATH"] = self.get_data_path()
        self.maxDiff = None
        self.createdReferences = []

    def tearDown(self):
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        self.assertEqual(self.createdReferences, [], "Created references: " + ", ".join(self.createdReferences).replace(self.get_data_path()+"/", ""))
    
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

    def get_temp_path(self, fn = None):
        tmpdir = tempfile.gettempdir()
        if fn:
            tmpdir = os.path.join(tmpdir, fn)
        return tmpdir

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
            return self.assertRun(cmd.split())
        if cmd[0].endswith(".py"):
            cmd = [sys.executable] + cmd
        print("Running:", " ".join(cmd))
        p = subprocess.Popen(cmd, 
            env = dict(os.environ, PYTHONPATH = os.pathsep.join(sys.path)), # Otherwise ".local" path might be missing
            stdout = subprocess.PIPE #, stderr = subprocess.PIPE
        )
        (stdout, stderr) = p.communicate()
        self.assertEqual(p.returncode, 0)
        return stdout.decode("utf-8")

    def assertEqualFile(self, file, reference):
        reference = self.get_data_path("expected/" + reference, check = False)
        if not os.path.isfile(reference):
            dirname = os.path.dirname(reference)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            shutil.copyfile(file, reference)
            self.createdReferences.append(reference)
        self.assertTrue(os.path.isfile(file))
        content = open(file, "r").read()
        reference_content = open(reference, "r").read()
        self.assertEqual(content, reference_content)