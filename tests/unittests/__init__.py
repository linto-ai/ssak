import unittest
import os
import sys

import pickle
import hashlib

class Test(unittest.TestCase):

    def setUp(self):
        os.environ["DATAPATH"] = self.get_data_path()
        #print("Running", self.__class__.__name__)

    def hash(self, obj):
        # Hash an object into a deterministic string
        return hashlib.md5(pickle.dumps(obj)).hexdigest()

    def get_data_path(self, fn = None):
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data"
        )
        if fn:
            data_path = os.path.join(data_path, fn)
        self.assertTrue(os.path.exists(data_path), f"Cannot find {data_path}")
        return data_path

    # def test_paths(self):
    #     print("test_paths")
    #     self.assertTrue(os.path.isdir(self.get_data_path("audio")))
    #     self.assertTrue(os.path.isdir(self.get_data_path("kaldi")))


# Import all classes in the current directory
# https://stackoverflow.com/questions/6246458/import-all-classes-in-directory (no chance with https://stackoverflow.com/questions/1057431/loading-all-modules-in-a-folder-in-python)
path = os.path.dirname(os.path.abspath(__file__))

for py in [f[:-3] for f in os.listdir(path) if f.endswith('.py') and f != '__init__.py']:
    mod = __import__('.'.join([__name__, py]), fromlist=[py])
    classes = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]
    for cls in classes:
        setattr(sys.modules[__name__], cls.__name__, cls)
