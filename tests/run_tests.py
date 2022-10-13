import os
import sys
import unittest

os.environ["HOME"] = os.path.dirname(__file__)

root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from unittests import *

if __name__ == '__main__':
    unittest.main()
