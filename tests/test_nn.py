import pyoptimize as pyopt

import numpy as np


def test_denserect():
    weights = np.random.rand(10000, 2)
    dr = pyopt.DenseRect(100, 3, weights)
    iv = np.random.rand(100)
    #dr.run(iv)
    #import pdb;pdb.set_trace()