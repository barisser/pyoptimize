import pyoptimize as pyopt

import numpy as np


def test_denserect():
    weights = np.random.rand(10000, 2) * 2 - 1.0
    dr = pyopt.DenseRect(100, 3, weights)
    iv = np.random.rand(100) * 2 - 1.0
    response = dr.run(iv)
    assert response.shape == (100,)
