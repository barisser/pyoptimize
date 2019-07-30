import pyoptimize as pyopt

import numpy as np


def test_denserect():
    weights = np.random.rand(10000, 2) * 2 - 1.0
    dr = pyopt.DenseRect(100, 3, weights)
    iv = np.random.rand(100) * 2 - 1.0
    response = dr.run(iv)
    assert response.shape == (100,)

def test_solve_gym_simple():
    constraints = []
    width = 20
    depth = 3
    environment = 'CartPole-v0'
    reward_function = pyopt.gym_reward_function(width, depth, environment)
    vector = np.random.rand(800)
    solution = pyopt.gradient_descent(vector, reward_function, constraints)
#    solution = pyopt.pop_descent(vector, reward_function, constraints, 10)
#    rtr = reward_function(vector)
    import pdb;pdb.set_trace()
#    assert True