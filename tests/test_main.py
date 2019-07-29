import pyoptimize as pyopt

import gym
import numpy as np


def compare_vectors(v1, v2, abs_tolerance):
    assert len(v1) == len(v2)
    return all([abs(v1[i]-v2[i])<abs_tolerance for i in range(len(v1))])


def test_simple_optimization():
    constraints = [lambda x: 5- x[0], lambda x: -1 - x[1], lambda x: -3 - x[2]]

    solution = pyopt.gradient_descent([0, -3, -6], lambda x: sum(x) + x[0], constraints)
    assert compare_vectors(solution, [5, -1, -3], 10**-3)

def test_optimization_starting_out_of_constraints():
    constraints = [lambda x: 5- x[0], lambda x: -1 - x[1], lambda x: -3 - x[2]]

    solution = pyopt.gradient_descent([9, -3, 16], lambda x: sum(x) + x[0], constraints)
    assert compare_vectors(solution, [5, -1, -3], 10**-5)    




def gym_reward_function(vector):
    env = gym.make('CartPole-v0')
    env.reset()
    assert vector.shape() == (10, 3)
#    model = 


def test_pop_descent():
    constraints = []
 #   reward_function = 
#    solution = pyopt.gradient_descent()    

#def test_penalty_function():
    #assert pyopt.penalty_function(0.01) == 10**-14
    #assert pyopt.penalty_function(0.00001) == 3