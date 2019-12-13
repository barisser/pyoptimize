import math

import pyoptimize as pyopt

import numpy as np


def compare_vectors(v1, v2, abs_tolerance):
    assert len(v1) == len(v2)
    return all([abs(v1[i]-v2[i])<abs_tolerance for i in range(len(v1))])


def test_simple_optimization():
    constraints = [lambda x: 5- x[0], lambda x: -1 - x[1], lambda x: -3 - x[2]]

    solution, _ = pyopt.gradient_descent([0, -3, -6], lambda x: sum(x) + x[0], constraints, show_time=True)
    assert compare_vectors(solution, [5, -1, -3], 10**-12)

def test_optimization_starting_out_of_constraints():
    constraints = [lambda x: 5- x[0], lambda x: -1 - x[1], lambda x: -3 - x[2]]

    solution, _ = pyopt.gradient_descent([9, -3, 16], lambda x: sum(x) + x[0], constraints, show_time=True)
    assert compare_vectors(solution, [5, -1, -3], 10**-12)    

def test_slightly_more_complex_optimization():
    constraints = []
    reward_function = lambda y: sum([math.sin(x) * math.exp(-(x-15.0)**2) for x in y])
    solution, _ = pyopt.gradient_descent([0]*2, reward_function, constraints, show_time=True)
    # This is a local optima only!
    assert compare_vectors(solution, [3.099599]*2, 10**-5)
    # note for some reason above finds global optima if dimensionality is much higher

def simple_nonconvex_problem(v):
    c = [1, 2, 3, 1, 2]
    cc = [-5, -11, -2, -5, -7]
    s = math.exp(-1 * (sum([(v[i]-c[i])**2 for i in range(5)])))*3 
    s += math.exp(-1 * (sum([(v[i]-cc[i])**2 for i in range(5)])))*20
    return s


# -20 to +20= range
# 10 centers 8 dimensions
size = 10
centers_n = 10
ndims = 4
centers = np.random.rand(centers_n, ndims) * size*2 - size

center_weights = np.arange(centers_n)
center_steepnesses = np.random.random(centers_n) * 4

def hard_nonconvex_problem(v):
    s = 0
    for n, c in enumerate(centers):
        d = np.linalg.norm(v-c)
        s += np.exp(-(d**2) * center_steepnesses[n]) * center_weights[n]
    return s

def test_hard_nonconvex():
    vector_ranges = [[-size, size]]*ndims
    solution, score = pyopt.multiple_gradient_descent(vector_ranges,
                                                      hard_nonconvex_problem, max_iterations=20)
    assert score >= 5 # this is probabilistic...


# def test_pop_grad_descent():
#     constraints = []
#     reward_function = lambda y: sum([math.sin(x) * math.exp(-(x-9.0)**2) for x in y])
#     solution = pyopt.pop_grad_descent([0]*2, reward_function, constraints, 200)
#     assert compare_vectors(solution, [14.69088]*2, 10**-5)

# def test_pop_grad_descent2():
#     constraints = []
#     solution = pyopt.pop_grad_descent([0]*5, simple_nonconvex_problem, constraints, 200)
#     import pdb;pdb.set_trace()

# def test_pop_descent():
#     constraints = [lambda x: 5- x[0], lambda x: -1 - x[1], lambda x: -3 - x[2]]
#     reward_function = lambda y: sum(y)
#     solution = pyopt.pop_descent([0]*3, reward_function, constraints, 20)
#     assert compare_vectors(solution, [5, -1, -3], 10**-5)


# def test_pop_descent2():
#     constraints = []
#     solution = pyopt.pop_descent([0]*5, simple_nonconvex_problem, constraints)
#     import pdb;pdb.set_trace()


def test_multi_grad_descent():
    constraints = []
    vector_ranges = [[-10, 10]] * 5
    solution, score = pyopt.multiple_gradient_descent(vector_ranges,
        simple_nonconvex_problem, constraints, max_iterations=100)
    assert abs(score - 20) < 10**-6
    assert compare_vectors(solution, [-5, -11, -2, -5, -7], 10**-6)


def test_saddle_point():
    constraints = [lambda x: 1.0 - sum([y**2 for y in x])] # unit circle


def test_buffer_pops():
    poplist = np.array([[1,2,3],[2,4,4], [1,1,1]])
    popscores= np.array([1, 1, 1000])
    pops2 = pyopt.main.buffer_pops(poplist, popscores, 0.01, 2.0)




#def test_penalty_function():
    #assert pyopt.penalty_function(0.01) == 10**-14
    #assert pyopt.penalty_function(0.00001) == 3
