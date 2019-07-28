import pyoptimize as pyopt


def compare_vectors(v1, v2, abs_tolerance):
    assert len(v1) == len(v2)
    return all([abs(v1[i]-v2[i])<abs_tolerance for i in range(len(v1))])


def test_simple_optimization():
    constraints = [lambda x: 5- x[0], lambda x: -1 - x[1], lambda x: -3 - x[2]]

    solution = pyopt.optimize([0, -3, -6], lambda x: sum(x) + x[0], constraints)
    assert compare_vectors(solution, [5, -1, -3], 10**-3)

def test_optimization_starting_out_of_constraints():
    constraints = [lambda x: 5- x[0], lambda x: -1 - x[1], lambda x: -3 - x[2]]

    solution = pyopt.optimize([9, -3, 16], lambda x: sum(x) + x[0], constraints)
    assert compare_vectors(solution, [5, -1, -3], 10**-6)    