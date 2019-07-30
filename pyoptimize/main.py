import math
import operator

import numpy as np


def penalty_function(nearness, buff=10**-6):
    """
    weak regime --> far from constraint violation
    strong regime --> close to but not doing constraint violation
    negative regime --> already in violation of constraints

    Penalty function must be continuous!
    """
    if nearness > buff: # weak
        return 1 / ((buff + 10**-32))
    elif nearness > 0 and nearness <= buff: # strong
        return 1 / ((nearness + 10**-32))
    else: # negative
        return 10**64 * (1 - nearness)


def constraint_penalty(vector, constraints):
    p = 0
    for constraint in constraints:
        # give exponentially exploding penalty 
        nearness = constraint(vector)
        p += penalty_function(nearness)
    return p


def sim(vector, reward_function, constraints):
    return reward_function(vector) - constraint_penalty(vector, constraints)
    

def mutate_vector(vector, distance):
    v = np.array(vector)
    return v + (2 * np.random.rand(*v.shape) - 1.0) * distance


def solutions_close(s1, s2):
    return np.linalg.norm(np.array(s1)-np.array(s2)) / len(s1) < 0.2

def remove_duplicate_solutions(solutions_dict):
    s = solutions_dict.keys()
    d = {}
    for i in range(len(s)):
        # FIX ME
        add = True
        for k in d:
#            import pdb;pdb.set_trace()
            if solutions_close(s[i], k):
                add = False
                break
        if add:
            d[s[i]] = solutions_dict[s[i]]
    return d



def pop_descent(vector, reward_function, constraints, pop_n,
    survivors=0.3, iterations=10, jump_distance=0.5):
    """
    Form a population of vectors (pop_n).
    For each vector, perform gradient descent.
    Out of end results, select top N survivors.

    Out of these survivors, send out 'scout' vectors mutated
    from each survivor.  The amount of randomness by which
    they are mutated is a temperature function which is variable.
    Each scout repeats the gradient descent process.
    """
    pop = [vector]

    for i in range(iterations):
        assert len(pop) <= pop_n
        while len(pop) < pop_n:
            pop.append(mutate_vector(vector, jump_distance))

        solutions = {}
        for n, p in enumerate(pop):
            solution = gradient_descent(p, reward_function, constraints)
            reward = reward_function(solution)
            solutions[tuple(solution)] = reward
        solutions = remove_duplicate_solutions(solutions)
        pops = [x[0] for x in sorted(solutions.items(), key=operator.itemgetter(1))[::-1][:int(survivors*pop_n)]]
        #import pdb;pdb.set_trace()
    return sorted(solutions.items(), key=operator.itemgetter(1))[::-1][0][0]


def gradient_descent(vector, reward_function, constraints, max_iterations=10**6):
    """
    Constraints are functions which must not return values zero or under.
    So if you want to constraint vector[2] < 5, use constraint 
    lambda vector: 5 - vector[2] 
    """
    learning_rate = 0.01
    best_vector = vector
    n = 0
    last_improvement = None


    while n < max_iterations:
        old_vector = list(best_vector)
        best_reward = sim(best_vector, reward_function, constraints)
        n += 1
        start_reward = best_reward

        for i in range(len(best_vector) * 2):
            new_vector = list(best_vector)
            new_vector[int(i/2)] += learning_rate * (-1)**i
            new_reward = sim(new_vector, reward_function, constraints)
            improvement = new_reward - best_reward

            if improvement > 0:
                best_vector = new_vector
                best_reward = new_reward
                last_improvement = improvement

        round_improvement = best_reward - start_reward
        if round_improvement > 0 and last_improvement:
            learning_rate = learning_rate * round_improvement / last_improvement
            last_improvement = round_improvement

        if best_vector == old_vector:
            if learning_rate < 0.00001:
                break
            else:
                learning_rate = learning_rate / 2.0

    return best_vector

