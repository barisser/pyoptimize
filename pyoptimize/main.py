import logging
import math
import multiprocessing as mp
import operator
import time

import numpy as np


def penalty_function(nearness, buff=10**-6):
    """
    weak regime --> far from constraint violation
    strong regime --> close to but not doing constraint violation
    negative regime --> already in violation of constraints

    Penalty function must be continuous!
    """
    if nearness > 0:
        return 0
    else: # negative
        return 10**64 * (1 - nearness)


def constraint_penalty(vector, constraints):
    if constraints is None:
        return 0

    p = 0
    for constraint in constraints:
        # give exponentially exploding penalty 
        nearness = constraint(vector)
        p += penalty_function(nearness)
    return p


def sim(vector, reward_function, constraints):
    return reward_function(vector) - constraint_penalty(vector, constraints)
    
def multisim(vectors, reward_function, constraints, pool=None):
    if pool is None:
        pool = mp.Pool(mp.cpu_count)

    def s(vector, constraints):
        return reward_function(vector) - constraint_penalty(vector, constraints)
    return pool.starmap(s, [tuple(v, constraints) for v in vectors])

def mutate_vector(vector, distance):
    v = np.array(vector)
    return v + (2 * np.random.rand(*v.shape) - 1.0) * distance


def solutions_close(s1, s2):
    return np.linalg.norm(np.array(s1)-np.array(s2)) / len(s1) < 0.2

def remove_duplicate_solutions(solutions_dict):
    s = solutions_dict.keys()
    d = {}
    for i in s:
        # FIX ME
        add = True
        for k in d:
            if solutions_close(i, k):
                add = False
                break
        if add:
            d[i] = solutions_dict[i]
    return d


def pop_grad_descent(vector, reward_function, constraints, pop_n,
    survivors=0.3, iterations=100, jump_distance=0.5):
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
    last_pops = []

    for i in range(iterations):
        assert len(pop) <= pop_n
        while len(pop) < pop_n:
            pop.append(mutate_vector(vector, jump_distance))

        solutions = {}
        for n, p in enumerate(pop):
            solution, _ = gradient_descent(p, reward_function, constraints)
            reward = reward_function(solution)
            solutions[tuple(solution)] = reward
        solutions = remove_duplicate_solutions(solutions)
        pops = [x[0] for x in sorted(solutions.items(), key=operator.itemgetter(1))[::-1][:int(survivors*pop_n)]]
        if set(pops) == set(last_pops):
            # our pops havent changed at all.
            # abort
            break
        last_pops = pops
    return sorted(solutions.items(), key=operator.itemgetter(1))[::-1][0][0]


def buffer_pops(pops, pop_scores, learning_vector, survival_factor, top_fr=0.1):
    """
    For a list of [vector, vector_score], add to
    the population by mutating.
    Mutate parents probabilistically.  Discard others.

    Mutations occur via the learning vector which has per-parameter mutation rates.
    """
    topn = max(int(top_fr * len(pops)), 1)
    mean = pop_scores.mean()
    std = pop_scores.std()
    normalized_scores = (pop_scores - mean) / std 
    osp = pop_scores.copy()
    osp.sort()
    nthbest = osp[-topn]

    best = np.array([pops[n] for n, x in enumerate(pop_scores) if x >= nthbest][:topn])

    probabilities = np.exp(normalized_scores * survival_factor)
    probabilities = probabilities / probabilities.sum()
    next_gen = np.array([pops[x] for x in np.random.choice(range(len(pops)),
        p=probabilities, size=len(pops) - topn)])

    next_gen = next_gen + np.multiply((np.random.normal(size=next_gen.shape) * 2 - 1.0), learning_vector)
    next_gen = np.append(next_gen, best, axis=0)

    return next_gen


def iterate_factors(sh, learning_vector, survival_factor, reset_n, max_fruitlessness=100, lookback=40):
    """
    Update learning parameters.
    Learning Vector is the amplitude of mutations along each
    parameter axis.

    Survival Factor is the exponential weight of score's relationship
    to survival probability.
    High survival_factor ---> higher score weighting --> stricter survival
    lower survival_factor --> laxer survival
    """

    if len(sh) < lookback+1 or sh.ndim == 1: # too little data
        return learning_vector, survival_factor, reset_n

    if len(sh) % 100 != 0: # only run occasionally
        return learning_vector, survival_factor, reset_n

    if sh[-(lookback + 1)].max() <= -1 * 10**32: # still outside of constraints
        return learning_vector, survival_factor, reset_n

    maxes = sh.max(axis=1)
    exploratory_attempts_max = 1
    means = sh.mean(axis=1)
    # if max rate of growth is positive but decreasing, decrease learningrate
    recent_growth = maxes[-1] - maxes[-(lookback+1)]
    dd = maxes[-1] - maxes[int(-(lookback/2 + 1))] * 2 + maxes[-(lookback+1)] # double derivative

    if recent_growth == 0: # no max progress !
        # either still converging
        if learning_vector.mean() < 10**-6:
            if reset_n >= exploratory_attempts_max:
                return -1, 1000, reset_n
            # reset at exploratory value
            learning_vector = np.maximum(learning_vector, 0.5)
            survival_factor = 5.0
            reset_n += 1
        else: # try to converge
            learning_vector = learning_vector * 0.5
            survival_factor = 5.0 # strict

    return learning_vector, survival_factor, reset_n


def pop_descent(vector, reward_function, constraints, pop_n=10, learning_rate=0.01,
    survival_factor=2.0, max_iterations=10**4):
    """
    """
    vector = np.array(vector)
    learning_vector = np.ones(vector.shape) * learning_rate
    pop_scores = np.ones(pop_n)
    pops = buffer_pops(np.array([vector for _ in range(pop_n)]),
        pop_scores,
        learning_rate, survival_factor)
    n = 0
    last_improvement = None
    best = None
    previous_scores = None
    previous_pops = None
    reset_n = 0
    scorehistory = None
    pool = mp.Pool(4)

    while n < max_iterations:
        previous_scores = pop_scores.copy()
        for i in range(len(pops)):
            pop_scores[i] = sim(pops[i], reward_function, constraints)
        if scorehistory is not None:
            scorehistory = np.append(scorehistory, [pop_scores], axis=0)
        else:
            scorehistory = np.array([pop_scores])
        previous_pops = pops
        pops = buffer_pops(pops, pop_scores, learning_vector, survival_factor)
        n += 1
        learning_vector, survival_factor, reset_n = iterate_factors(scorehistory,
            learning_vector, survival_factor, reset_n)
        if isinstance(learning_vector, int) and learning_vector == -1:
            break

    sh = np.array(scorehistory).reshape(n, pop_n)
    return [pops[n] for n, x in enumerate(pop_scores) if x == pop_scores.max()][0]

def multiple_gradient_descent(vector_ranges, reward_function,
    constraints=None, max_iterations=100, learning_rate=0.01):
    """
    Performs gradient_descent multiple times 
    from random starting points within the vector ranges.

    A vector range looks like
    [[-3, 3], [-4, 4],...[-7, -3]]
    """
    scores = {}
    size = tuple([len(vector_ranges)])
    vector_magnitudes = np.array([[x[1] - x[0]] for x in vector_ranges])[:, 0]
    vector_mins = np.array([x[0] for x in vector_ranges])

    for i in range(max_iterations):
        rv = np.random.uniform(0, 1, size=size)
        vector = np.multiply(rv, vector_magnitudes) + vector_mins
        optima_vector, score = gradient_descent(vector, reward_function,
            constraints=constraints, learning_rate=0.01)
        scores[tuple(optima_vector)] = score

    best = max(scores, key=scores.get)
    return best, scores[best]


def gradient_descent(vector, reward_function, constraints=None,
    max_iterations=10**6, learning_rate=0.01, break_learning_rate=10**-32,
    show_time=False, loud=False):
    """
    Constraints are functions which must not return values zero or under.
    So if you want to constraint vector[2] < 5, use constraint 
    lambda vector: 5 - vector[2] 
    """
    best_vector = np.array(vector).astype(float)
    start = time.time()
    n = 0
    last_improvement = None

    while n < max_iterations:
        old_vector = best_vector.copy()
        best_reward = sim(best_vector, reward_function, constraints)
        n += 1
        start_reward = best_reward

        for i in range(len(best_vector) * 2):
            new_vector = best_vector.copy()
            new_vector[int(i/2)] += learning_rate * (-1)**i
            new_reward = sim(new_vector, reward_function, constraints)
            improvement = new_reward - best_reward

            if improvement > 0:
                best_vector = new_vector
                best_reward = new_reward
                last_improvement = improvement
                if loud:
                    print("Found new best vector: {} at reward {}".format(best_vector, best_reward))

                    
        if (best_vector == old_vector).all():
            if learning_rate < break_learning_rate:
                break
            else:
                learning_rate = learning_rate / 2.0
        elif improvement > 0:
            learning_rate = learning_rate * 1.1

    duration = time.time() - start
    if show_time:
        print("Solved in {0} seconds.".format(duration))
    return np.array(best_vector), best_reward

