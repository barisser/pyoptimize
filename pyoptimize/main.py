import math
import operator

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression as PLSR
from sklearn.linear_model import LinearRegression as LR


def penalty_function(nearness, buff=10**-6):
    """
    weak regime --> far from constraint violation
    strong regime --> close to but not doing constraint violation
    negative regime --> already in violation of constraints

    Penalty function must be continuous!
    """
    # if nearness > buff: # weak
    #     return 1 / ((buff + 10**-32))
    # elif nearness > 0 and nearness <= buff: # strong
    #     return 1 / ((nearness + 10**-32))
    if nearness > 0:
        return 0
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
            solution = gradient_descent(p, reward_function, constraints)
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
    mean = pop_scores.mean()
    std = pop_scores.std()
    normalized_scores = (pop_scores - mean) / std 
    osp = pop_scores.copy()
    osp.sort()
    nthbest = osp[-top_n]

    topn = max(int(top_fr * len(pops)), 1)
    best = np.array([pops[n] for n, x in enumerate(pop_scores) if x >= nthbest][:topn])

    probabilities = np.exp(normalized_scores * survival_factor)
    probabilities = probabilities / probabilities.sum()
    next_gen = np.array([pops[x] for x in np.random.choice(range(len(pops)),
        p=probabilities, size=len(pops) - topn)])

    next_gen = next_gen + np.multiply((np.random.normal(size=next_gen.shape) * 2 - 1.0), learning_vector)
    next_gen = np.append(next_gen, best, axis=0)

    return next_gen


def iterate_factors(scores, previous_scores,
    pops, previous_pops,
    learning_vector, survival_factor):
    """
    Update learning parameters.
    Learning Vector is the amplitude of mutations along each
    parameter axis.

    Survival Factor is the exponential weight of score's relationship
    to survival probability.
    High survival_factor ---> higher score weighting --> stricter survival
    lower survival_factor --> laxer survival

    We want high SF when we are converging on a solution.
    We want low SF when we are stuck/converged, or in exploratory mode.
    """
    # smean = scores.mean()
    # nm = scores.max()
    # psmean = previous_scores.mean()
    # pm = previous_scores.max()
    # sstd = scores.std()
    # pstd = previous_scores.std()
    # if psmean == 1 and pstd == 0:
    #     return learning_vector, survival_factor

    # max_improvement = nm - pm
    # if max_improvement < 0:
    #     # we're not making ANY progress
    #     # try to converge
    #     survival_factor = 0.5

    #     if learning_vector < 10**-6:
    #         learning_vector = 0.1
    #     else:
    #         learning_vector = learning_vector 

    # elif smean - psmean <= 0:
    #     # max's are making progress but average's aren't.
    #     # get STRICT
    #     survival_factor = 20.0
    #     learning_vector = learning_vector * 1.0
    # else:
    #     # max's and average's are both making progress.
    #     # stay semilax
    #     # learning vectors can get expansive
    #     survival_factor = 3.0
    #     learning_vector = learning_vector * 2.0


    return learning_vector, survival_factor


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
    scorehistory = np.array([])

    while n < max_iterations:
        previous_scores = pop_scores.copy()
        for i in range(len(pops)):
            pop_scores[i] = sim(pops[i], reward_function, constraints)
        scorehistory = np.append(scorehistory, [pop_scores])
#        print("mean reward " + str(np.array([x for x in pop_scores if x > -10**32]).mean()))
        previous_pops = pops
        pops = buffer_pops(pops, pop_scores, learning_vector, survival_factor)
        n += 1
        learning_vector, survival_factor = iterate_factors(pop_scores, previous_scores,
            pops, previous_pops,
            learning_vector, survival_factor)
        if np.linalg.norm(learning_vector) < 10**-6:
            break

        print(n)
    sh = np.array(scorehistory).reshape(n, pop_n)
    import pdb;pdb.set_trace()
    return [pops[n] for n, x in enumerate(pop_scores) if x == pop_scores.max()][0]



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
#                print("new best:"+str(best_reward))
                last_improvement = improvement

        round_improvement = best_reward - start_reward
        if round_improvement > 0 and last_improvement:
            learning_rate = learning_rate * round_improvement / last_improvement
            last_improvement = round_improvement

        if best_vector == old_vector:
            if learning_rate < 10**-32:
                break
            else:
                learning_rate = learning_rate / 2.0

    return best_vector

