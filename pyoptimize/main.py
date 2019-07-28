import math


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
    

def optimize(vector, reward_function, constraints):
    """
    Constraints are functions which must not return values zero or under.
    So if you want to constraint vector[2] < 5, use constraint 
    lambda vector: 5 - vector[2] 
    """
    learning_rate = 0.01
    best_vector = vector
    n = 0
    last_improvement = None


    while n < 100000:
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

