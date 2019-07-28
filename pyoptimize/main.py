import math


def constraint_penalty(vector, constraints, steepness=10):
    p = 0
    for constraint in constraints:
        # give exponentially exploding penalty 
        nearness = constraint(vector)
        if nearness > 0:
#            import pdb;pdb.set_trace()
            p += (1 / (nearness*10**16 + 10**-32))
        else:
            p += 10**32 - nearness*10**16 #math.exp((steepness*-nearness)**2) - 1.0
        # try:
        #     p += math.exp(steepness * -nearness) - 1.0
        # except:
        #     print("TOO STEEP")
        #     p += math.exp(steepness / 10 * -nearness) - 1.0
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


    while n<100000:
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
                #print("Found new best vector {0}  {1}".format(best_vector, best_reward))
                last_improvement = improvement

        round_improvement = best_reward - start_reward
        if round_improvement > 0 and last_improvement:
            learning_rate = learning_rate * round_improvement / last_improvement
            print(learning_rate)
            last_improvement = round_improvement

        if best_vector == old_vector:
            if learning_rate < 0.00001:
                break
            else:
                learning_rate = learning_rate / 2.0

    print("Final best vector/score {0} / {1}".format(best_vector, best_reward))
    return best_vector

