# Define a reward function
# reward functions always take a vector as an arg and return a float score to be maximized.
def reward(vector):
    distance = sum([(x-3)**2 for x in vector])**0.5
    return 2**(-1 * distance **2)

# Optionally define constraints as functions.
# constraint functions always take a vector as an arg and return a float.
# a positive float signifies compliance with the constraint, zero or negative values are not compliant.
constraints = [lambda vector: -3 - vector[2], lambda vector: 4 - sum(vector)]

# Choose a starting vector.
starting_vector = [1, 2, 3] # note this vector is actually not compliant with above constraints

# find a local maxima
import pyoptimize

solution, score = pyoptimize.gradient_descent(starting_vector, reward, constraints)

print("Found solution: {0}".format(solution))
