import gym
import numpy as np



class DenseRect(object):
    def __init__(self, width, depth, weights):
        self.width = width
        self.depth = depth
        #if not weights.shape == (width**2, depth - 1):
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)
        self.weights = weights.reshape(depth - 1, width, width)


    def run(self, input_vector):
        assert input_vector.shape[0] <= self.width
        if input_vector.shape[0] < self.width:
            input_vector = np.pad(input_vector, (self.width - len(input_vector))/2, 'constant')
            #import pdb;pdb.set_trace()

        v = input_vector.copy()
        for i in range(self.depth - 1):
            v = np.dot(v, self.weights[i])
        return v

def logistic(x):
    return 1 / (1 + np.exp(x))

def convert_to_action(response, env):
    if isinstance(env.action_space, gym.spaces.discrete.Discrete):
        return int(np.rint(logistic(response[:1]))[0])#env.action_space.n]))
    import pdb;pdb.set_trace()


def gym_reward_function(model, environment):
    env = gym.make(environment)
    def func(vector):
        observation = env.reset()
        r = 0
        for i in range(10**4):
            response = model.run(observation)
            action = convert_to_action(response, env)
            observation, reward, done, info = env.step(action)
            r += reward
            if done:
                break
        return r

    return func





