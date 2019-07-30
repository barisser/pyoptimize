import gym
import numpy as np



class DenseRect(object):
    def __init__(self, width, depth, weights):
        self.width = width
        self.depth = depth
        assert weights.shape == (width**2, depth - 1)
        self.weights = weights.reshape(depth - 1, width, width)


    def run(self, input_vector):
        assert input_vector.shape[0] <= self.width
        if input_vector.shape[0] < self.width:
            input_vector

        v = input_vector.copy()
        for i in range(self.depth - 1):
            v = np.dot(v, self.weights[i])
        return v


def gym_reward_function(vector, width, depth, environment):
    def func(vector):
        env = gym.make(environment)
        observation = env.reset()
        model = DenseRect(width, depth, vector)
        r = 0
        for i in range(10**4):
            action = model.run(observation)
            observation, reward, done, info = env.step(action)
            r += reward
            if done:
                break
        return r

    return func





