

class DenseRect(object):
    def __init__(self, width, depth, weights):
        self.width = width
        self.depth = depth
        assert weights.shape == (width**2, depth-1)
        self.weights = weights

    def run(self, input_vector):
        assert input_vector.shape == (self.width,)
        v = input_vector.copy()
        for i in range(self.depth):

            import pdb;pdb.set_trace()
