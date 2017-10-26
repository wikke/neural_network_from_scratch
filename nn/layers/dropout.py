import numpy as np

class dropout():
    def __init__(self, ratio=0.5):
        self.pass_ratio = 1 - ratio

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape

    def get_output_shape(self):
        return self.input_shape

    def get_weights(self):
        return [self.input_shape]

    def set_weights(self, weights):
        self.input_shape = weights[0]

    def get_weight_squared_sum(self):
        return 0.0

    def forward(self, x):
        return x

    def backward(self, grad, lr=0.01, momentum=None, l2_lambda=0.1):
        mask = np.zeros(grad.shape).flatten()
        l = mask.shape[0]
        mask[np.random.choice(l, int(l * self.pass_ratio), replace=False)] = 1
        return grad * mask.reshape(grad.shape)
