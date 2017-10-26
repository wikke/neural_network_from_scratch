import numpy as np

class flatten():
    def set_input_shape(self, input_shape):
        self.input_shape = input_shape

    def get_output_shape(self):
        n = 1
        for s in self.input_shape:
            n *= s

        return (n,)

    def get_weights(self):
        return [self.input_shape]

    def set_weights(self, weights):
        self.input_shape = weights[0]

    def get_weight_squared_sum(self):
        return 0.0

    def forward(self, x):
        self.last_shape = x.shape
        return np.reshape(x, (x.shape[0], -1))

    def backward(self, grad, lr=0.01, momentum=None, l2_lambda=0.1):
        return np.reshape(grad, self.last_shape)
