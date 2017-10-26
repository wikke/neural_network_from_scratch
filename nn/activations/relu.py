import numpy as np

class relu():
    def set_input_shape(self, input_shape):
        self.input_shape = input_shape

    def get_output_shape(self):
        return self.input_shape

    def get_weights(self):
        return [self.input_shape]

    def set_weights(self, weights):
        self.input_shape = weights[0]

    def get_l2_loss(self):
        return (0.0, 0)

    def forward(self, x):
        x[x < 0] = 0
        return x

    def backward(self, grad, lr=0.01, momentum=None, l2_lambda=0.1):
        grad[grad < 0] = 0
        return grad
