import numpy as np

class softmax():
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
        # stable softmax
        x = x - np.max(x)
        exps = np.exp(x)
        exps_sum = exps.sum(axis=1)

        for i in range(exps.shape[0]):
            exps[i] /= exps_sum[i]

        self.last_out = exps
        return exps

    # softmax derivative: s * (1 - s), same with sigmoid
    # since "a softmax for two dimensions (events) is exactly the sigmoid function"
    def backward(self, grad, lr=0.1, momentum=None, l2_lambda=0.1):
        return grad * self.last_out * (1 - self.last_out)
