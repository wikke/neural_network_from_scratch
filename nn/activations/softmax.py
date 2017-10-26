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
            # FIXME RuntimeWarning: invalid value encountered in true_divide exps[i] /= exps_sum[i]
            # FIXME x.min = -14878338.6839, x.max = -1010.88076834
            exps[i] /= exps_sum[i]

        return exps

    # https://stackoverflow.com/questions/42934036/building-the-derivative-of-softmax-in-tensorflow-from-a-numpy-version
    def backward(self, grad, lr=0.01, momentum=None, l2_lambda=0.1):
        J = - grad[..., None] * grad[:, None, :] # off-diagonal Jacobian
        iy, ix = np.diag_indices_from(J[0])
        J[:, iy, ix] = grad * (1. - grad) # diagonal
        return J.sum(axis=1)
