import numpy as np

class fc():
    def __init__(self, units):
        self.units = units

        self.W, self.last_dw = None, None
        self.b = np.random.uniform(-1e-4, 1e-4, (units,))
        self.last_db = np.zeros((units,))

    def get_weights(self):
        return [self.input_shape, self.W, self.b, self.last_dw, self.last_db]

    def set_weights(self, weights):
        self.input_shape, self.W, self.b, self.last_dw, self.last_db = weights

    def get_l2_loss(self):
        # TODO check if no this.b considered
        return np.sum(self.W * self.W), self.input_shape[0] * self.units

    def set_input_shape(self, input_shape):
        assert len(input_shape) == 1
        self.input_shape = input_shape

        self.W = np.random.uniform(-1e-4, 1e-4, (input_shape[0], self.units))
        self.last_dw = np.zeros((input_shape[0], self.units))

    def get_output_shape(self):
        return (self.units,)

    def forward(self, x):
        self.last_input = x
        return np.dot(x, self.W) + self.b

    def backward(self, grad, lr=0.01, momentum=0.9, l2_lambda=0.1):
        dw = np.dot(self.last_input.T, grad)
        # FIXME mean ?
        db = np.mean(grad, axis=0)

        # l2 Regularization
        dw += (l2_lambda * self.W)

        # momentum
        dw += momentum * self.last_dw
        db += momentum * self.last_db
        self.last_dw = dw
        self.last_db = db

        grad_out = np.dot(grad, self.W.T)

        # udpate at last
        self.W += lr * dw
        self.b += lr * db

        return grad_out
