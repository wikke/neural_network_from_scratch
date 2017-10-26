import numpy as np

class conv2d():
    def __init__(self, filters):
        self.filters = filters
        self.kernel_size = 3
        self.resize = self.filters * self.kernel_size * self.kernel_size

        self.weights, self.last_delta_weight = None, None

    def get_weights(self):
        return [self.input_shape, self.filters, self.kernel_size, self.resize, self.weights, self.last_delta_weight]

    def set_weights(self, weights):
        self.input_shape, self.filters, self.kernel_size, self.resize, self.weights, self.last_delta_weight = weights

    def get_l2_loss(self):
        return np.sum(self.weights * self.weights), self.filters * self.kernel_size * self.kernel_size * self.input_shape[2]

    def set_input_shape(self, input_shape):
        assert len(input_shape) == 3
        self.input_shape = input_shape

        self.weights = np.random.uniform(-1e-4, 1e-4, (self.filters, self.kernel_size, self.kernel_size, self.input_shape[2]))
        self.last_delta_weight = np.zeros((self.filters, self.kernel_size, self.kernel_size, self.input_shape[2]))

    def get_output_shape(self):
        return (self.input_shape[0] - self.kernel_size + 1, self.input_shape[1] - self.kernel_size + 1, self.filters)

    def forward(self, x):
        # x: (None, h, w, channel)
        # weight: (filters, channel, kernel_size, kernel_size)
        # out: (None, out_h, out_w, filters)

        self.last_input = x
        batch_size, h, w, channels = x.shape
        out_h, out_w = h - self.kernel_size + 1, w - self.kernel_size + 1
        out = np.zeros((batch_size, out_h, out_w, self.filters))

        # for b in range(batch_size):
        for f in range(self.filters):
            for i in range(out_h):
                for j in range(out_w):
                    # kernel_size, kernel_size, channel
                    # x[b, i:i+self.kernel_size, j:j+self.kernel_size]

                    # kernel_size, kernel_size, channel
                    # self.weights[f]
                    # out[:, i, j, f] = np.sum(x[:, i:i+self.kernel_size, j:j+self.kernel_size] * self.weights[f], axis=0)

                    tmp = x[:, i:i+self.kernel_size, j:j+self.kernel_size] * self.weights[f]
                    tmp = np.reshape(tmp, (tmp.shape[0], -1))
                    out[:, i, j, f] = np.sum(tmp, axis=1)

        return out

    def backward(self, grad, lr=0.01, momentum=0.9, l2_lambda=0.1):
        # error: (None, out_h, out_w, filters)
        # dw: (filters, kernel_size, kernel_size, channel)
        # grad_out: (None, h, w, channel)

        batch_size, error_h, error_w, _ = grad.shape
        grad_out = np.zeros((batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        dw = np.zeros((self.filters, self.kernel_size, self.kernel_size, self.input_shape[2]))

        # for b in range(batch_size):
        for f in range(self.filters):
            for h in range(error_h):
                for w in range(error_w):
                    # error @ output
                    # index 1 is out of bounds for axis 3 with size 1
                    e = grad[:, h, w, f]

                    for h_inc in range(self.kernel_size):
                        for w_inc in range(self.kernel_size):
                            for c in range(self.input_shape[2]):
                                # error-out = weights * e

                                # weight: (filters, kernel_size, kernel_size, channel)
                                # e: (batch_size, error_h, error_w, filters)
                                # grad_out: (batch_size, kernel_size, kernel_size, channel)

                                grad_out[:, h+h_inc, w+w_inc, c] += self.weights[f, h_inc, w_inc, c] * e

                                # weight-delta = input-image * e
                                # self.last_input: (None, h, w, channel)
                                dw[f, h_inc, w_inc, c] += np.sum(self.last_input[:, h+h_inc, w+w_inc, c] * e)

        dw = dw / (batch_size * self.resize)

        # l2 Regularization
        dw += (l2_lambda * self.weights)

        # momentum
        dw += momentum * self.last_delta_weight

        self.weights += lr * dw
        self.last_delta_weight = dw

        grad_out = grad_out / (batch_size * self.resize)
        return grad_out
