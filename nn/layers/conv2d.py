import numpy as np

class conv2d():
    def __init__(self, filters):
        self.filters = filters
        self.kernel_size = 3
        self.strides = self.kernel_size
        self.resize = self.filters * self.kernel_size * self.kernel_size

        self.weights, self.last_dw = None, None

    def get_weights(self):
        return [self.input_shape, self.filters, self.kernel_size, self.resize, self.weights, self.last_dw]

    def set_weights(self, weights):
        self.input_shape, self.filters, self.kernel_size, self.resize, self.weights, self.last_dw = weights

    def get_weight_squared_sum(self):
        return np.sum(self.weights * self.weights)
            #, self.filters * self.kernel_size * self.kernel_size * self.input_shape[2]

    def set_input_shape(self, input_shape):
        assert len(input_shape) == 3
        self.input_shape = input_shape

        self.weights = np.random.uniform(-1e-4, 1e-4, (self.filters, self.kernel_size, self.kernel_size, self.input_shape[2]))
        self.last_dw = np.zeros((self.filters, self.kernel_size, self.kernel_size, self.input_shape[2]))

    def get_output_shape(self):
        return (self.input_shape[0] // self.strides, self.input_shape[1] // self.strides, self.filters)

    def forward(self, x):
        # x: (None, h, w, channel)
        # weight: (filters, channel, kernel_size, kernel_size)
        # out: (None, out_h, out_w, filters)

        self.last_input = x
        batch_size, h, w, channels = x.shape
        out_h, out_w = h // self.strides, w // self.strides
        out = np.zeros((batch_size, out_h, out_w, self.filters))

        for f in range(self.filters):
            for i in range(out_h):
                for j in range(out_w):
                    ii, jj = i * self.strides, j * self.strides
                    tmp = x[:, ii:ii + self.kernel_size, jj:jj + self.kernel_size] * self.weights[f]
                    tmp = np.reshape(tmp, (tmp.shape[0], -1))
                    out[:, i, j, f] = np.sum(tmp, axis=1)

        return out

    def backward(self, grad, lr=0.01, momentum=None, l2_lambda=0.1):
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
                                grad_out[:, h+h_inc, w+w_inc, c] += self.weights[f, h_inc, w_inc, c] * e

                                dw[f, h_inc, w_inc, c] += np.sum(self.last_input[:, h+h_inc, w+w_inc, c] * e)

        dw = dw / (batch_size * self.resize)

        # l2 Regularization
        dw += (l2_lambda * self.weights)

        dw *= lr

        # momentum
        if momentum is not None:
            dw += momentum * self.last_dw
            self.last_dw = dw

        self.weights += dw

        grad_out = grad_out / (batch_size * self.resize)
        return grad_out
