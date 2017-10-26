import numpy as np

class maxpooling2d():
    def __init__(self, pool_size=2, strides=2):
        self.pool_size = pool_size
        self.strides = strides

    def get_weights(self):
        return [self.pool_size, self.strides, self.input_shape]

    def set_weights(self, weights):
        self.pool_size, self.strides, self.input_shape = weights

    def get_l2_loss(self):
        return (0.0, 0)

    def set_input_shape(self, input_shape):
        assert len(input_shape) == 3
        self.input_shape = input_shape

    def get_output_shape(self):
        return (self.input_shape[0] // self.strides, self.input_shape[1] // self.strides, self.input_shape[2])

    def forward(self, x):
        s = x.shape
        self.last_shape = s
        height, width = s[1] // self.strides, s[2] // self.strides

        out = np.zeros((s[0], height, width, s[3]))
        self.last_max_pos = np.zeros((s[0], height, width, s[3]))

        for i in range(s[0]):
            for j in range(s[3]):
                for h in range(height):
                    for w in range(width):
                        zone = x[i, h:h+self.pool_size, w:w+self.pool_size, j]
                        max_value = np.max(zone)
                        max_idx = (zone == max_value).flatten()

                        if np.sum(max_idx) == 1:
                            self.last_max_pos[i, h, w, j] = int(np.argmax(zone))
                        else:
                            pos = np.random.choice(np.argwhere(max_idx==True).flatten())
                            self.last_max_pos[i, h, w, j] = int(pos)

                        out[i, h, w, j] = max_value

        return out

    def backward(self, grad, lr=0.01, momentum=None, l2_lambda=0.1):
        error_out = np.zeros(self.last_shape)

        for i in range(error_out.shape[0]):
            for j in range(error_out.shape[3]):
                for h in range(grad.shape[1]):
                    for w in range(grad.shape[2]):
                        h_offset = int(self.last_max_pos[i, h, w, j] // self.pool_size)
                        w_offset = int(self.last_max_pos[i, h, w, j] % self.pool_size)
                        error_out[i, h+h_offset, w+w_offset, j] = grad[i, h, w, j]

        return error_out
