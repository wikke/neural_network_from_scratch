import numpy as np

# TODO better initialization
E = 0.2

class rnn():
    def __init__(self, units):
        self.units = units

        # self.W = np.random.uniform(-1e-4, 1e-4, (units, units))
        self.W = np.random.uniform(-E, E, (units, units))
        self.U = None
        self.last_dw = np.zeros((units, units))
        self.last_du = None

    def set_input_shape(self, input_shape):
        assert len(input_shape) == 2
        self.input_shape = input_shape
        self.time_steps  =input_shape[0]
        self.input_dim = input_shape[1]

        # self.U = np.random.uniform(-1e-4, 1e-4, (self.input_dim, self.units))
        self.U = np.random.uniform(-E, E, (self.input_dim, self.units))
        self.last_du = np.zeros((self.input_dim, self.units))

    def get_output_shape(self):
        return (self.units,)

    def forward(self, x):
        # x: <None, timesteps, input_dim>
        # self.last_time_steps = x.shape[1]
        batch_size = x.shape[0]
        status = np.zeros((batch_size, self.units))

        # record for bptt
        self.last_x = x
        self.last_all_status = np.zeros((batch_size, self.time_steps+1, self.units))
        # record the initial status, all zeros
        self.last_all_status[:, 0] = status

        for ts in range(self.time_steps):
            status = self.do_forward(status, x[:, ts])
            self.last_all_status[:, ts + 1] = status

        return status

    def do_forward(self, pre_status, x_ts):
        # pre_status: <None, self.units>
        # W: units, units
        #
        # x_t: <None, input_dim>
        # U: units, input_dim
        #
        # status: <None, self.units>

        status = np.tanh(np.dot(x_ts, self.U) + np.dot(pre_status, self.W))

        return status

    def get_weights(self):
        return [self.W, self.U, self.input_shape, self.time_steps, self.input_dim]

    def set_weights(self, weights):
        self.W, self.U, self.input_shape, self.time_steps, self.input_dim = weights

    def get_weight_squared_sum(self):
        return np.sum(self.W * self.W) * np.sum(self.U * self.U)

    def backward(self, grad, lr=0.01, momentum=None, l2_lambda=0.1):
        # grad/grad_next: <None, self.units>
        # grad_out: <None, timesteps, input_dim>

        grad_out = np.zeros((grad.shape[0], self.time_steps, self.input_dim))
        grad_status = grad

        # clear
        self.dw = np.zeros((self.units, self.units))
        self.du = np.zeros((self.input_dim, self.units))
        for ts in reversed(range(self.time_steps)):
            grad_out[:, ts], grad_status = self.do_bptt(ts, grad_status)

        # l2 Regularization
        self.dw += (l2_lambda * self.W)

        # update weight
        self.dw *= lr
        self.du *= lr

        # momentum
        if momentum is not None:
            self.dw += momentum * self.last_dw
            self.du += momentum * self.last_du

            self.last_dw = self.dw
            self.last_du = self.du

        self.W += self.dw
        self.U += self.du

        return grad_out

    def do_bptt(self, ts, grad_status):
        # self.last_x: <None, timesteps, input_dim>
        # grad_status: <None, self.units>
        # self.last_all_status = np.zeros((batch_size, self.time_steps, self.units))

        # formula: grad * (1-status ^ 2) * W

        # tanh_derivative: <None, self.units>
        tanh_derivative = 1 - np.square(self.last_all_status[:, ts])
        grad_tanh_d = grad_status * tanh_derivative

        grad_ts = np.dot(grad_tanh_d, self.U.T)
        grad_status_next = np.dot(grad_tanh_d, self.W.T)

        # Attention, self.last_all_status has shape <batch_size, self.time_steps+1, self.units>
        # which record the initial all zeros status at index 0
        self.dw += np.dot(self.last_all_status[:, ts].T, grad_tanh_d)
        self.du += np.dot(self.last_x[:, ts].T, grad_tanh_d)

        return grad_ts, grad_status_next
