import numpy as np
import pickle

class seq():
    def __init__(self, input_shape, lr=0.01, momentum=None, l2_lambda=0.1):
        self.input_shape = input_shape
        self.lr = lr
        self.momentum = momentum
        self.l2_lambda = l2_lambda
        self.layers = []
        self.trained_samples = 0

    def get_lr(self):
        return self.lr

    def set_lr(self, lr):
        self.lr = lr

    def add(self, layer):
        last_shape = self.input_shape if len(self.layers) == 0 else self.layers[-1].get_output_shape()
        layer.set_input_shape(last_shape)
        self.layers.append(layer)

    def get_l2_loss(self):
        loss = 0.0
        for l in self.layers:
            loss += l.get_weight_squared_sum()

        return (loss * self.l2_lambda) / 2.0

    def summary(self):
        print(self.input_shape)
        for l in self.layers:
            print(l)
            print(l.get_output_shape())

    def save_weights(self, file):
        weights = []
        for l in self.layers:
            weights.append(l.get_weights())

        with open(file, 'wb') as f:
            pickle.dump(weights, f, protocol=True)

    def load_weights(self, file):
        with open(file, 'rb') as f:
            weights = pickle.load(f)

            for l, w in zip(self.layers, weights):
                l.set_weights(w)

    def forward(self, input):
        for l in self.layers:
            input = l.forward(input)
        return input
    
    def backward(self, grad):
        self.trained_samples += grad.shape[0]

        layer_num = len(self.layers)
        for i in reversed(range(layer_num)):
            grad = np.clip(grad, -1.0, 1.0)
            grad = self.layers[i].backward(grad, lr=self.lr, momentum=self.momentum, l2_lambda=self.l2_lambda)
