import numpy as np
from nn.models.seq import seq
from nn.layers import fc, conv2d, flatten, dropout, maxpooling2d
from nn.activations import sigmoid, relu, softmax
import mnist_utils

LR = 0.001
MOMENTUM = 0.9
L2_LAMBDA = 0.1

def build_model():
    model = seq(input_shape=(28, 28, 1), lr=LR, momentum=MOMENTUM, l2_lambda=L2_LAMBDA)

    model.add(conv2d.conv2d(filters=4))
    model.add(relu.relu())
    # model.add(conv2d.conv2d(filters=4))
    # model.add(relu.relu())
    # model.add(maxpooling2d.maxpooling2d(pool_size=2, stride=2))
    model.add(flatten.flatten())
    # model.add(dropout.dropout(0.5))
    model.add(fc.fc(units=10))
    model.add(softmax.softmax())

    return model

def evaluate_model(model):
    X, y = mnist_utils.get_batch(dataset='cv')
    pred = model.forward(X)
    loss, entropy_loss, l2_reg_loss = mnist_utils.cal_loss(y, pred, model)

    accuracy = 0.0
    for i in range(pred.shape[0]):
        if np.argmax(pred[i]) == np.argmax(y[i]):
            accuracy += 1.0 / X.shape[0]

    return np.mean(loss), accuracy
