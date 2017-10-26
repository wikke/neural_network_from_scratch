import numpy as np
from nn.models.seq import seq
from nn.layers import fc, conv2d, flatten, dropout, maxpooling2d
from nn.activations import sigmoid, relu, softmax
import utils

LR = 0.01
MOMENTUM = 0.5
L2_LAMBDA = 0.1

def build_model():
    model = seq(input_shape=(28, 28, 1), lr=LR, momentum=MOMENTUM, l2_lambda=L2_LAMBDA)

    model.add(conv2d.conv2d(filters=4))
    model.add(relu.relu())
    # model.add(conv2d.conv2d(filters=4))
    # model.add(relu.relu())
    # model.add(maxpooling2d.maxpooling2d(pool_size=2, stride=2))
    #
    # model.add(conv2d.conv2d(filters=4))
    # model.add(relu.relu())
    # model.add(conv2d.conv2d(filters=4))
    # model.add(relu.relu())
    # model.add(maxpooling2d.maxpooling2d())

    model.add(flatten.flatten())

    # model.add(fc.fc(i=128, o=128))
    # model.add(relu.relu())

    # model.add(dropout.dropout(0.2))
    model.add(fc.fc(units=10))
    model.add(softmax.softmax())
    # model.add(sigmoid.sigmoid())

    return model

def evaluate_model(model):
    X, y = utils.get_batch(dataset='cv')
    pred = model.forward(X)
    l2_reg_loss = model.get_l2_loss()
    entropy_loss = np.mean(utils.cross_entropy(y, pred))
    loss = entropy_loss + l2_reg_loss

    accuracy = 0.0
    for i in range(pred.shape[0]):
        if np.argmax(pred[i]) == np.argmax(y[i]):
            accuracy += 1 / X.shape[0]

    return loss, accuracy
