import numpy as np
from nn.models.seq import seq
from nn.layers import rnn, fc
from nn.activations import softmax

EPOCHS = 1000000
BATCH_SIZE = 16
LR = 0.001
MOMENTUM = 0.9
L2_LAMBDA = 0.1
PATIENCE = 5

TIME_STEPS = 5
INPUT_DIM = 1

RNN_UNITS = 8
OUTPUT_DIM = 2

def get_model():
    model = seq(input_shape=(TIME_STEPS, INPUT_DIM), lr=LR, momentum=MOMENTUM, l2_lambda=L2_LAMBDA)
    model.add(rnn.rnn(units=RNN_UNITS))
    model.add(fc.fc(units=OUTPUT_DIM))
    model.add(softmax.softmax())
    # model.summary()

    return model

def gen(batch_size):
    X = np.zeros((batch_size, TIME_STEPS, INPUT_DIM))
    y = np.zeros((batch_size, OUTPUT_DIM))
    y[:, 1] = 1

    idx = np.random.choice(batch_size, batch_size // 2, replace=False)
    X[idx, 0, 0] = 1
    y[idx, 0] = 1
    y[idx, 1] = 0

    return X, y

def cal_loss(true, pred, model):
    n = true.shape[0]

    entropy_loss = -np.sum(true * np.log(pred) + (1 - true) * np.log(1 - pred)) / n
    l2_loss = model.get_l2_loss() / n
    loss = entropy_loss + l2_loss

    return loss, entropy_loss, l2_loss


def evaluate(model, batch_size=BATCH_SIZE, verbose=False):
    X, y = gen(batch_size)
    pred = model.forward(X)
    loss, entropy_loss, l2_loss = cal_loss(y, pred, model)

    accuracy = 0.0
    for b in range(batch_size):
        if np.argmax(y[b]) == np.argmax(pred[b]):
            accuracy += 1.0 / batch_size

    if verbose:
        print(X)
        print(y)
        print(pred)
        print('loss {}, entropy_loss {}, l2_loss {}, accuracy {}'.format(loss, entropy_loss, l2_loss, accuracy))

    return loss, accuracy

def main():
    model = get_model()

    stop_num, min_loss = 0, 99999999
    for e in range(EPOCHS):
        X, y = gen(BATCH_SIZE)
        pred = model.forward(X)
        loss, entropy_loss, l2_loss = cal_loss(y, pred, model)
        # print('loss {:.8f} = entropy {:.8f} + l2 {:.8f} | {} samples'
        #       .format(np.mean(loss), np.mean(entropy_loss), np.mean(l2_loss), (e + 1) * BATCH_SIZE))

        grad = y - pred
        model.backward(grad)

        if (e + 1) % 10 == 0:
            loss, accuracy = evaluate(model, batch_size=BATCH_SIZE * 10)
            print('validate loss {}, accuracy {}'.format(loss, accuracy))

            if min_loss >= loss:
                min_loss = loss
                stop_num = 0
            else:
                stop_num += 1

                if stop_num >= PATIENCE:
                    print('early stop')
                    break

    evaluate(model, batch_size=1, verbose=True)

if __name__ == '__main__':
    main()
