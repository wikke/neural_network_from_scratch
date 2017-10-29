import numpy as np
import mnist_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import SGD

TRAIN_SIZE = 32
EPOCHS = 1000000

def train():
    model = Sequential()
    model.add(Conv2D(filters=4, kernel_size=3, strides=3, padding='valid', input_shape=(28,28,1), activation='relu'))
    model.add(Conv2D(filters=4, kernel_size=3, strides=3, padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=sgd, loss='binary_crossentropy')
    model.summary()

    for e in range(EPOCHS):
        X, y = mnist_utils.get_batch(TRAIN_SIZE, 'train')
        loss = model.train_on_batch(X, y)

        if (e+1) % 20 == 0:
            X, y = mnist_utils.get_batch(dataset='cv')
            loss = model.test_on_batch(X, y)

            pred = model.predict_on_batch(X)

            accuracy = 0.0
            for i in range(pred.shape[0]):
                if np.argmax(pred[i]) == np.argmax(y[i]):
                    accuracy += 1.0 / pred.shape[0]

            print('validate loss {}, acc {}'.format(loss, accuracy))

if __name__ == '__main__':
    train()
