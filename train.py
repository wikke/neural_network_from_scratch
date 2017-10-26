import numpy as np
import utils
import mnist_model

TRAIN_SIZE = 32
EPOCHS = 1000000

def train():
    model = mnist_model.build_model()
    model.summary()

    for e in range(EPOCHS):
        X, y = utils.get_batch(TRAIN_SIZE, 'train')

        pred = model.forward(X)
        l2_reg_loss = model.get_l2_loss()
        entropy_loss = utils.cross_entropy(y, pred)
        loss = entropy_loss + l2_reg_loss
        print('loss {:.8f} = entropy {:.8f} + l2 {:.8f} | {} samples'
              .format(np.mean(loss), np.mean(entropy_loss), np.mean(l2_reg_loss), (e+1) * TRAIN_SIZE))

        grad = y - pred
        model.backward(grad)

        if (e+1) % 20 == 0:
            validate(model)

min_loss = 999999
lr_diminished = False
def validate(model):
    global min_loss, lr_diminished

    loss, accuracy = mnist_model.evaluate_model(model)
    print('-'*50)
    print('Validate Dataset, loss {:.8f}, acc {:.4f}'.format(loss, accuracy))

    if loss < 0.05 and not lr_diminished:
        print('\n\n******* Decrease LR to {} *******\n\n'.format(model.get_lr() / 50))
        model.set_lr(model.get_lr() / 50)
        lr_diminished = True

    if min_loss > loss:
        weight_file = './weights/weights-{}-{:.6f}-{:.4f}'.format(model.trained_samples, loss, accuracy)
        model.save_weights(weight_file)
        print('weight saved as {}'.format(weight_file))
        min_loss = loss

    print('-'*50)

if __name__ == '__main__':
    train()
