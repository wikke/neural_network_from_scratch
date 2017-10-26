import gzip, struct, random
import numpy as np

def one_hot_10(x):
    r = np.zeros((x.shape[0], 10))
    for i in range(x.shape[0]):
        r[i, x[i]] = 1
    return r.astype(np.uint8)

def load_mnist(dir):
    with gzip.open('{}/train-labels-idx1-ubyte.gz'.format(dir)) as f:
        magic, num = struct.unpack(">II", f.read(8))
        train_labels = np.fromstring(f.read(), dtype=np.int8)

    with gzip.open('{}/train-images-idx3-ubyte.gz'.format(dir)) as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        train_images = np.fromstring(f.read(), dtype=np.uint8).reshape(num, rows, cols, 1)

    with gzip.open('{}/t10k-labels-idx1-ubyte.gz'.format(dir)) as f:
        magic, num = struct.unpack(">II", f.read(8))
        test_labels = np.fromstring(f.read(), dtype=np.int8)

    with gzip.open('{}/t10k-images-idx3-ubyte.gz'.format(dir)) as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        test_images = np.fromstring(f.read(), dtype=np.uint8).reshape(num, rows, cols, 1)

    return train_images / 255.0, one_hot_10(train_labels), test_images / 255.0, one_hot_10(test_labels)

loaded = False
train_images, train_labels, cv_images, cv_labels, test_images, test_labels = None, None, None, None, None, None
def get_batch(batch_size=16, dataset='train'):
    global loaded, train_images, train_labels, cv_images, cv_labels, test_images, test_labels

    if not loaded:
        train_images, train_labels, test_images, test_labels = load_mnist('./datasets/MNIST')
        cv_images, cv_labels = train_images[:10000], train_labels[:10000]
        train_images, train_labels = train_images[10000:], train_labels[10000:]
        loaded = True

    if dataset == 'train':
        idx = random.randint(0, train_images.shape[0] - batch_size)
        return train_images[idx:idx+batch_size].copy(), train_labels[idx:idx+batch_size].copy()
    elif dataset == 'cv':
        return cv_images, cv_labels
    elif dataset == 'test':
        return test_images, test_labels
    else:
        return None, None

def cal_loss(true, pred, model):
    n = true.shape[0]

    entropy_loss = -np.sum(true * np.log(pred) + (1 - true) * np.log(1 - pred)) / n
    l2_loss = model.get_l2_loss() / n
    # squared_weights, param_num = model.get_l2_weights_num()
    # l2_loss = np.sum(squared_weights) / (2.0 * param_num)

    loss = entropy_loss + l2_loss

    return loss, entropy_loss, l2_loss
