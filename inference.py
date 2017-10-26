from glob import glob
import argparse
import cv2
import numpy as np
import mnist_model

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="dir of pictures")
parser.add_argument("--weight", help="path of weights")
args = parser.parse_args()

model = mnist_model.build_model()
model.load_weights(args.weight)

for f in glob('{}/*.png'.format(args.dir)):
    img = cv2.resize(cv2.imread(f), (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    X = np.reshape(img, (1, 28, 28, 1))
    pred = model.forward(X)
    num = np.argmax(pred[0])

    print('{} {}'.format(num, f))
