from glob import glob
import argparse
import cv2
import numpy as np

import mnist_model

# import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="dir of pictures")
args = parser.parse_args()

file_glob = glob('{}/*.png'.format(args.dir))

model = mnist_model.build_model()
model.load_weights('./weights/weights-30720-0.039668-0.8929')

for f in file_glob:
    img = cv2.resize(cv2.imread(f), (28, 28))
    # plt.imshow(img, cmap='gray')
    # plt.show()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    X = np.reshape(img, (1, 28, 28, 1))
    pred = model.forward(X)
    num = np.argmax(pred[0])

    print('{} {}'.format(num, f))
