import mnist_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--weight", help="path of weights")
args = parser.parse_args()

model = mnist_model.build_model()
# model.load_weights('./weights/weights-252160-0.797545-0.8575')
model.load_weights(args.weight)
loss, accuracy = mnist_model.evaluate_model(model)

print("accuracy {} over {} samples".format(accuracy, 10000))
