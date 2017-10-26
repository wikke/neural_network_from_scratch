import mnist_model

model = mnist_model.build_model()
model.load_weights('./weights/weights-30720-0.039668-0.8929')
loss, accuracy = mnist_model.evaluate_model(model)

print("accuracy {} over {} samples".format(accuracy, 10000))
