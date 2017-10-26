# Neural Network from scratch

> It's really challenging!!! 
> 
> I'm just feeling that: **When neural network goes deep into code, you have to go back to mathematics.**

## Features Implemented

### Back Bropagation Algorithm

Definitely

### Model

Sequential(seq)

### Layers

dense(fc), conv2d, flatten, maxpooling2d, dropout

### Activations

relu, sigmoid, softmax

### Optimizor & Training

- Initial learning rate 0.001, set to 5e-5 after loss < 1.0
- SGD(with momentum)
- Gradients Clipping
- l2 Regularization

## MNIST Recognizer Example

### Network

- Input:(None, 28, 28, 1)
- Conv2D(relu)
- Flatten
- Dropout(0.5)
- Full-Connect(softmax)

### Performance

85% accuracy after 25w samples with a simple network above. 

Actually I had achieved 90% accuracy, but somehow I lost it. It's obviously that, complex model with high accuracy and low loss is harder to train, especially in Python.

### Env

- Ubuntu 16.04.2 LTS
- Python 2.7.12(Python 3.6 Also works)

### Scripts

- train:

`python train.py`

- test

`python test.py --weight ./weights/weights-252160-0.797545-0.8575`

- inference

`python inference.py --dir ./datasets/pics --weight ./weights/weights-252160-0.797545-0.8575`

### Attentions

If you meet

`ImportError: libSM.so.6: cannot open shared object file: No such file or directory`

on Ubuntu Server, just run command below:

`apt-get install libsm6 libxrender1 libfontconfig1`

