# Neural Network from scratch

> It's really challenging!!! 
> 
> I'm just feeling that: **When neural network goes deep into code, you have to go back to mathematics.**

Implement a neural network framework from scratch, and train with 2 examples:

- MNIST Classifier
- Time Series Prediction

## Neural Network Framework

- Back Propagation Algorithm
- Model: Sequential(seq)
- Layers: dense(fc), conv2d, flatten, maxpooling2d, dropout
- Activations: relu, sigmoid, softmax
- Optimizor & Training
    - Initial learning rate 0.001, set to 5e-5 after loss < 1.0
    - SGD(with momentum)
    - Gradients Clipping
    - l2 Regularization

## Examples 1. MNIST Classifier

### Netwrok Architecture

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

- train MNIST

`python train_mnist.py`

- test MNIST

`python test_mnist.py --weight ./weights/weights-252160-0.797545-0.8575`

- inference hand written digits

`python inference_mnist.py --dir ./datasets/pics --weight ./weights/weights-252160-0.797545-0.8575`

## Examples 2. RNN Example

Training dataset goes like this:

```
     X             y
1 0 0 ... 0      1  0
0 0 0 ... 0      0  1
0 0 0 ... 0      0  1
1 0 0 ... 0      1  0
1 0 0 ... 0      1  0
```

Pattern is obviously, right?

### Network Architecture

- Input:(None, TIME_STEPS, INPUT_DIM)
- RNN(RNN_UNITS)
- Full-Connect(2, with softmax)

### Scripts

`python train_rnn.py`
