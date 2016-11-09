# digit-recognizer

Design and train a multilayer perceptron (MLP) and an SVM engine, to recognize handwritten digits.
Using the MNIST dataset, one of the most widely-studied datasets in machine learning and pattern recognition literature http://yann.lecun.com/exdb/mnist/.

MNIST consists of a training dataset and a testing dataset, with 60,000 and 10,000 images of handwritten digits respectively. To cut down the training and testing time, I will choose to limit my training to the first 20,000 instances of the training dataset and to limit my testing to the first 5,000 instances of the testing dataset. The instances in the training and testing datasets are not ordered according to digit.
