import pylab
from numpy import *
import pandas as pd
import sklearn
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import numpy as np
import mnist
import matplotlib.pyplot as plt

images, labels = mnist.load_mnist('training',selection=slice(0, 20000))
images = images.reshape(len(images),-1)

testing_images, testing_labels = mnist.load_mnist('testing',selection=slice(0, 5000))
testing_images = testing_images.reshape(len(testing_images),-1)

# mlp_classifier = MLPClassifier()
# mlp_classifier = MLPClassifier(hidden_layer_sizes= 12)
# mlp_classifier = OneVsRestClassifier(mlp_classifier)
# mlp_classifier.fit(images,labels)

x_axis = range(1,96,5)
print(x_axis)
y_axis = []

for i in range(1,96,5):
    mlp_classifier = MLPClassifier(hidden_layer_sizes=i)
    mlp_classifier = OneVsRestClassifier(mlp_classifier)
    mlp_classifier.fit(images,labels)
    y_axis.append(mlp_classifier.score(testing_images,testing_labels))
    print(i)

pylab.plot(x_axis,y_axis)
pylab.show()
