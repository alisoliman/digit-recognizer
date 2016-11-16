from pylab import *
from numpy import *
import mnist

images, labels = mnist.load_mnist('training', digits=[2])
imshow(images.mean(axis=0), cmap=cm.gray)
show()
