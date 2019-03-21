import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def rel_error(x, y):
   return np.max( np.abs(x-y) / np.maximum(1e-8, np.abs(x)+np.abs(y)) )

data = get_CIFAR10_data()
for k, v in list(data.item()):
   print(('%s: ' % k, v.shape))

# test affine_forward

num_inputs = 2
input_shape = (4,5,6)
output_dim = 3

input_size = num_inputs * np.prod(input_shape)
weight_size = output_dim * np.prod(input_shape)

x = np.linspace(-0.1, 0.5, num = input_size).reshape(num_inputs, *input_shape)
w = np.linspace(-0.2, 0.3, num = weight_size).reshape(np.prob(input_shape), output_dim)
b = np.linspace(-0.3, 0.1






