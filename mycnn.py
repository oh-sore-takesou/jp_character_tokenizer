import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class CharCNN(Chain):
    def __init__(self, input_channel, output_channel, filter_height, filter_width, n_units, n_out):
        super(CharCNN, self).__init__(
            conv1=L.Convolution2D(None, output_channel, 1),
            l1=L.Linear(None, n_units),
            l2=L.Linear(None, n_out),
        )

    def __call__(self, x, y):
        return F.mean_squared_error(self.fwd(x), y)
    
    def fwd(self, x, train=True):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), 1)
        h2 = F.dropout(F.relu(self.l1(h1)), train=train)
        y = self.l2(h2)
        return y
