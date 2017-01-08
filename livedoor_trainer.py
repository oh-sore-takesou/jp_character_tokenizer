import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

from tqdm import tqdm
from sys import exit as e

from mynet import CharLinear
from livedoor_datareader import LivedoorReader

def train():
    livedoor = LivedoorReader()
    smax_encoded_datasets = livedoor('smax')
    # print(len(smax_encoded_datasets)) # 860
    # print(len(smax_encoded_datasets[0])) # 1024
    # print(len(smax_encoded_datasets[0][0])) # 66
    ithack_encoded_datasets = livedoor('it-life-hack')
    smax_data_tuples = []
    ithack_data_tuples = []
    # make tuple with y
    for smax_encoded_dataset in smax_encoded_datasets:
        smax_data_tuples.append(tuple([np.array(smax_encoded_dataset, dtype=np.float32), [1,0]]))
    for ithack_encoded_dataset in ithack_encoded_datasets:
        ithack_data_tuples.append(tuple([np.array(ithack_encoded_dataset, dtype=np.float32), [0,1]]))
    # separate train and test
    train = smax_data_tuples[:800] + ithack_data_tuples[:800]
    test = smax_data_tuples[800:] + ithack_data_tuples[800:]
    xtrain = np.array([ x for x, t in train ], dtype=np.float32)
    ytrain = np.array([ t for x, t in train ], dtype=np.float32)
    xtest = np.array([ x for x, t in test ], dtype=np.float32)
    ytest = np.array([ t for x, t in test ], dtype=np.float32)
    # setup model
    model = CharLinear(100, 2)
    optimizer = optimizers.SGD()
    optimizer.setup(model)
    # train
    n = len(xtrain)
    bs = 10
    losses = []
    for j in tqdm(range(10)):
        sffindx = np.random.permutation(n)
        for i in range(0, n, bs):
            x = Variable(xtrain[sffindx[i:(i+bs) if (i+bs) < n else n]])
            y = Variable(ytrain[sffindx[i:(i+bs) if (i+bs) < n else n]])
            model.zerograds()
            loss = model(x, y)
            loss.backward()
            optimizer.update()
            losses.append(loss)

    # test
    n_test = len(xtest)
    bs_test = 5
    sffindx = np.random.permutation(n_test)
    n_correct = 0
    n_incorrect = 0
    for i in range(0, n_test, bs_test):
        x = Variable(xtest[sffindx[i:(i+bs) if (i+bs) < n else n]])
        y = ytest[sffindx[i:(i+bs) if (i+bs) < n else n]]
        result = model.fwd(x)
        for r, ans in zip(result, y):
            if r.data.argmax() == ans.argmax():
                n_correct += 1
            else:
                n_incorrect += 1
    print('correct: {} \nincorrect: {}'.format(n_correct, n_incorrect))


if __name__ == '__main__':
    train()
