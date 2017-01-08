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

from mycnn import CharCNN
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
    train = smax_data_tuples[:200] + ithack_data_tuples[:200]
    test = smax_data_tuples[800:] + ithack_data_tuples[800:]
    xtrain = np.array([ x for x, t in train ], dtype=np.float32).reshape(len(train), 1, 66, 1024)
    ytrain = np.array([ t for x, t in train ], dtype=np.float32)
    xtest = np.array([ x for x, t in test ], dtype=np.float32).reshape(len(test), 1, 66, 1024)
    ytest = np.array([ t for x, t in test ], dtype=np.float32)
    # setup model
    model = CharCNN(input_channel=None, output_channel=5, filter_height=10, filter_width=10, n_units=5, n_out=2)
    optimizer = optimizers.RMSpropGraves()
    optimizer.setup(model)
    # train
    n = len(xtrain)
    bs = 10
    losses = []
    for j in range(100):
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
            x = Variable(xtest[sffindx[i:(i+bs_test) if (i+bs_test) < n_test else n_test]])
            y = ytest[sffindx[i:(i+bs_test) if (i+bs_test) < n_test else n_test]]
            result = model.fwd(x, train=False)
            for r, ans in zip(result, y):
                if r.data.argmax() == ans.argmax():
                    n_correct += 1
                else:
                    n_incorrect += 1
        # print('correct: {} \nincorrect: {}'.format(n_correct, n_incorrect))
        print('correct: {}'.format(n_correct/(n_incorrect+n_correct)))


if __name__ == '__main__':
    train()
