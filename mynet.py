import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

from get_data import get_data

# class CharLinear(Chain):
#     def __init__(self, n_units, n_out):
#         super(CharLinear, self).__init__(
#             l1=L.Linear(None, n_units),
#             l2=L.Linear(None, n_units),
#             l3=L.Linear(None, n_out),
#         )
#     def __call__(self, x):
#         h1 = F.relu(self.l1(x))
#         h2 = F.relu(self.l2(h1))
#         y = self.l3(h2)
#         return y
class CharLinear(Chain):
    def __init__(self, n_units, n_out):
        super(CharLinear, self).__init__(
            l1=L.Linear(None, n_units),
            l2=L.Linear(None, n_units),
            l3=L.Linear(None, n_out),
        )

    def __call__(self, x, y):
        return F.mean_squared_error(self.fwd(x), y)
    
    def fwd(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y

if __name__ == '__main__':
    train, test = get_data()
    xtrain = [ x for x, t in train ]
    ytrain = [ t for x, t in train ]
    xtest = [ x for x, t in test ]
    ytest = [ t for x, t in test ]
    print(xtrain)
    import sys 
    sys.exit()
    model = CharLinear(100, 2)
    optimizer = optimizers.SGD()
    optimizer.setup(model)
    n = 10 
    bs = 2
    for j in range(500):
        sffindx = np.random.permutation(n)
        for i in range(0, n, bs):
            # x = Variable(xtrain[sffindx[i:(i+bs) if (i+bs) < n else n]])
            # y = Variable(ytrain[sffindx[i:(i+bs) if (i+bs) < n else n]])
            x = Variable(np.array(xtrain[i:i+bs], dtype=np.float32))
            y = Variable(np.array(ytrain[i:i+bs], dtype=np.float32))
            model.zerograds()
            loss = model(x, y)
            loss.backward()
            optimizer.update()
            if j % 50 == 0:
                print(loss.data)
    # train_iter = iterators.SerialIterator(train, batch_size=2, shuffle=True)
    # test_iter = iterators.SerialIterator(test, batch_size=2, repeat=False, shuffle=False)
    # model = L.Classifier(CharLinear(100, 2))
    # optimizer = optimizers.SGD()
    # optimizer.setup(model)
    # updater = training.StandardUpdater(train_iter, optimizer)
    # trainer = training.Trainer(updater, (50, 'epoch'), out='result')
    # trainer.extend(extensions.Evaluator(test_iter, model))
    # trainer.extend(extensions.LogReport())
    # trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
    # trainer.extend(extensions.ProgressBar())
    # trainer.run()
    
    results = model.fwd(Variable(np.array(xtest, dtype=np.float32)))
    print(results.data)
    for r, y in zip(results, np.array(ytest)):
        print('{} has to be {}'.format(y.argmax(), r.data.argmax()))
