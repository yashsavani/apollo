import apollo.jobs as J
import apollo
import numpy as np

def get_data():
    example = np.array(np.random.random()).reshape((1, 1, 1, 1))
    return {'data': example, 'label': example * 3}

net = apollo.Net()
net.add(J.NumpyJob('label'))
net.add(J.NumpyJob('data'))
net.add(J.Convolution('conv', (1, 1), 1, bottoms=['data']))
net.add(J.EuclideanLoss('loss', bottoms=['conv', 'label']))

trainer = apollo.solvers.SGD(net, 0.1,
    max_iter=1000, loggers=[apollo.loggers.DisplayLogger(100)])

trainer.fit([get_data() for _ in xrange(1000)])
