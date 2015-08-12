from apollo.jobs.layers import Convolution, EuclideanLoss
from apollo.jobs.generators import InputData
import apollo
import numpy as np

apollo.set_gpu_device(-1)

def get_data():
    example = np.random.random((3, 5, 5))
    return {'data': example, 'label': example * 3}

apollo.set_mode_gpu()
net = apollo.Net()
net.add(InputData('label'))
net.add(InputData('data'))
net.add(Convolution('conv', (1, 1), 3, bottoms=['data']))
net.add(EuclideanLoss('loss', bottoms=['conv', 'label']))

trainer = apollo.solvers.SGD(net, 0.01,
    max_iter=100000, loggers=[apollo.loggers.TrainLogger(100), apollo.loggers.ValLogger(1000)])

trainer.fit([get_data() for _ in xrange(1000)]
