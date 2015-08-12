from apollo.jobs.layers import Convolution, EuclideanLoss
from apollo.jobs.generators import InputData
import apollo
import numpy as np

apollo.set_gpu_device(2)

def get_data():
    example = np.random.random((1, 1, 1))
    return {'data': example, 'label': example * 3}

def get_number():
    number = raw_input("Number: ")
    return {'data': [np.array(number).reshape((1,1,1))]}

apollo.set_mode_gpu()
net = apollo.Net()
net.add(InputData('label'), states=['train', 'val'])
net.add(InputData('data'))
net.add(Convolution('conv', (1, 1), 1, bottoms=['data']))
net.add(EuclideanLoss('loss', bottoms=['conv', 'label']))

trainer = apollo.solvers.SGD(net, 0.1,
        max_iter=3100, loggers=[apollo.loggers.TrainLogger(100), 
        apollo.loggers.ValLogger(1000)])

trainer.fit([get_data() for _ in xrange(1000)])

net.forward('deploy', get_number())
print net.tops['conv'].data.flatten()[0]


