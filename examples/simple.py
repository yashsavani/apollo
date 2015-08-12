from apollo.jobs.layers import Convolution, EuclideanLoss, ReLU
from apollo.jobs.generators import InputData
import apollo
import numpy as np

apollo.set_gpu_device(2)

def get_data():
    example = np.random.random((1, 1, 1))
    return {'data': example, 'label': example ** 2}

def get_number():
    number = raw_input("Number: ")
    return {'data': np.array(number).reshape((1,1,1))}

apollo.set_mode_gpu()
net = apollo.Net()
net.add(InputData('label'), states=['train', 'val'])
net.add(InputData('data'))
net.add(Convolution('conv', (1, 1), 10, bottoms=['data']))
net.add(ReLU('relu', bottoms=['conv'], tops=['conv']))
net.add(Convolution('conv2', (1, 1), 1, bottoms=['conv']))
net.add(EuclideanLoss('loss', bottoms=['conv2', 'label']))

net.solver = apollo.solvers.SGD(net, 0.1,
        max_iter=3100, loggers=[apollo.loggers.TrainLogger(100), 
        apollo.loggers.ValLogger(1000)], batch_size=5)


net.fit([get_data() for _ in xrange(1000)])

print net.predict([get_number()], tops=["conv2"])['conv2'].flatten()[0]


