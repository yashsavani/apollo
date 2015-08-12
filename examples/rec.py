from apollo.jobs.layers import Convolution, EuclideanLoss, InnerProduct, Dropout
from apollo.jobs.generators import InputDataSeq, InputData, Recurrent, LstmUnit
import apollo
import numpy as np
import random

def get_data():
    length = random.randrange(5, 15)
    example = np.random.random((length, 1))
    return {'data': example, 'label': np.cumsum(example)}
    #return {'data': example, 'label': example * 3}

apollo.set_mode_gpu()

num_steps = [0]
net = apollo.Net()
net.add(InputDataSeq('data', num_steps))
net.add(InputDataSeq('label'))

r = Recurrent(num_steps)
r.add_job(LstmUnit, name='lstm0', mem_cells=1000, bottoms=['data'])
r.add_job(Dropout, name='drop0', dropout_ratio=0.5, bottoms=['lstm0'])
r.add_job(LstmUnit, name='lstm', mem_cells=1000, bottoms=['drop0'])
r.add_job(InnerProduct, name='ip', num_output=1, bottoms=['lstm'])
r.add_job(EuclideanLoss, name='loss', bottoms=['ip', 'label'])
net.add(r)

trainer = apollo.solvers.SGD(net, 0.1, gamma_lr=0.9, gamma_stepsize=1000,
    loggers=[apollo.loggers.DisplayLogger(100)])

trainer.fit([get_data() for _ in xrange(5000)])
