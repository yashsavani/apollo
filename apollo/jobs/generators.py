from .layers import NumpyData
from .job import Job

class NumpyJob(Job):
    def __init__(self, name, fn=None):
        if fn is None:
            fn = lambda x: x
        self.fn = fn
        self.name = name
    def forward(self, apollo_net, input_data):
        #print input_data
        return apollo_net.forward_layer(NumpyData(name=self.name, data=self.fn(input_data[self.name])))
        #apollo_net.forward_layer(NumpyData(name=self.name, data=self.fn(input_data[self.name])))
        #import numpy as np
        #apollo_net.tops[self.name].data[:] = np.zeros((1,1,1,1))
        #print apollo_net.tops[self.name].data.shape
        #return

        #data = self.fn(input_data[self.name])
        #print data
        #import numpy
        #print self.name
        #out = apollo_net.forward_layer(NumpyData(name=self.name, data=(numpy.zeros((1,1,1,1)))))
        #print apollo_net.tops[self.name].shape
        #print apollo_net.tops.keys()
        #return 0.
