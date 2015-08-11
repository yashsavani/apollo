import apollo
import apollo.layers as L
import numpy as np

net = apollo.Net()
for i in range(1000):
    example = np.array(np.random.random()).reshape((1, 1, 1, 1))
    net.forward_layer(L.NumpyData('data', example))
    net.forward_layer(L.NumpyData('label', example*3))
    net.forward_layer(L.Convolution('conv', (1, 1), 1, bottoms=['data']))
    loss = net.forward_layer(L.EuclideanLoss('loss', bottoms=['conv', 'label']))
    net.backward()
    net.update(lr=0.1)
    if i % 100 == 0:
        print loss
