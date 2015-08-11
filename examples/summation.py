import apollo.jobs.layers import Convolution, EuclideanLoss
import apollo.jobs.generators import NumpyDataJob
import apollo
import numpy as np

def get_data():
    example = np.array(np.random.random()).reshape((1, 1, 1, 1))
    return {'data': example, 'label': example * 3}

net = apollo.Net()
net.add(NumpyDataJob('label'))
net.add(NumpyDataJob('data'))
net.add(Convolution('conv', (1, 1), 1, bottoms=['data']))
net.add(EuclideanLoss('loss', bottoms=['conv', 'label']))

trainer = apollo.solvers.SGD(net, 0.1,
    max_iter=1000, loggers=[apollo.loggers.DisplayLogger(100)])

trainer.fit([get_data() for _ in xrange(1000)])




























#import logging
#import numpy as np
#import random
#import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
#import argparse

#import apollo
#from apollo import layers
#from apollo.layers import InnerProduct, NumpyData

#def seq_name(name, idx):
    #return '%s/o%d' % (name, idx)

#class NumpyDataSeq(object):
    #def __init__(self, name, data):
        #self.name = name
        #self.data = data
    #def forward(self, net):
        #for idx in range(len(self.data)):
            #net.forward_layer(layers.NumpyData(name=seq_name(self.name, idx),
                #data=self.data[idx]))
#class LstmSeq(object):
    #def __init__(self, name, bottoms, mem_cells, init_range):
        #self.name = name
        #self.bottoms = bottoms
        #self.mem_cells = mem_cells
        #self.filler = layers.Filler(type='uniform', min=-init_range,
            #max=init_range)
    #def forward(self, net):
        #idx = 0
        #init_bottom_name = seq_name(self.bottoms[0], idx)
        #batch_size = net.tops[init_bottom_name].shape[0]
        #right_pad = max(len(net.tops[init_bottom_name].shape) - 2, 0)
        #net.forward_layer(layers.NumpyData(name='%s/lstm_seed' % self.name,
            #data=np.zeros([batch_size, self.mem_cells] + [1] * right_pad)))
        ##print [batch_size, self.mem_cells] + [1] * right_pad
        ##print 'shape', net.tops['%s/lstm_seed' % self.name].shape
        #while seq_name(self.bottoms[0], idx) in net.active_layer_names():
            #bottom_name = seq_name(self.bottoms[0], idx)
            #if idx == 0:
                #prev_hidden = '%s/lstm_seed' % self.name
                #prev_mem = '%s/lstm_seed' % self.name
            #else:
                #prev_hidden = '%s/o%d' % (self.name, idx - 1)
                #prev_mem = '%s/mem%d' % (self.name, idx - 1)
            ##print net.tops[prev_hidden].shape
            ##print net.tops[bottom_name].shape
            #net.forward_layer(layers.Concat(name='%s/concat%d' % (self.name, idx),
                #bottoms=[prev_hidden, bottom_name]))
            ## Run the LSTM for one more step
            #net.forward_layer(layers.Lstm(name='lstm%d' % idx,
                #bottoms=['%s/concat%d' % (self.name, idx), prev_mem],
                #param_names=['%s/input_value', '%s/input_gate',
                    #'%s/forget_gate', '%s/output_gate'],
                #tops=['%s/o%d' % (self.name, idx), '%s/mem%d' % (self.name, idx)],
                #num_cells=self.mem_cells, weight_filler=self.filler))
            #idx += 1

#def forward(net, hyper):
    #length = random.randrange(5, 15)

    ## initialize all weights in [-0.1, 0.1]
    #filler = layers.Filler(type='uniform', min=-hyper['init_range'],
        #max=hyper['init_range'])
    ## Begin recurrence through 5 - 15 inputs
    #values = [np.reshape(np.array([random.random() for _ in range(hyper['batch_size'])]),
        #(hyper['batch_size'], 1)) for _ in range(length)]
    #accum = sum(values)
    #NumpyDataSeq(name='value', data=values).forward(net)
    #LstmSeq(name='lstm', bottoms=['value'], init_range=hyper['init_range'],
        #mem_cells=hyper['mem_cells']).forward(net)
    ## Add a fully connected layer with a bottom blob set to be the last used LSTM cell
    ## Note that the network structure is now a function of the data
    #net.forward_layer(InnerProduct(name='ip', bottoms=['lstm/o' + str(length - 1)],
        #num_output=1, weight_filler=filler))
    ## Add a label for the sum of the inputs
    #net.forward_layer(NumpyData(name='label', data=accum))
    ## Compute the Euclidean loss between the preiction and label, used for backprop
    #loss = net.forward_layer(layers.EuclideanLoss(name='euclidean',
        #bottoms=['ip', 'label']))
    #return loss

#def train(hyper):
    #apollo.set_random_seed(hyper['random_seed'])
    #if hyper['gpu'] is None:
        #apollo.set_mode_cpu()
        #logging.info('Using cpu device (pass --gpu X to train on the gpu)')
    #else:
        #apollo.set_mode_gpu()
        #apollo.set_device(hyper['gpu'])
        #logging.info('Using gpu device %d' % hyper['gpu'])
    #apollo.set_logging_verbosity(hyper['loglevel'])

    #net = apollo.Net()
    #forward(net, hyper)
    #network_path = '%s/network.jpg' % hyper['schematic_prefix']
    #net.draw_to_file(network_path)
    #logging.info('Drawing network to %s' % network_path)
    #net.reset_forward()
    #if 'weights' in hyper:
        #logging.info('Loading weights from %s' % hyper['weights'])
        #net.load(hyper['weights'])

    #train_loss_hist = []
    #for i in xrange(hyper['start_iter'], hyper['max_iter']):
        #train_loss_hist.append(forward(net, hyper))
        #net.backward()
        #lr = (hyper['base_lr'] * hyper['gamma']**(i // hyper['stepsize']))
        #net.update(lr=lr, momentum=hyper['momentum'],
            #clip_gradients=hyper['clip_gradients'])
        #if i % hyper['display_interval'] == 0:
            #logging.info('Iteration %d: %s' % (i, np.mean(train_loss_hist[-hyper['display_interval']:])))
        #if i % hyper['snapshot_interval'] == 0 and i > hyper['start_iter']:
            #filename = '%s/%d.h5' % (hyper['snapshot_prefix'], i)
            #logging.info('Saving net to: %s' % filename)
            #net.save(filename)
        #if i % hyper['graph_interval'] == 0 and i > hyper['start_iter']:
            #sub = hyper.get('sub', 100)
            #plt.plot(np.convolve(train_loss_hist, np.ones(sub)/sub)[sub:-sub])
            #filename = '%s/train_loss.jpg' % hyper['graph_prefix']
            #logging.info('Saving figure to: %s' % filename)
            #plt.savefig(filename)

#def evaluate_forward(net):
    #length = 20
    #net.forward_layer(layers.NumpyData(name='prev_hidden',
        #data=np.zeros((1, hyper['mem_cells'], 1, 1))))
    #net.forward_layer(layers.NumpyData(name='prev_mem',
        #data=np.zeros((1, hyper['mem_cells'], 1, 1))))
    #filler = layers.Filler(type='uniform', min=-hyper['init_range'], max=hyper['init_range'])
    #accum = np.array([0.])
    #predictions = []
    #for step in range(length):
        #value = 0.5
        #net.forward_layer(layers.NumpyData(name='value',
            #data=np.array(value).reshape((1, 1, 1, 1))))
        #accum += value
        #prev_hidden = 'prev_hidden'
        #prev_mem = 'prev_mem'
        #net.forward_layer(layers.Concat(name='lstm_concat', bottoms=[prev_hidden, 'value']))
        #net.forward_layer(layers.Lstm(name='lstm', bottoms=['lstm_concat', prev_mem],
            #param_names=['input_value', 'input_gate', 'forget_gate', 'output_gate'],
            #weight_filler=filler,
            #tops=['next_hidden', 'next_mem'], num_cells=hyper['mem_cells']))
        #net.forward_layer(layers.InnerProduct(name='ip', bottoms=['next_hidden'],
            #num_output=1))
        #predictions.append(float(net.tops['ip'].data.flatten()[0]))
        ## set up for next prediction by copying LSTM outputs back to inputs
        #net.tops['prev_hidden'].data_tensor.copy_from(net.tops['next_hidden'].data_tensor)
        #net.tops['prev_mem'].data_tensor.copy_from(net.tops['next_mem'].data_tensor)
        #net.reset_forward()
    #return predictions

#def eval(hyper):
    #eval_net = apollo.Net()
    ## evaluate the net once to set up structure before loading parameters
    #evaluate_forward(eval_net)
    #eval_net.load('%s/%d.h5' % (hyper['snapshot_prefix'], hyper['max_iter'] - 1))
    #print evaluate_forward(eval_net)

#def main():
    #hyper = {}
    #hyper['gpu'] = None
    #hyper['batch_size'] = 32
    #hyper['init_range'] = 0.1
    #hyper['base_lr'] = 0.03
    #hyper['momentum'] = 0.9
    #hyper['clip_gradients'] = 0.1
    #hyper['display_interval'] = 100
    #hyper['max_iter'] = 5001
    #hyper['snapshot_prefix'] = '/tmp'
    #hyper['schematic_prefix'] = '/tmp'
    #hyper['snapshot_interval'] = 1000
    #hyper['random_seed'] = 21
    #hyper['gamma'] = 0.5
    #hyper['stepsize'] = 1000
    #hyper['solver_mode'] = 'gpu'
    #hyper['mem_cells'] = 1000
    #hyper['graph_interval'] = 1000
    #hyper['graph_prefix'] = '/tmp'

    #parser = argparse.ArgumentParser()
    #parser.add_argument('--gpu', type=int)
    #parser.add_argument('--loglevel', default=3, type=int)
    #parser.add_argument('--start_iter', default=0, type=int)
    #parser.add_argument('--weights', default=None, type=str)
    #args = parser.parse_args()
    #hyper.update({k:v for k, v in vars(args).iteritems() if v is not None})
    #train(hyper)
    #eval(hyper)

#if __name__ == '__main__':
    #main()
