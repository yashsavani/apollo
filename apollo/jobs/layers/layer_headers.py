import caffe_pb2
import layer_helpers
from layer_helpers import assign_proto
from apollo.jobs.job import Job

class Layer(Job):
    def __init__(self, sublayer, name, kwargs):
        self.parse(sublayer, name, kwargs)

    def run(self, apollo_net):
        return apollo_net.forward_layer(self)

    def parse(self, sublayer, name, kwargs):
        self.p = caffe_pb2.LayerParameter()
        self.p.type = type(sublayer).__name__
        param_type = type(sublayer).__name__
        if 'bottoms' not in kwargs:
            raise AttributeError('Layer %s must specify bottoms=["list", "of", "bottoms"]' % param_type)
        bottoms = kwargs['bottoms']
        if type(bottoms) == str:
            raise AttributeError('Layer %s type bottoms="%s" argument must be a list of strings, not a string' % (param_type, bottoms))
        tops = kwargs.get('tops', [name])
        if type(tops) == str:
            raise AttributeError('Layer %s type tops="%s" argument must be a list of strings, not a string' % (param_type, tops))
        self.p.name = name
        for blob_name in tops:
            self.p.top.append(blob_name)
        for blob_name in bottoms:
            self.p.bottom.append(blob_name)

        param_names = kwargs.get('param_names', [])
        param_lr_mults = kwargs.get('param_lr_mults', [])
        param_decay_mults = kwargs.get('param_decay_mults', [])
        assert type(param_names) != str
        for i in range(max(len(param_names), len(param_lr_mults), len(param_decay_mults))):
            self.p.param.add()
            if param_names:
                self.p.param[-1].name = param_names[i]
            if param_lr_mults:
                self.p.param[-1].lr_mult = param_lr_mults[i]
            if param_decay_mults:
                self.p.param[-1].decay_mult = param_decay_mults[i]
        if 'train' in kwargs:
            assign_proto(self.p, 'train', kwargs['train'])
        if 'deploy' in kwargs:
            assign_proto(self.p, 'deploy', kwargs['deploy'])
        default_params = set(['name', 'bottoms', 'tops',
                             'deploy', 'train',
                             'param_names, param_lr_mults',
                             'param_decay_mults'])

        if param_type in layer_helpers.param_names:
            proto_param = getattr(self.p, layer_helpers.param_names[param_type] + '_param')
            for k, v in kwargs.iteritems():
                if k in default_params:
                    continue
                try:
                    assign_proto(proto_param, k, v)
                except AttributeError:
                    raise AttributeError('Layer %s has no keyword argument %s=%s' % (param_type, k, v))
        else:
            for k, v in kwargs.iteritems():
                if k not in default_params:
                    raise AttributeError('Layer %s has no keyword argument %s=%s' % (param_type, k, v))

class PyLayer(Layer):
    def __init__(self, **kwargs):
        super(PyLayer ,self).__init__(kwargs)
        self.kwargs = kwargs
        self.p.type = 'Py'
        if 'param_shapes' in kwargs:
            for shape in kwarg['param_shapes']:
                param_shape = self.p.py_param.param_shapes.add()
                for dimension in shape:
                    param_shape.dimension.append(dimension)
        if 'param_fillers' in kwargs:
            assert len(kwargs['param_shapes']) == len(kwargs['param_filler'])
            for filler in kwarg['param_fillers']:
                filler_param = self.p.py_param.param_fillers.add()
                filler_param.CopyFrom(filler.filler_param)
    def setup(self, bottom_vec, top_vec):
        pass
    def forward(self, bottom_vec, top_vec):
        pass
    def backward(self, bottom_vec, top_vec):
        pass

class LossLayer(Layer):
    def __init__(self, sublayer, name, kwargs):
        kwargs['deploy'] = kwargs.get('deploy', False)
        super(LossLayer, self).__init__(sublayer, name, kwargs)
        loss_weight = kwargs.get('loss_weight', 1.)
        self.p.loss_weight.append(loss_weight)
        assert 'tops' not in kwargs or kwargs['tops'] == 1

class DataLayer(Layer):
    def __init__(self, sublayer, name, kwargs):
        kwargs['bottoms'] = kwargs.get('bottoms', [])
        super(DataLayer, self).__init__(sublayer, name, kwargs)
