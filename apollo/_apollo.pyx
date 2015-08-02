cimport numpy as cnp
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.set cimport set
from libcpp.map cimport map
from cython.operator cimport postincrement, dereference
from definitions cimport Tensor as CTensor, Blob as CBlob, Layer as CLayer, shared_ptr, LayerParameter, ApolloNet, TRAIN, TEST

import numpy as pynp
import utils.draw
import h5py
import os
import caffe_pb2

cdef extern from "caffe/caffe.hpp" namespace "caffe::Caffe":
    void set_random_seed(unsigned int)
    enum Brew:
        CPU = 0
        GPU = 1
    void set_mode(Brew)
    void SetDevice(int) except+
    void set_logging_verbosity(int level)

cdef class Caffe:
    def __cinit__(self):
        pass
    @staticmethod
    def set_random_seed(seed):
        set_random_seed(seed)
    @staticmethod
    def set_device(device_id):
        SetDevice(device_id)
    @staticmethod
    def set_mode_cpu():
        set_mode(CPU)
    @staticmethod
    def set_mode_gpu():
        set_mode(GPU)
    @staticmethod
    def set_logging_verbosity(level):
        set_logging_verbosity(level)

cdef class Tensor:
    """Tensor class accessible from python"""
    cdef shared_ptr[CTensor] thisptr
    def __cinit__(self):
        self.thisptr.reset(new CTensor())
    cdef void Init(self, shared_ptr[CTensor] other):
        self.thisptr = other
    cdef void AddFrom(Tensor self, Tensor other) except +:
        self.thisptr.get().AddFrom(other.thisptr.get()[0])
    cdef void MulFrom(Tensor self, Tensor other) except +:
        self.thisptr.get().MulFrom(other.thisptr.get()[0])
    cdef void AddMulFrom(Tensor self, Tensor other, float alpha) except +:
        self.thisptr.get().AddMulFrom(other.thisptr.get()[0], alpha)
    cdef void CopyFrom(Tensor self, Tensor other) except +:
        self.thisptr.get().CopyFrom(other.thisptr.get()[0])
    cdef void CopyChunkFrom(Tensor self, Tensor other, int count, int this_offset, int other_offset) except +:
        self.thisptr.get().CopyChunkFrom(other.thisptr.get()[0], count, this_offset, other_offset)
    cdef float DotPFrom(Tensor self, Tensor other) except +:
        return self.thisptr.get().DotPFrom(other.thisptr.get()[0])
    def reshape(self, pytuple):
        cdef vector[int] shape
        for x in pytuple:
            shape.push_back(x)
        self.thisptr.get().Reshape(shape)
    property shape:
        def __get__(self):
            return self.thisptr.get().shape()
    def count(self):
        return self.thisptr.get().count()
    def dot(self, other):
        return self.DotPFrom(other)
    def cosine(self, other):
        num = self.dot(other) 
        denom = (self.dot(self) * other.dot(other)) ** 0.5
        if denom > 0.:
            return 1 - num / denom
        else:
            return 2.
    def copy_from(self, other):
        self.CopyFrom(other)
    def copy_chunk_from(self, other, count, this_offset, other_offset):
        self.CopyChunkFrom(other, count, this_offset, other_offset)
    def set_values(self, other):
        self.thisptr.get().SetValues(other)
    def axpy(self, other, alpha):
        self.AddMulFrom(other, alpha)
    def __iadd__(self, other):
        self.AddFrom(other)
        return self
    def __isub__(self, other):
        self.AddMulFrom(other, -1.)
    def __imul__(self, other):
        if type(other) == type(self):
            self.MulFrom(other)
        else:
            self.thisptr.get().scale(other)
        return self
    def get_mem(self):
        result = tonumpyarray(self.thisptr.get().mutable_cpu_mem(),
                    self.thisptr.get().count())
        sh = self.shape
        result.shape = sh if len(sh) > 0 else (1,)
        return pynp.copy(result)
    def set_mem(self, other):
        result = tonumpyarray(self.thisptr.get().mutable_cpu_mem(),
                    self.thisptr.get().count())
        sh = self.shape
        result.shape = sh if len(sh) > 0 else (1,)
        result[:] = other

cdef class Blob(object):
    cdef shared_ptr[CBlob] thisptr
    def __cinit__(self):
        pass
    cdef void Init(self, shared_ptr[CBlob] other):
        self.thisptr = other
    def count(self):
        return self.thisptr.get().count()
    def reshape(self, pytuple):
        cdef vector[int] shape
        for x in pytuple:
            shape.push_back(x)
        self.thisptr.get().Reshape(shape)
    property shape:
        def __get__(self):
            return self.thisptr.get().shape()
    property data:
        def __get__(self):
            result = tonumpyarray(self.thisptr.get().mutable_cpu_data(),
                        self.thisptr.get().count())
            sh = self.shape
            result.shape = sh if len(sh) > 0 else (1,)
            return result
    property diff:
        def __get__(self):
            result = tonumpyarray(self.thisptr.get().mutable_cpu_diff(),
                        self.thisptr.get().count())
            sh = self.shape
            result.shape = sh if len(sh) > 0 else (1,)
            return result
    property diff_tensor:
        def __get__(self):
            cdef shared_ptr[CTensor] ctensor = self.thisptr.get().diff()
            diff = Tensor()
            diff.Init(ctensor)
            return diff
        def __set__(self, other):
            self.diff_tensor.copy_from(other)

    property data_tensor:
        def __get__(self):
            cdef shared_ptr[CTensor] ctensor = self.thisptr.get().data()
            data = Tensor()
            data.Init(ctensor)
            return data
        def __set__(self, other):
            self.data_tensor.copy_from(other)

cdef class Layer(object):
    cdef shared_ptr[CLayer] thisptr
    def __cinit__(self):
        pass
    cdef void Init(self, shared_ptr[CLayer] other):
        self.thisptr = other
    def layer_param(self):
        param = caffe_pb2.LayerParameter()
        cdef string s
        self.thisptr.get().layer_param().SerializeToString(&s)
        param.ParseFromString(s)
        return param
    property buffers:
        def __get__(self):
            buffers = []
            cdef vector[shared_ptr[CBlob]] cbuffers
            (&cbuffers)[0] = self.thisptr.get().buffers()
            for i in range(cbuffers.size()):
                new_blob = Blob()
                new_blob.Init(cbuffers[i])
                buffers.append(new_blob)
            return buffers
    property blobs:
        def __get__(self):
            blobs = []
            cdef vector[shared_ptr[CBlob]] cblobs
            (&cblobs)[0] = self.thisptr.get().blobs()
            for i in range(cblobs.size()):
                new_blob = Blob()
                new_blob.Init(cblobs[i])
                blobs.append(new_blob)
            return blobs
        
cdef class Net:
    cdef ApolloNet* thisptr
    python_layers = {}
    def __cinit__(self):
        self.thisptr = new ApolloNet()
    def __dealloc__(self):
        del self.thisptr
    property phase:
        def __get__(self):
            if self.thisptr.phase() == TRAIN:
                return 'train'
            elif self.thisptr.phase() == TEST:
                return 'test'
            else:
                raise ValueError("phase must be one of ['train', 'test']")
        def __set__(self, value):
            if value == 'train':
                self.thisptr.set_phase_train()
            elif value == 'test':
                self.thisptr.set_phase_test()
            else:
                raise ValueError("phase must be one of ['train', 'test']")
    def forward_layer(self, layer):
        if layer.p.type == 'Py':
            new_layer = (layer.p.name not in self.layers)
            self.thisptr.ForwardLayer(layer.p.SerializeToString(), layer.r.SerializeToString())
            tops = self.tops
            bottom_vec = [tops[name] for name in layer.p.bottom]
            top_vec = [tops[name] for name in layer.p.top]
            if new_layer:
                layer.blobs = self.layers[layer.p.name].blobs
                layer.net = self
                self.python_layers[layer.p.name] = layer
            cached_layer = self.python_layers[layer.p.name]
            cached_layer.kwargs = layer.kwargs
            if new_layer:
                cached_layer.setup(bottom_vec, top_vec)
            else:
                cached_layer.p.ClearField('bottom')
                for bottom_name in layer.p.bottom:
                    cached_layer.p.bottom.append(bottom_name)
                cached_layer.r.CopyFrom(layer.r)
            loss = cached_layer.forward(bottom_vec, top_vec)
        else:
            loss = self.thisptr.ForwardLayer(layer.p.SerializeToString(), layer.r.SerializeToString())
        return loss
    def backward_layer(self, layer_name):
        if layer_name in self.python_layers:
            cached_layer = self.python_layers[layer_name]
            self.thisptr.BackwardLayer(layer_name)
            tops = self.tops
            bottom_vec = [tops[name] for name in cached_layer.p.bottom]
            top_vec = [tops[name] for name in cached_layer.p.top]
            cached_layer.backward(top_vec, bottom_vec)
            self.thisptr.BackwardLayer(layer_name)
        else:
            self.thisptr.BackwardLayer(layer_name)
    def backward(self):
        for layer_name in self.active_layer_names()[::-1]:
            self.backward_layer(layer_name)
    def update(self, lr, momentum=0., clip_gradients=-1, weight_decay=0.):
        diffnorm = self.diff_l2_norm() 
        clip_scale = 1.
        if clip_gradients > 0:
            if diffnorm > clip_gradients:
                clip_scale = clip_gradients / diffnorm
        params = self.params
        for param_name in self.active_param_names():
            self.update_param(params[param_name],
                              lr * clip_scale * self.param_lr_mults(param_name),
                              momentum,
                              weight_decay * self.param_decay_mults(param_name))
        self.reset_forward()
    def update_param(self, param, lr, momentum, weight_decay):
        param.diff_tensor.axpy(param.data_tensor, weight_decay)
        param.data_tensor.axpy(param.diff_tensor, -lr)
        param.diff_tensor *= momentum
    def diff_l2_norm(self):
        return self.thisptr.DiffL2Norm()
    def reset_forward(self):
        """Clears vector of layers to backpropped through after each forward pass"""
        self.thisptr.ResetForward()
    def active_layer_names(self):
        cdef vector[string] layer_names
        layer_names = self.thisptr.active_layer_names()
        return layer_names
    def active_param_names(self):
        cdef set[string] param_set
        (&param_set)[0] = self.thisptr.active_param_names()
        cdef set[string].iterator it = param_set.begin()
        cdef set[string].iterator end = param_set.end()
        param_names = []
        while it != end:
            param_names.append(dereference(it))
            postincrement(it)
        return param_names
    def param_lr_mults(self, name):
        cdef map[string, float] lr_mults
        (&lr_mults)[0] = self.thisptr.param_lr_mults()
        return lr_mults[name]

    def param_decay_mults(self, name):
        cdef map[string, float] decay_mults
        (&decay_mults)[0] = self.thisptr.param_decay_mults()
        return decay_mults[name]
    def net_param(self):
        # will be empty if called before the forward pass or after reset_forward
        param = caffe_pb2.NetParameter()
        param.name = 'name'
        layers = self.layers
        for layer_name in self.active_layer_names():
            layer = param.layer.add()
            layer.CopyFrom(layers[layer_name].layer_param())
        return param
    def draw_to_file(self, filename, rankdir='LR', require_nonempty=True):
        net_param = self.net_param()
        if len(net_param.layer) == 0 and require_nonempty:
            raise ValueError('Cowardly refusing to draw net with no active layers. HINT: call this function before reset_forward()')
        utils.draw.draw_net_to_file(self.net_param(), filename, rankdir)
    property layers:
        def __get__(self):
            cdef map[string, shared_ptr[CLayer]] layers_map
            (&layers_map)[0] = self.thisptr.layers()

            layers = {}
            cdef map[string, shared_ptr[CLayer]].iterator it = layers_map.begin()
            cdef map[string, shared_ptr[CLayer]].iterator end = layers_map.end()
            cdef string layer_name
            cdef shared_ptr[CLayer] layer
            while it != end:
                layer_name = dereference(it).first
                layer = dereference(it).second
                py_layer = Layer()
                py_layer.Init(layer)
                layers[layer_name] = py_layer
                postincrement(it)

            return layers

    property params:
        def __get__(self):
            cdef map[string, shared_ptr[CBlob]] param_map
            (&param_map)[0] = self.thisptr.params()

            blobs = {}
            cdef map[string, shared_ptr[CBlob]].iterator it = param_map.begin()
            cdef map[string, shared_ptr[CBlob]].iterator end = param_map.end()
            cdef string blob_name
            cdef shared_ptr[CBlob] blob_ptr
            while it != end:
                blob_name = dereference(it).first
                blob_ptr = dereference(it).second
                new_blob = Blob()
                new_blob.Init(blob_ptr)
                blobs[blob_name] = new_blob
                postincrement(it)

            return blobs

    property tops:
        def __get__(self):
            cdef map[string, shared_ptr[CBlob]] top_map
            (&top_map)[0] = self.thisptr.tops()

            tops = {}
            cdef map[string, shared_ptr[CBlob]].iterator it = top_map.begin()
            cdef map[string, shared_ptr[CBlob]].iterator end = top_map.end()
            cdef string top_name
            cdef shared_ptr[CBlob] top_ptr
            while it != end:
                top_name = dereference(it).first
                top_ptr = dereference(it).second
                new_blob = Blob()
                new_blob.Init(top_ptr)
                tops[top_name] = new_blob
                postincrement(it)

            return tops

    def save(self, filename):
        assert filename.endswith('.h5'), "saving only supports h5 files"
        with h5py.File(filename, 'w') as f:
            for name, value in self.params.items():
                f[name] = pynp.copy(value.data)

    def load(self, filename):
        if len(self.params) == 0:
            raise ValueError('WARNING, loading into empty net.')
        _, extension = os.path.splitext(filename)
        if extension == '.h5':
            with h5py.File(filename, 'r') as f:
                params = self.params
                names = []
                f.visit(names.append)
                for name in names :
                    if name in params:
                        params[name].data[:] = f[name]
        elif extension == '.caffemodel':
            self.thisptr.CopyTrainedLayersFrom(filename)
        else:
            assert False, "Error, filename is neither h5 nor caffemodel: %s, %s" % (filename, extension)
    def copy_params_from(self, other):
        self_params = self.params
        if len(self_params) == 0:
            raise ValueError('WARNING, loading into empty net.')
        for name, value in other.params.items():
            if name in self_params:
                self_params[name].data[:] = pynp.copy(value.data)


cdef extern from "caffe/proto/caffe.pb.h" namespace "caffe":
    cdef cppclass NumpyDataParameter:
        void add_data(float data)
        void add_shape(unsigned int shape)
        string DebugString()
    cdef cppclass RuntimeParameter:
        NumpyDataParameter* mutable_numpy_data_param()
        NumpyDataParameter numpy_data_param()
        bool SerializeToString(string*)
        string DebugString()

class PyRuntimeParameter(object):
    def __init__(self, result):
        self.result = result
    def SerializeToString(self):
        return self.result

def make_numpy_data_param(numpy_array):
    """Serialize numpy array to prototxt"""
    assert numpy_array.dtype == pynp.float32
    cdef vector[int] v
    for x in numpy_array.shape:
        v.push_back(x)
    cdef string s = make_numpy_data_param_fast(pynp.ascontiguousarray(numpy_array.flatten()), v)
    return PyRuntimeParameter(str(s)) #s.encode('utf-8'))

cdef string make_numpy_data_param_fast(cnp.ndarray[cnp.float32_t, ndim=1] numpy_array, vector[int] v):
    cdef RuntimeParameter runtime_param
    cdef NumpyDataParameter* numpy_param
    numpy_param = runtime_param.mutable_numpy_data_param()
    cdef int length = len(numpy_array)
    cdef int i
    for i in range(v.size()):
        numpy_param[0].add_shape(v[i])
    for i in range(length):
        numpy_param[0].add_data(numpy_array[i])
    cdef string s
    runtime_param.SerializeToString(&s)
    return s

cnp.import_array()
cdef public api tonumpyarray(float* data, long long size) with gil:
    #if not (data and size >= 0): raise ValueError
    cdef cnp.npy_intp dims = size
    #NOTE: it doesn't take ownership of `data`. You must free `data` yourself
    return cnp.PyArray_SimpleNewFromData(1, &dims, cnp.NPY_FLOAT, <void*>data)
