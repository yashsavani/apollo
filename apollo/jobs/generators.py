import layers as L
from .job import Job

class InputData(Job):
    def __init__(self, name, fn=None):
        if fn is None:
            fn = lambda x: x
        self.name = name
        self.fn = fn
    def is_data_job(self):
        return True
    def run(self, apollo_net, input_data):
        data = self.fn(input_data[self.name])
        apollo_net.forward_layer(L.NumpyData(name=self.name, data=data))

class InputDataSeq(object):
    def __init__(self, name, length_ref, fn=None):
        self.name = name
        self.length_ref = length_ref
        if fn is None:
            fn = lambda x: x
        self.fn = fn
    def is_data_job(self):
        return True
    def run(self, apollo_net, input_data):
        data = self.fn(input_data[self.name])
        self.length_ref = [len(data[0])]
        for idx in range(self.length_ref[0]):
            apollo_net.forward_layer(L.NumpyData(name=self.name, data=data[:, idx]))

def LstmUnit(Job):
    def __init__(self, name, idx, bottoms, tops, mem_cells, init_range, seed=None):
        self.name = name
        self.idx = idx
        self.bottoms = bottoms
        self.tops = tops
        self.mem_cells = mem_cells
        self.init_range = init_range
        self.seed = seed

    def run(self, apollo_net):
        if self.idx == 0:
            batch_size = apollo_net.tops[bottom_name].shape[0]
            net.forward_layer(L.NumpyData(name='%s:lstm_seed' % self.name,
                data=np.zeros([batch_size, self.mem_cells])))
            prev_mem = '%s:lstm_seed' % self.name
            if self.seed is None:
                prev_hidden = '%s:lstm_seed' % self.name
            else:
                prev_hidden = self.seed
        else:
            prev_hidden = '%s:%d' % (self.name, self.idx - 1)
            prev_mem = '%s:mem%d' % (self.name, self.idx - 1)

        # Concatenate the input with the previous memory
        apollo_net.forward_layer(L.Concat(name='%s:concat%d' % (self.name, self.idx),
            bottoms=[prev_hidden, bottoms[0]]))
        # Run the LSTM
        apollo_net.forward_layer(L.Lstm(name='lstm%d' % self.idx,
            bottoms=['%s:concat%d' % (self.name, self.idx), prev_mem],
            param_names=['%s:input_value', '%s:input_gate',
                '%s:forget_gate', '%s:output_gate'],
            tops=[self.tops[0], '%s:mem%d' % (self.name, self.idx)],
            num_cells=self.mem_cells, weight_filler=Filler('uniform', init_range)))
        return 0.
        
class Recurrent(Job):
    def __init__(self, length_ref):
        """
        length_ref: the number of steps the recurrent job will run for.
            Stored in the first element of a list for pass by reference semantics.
        """
        self.job_lambdas = []
        self.length_ref = length_ref
    def add_job(self, job, **kwargs):
        def make_job(idx):
            kw = kwargs.copy()
            if 'name' in kw:
                kw['name'] = '%s:%d' % (kw['name'], idx)
            if 'tops' in kw:
                for i in range(len(kw['tops'])):
                    kw['tops'][i] = '%s:%d' % (kw['tops'][i], idx)
            if 'bottoms' in kw:
                for i in range(len(kw['bottoms'])):
                    kw['bottoms'][i] = '%s:%d' % (kw['bottoms'][i], idx)
            return job(**kw)
        self.add_job_lambda(self, make_job)
    def add_job_lambda(self, job_lambda):
        self.job_lambdas.append(job_lambda)
    def run(self, apollo_net):
        for idx in range(length_ref[0]):
            for f in self.job_lambdas:
                f(idx).run(apollo_net)
