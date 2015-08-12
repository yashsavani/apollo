import layers as L
import numpy as np
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
        apollo_net.forward_layer(L.NumpyData(self.name, data=data))

class InputDataSeq(Job):
    def __init__(self, name, num_steps=[0], sequence_lengths=[], fn=None, pad_value=0.):
        self.name = name
        self.num_steps = num_steps
        if fn is None:
            fn = lambda x: x
        self.fn = fn
        self.pad_value = pad_value
        self.sequence_lengths = sequence_lengths
    def is_data_job(self):
        return True
    def run(self, apollo_net, input_data):
        data_batch = self.fn(input_data[self.name])
        del self.sequence_lengths[:]
        for x in data_batch:
            self.sequence_lengths.append(len(x))
        self.num_steps[0] = max(self.sequence_lengths)
        if isinstance(data_batch, np.ndarray):
            self.num_steps[0] = data_batch.shape[1]
            padded_data = data_batch
        else:
            data = [np.array(x) for x in data_batch]
            padded_data = np.zeros([len(data), self.num_steps[0]] + list(data[0].shape)[1:])
            for idx in range(len(data)):
                length = self.sequence_lengths[idx]
                padded_data[idx, :length] = data[idx]
                padded_data[idx, length:] = self.pad_value
        for step in range(self.num_steps[0]):
            apollo_net.forward_layer(L.NumpyData('%s:%d' % (self.name, step), padded_data[:, step]))

class LstmUnit(Job):
    def __init__(self, name, step, bottoms, mem_cells, init_range=0.1, tops=None, seed=None):
        self.raw_name = name
        self.name = '%s:%d' % (name, step)
        self.step = step
        self.bottoms = ['%s:%d' % (b, step) for b in bottoms]
        self.mem_cells = mem_cells
        self.init_range = init_range
        self.seed = seed
        if tops is None:
            self.tops = [self.name]

    def run(self, apollo_net):
        if self.step == 0:
            batch_size = apollo_net.tops[self.bottoms[0]].shape[0]
            apollo_net.forward_layer(L.NumpyData('%s:lstm_seed' % self.raw_name,
                np.zeros([batch_size, self.mem_cells])))
            prev_mem = '%s:lstm_seed' % self.raw_name
            if self.seed is None:
                prev_hidden = '%s:lstm_seed' % self.raw_name
            else:
                prev_hidden = self.seed
        else:
            prev_hidden = '%s:%d' % (self.raw_name, self.step - 1)
            prev_mem = '%s:%d:mem' % (self.raw_name, self.step - 1)

        # Concatenate the input with the previous memory
        apollo_net.forward_layer(L.Concat(name='%s:concat' % (self.name),
            bottoms=([prev_hidden] + self.bottoms)))
        # Run the LSTM
        apollo_net.forward_layer(L.Lstm(name=self.name,
            bottoms=['%s:concat' % self.name, prev_mem],
            param_names=[x % self.name for x in ['%s:input_value', '%s:input_gate',
                '%s:forget_gate', '%s:output_gate']],
            tops=[self.tops[0], '%s:mem' % self.name],
            num_cells=self.mem_cells, weight_filler=L.Filler('uniform', self.init_range)))
        
class Recurrent(Job):
    def __init__(self, num_steps):
        """
        num_steps: the number of steps the recurrent job will run for.
            Stored in the first element of a list for pass by reference semantics.
        """
        self.job_lambdas = []
        self.num_steps = num_steps
    def add_job(self, job, *args, **kwargs):
        def make_job(step):
            kw_mod = {}
            if job in [LstmUnit]:
                # the LstmUnit will handle recurrent name mangling itself
                kw_mod['step'] = step
                for k, v in kwargs.iteritems():
                    kw_mod[k] = v
            else:
                for k, v in kwargs.iteritems():
                    if k == 'name':
                        kw_mod[k] = '%s:%d' % (v, step)
                    elif k == 'tops':
                        kw_mod[k] = []
                        for i in range(len(v)):
                            kw_mod[k].append('%s:%d' % (v[i], step))
                    elif k == 'bottoms':
                        kw_mod[k] = []
                        for i in range(len(v)):
                            kw_mod[k].append('%s:%d' % (v[i], step))
                    else:
                        kw_mod[k] = v
            return job(**kw_mod)
        self.add_job_lambda(make_job)
    def add_job_lambda(self, job_lambda):
        self.job_lambdas.append(job_lambda)
    def run(self, apollo_net):
        for step in range(self.num_steps[0]):
            for f in self.job_lambdas:
                job = f(step)
                job.run(apollo_net)
