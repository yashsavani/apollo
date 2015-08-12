from apollo import ApolloNet
import numpy as np

class Net:
    def __init__(self):
        self.jobs = []
        self.active_jobs = []
        self.apollo_net = ApolloNet()
        self.solver = None
    def add(self, job, states=None):
        if states is not None:
            job.states = lambda: states
        self.jobs.append(job)
    def forward(self, state, input_data={}):
        for job in self.jobs:
            if state in job.states():
                if job.is_data_job():
                    job.run(self.apollo_net, input_data)
                else:
                    job.run(self.apollo_net)
                self.active_jobs.append(job)
    def backward(self):
        self.apollo_net.backward()
    @property
    def params(self):
        return self.apollo_net.params
    @property
    def tops(self):
        return self.apollo_net.tops
    @property
    def layers(self):
        return self.apollo_net.layers
    def load(self, weight_file):
        self.apollo_net.load(weight_file)
    def fit(self, train_data, val_data=None):
        if self.solver is None:
            raise AttributeError("Solver must be set in order to train.")
        self.solver.solve(train_data, val_data)
    def predict(self, data, tops=[], batch_size=1):
        assert not isinstance(tops, str)
        batch = None
        outputs = None
        for instance in data:
            if batch == None:
                batch = {k: [v] for k, v in instance.items()}
            else:
                for k, v in instance.iteritems():
                    batch[k].append(v)
            if len(batch.values()[0]) == batch_size:
                self.apollo_net.clear_forward()
                self.forward('deploy', batch)
                if outputs is None:
                    outputs = {k: np.copy(self.tops[k].data) for k in tops}
                else:
                    for k in tops:
                        outputs[k] = np.vstack(
                                (outputs[k], np.copy(self.tops[k].data)))
                batch = None
        return outputs

    def save(self, weight_file):
        self.apollo_net.save(weight_file)
    def draw_to_file(self, filename, rankdir='LR', require_nonempty=True):
        self.apollo_net.draw_to_file(self, filename, rankdir='LR', require_nonempty=True)
