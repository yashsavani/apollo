from apollo import ApolloNet

class Net:
    def __init__(self):
        self.jobs = []
        self.active_jobs = []
        self.apollo_net = ApolloNet()
    def add(self, job):
        self.jobs.append(job)
    def forward(self, state, input_data={}):
        loss = 0.
        for job in self.jobs:
            if state in job.states():
                loss += job.forward(self.apollo_net, input_data)
                self.active_jobs.append(job)
        return loss
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
    def save(self, weight_file):
        self.apollo_net.save(weight_file)
    def draw_to_file(self, filename, rankdir='LR', require_nonempty=True):
        self.apollo_net.draw_to_file(self, filename, rankdir='LR', require_nonempty=True)
