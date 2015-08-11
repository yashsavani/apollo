class Job(object):
    def forward(self, apollo_net, input_data):
        raise NotImplementedError('Not Implemented')
    def backward(self, apollo_net):
        pass
    def states(self):
        return ['train', 'val', 'deploy']
