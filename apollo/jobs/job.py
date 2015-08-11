class Job(object):
    def run(self, apollo_net):
        raise NotImplementedError('Not Implemented')
    def is_data_job(self):
        return False
    def states(self):
        return ['train', 'val', 'deploy']
