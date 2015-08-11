class DisplayLogger(object):
    def __init__(self, display_interval):
        self.display_interval = display_interval
    def log(self, idx, meta_data):
        if idx % self.display_interval == 0:
            print 'train loss: ', meta_data['train_loss'][-1]
