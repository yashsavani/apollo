from time import strftime

class TrainLogger(object):
    def __init__(self, display_interval):
        self.display_interval = display_interval
    def log(self, idx, meta_data):
        if idx % self.display_interval == 0:
            print "%s - Iteration %4d - Train Loss: %g" % (strftime("%Y-%m-%d %H:%M:%S"), idx, meta_data['train_loss'][-1])

class ValLogger(object):
    def __init__(self, display_interval):
        self.display_interval = display_interval
    def log(self, idx, meta_data):
        if idx % self.display_interval == 0:
            print "%s - Iteration %4d - Validation Loss: %g" % (strftime("%Y-%m-%d %H:%M:%S"), idx, meta_data['val_loss'][-1])
