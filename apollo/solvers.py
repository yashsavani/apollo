import apollo
import sys
import random
import types

def make_generator(data):
    if isinstance(data, list):
        while True:
            random.shuffle(data)
            for x in data:
                yield x
    elif isinstance(data, types.GeneratorType):
        for x in data:
            yield x
    else:
        raise ValueError("data must be a list or a generator of dictionaries")

class SGD(object):
    def __init__(self, net, base_lr, momentum=0.0, clip_gradients=-1.,
                 weight_decay=0., max_iter=sys.maxint, start_iter=0,
                 val_interval=1000, val_iter=100, random_seed=91,
                 gamma_lr=1.0, gamma_stepsize=sys.maxint, loggers=[]):
        self.net = net
        self.base_lr = base_lr
        self.momentum = momentum
        self.clip_gradients = clip_gradients
        self.weight_decay = weight_decay
        self.max_iter = max_iter
        self.start_iter = start_iter
        self.val_interval = val_interval
        self.val_iter = val_iter
        self.random_seed = random_seed
        self.gamma_lr = gamma_lr
        self.gamma_stepsize = gamma_stepsize
        self.loggers = loggers

    def fit(self, train_data, val_data=None):
        apollo.set_random_seed(self.random_seed)
        train_data = make_generator(train_data)
        if val_data is not None:
            val_data = make_generator(val_data)
        train_loss = []
        val_loss = []

        for idx in xrange(self.start_iter, self.max_iter):
            if idx % self.val_interval == 0 and val_data is not None:
                for val_idx in xrange(self.val_iter):
                    val_loss.append(self.net.forward("val", val_data.next()))
                    self.net.apollo_net.clear_forward()
            train_loss.append(self.net.forward("train", train_data.next()))
            self.net.apollo_net.backward()
            lr = self.base_lr * self.gamma_lr**(idx // self.gamma_stepsize)
            self.net.apollo_net.update(lr, momentum=self.momentum,
                clip_gradients=self.clip_gradients, weight_decay=self.weight_decay)
            for logger in self.loggers:
                logger.log(idx, {"train_loss": train_loss, "val_loss": val_loss, "net": self.net, "start_iter": self.start_iter})
