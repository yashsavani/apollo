import logging

class StepTraining:
    def __init__(net, batch_iter, iter = 0):
        """
        Implements step learning rule.
        
        Parameters 
        ----------
        batch_iter : iter of dictionaries
            Yields batches of training data.  

        iter: int, optional, default 0
            Initial number of training iterations,
            useful for resuming training.  
        """
        self.net = net
        self.batch_iter
        self.iter = 0

    def set_iter(self, iter_max, base_lr = 0.01, weight_decay = 0.0005, 
            momentum = 0.9, display_iter = 100, gamma = 0.0001, power = 0.75):
        """
        Advances iteration count to value specified.  Takes learning hyperparameters
        as input.  Returns list of loss values.  
        
        Parameters 
        ----------
        iter_max: int 
            Target number of training iterations.  

        display_iter: int
            Frequency of logging loss information to standard ouput. 

        base_lr: float

        weight_decay: float

        momentum: float

        gamma: float

        power: float

        Returns 
        -------
        loss_list : list of doubles
            Values of loss for each training iteration.  
        """
        self.net.phase = "test"
        batch_iter = iter(self.batch_iter)
        loss_list = []
        while self.iter < iter_max:
            try:
                batch = batch_iter.next()
            except StopIteration:
                batch_iter = iter(batches)
                batch = batch_iter.next()
                
            # push data to network
            for key, item in batch.iteritems():
                self.net.forward_layer(NumpyData(name = key, data = item))

            loss = 0
            for layer in self.layer_list:
                if layer.train:
                    loss += self.net.forward_layer(layer)

            self.net.backward()
            lr = base_lr * (1. + gamma * self.iter) ** (-power)
            self.net.update(lr=lr, momentum = momentum, weight_decay = weight_decay)
            if self.iter % display_iter == 0:
                logging.info("Iteration %d loss: %f" %(self.iter, loss))
                logging.info("Iteration %d lr: %f" %(self.iter, lr))
            self.iter += 1
            loss_list.append(loss)

        return loss_list

