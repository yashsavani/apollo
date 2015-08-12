from time import strftime
import os

class TrainLogger(object):
    def __init__(self, display_interval, log_file="/tmp/apollo_log.txt"):
        self.display_interval = display_interval
        self.log_file = log_file
        os.system("touch %s" % self.log_file)
    def log(self, idx, meta_data):
        if idx % self.display_interval == 0:
            log_line = ""
            try:
                log_line = "%s - Iteration %4d - Train Loss: %g" % \
                    (strftime("%Y-%m-%d %H:%M:%S"), idx, meta_data['train_loss'][-1])
            except:
                log_line = "Skipping training log: Unknown Error"

            try:
                with open(self.log_file, 'ab+') as lfile:
                    lfile.write("%s\n" % log_line)
            except IOError:
                print "Trainer Logger Error: %s does not exist." % self.log_file
            except Exception as e:
                print e
            print log_line

class ValLogger(object):
    def __init__(self, display_interval, log_file="/tmp/apollo_log.txt"):
        self.display_interval = display_interval
        self.log_file = log_file
        os.system("touch %s" % self.log_file)
    def log(self, idx, meta_data):
        if idx % self.display_interval == 0:
            try:
                log_line = "%s - Iteration %4d - Validation Loss: %g" % \
                    (strftime("%Y-%m-%d %H:%M:%S"), idx, meta_data['val_loss'][-1])
            except IndexError:
                log_line = "Skipping validation log: \
You have not provided a Validation Set"
            else:
                log_line =  "Skipping validation log: Unknown Error"

            try:
                with open(self.log_file, 'ab+') as lfile:
                    lfile.write("%s\n" % log_line)
            except IOError:
                print "Validation Logger Error: %s does not exist." % self.log_file
            except Exception as e:
                print e
            print log_line
